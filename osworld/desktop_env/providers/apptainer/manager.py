"""VMManager for the apptainer provider.

Mirrors :class:`desktop_env.providers.docker.manager.DockerVMManager`: it
returns the path to a downloaded Ubuntu qcow2 image. Downloads happen on first
use; subsequent calls reuse the cached file.
"""

from __future__ import annotations

import logging
import os
import zipfile
from time import sleep

import requests
from tqdm import tqdm

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.apptainer.ApptainerVMManager")
logger.setLevel(logging.INFO)


UBUNTU_X86_URL = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
)
WINDOWS_X86_URL = (
    "https://huggingface.co/datasets/xlangai/windows_osworld/resolve/main/Windows-10-x64.qcow2.zip"
)

DEFAULT_VMS_DIR = os.environ.get(
    "OSWORLD_APPTAINER_VMS_DIR",
    os.path.abspath("osworld_vms"),
)

# Where per-call writable overlays live. They have to outlive the qemu process
# AND the CUA subprocess, because the OSWorld evaluator is a separate process
# that re-attaches to the same overlay file to score the agent's modifications.
DEFAULT_OVERLAY_DIR = os.environ.get(
    "OSWORLD_APPTAINER_OVERLAY_DIR",
    os.path.abspath("apptainer/overlays"),
)

RETRY_INTERVAL = 5


def _download_vm(vms_dir: str, url: str) -> str:
    os.makedirs(vms_dir, exist_ok=True)
    file_name = url.split("/")[-1]
    dest = os.path.join(vms_dir, file_name)
    logger.info("Downloading VM image %s -> %s", url, dest)
    while True:
        headers = {}
        downloaded_size = 0
        if os.path.exists(dest):
            downloaded_size = os.path.getsize(dest)
            headers["Range"] = f"bytes={downloaded_size}-"
        with requests.get(url, headers=headers, stream=True) as response:
            if response.status_code == 416:
                logger.info("Already fully downloaded.")
                break
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(dest, "ab") as fh, tqdm(
                desc="Progress",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                initial=downloaded_size,
                ascii=True,
            ) as bar:
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        bar.update(fh.write(chunk))
                except (requests.exceptions.RequestException, IOError) as exc:
                    logger.error("Download error: %s, retrying in %ds", exc, RETRY_INTERVAL)
                    sleep(RETRY_INTERVAL)
                    continue
                logger.info("Download complete.")
                break
    if file_name.endswith(".zip"):
        logger.info("Unzipping %s ...", dest)
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(vms_dir)
    return dest


class ApptainerVMManager(VMManager):
    def __init__(self, registry_path: str = ""):
        self.vms_dir = DEFAULT_VMS_DIR
        self.overlay_dir = DEFAULT_OVERLAY_DIR
        os.makedirs(self.overlay_dir, exist_ok=True)

    def add_vm(self, vm_path, **kwargs):
        pass

    def check_and_clean(self, **kwargs):
        pass

    def delete_vm(self, vm_path, region=None, **kwargs):
        pass

    def initialize_registry(self, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return os.path.join(self.vms_dir, "Ubuntu.qcow2")

    def occupy_vm(self, vm_path, pid, region=None, **kwargs):
        pass

    def release_vm(self, vm_path, **kwargs):
        """Return VM to pool without stopping it.

        For the apptainer provider each overlay is single-use, so there is no
        registry to update.  The qemu process keeps running so a subsequent
        ``DesktopEnv`` (e.g. the OSWorld evaluator) can reconnect via the
        sidecar ``{overlay}.conn.json`` that ``ApptainerProvider`` writes.
        """
        pass

    def _resolve_base_image(self, os_type: str) -> str:
        if os_type == "Ubuntu":
            url = UBUNTU_X86_URL
        elif os_type == "Windows":
            url = WINDOWS_X86_URL
        else:
            raise ValueError(f"Unsupported os_type: {os_type}")
        file_name = url.split("/")[-1]
        unpacked = file_name[:-4] if file_name.endswith(".zip") else file_name
        target = os.path.join(self.vms_dir, unpacked)
        if not os.path.exists(target):
            _download_vm(self.vms_dir, url)
        return target

    def _create_overlay(self, base_path: str) -> str:
        """Mint a fresh writable qcow2 overlay backed by ``base_path``.

        Returns the absolute path to the new overlay. The overlay outlives the
        qemu process AND the CUA subprocess so the OSWorld evaluator can
        re-attach to it. Cleanup is the caller's responsibility (the overlay
        directory grows otherwise).
        """
        import subprocess
        import time as _time
        import uuid

        # Try to import the SIF path lazily; we need qemu-img and the apptainer
        # provider's SIF is the only place we know it lives on this host.
        from desktop_env.providers.apptainer.provider import DEFAULT_SIF_PATH

        os.makedirs(self.overlay_dir, exist_ok=True)
        overlay_name = f"overlay_{int(_time.time())}_{uuid.uuid4().hex[:8]}.qcow2"
        overlay_path = os.path.join(self.overlay_dir, overlay_name)
        base_dir = os.path.dirname(os.path.abspath(base_path))
        try:
            subprocess.run(
                [
                    "apptainer", "exec",
                    "--bind", f"{self.overlay_dir}:{self.overlay_dir}",
                    "--bind", f"{base_dir}:{base_dir}",
                    DEFAULT_SIF_PATH,
                    "qemu-img", "create",
                    "-f", "qcow2",
                    "-b", os.path.abspath(base_path),
                    "-F", "qcow2",
                    overlay_path,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create overlay {overlay_path}: "
                f"{exc.stderr.decode(errors='ignore')}"
            )
        logger.info("Created overlay %s (backing %s)", overlay_path, base_path)
        return overlay_path

    def get_vm_path(self, os_type: str = "Ubuntu", region=None, screen_size=(1920, 1080), **kwargs):
        """Return a freshly-minted overlay qcow2 backed by the base image.

        Per call we create a NEW overlay so each VM session has its own
        writable layer. The overlay survives qemu termination so the OSWorld
        evaluator (a separate process) can re-attach to the same on-disk
        state by passing the overlay path back through DesktopEnv's
        ``path_to_vm`` argument.
        """
        base = self._resolve_base_image(os_type)
        return self._create_overlay(base)
