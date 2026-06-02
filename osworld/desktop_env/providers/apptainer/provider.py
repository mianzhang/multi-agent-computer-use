"""Apptainer-based OSWorld provider.

Boots the same Ubuntu qcow2 image used by the docker provider, but launches
``qemu-system-x86_64`` inside an apptainer (Singularity) container so we don't
need docker on the host. The host only needs apptainer + access to ``/dev/kvm``.

Each VM instance is fully isolated: own ports, own qemu process, own pidfile.
"""

from __future__ import annotations

import json as _json
import logging
import os
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import psutil
import requests
from filelock import FileLock

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.apptainer.ApptainerProvider")
logger.setLevel(logging.INFO)


WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 30
DEFAULT_BOOT_TIMEOUT = 120

DEFAULT_SIF_PATH = os.environ.get(
    "OSWORLD_APPTAINER_SIF",
    os.path.abspath("apptainer/images/osworld-docker.sif"),
)
DEFAULT_RUN_DIR = os.environ.get(
    "OSWORLD_APPTAINER_RUN_DIR",
    os.path.abspath("apptainer/run"),
)


class _QMP:
    """Minimal QMP (QEMU Machine Protocol) client."""

    def __init__(self, sock_path: str, timeout: float = 60):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(sock_path)
        self._buf = b""
        self._recv()  # greeting
        self.execute("qmp_capabilities")

    def _recv(self):
        """Receive one non-event QMP message."""
        while True:
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                msg = _json.loads(line)
                if "event" not in msg:
                    return msg
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("QMP connection closed")
            self._buf += chunk

    def execute(self, cmd: str, **kwargs):
        """Send a QMP execute command and return the response."""
        data: dict = {"execute": cmd}
        if kwargs:
            data["arguments"] = kwargs
        self._sock.sendall(_json.dumps(data).encode() + b"\n")
        return self._recv()

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PortAllocationError(Exception):
    pass


class ApptainerProvider(Provider):
    """Run an OSWorld VM via apptainer + qemu."""

    def __init__(self, region: Optional[str] = None):
        super().__init__(region=region)
        self.sif_path: str = DEFAULT_SIF_PATH
        self.run_dir: str = DEFAULT_RUN_DIR
        os.makedirs(self.run_dir, exist_ok=True)

        self.server_port: Optional[int] = None
        self.chromium_port: Optional[int] = None
        self.vnc_port: Optional[int] = None
        self.vlc_port: Optional[int] = None

        self.process: Optional[subprocess.Popen] = None
        self.instance_dir: Optional[str] = None

        self.ram_size = os.environ.get("OSWORLD_APPTAINER_RAM", "4G")
        self.cpu_cores = int(os.environ.get("OSWORLD_APPTAINER_CPUS", "4"))

        self.lock_file = Path(self.run_dir) / "port_allocation.lck"

    # ------------------------------------------------------------------ utils
    def _get_used_ports(self) -> set:
        used = set()
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr:
                    used.add(conn.laddr.port)
        except psutil.AccessDenied:
            # Fall back: just bind-test ports below
            pass
        return used

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _get_available_port(self, start_port: int, used: set) -> int:
        port = start_port
        while port < 65500:
            if port not in used and self._is_port_free(port):
                used.add(port)
                return port
            port += 1
        raise PortAllocationError(f"No available ports starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = DEFAULT_BOOT_TIMEOUT) -> None:
        start = time.time()
        url = f"http://localhost:{self.server_port}/screenshot"
        while time.time() - start < timeout:
            if self.process is not None and self.process.poll() is not None:
                # qemu died
                raise RuntimeError(
                    f"qemu process exited with code {self.process.returncode} "
                    f"before VM became ready (see {self.instance_dir}/qemu.log)"
                )
            try:
                resp = requests.get(url, timeout=(5, 5))
                if resp.status_code == 200:
                    logger.info("VM is ready (server on %s).", self.server_port)
                    return
            except Exception:
                pass
            logger.info("Waiting for VM to be ready on port %s...", self.server_port)
            time.sleep(RETRY_INTERVAL)
        raise TimeoutError(
            f"VM failed to become ready within {timeout}s (see {self.instance_dir}/qemu.log)"
        )

    # -------------------------------------------------------- sidecar helpers
    @staticmethod
    def _sidecar_path(path_to_vm: str) -> str:
        """Return the path of the connection-info sidecar for *path_to_vm*."""
        return os.path.abspath(path_to_vm) + ".conn.json"

    def _write_sidecar(self, path_to_vm: str) -> None:
        """Persist port/instance info so a later process can reconnect."""
        data = {
            "server_port": self.server_port,
            "chromium_port": self.chromium_port,
            "vnc_port": self.vnc_port,
            "vlc_port": self.vlc_port,
            "instance_dir": self.instance_dir,
            "pid": self.process.pid if self.process else None,
            "qmp_sock": os.path.join(self.instance_dir, "qmp.sock") if self.instance_dir else None,
        }
        path = self._sidecar_path(path_to_vm)
        with open(path, "w") as f:
            _json.dump(data, f)
        logger.info("Wrote VM sidecar %s", path)

    def _try_reconnect(self, path_to_vm: str) -> bool:
        """Try to reuse an already-running VM via its sidecar file.

        Returns True (and populates self.*) if the VM is still reachable,
        False otherwise.
        """
        sidecar = self._sidecar_path(path_to_vm)
        if not os.path.exists(sidecar):
            return False
        try:
            with open(sidecar) as f:
                data = _json.load(f)
            port = data["server_port"]
            # Quick health-check: can we reach the in-VM HTTP server?
            resp = requests.get(f"http://localhost:{port}/screenshot", timeout=(5, 5))
            if resp.status_code != 200:
                logger.info("Sidecar exists but VM not reachable (status %s); will relaunch.", resp.status_code)
                return False
        except Exception as exc:
            logger.info("Sidecar exists but VM not reachable (%s); will relaunch.", exc)
            try:
                os.remove(sidecar)
            except OSError:
                pass
            return False

        # VM is alive — adopt its connection info.
        self.server_port = data["server_port"]
        self.chromium_port = data["chromium_port"]
        self.vnc_port = data["vnc_port"]
        self.vlc_port = data["vlc_port"]
        self.instance_dir = data.get("instance_dir")
        self._adopted_pid = data.get("pid")
        logger.info(
            "Reconnected to existing VM on ports %s/%s/%s/%s pid=%s (sidecar %s)",
            self.server_port, self.chromium_port, self.vnc_port, self.vlc_port,
            self._adopted_pid, sidecar,
        )
        return True

    def _remove_sidecar(self, path_to_vm: str) -> None:
        sidecar = self._sidecar_path(path_to_vm)
        try:
            os.remove(sidecar)
            logger.info("Removed VM sidecar %s", sidecar)
        except FileNotFoundError:
            pass

    # --------------------------------------------------------- Provider impl
    MAX_LAUNCH_RETRIES = 3

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        # If a previous process (e.g. the CUA agent) left the VM running,
        # reconnect to it instead of cold-booting a new qemu.
        if self._try_reconnect(path_to_vm):
            return

        if not os.path.exists(self.sif_path):
            raise FileNotFoundError(
                f"Apptainer SIF not found at {self.sif_path}. Set OSWORLD_APPTAINER_SIF "
                f"to override."
            )

        if not os.path.exists(path_to_vm):
            raise FileNotFoundError(f"VM disk not found: {path_to_vm}")

        # Check for a saved VM state in the overlay's backing chain.
        # If found, we restore via incoming migration instead of cold-booting.
        vmstate = self._find_vmstate_in_chain(path_to_vm)
        if vmstate:
            logger.info(
                "Found VM state file %s; will restore via incoming migration",
                vmstate,
            )

        # Retry loop: port allocation is inherently racy when multiple VMs
        # start on the same host.  If qemu fails to bind (e.g. port stolen
        # between our check and its bind()), we pick fresh ports and retry.
        # We use random base ports (server/etc. from 10000-60000, VNC display
        # from 0-999 => ports 5900-6899) so that concurrent VMs on the same
        # host virtually never collide -- the FileLock + bind-test is kept as
        # a safety net.
        for attempt in range(1, self.MAX_LAUNCH_RETRIES + 1):
            lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)
            with lock:
                used = self._get_used_ports()
                base = random.randint(10000, 60000)
                self.server_port = self._get_available_port(base, used)
                self.chromium_port = self._get_available_port(base + 1, used)
                self.vnc_port = self._get_available_port(base + 2, used)
                self.vlc_port = self._get_available_port(base + 3, used)
                vnc_display_base = random.randint(0, 999)
                qemu_vnc_display = self._get_available_port(5900 + vnc_display_base, used) - 5900

                instance_id = f"vm_{self.server_port}_{int(time.time())}"
                self.instance_dir = os.path.join(self.run_dir, instance_id)
                os.makedirs(self.instance_dir, exist_ok=True)

                self._launch_qemu(path_to_vm, qemu_vnc_display,
                                  incoming_state=vmstate)

            try:
                if vmstate:
                    self._resume_incoming_migration()
                self._wait_for_vm_ready()
                break  # success
            except Exception as exc:
                self._kill_process()
                if vmstate and attempt == 1:
                    logger.warning(
                        "Incoming migration failed (%s); retrying with cold boot...",
                        exc,
                    )
                    vmstate = None
                    continue
                if attempt < self.MAX_LAUNCH_RETRIES:
                    logger.warning(
                        "VM launch attempt %d/%d failed (%s); retrying with fresh ports...",
                        attempt, self.MAX_LAUNCH_RETRIES, exc,
                    )
                    time.sleep(WAIT_TIME)
                else:
                    raise

        self._write_sidecar(path_to_vm)

    def _overlay_backing_dir(self, overlay_path: str) -> Optional[str]:
        """Return the directory containing the overlay's backing image, or
        None if the overlay has no backing file."""
        try:
            res = subprocess.run(
                [
                    "apptainer", "exec",
                    "--bind", f"{os.path.dirname(os.path.abspath(overlay_path))}:{os.path.dirname(os.path.abspath(overlay_path))}",
                    self.sif_path,
                    "qemu-img", "info", "--output=json", os.path.abspath(overlay_path),
                ],
                check=True,
                capture_output=True,
            )
            import json as _json
            info = _json.loads(res.stdout.decode())
            backing = info.get("full-backing-filename") or info.get("backing-filename")
            if not backing:
                return None
            if not os.path.isabs(backing):
                backing = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(overlay_path)), backing)
                )
            return os.path.dirname(backing)
        except Exception:
            return None

    def _has_backing_file(self, qcow2_path: str) -> bool:
        """Return True if qcow2_path is a qcow2 with a backing file (i.e. is
        already an overlay we can boot directly)."""
        return self._get_backing_file(qcow2_path) is not None

    def _get_backing_file(self, qcow2_path: str) -> Optional[str]:
        """Return the full path of a qcow2's backing file, or None."""
        qcow2_abs = os.path.abspath(qcow2_path)
        qcow2_dir = os.path.dirname(qcow2_abs)
        try:
            res = subprocess.run(
                [
                    "apptainer", "exec",
                    "--bind", f"{qcow2_dir}:{qcow2_dir}",
                    self.sif_path,
                    "qemu-img", "info", "--output=json", qcow2_abs,
                ],
                check=True,
                capture_output=True,
            )
            info = _json.loads(res.stdout.decode())
            backing = info.get("full-backing-filename") or info.get("backing-filename")
            if not backing:
                return None
            if not os.path.isabs(backing):
                backing = os.path.normpath(os.path.join(qcow2_dir, backing))
            return backing
        except Exception as exc:
            logger.warning("qemu-img info failed for %s: %s", qcow2_path, exc)
            return None

    def _find_vmstate_in_chain(self, overlay_path: str) -> Optional[str]:
        """Walk the qcow2 backing chain looking for a .vmstate file."""
        path = os.path.abspath(overlay_path)
        for _ in range(5):
            vmstate = path + ".vmstate"
            if os.path.exists(vmstate):
                return vmstate
            backing = self._get_backing_file(path)
            if not backing:
                break
            path = backing
        return None

    def _resume_incoming_migration(self, timeout: int = 120) -> None:
        """After launching with ``-incoming``, wait for state load and resume."""
        qmp_sock = os.path.join(self.instance_dir, "qmp.sock")

        # Wait for QMP socket to appear
        deadline = time.time() + 30
        while time.time() < deadline:
            if os.path.exists(qmp_sock):
                break
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"qemu exited with {self.process.returncode} "
                    f"before QMP socket appeared"
                )
            time.sleep(0.5)
        else:
            raise TimeoutError("QMP socket never appeared")

        time.sleep(1)  # let socket become ready for connections

        with _QMP(qmp_sock, timeout=timeout) as qmp:
            deadline = time.time() + timeout
            while time.time() < deadline:
                resp = qmp.execute("query-status")
                status = resp.get("return", {}).get("status")
                if status in ("paused", "prelaunch", "postmigrate"):
                    break
                if status == "inmigrate":
                    time.sleep(0.5)
                    continue
                logger.warning(
                    "Unexpected VM status during incoming migration: %s", status
                )
                time.sleep(0.5)
            else:
                raise TimeoutError("Incoming migration did not complete")

            qmp.execute("cont")
            logger.info("VM resumed after incoming migration")

    def _launch_qemu(self, path_to_vm: str, vnc_display: int,
                     incoming_state: Optional[str] = None) -> None:
        # Build qemu argv. We use user-mode networking with hostfwd so we don't
        # need NET_ADMIN.
        hostfwd_parts = [
            f"hostfwd=tcp:127.0.0.1:{self.server_port}-:5000",
            f"hostfwd=tcp:127.0.0.1:{self.chromium_port}-:9222",
            f"hostfwd=tcp:127.0.0.1:{self.vlc_port}-:8080",
        ]
        netdev = "user,id=net0," + ",".join(hostfwd_parts)

        # Per-instance writable copy of the OVMF UEFI firmware (split CODE/VARS).
        # The OSWorld Ubuntu qcow2 was built to boot under UEFI (q35 + edk2);
        # using SeaBIOS or skipping UEFI leaves qemu sitting at the firmware
        # boot menu and never reaches the in-VM agent server.
        code_dst = os.path.join(self.instance_dir, "OVMF_CODE.fd")
        vars_dst = os.path.join(self.instance_dir, "OVMF_VARS.fd")

        # If the caller handed us a qcow2 that already has a backing file, it
        # is already an overlay (most likely created by ApptainerVMManager.
        # get_vm_path()) and we boot it in place so writes accumulate on the
        # same on-disk file. Otherwise we treat path_to_vm as a base image and
        # wrap it in a fresh per-instance overlay (back-compat path for callers
        # that pass the read-only Ubuntu.qcow2 directly).
        path_to_vm_abs = os.path.abspath(path_to_vm)
        if self._has_backing_file(path_to_vm_abs):
            self.boot_overlay = path_to_vm_abs
            logger.info("Reusing existing overlay %s in-place", self.boot_overlay)
            create_overlay_cmd = ""
        else:
            self.boot_overlay = os.path.join(self.instance_dir, "boot.qcow2")
            logger.info(
                "Creating per-instance overlay %s on top of base %s",
                self.boot_overlay, path_to_vm_abs,
            )
            create_overlay_cmd = (
                f" && qemu-img create -f qcow2 -b {path_to_vm_abs} "
                f"-F qcow2 {self.boot_overlay}"
            )

        overlay_dir = os.path.dirname(self.boot_overlay)
        try:
            subprocess.run(
                [
                    "apptainer", "exec",
                    "--bind", f"{self.instance_dir}:{self.instance_dir}",
                    "--bind", f"{os.path.dirname(path_to_vm_abs)}:{os.path.dirname(path_to_vm_abs)}",
                    "--bind", f"{overlay_dir}:{overlay_dir}",
                    self.sif_path,
                    "bash", "-c",
                    (
                        f"cp /usr/share/OVMF/OVMF_CODE_4M.fd {code_dst} && "
                        f"cp /usr/share/OVMF/OVMF_VARS_4M.fd {vars_dst} && "
                        f"chmod 644 {code_dst} {vars_dst}"
                        + create_overlay_cmd
                    ),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to prepare UEFI firmware / boot overlay: {exc.stderr.decode(errors='ignore')}"
            )

        qemu_argv = [
            "qemu-system-x86_64",
            "-name", "osworld-apptainer",
            "-machine",
            "type=q35,smm=off,graphics=off,vmport=off,dump-guest-core=off,hpet=off,accel=kvm",
            "-cpu", "host,kvm=on,l3-cache=on,+hypervisor,migratable=on",
            "-enable-kvm",
            "-global", "kvm-pit.lost_tick_policy=discard",
            "-smp", f"{self.cpu_cores},sockets=1,cores={self.cpu_cores},threads=1",
            "-m", self.ram_size,
            "-drive", f"if=pflash,format=raw,readonly=on,file={code_dst}",
            "-drive", f"if=pflash,format=raw,file={vars_dst}",
            "-hda", self.boot_overlay,
            "-netdev", netdev,
            "-device", "virtio-net-pci,netdev=net0",
            "-device", "qemu-xhci,id=xhci",
            "-device", "usb-tablet",
            "-display", "none",
            "-vnc", f"127.0.0.1:{vnc_display}",
            "-qmp", f"unix:{self.instance_dir}/qmp.sock,server,nowait",
            "-serial", f"file:{self.instance_dir}/serial.log",
            "-pidfile", f"{self.instance_dir}/qemu.pid",
        ]

        if incoming_state:
            qemu_argv.extend(["-incoming", f"exec:cat < {incoming_state}"])

        # Bind /dev/kvm, the qcow2 directory, the instance dir, AND the
        # overlay's backing file directory (qemu has to be able to follow the
        # qcow2 backing-file pointer when booting an overlay). Apptainer shares
        # host network so hostfwd ports map onto the host directly.
        binds = [
            "/dev/kvm",
            f"{os.path.dirname(path_to_vm_abs)}:{os.path.dirname(path_to_vm_abs)}",
            f"{self.instance_dir}:{self.instance_dir}",
            f"{os.path.dirname(self.boot_overlay)}:{os.path.dirname(self.boot_overlay)}",
        ]
        # If the overlay backs a base image in a different dir, bind that too.
        try:
            base_dir = self._overlay_backing_dir(self.boot_overlay)
            if base_dir:
                binds.append(f"{base_dir}:{base_dir}")
        except Exception as exc:
            logger.warning("Could not determine overlay backing dir: %s", exc)
        # Bind the vmstate file directory for incoming migration.
        if incoming_state:
            incoming_dir = os.path.dirname(os.path.abspath(incoming_state))
            binds.append(f"{incoming_dir}:{incoming_dir}")

        cmd = [
            "apptainer", "exec",
            "--writable-tmpfs",
            "--containall",
            "--cleanenv",
            "--no-home",
        ]
        for b in binds:
            cmd.extend(["--bind", b])
        cmd.append(self.sif_path)
        cmd.extend(qemu_argv)

        log_path = os.path.join(self.instance_dir, "qemu.log")
        logger.info("Launching qemu via apptainer: %s", " ".join(cmd))
        log_fh = open(log_path, "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _kill_process(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception as exc:
            logger.warning("Error while killing qemu: %s", exc)
        finally:
            self.process = None

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return (
            f"127.0.0.1:{self.server_port}:{self.chromium_port}:"
            f"{self.vnc_port}:{self.vlc_port}"
        )

    def save_state(self, path_to_vm: str, snapshot_name: str):
        """Save full VM state via QMP migration (snapshot_name is ignored)."""
        result = save_vm_state_for_overlay(os.path.abspath(path_to_vm))
        if result is None:
            raise RuntimeError(f"Failed to save VM state for {path_to_vm}")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        # Same semantics as the docker provider: stop and restart from clean
        # backing image.
        self.stop_emulator(path_to_vm)
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        self._remove_sidecar(path_to_vm)
        if self.process is not None:
            logger.info("Stopping qemu VM (pid=%s)", self.process.pid)
            self._kill_process()
            time.sleep(WAIT_TIME)
        elif getattr(self, "_adopted_pid", None):
            # We reconnected to a VM started by another process — kill it by PID.
            pid = self._adopted_pid
            logger.info("Stopping adopted qemu VM (pid=%s)", pid)
            try:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(WAIT_TIME)
                try:
                    os.kill(pid, 0)  # check if still alive
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                logger.info("Adopted qemu pid=%s already gone", pid)
            except Exception as exc:
                logger.warning("Error killing adopted qemu pid=%s: %s", pid, exc)
            self._adopted_pid = None
        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None


# ---------------------------------------------------------------------------
# Standalone helper — usable without an ApptainerProvider instance.
# ---------------------------------------------------------------------------

def save_vm_state_for_overlay(overlay_path: str) -> Optional[str]:
    """Save full VM state for the qemu process running on *overlay_path*.

    Reads the sidecar (``.conn.json``) to locate the QMP socket, then uses
    QMP ``migrate`` to dump the VM state (CPU + RAM + devices) to a file.
    Returns the vmstate file path on success, or ``None`` on failure.
    """
    overlay_abs = os.path.abspath(overlay_path)
    sidecar_path = overlay_abs + ".conn.json"
    if not os.path.exists(sidecar_path):
        logger.warning("No sidecar for %s, cannot save VM state", overlay_abs)
        return None

    with open(sidecar_path) as fh:
        data = _json.load(fh)

    instance_dir = data.get("instance_dir")
    qmp_sock = data.get("qmp_sock")
    if not qmp_sock and instance_dir:
        qmp_sock = os.path.join(instance_dir, "qmp.sock")
    if not qmp_sock or not os.path.exists(qmp_sock):
        logger.warning("QMP socket not found for %s (tried %s)", overlay_abs, qmp_sock)
        return None

    vmstate_path = overlay_abs + ".vmstate"

    try:
        with _QMP(qmp_sock, timeout=120) as qmp:
            # Pause the VM so the state is consistent.
            qmp.execute("stop")
            logger.info("VM paused; saving state to %s", vmstate_path)

            # Kick off migration.  The exec: URI pipes through cat inside the
            # (still-running) apptainer container whose bind-mounts include
            # the overlay directory.
            qmp.execute("migrate", uri=f"exec:cat > {vmstate_path}")

            # Poll until migration completes.
            for _ in range(240):  # up to ~120 s
                resp = qmp.execute("query-migrate")
                status = resp.get("return", {}).get("status")
                if status == "completed":
                    logger.info("VM state saved to %s", vmstate_path)
                    return vmstate_path
                if status == "failed":
                    err = resp.get("return", {}).get("error-desc", "unknown")
                    logger.error("VM state migration failed: %s", err)
                    return None
                time.sleep(0.5)

            logger.error("VM state migration timed out after 120 s")
            return None
    except Exception as exc:
        logger.error("Failed to save VM state for %s: %s", overlay_abs, exc)
        return None
