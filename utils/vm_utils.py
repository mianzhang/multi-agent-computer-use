from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from filelock import FileLock


logger = logging.getLogger("macu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VMWARE_REGISTRY_FILENAME = ".vmware_vms"
VMWARE_LOCK_FILENAME = ".vmware_lck"


def _vmrun_type_args() -> list[str]:
    if platform.system() in ("Windows", "Linux"):
        return ["-T", "ws"]
    if platform.system() == "Darwin":
        return ["-T", "fusion"]
    raise RuntimeError(f"Unsupported OS for vmrun: {platform.system()}")


def _read_registry_paths(registry_path: str | Path) -> list[str]:
    """Return VM paths from the OSWorld pool registry."""
    path = Path(registry_path)
    if not path.exists():
        return []
    paths: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        vm_path = line.split("|", 1)[0].strip()
        if vm_path:
            paths.append(vm_path)
    return paths


def _mark_all_free(registry_path: str | Path, paths: list[str]) -> None:
    """Rewrite the VM registry so every known path is marked free."""
    if not paths:
        return
    Path(registry_path).write_text(
        "".join(f"{path}|free\n" for path in paths),
        encoding="utf-8",
    )


def revert_vm_pool(
    registry_path: str | Path,
    snapshot_name: str,
    vm_paths_override: Optional[list[str]] = None,
    timeout: int = 600,
    log: Optional[logging.Logger] = None,
    cwd: Optional[str | Path] = None,
) -> tuple[int, int]:
    """Revert every VM in the pool to ``snapshot_name``.

    Returns ``(num_reverted, num_failed)``. Relative VM paths are interpreted
    relative to ``cwd`` to match OSWorld's registry behavior.
    """
    log = log or logger
    paths = list(vm_paths_override) if vm_paths_override else _read_registry_paths(registry_path)
    if not paths:
        log.warning("No VMs found in registry %s; nothing to reset.", registry_path)
        return 0, 0

    log.info("Reverting %d VM(s) to snapshot %r in parallel...", len(paths), snapshot_name)
    vmrun_args = _vmrun_type_args()
    popen_cwd = str(cwd) if cwd is not None else None

    procs: list[tuple[str, subprocess.Popen]] = []
    for vm_path in paths:
        cmd = ["vmrun", *vmrun_args, "revertToSnapshot", vm_path, snapshot_name]
        log.info("[%s] launching: %s", vm_path, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=popen_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        procs.append((vm_path, proc))

    deadline = time.time() + timeout
    num_ok = 0
    num_fail = 0
    for vm_path, proc in procs:
        remaining = max(1, int(deadline - time.time()))
        try:
            stdout, _ = proc.communicate(timeout=remaining)
            if proc.returncode == 0:
                log.info("[%s] reverted OK", vm_path)
                num_ok += 1
            else:
                log.error(
                    "[%s] vmrun returned %d:\n%s",
                    vm_path,
                    proc.returncode,
                    (stdout or "").strip(),
                )
                num_fail += 1
        except subprocess.TimeoutExpired:
            log.error("[%s] vmrun timed out after %ds, killing...", vm_path, timeout)
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] kill failed: %s", vm_path, exc)
            num_fail += 1
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] revert raised: %s", vm_path, exc)
            num_fail += 1

    if not vm_paths_override:
        try:
            _mark_all_free(registry_path, paths)
            log.info("Marked %d VM(s) as |free in registry %s", len(paths), registry_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to rewrite registry %s: %s", registry_path, exc)

    return num_ok, num_fail


def take_vm_screenshot(
    osworld_root: str | Path,
    provider_name: str,
    path_to_vm: str,
    task_config: dict,
    output: str | Path,
    wait_seconds: float = 90.0,
    headless: bool = False,
) -> bool:
    """Boot an OSWorld VM, save an initial screenshot, and stop the emulator."""
    osworld_root = str(osworld_root)
    project_root = str(PROJECT_ROOT)
    for import_path in (osworld_root, project_root):
        if import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)

    from osworld.patches import apply as apply_patches

    apply_patches()

    from desktop_env.desktop_env import DesktopEnv

    env = DesktopEnv(
        provider_name=provider_name,
        path_to_vm=path_to_vm,
        headless=headless,
        require_a11y_tree=False,
    )

    try:
        env.reset(task_config=task_config)
        time.sleep(wait_seconds)
        obs = env._get_obs()

        screenshot = obs.get("screenshot")
        if not screenshot:
            logger.warning("No screenshot obtained from initial VM state.")
            return False

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(screenshot)
        logger.info("Screenshot saved to %s", output_path)

        if provider_name == "apptainer":
            try:
                from desktop_env.providers.apptainer.provider import (
                    save_vm_state_for_overlay,
                )

                vmstate = save_vm_state_for_overlay(path_to_vm)
                if vmstate:
                    logger.info("Scout vmstate saved to %s", vmstate)
                else:
                    logger.warning("Scout vmstate save returned None")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scout vmstate save raised: %s", exc)
        return True
    finally:
        try:
            env.provider.stop_emulator(env.path_to_vm)
        except Exception:  # noqa: BLE001
            pass


def _run_vmrun(
    args: argparse.Namespace,
    command: list[str],
    label: str,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """Run a vmrun command and raise on failure."""
    cmd = ["vmrun", *_vmrun_type_args(), *command]
    logger.info("[%s] vmrun: %s", label, " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=args.osworld_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"vmrun failed for {label} (exit={completed.returncode})\n"
            f"  stdout: {(completed.stdout or '').strip()}\n"
            f"  stderr: {(completed.stderr or '').strip()}"
        )
    return completed


def _list_vm_snapshots(
    args: argparse.Namespace, vm_path: str, label: str
) -> list[str]:
    """Return snapshot names reported by vmrun for a VM."""
    completed = _run_vmrun(
        args,
        ["listSnapshots", vm_path],
        label=f"{label}:listSnapshots",
        timeout=60,
    )
    snapshots: list[str] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("Warning:")
            or line.startswith("Total snapshots:")
        ):
            continue
        snapshots.append(line)
    return snapshots


def _create_vm_snapshot(
    args: argparse.Namespace,
    vm_path: str,
    snapshot_name: str,
    label: str,
    timeout: int = 300,
) -> None:
    """Create a snapshot, tolerating vmrun hanging after the snapshot is taken."""
    cmd = ["vmrun", *_vmrun_type_args(), "snapshot", vm_path, snapshot_name]
    logger.info("[%s] vmrun: %s", label, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=args.osworld_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    deadline = time.time() + timeout
    snapshot_visible = False

    try:
        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                stdout, stderr = proc.communicate()
                if ret == 0:
                    return
                raise RuntimeError(
                    f"vmrun snapshot failed for {label} (exit={ret})\n"
                    f"  stdout: {(stdout or '').strip()}\n"
                    f"  stderr: {(stderr or '').strip()}"
                )

            try:
                snapshots = _list_vm_snapshots(args, vm_path, label)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[%s] listSnapshots while creating snapshot failed: %s", label, exc)
                snapshots = []

            if snapshot_name in snapshots:
                snapshot_visible = True
                logger.info(
                    "[%s] Snapshot %s is visible; waiting briefly for vmrun client to exit",
                    label, snapshot_name,
                )
                time.sleep(5)
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.communicate(timeout=10)
                        except Exception:  # noqa: BLE001
                            pass
                    logger.info(
                        "[%s] Proceeding after snapshot became visible (vmrun client terminated)",
                        label,
                    )
                else:
                    proc.communicate()
                return

            time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.communicate(timeout=10)
            except Exception:  # noqa: BLE001
                pass

    if snapshot_visible:
        return
    raise RuntimeError(
        f"vmrun snapshot timed out for {label}: snapshot {snapshot_name!r} "
        f"did not become visible within {timeout}s"
    )


def _stop_vm_if_needed(
    args: argparse.Namespace, vm_path: Optional[str], label: str
) -> None:
    """Best-effort hard stop. A no-op if the VM is already powered off."""
    if not vm_path:
        return
    try:
        _run_vmrun(args, ["stop", vm_path, "hard"], label=f"{label}:stop", timeout=120)
    except Exception as exc:  # noqa: BLE001
        logger.info("[%s] stop skipped/non-fatal: %s", label, exc)


def _clone_vm_from_source(
    args: argparse.Namespace,
    source_vm_path: str,
    clone_vmx_path: Path,
    clone_name: str,
    label: str,
) -> None:
    """Full-clone a stopped VM. Source must be powered off."""
    clone_vmx_path.parent.mkdir(parents=True, exist_ok=True)
    _run_vmrun(
        args,
        [
            "clone",
            source_vm_path,
            str(clone_vmx_path),
            "full",
            f"-cloneName={clone_name}",
        ],
        label=f"{label}:clone",
        timeout=900,
    )


def _delete_vm_tree(vm_path: Optional[str], label: str) -> bool:
    """Delete a VM tree or overlay file. Returns True if the path is gone."""
    if not vm_path:
        return True
    resolved = Path(vm_path).resolve()
    if resolved.suffix == ".qcow2":
        if resolved.is_file():
            try:
                resolved.unlink()
                # Delete companion sidecars too. The .vmstate file is a ~4GB
                # RAM snapshot; not deleting it leaks ~4GB per cleaned overlay
                # and fills the disk over a batch run. Safe to delete here:
                # cleanup runs at task teardown in leaf-first topological order
                # (an overlay is only removed once all overlays backing onto it
                # are gone), so no init_from successor can still need this
                # overlay's vmstate by the time the overlay itself is deleted.
                for suffix in (".conn.json", ".vmstate"):
                    sidecar = Path(str(resolved) + suffix)
                    if sidecar.exists():
                        sidecar.unlink()
                logger.info("[%s] Deleted overlay file %s", label, resolved)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Failed to delete overlay file %s: %s", label, resolved, exc)
                return False
        logger.info("[%s] Overlay already removed: %s", label, resolved)
        return True
    vm_dir = resolved.parent
    if not vm_dir.exists():
        return True
    try:
        shutil.rmtree(vm_dir)
        logger.info("[%s] Deleted VM directory %s", label, vm_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Failed to delete VM directory %s: %s", label, vm_dir, exc)
        return False


def _vmware_registry_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    osworld_root = Path(args.osworld_root)
    return osworld_root / VMWARE_REGISTRY_FILENAME, osworld_root / VMWARE_LOCK_FILENAME


def _registry_path_to_absolute(args: argparse.Namespace, registered_vm_path: str) -> Path:
    """Resolve a registry-style VM path to an absolute filesystem path."""
    path = Path(registered_vm_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(args.osworld_root) / path).resolve()


def _orchestrator_alloc_vm(args: argparse.Namespace) -> str:
    """Atomically claim a free VM from the OSWorld VMware pool registry."""
    registry_path, lock_path = _vmware_registry_paths(args)
    pid_str = str(os.getpid())
    with FileLock(str(lock_path), timeout=600):
        if not registry_path.exists():
            raise RuntimeError(
                f"VMware registry {registry_path} does not exist. The pool "
                f"must be initialized (run a CUA subprocess once to create it)."
            )
        lines = registry_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        chosen: Optional[str] = None
        for line in lines:
            if "|" not in line:
                new_lines.append(line)
                continue
            registered_vm_path, state = line.split("|", 1)
            if chosen is None and state.strip() == "free":
                chosen = registered_vm_path
                new_lines.append(f"{registered_vm_path}|{pid_str}")
            else:
                new_lines.append(line)
        if chosen is None:
            raise RuntimeError(
                f"No free VM available in {registry_path}. Wait for an "
                f"existing VM to be released or grow the pool."
            )
        registry_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info("Orchestrator claimed pool VM: %s", chosen)
    return chosen


def _update_vm_registry(
    args: argparse.Namespace, vm_path: str, state: str
) -> None:
    """Set a VM's registry entry to ``state`` (``\"free\"`` or a pid string)."""
    registry_path, lock_path = _vmware_registry_paths(args)
    if not registry_path.exists():
        return
    try:
        target = _registry_path_to_absolute(args, vm_path)
    except Exception:  # noqa: BLE001
        target = Path(vm_path)
    try:
        with FileLock(str(lock_path), timeout=600):
            lines = registry_path.read_text(encoding="utf-8").splitlines()
            new_lines: list[str] = []
            for line in lines:
                if "|" not in line:
                    new_lines.append(line)
                    continue
                registered_vm_path, _ = line.split("|", 1)
                try:
                    matches = _registry_path_to_absolute(args, registered_vm_path) == target
                except Exception:  # noqa: BLE001
                    matches = registered_vm_path == vm_path
                if matches:
                    new_lines.append(f"{registered_vm_path}|{state}")
                else:
                    new_lines.append(line)
            registry_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update VM registry for %s -> %s: %s", vm_path, state, exc)


def _orchestrator_release_vm(args: argparse.Namespace, vm_path: str) -> None:
    _update_vm_registry(args, vm_path, "free")
    logger.info("Orchestrator released pool VM: %s", vm_path)


def _orchestrator_reclaim_vm(args: argparse.Namespace, vm_path: str) -> None:
    """Re-mark an already-allocated pool VM as owned by this orchestrator."""
    pid_str = str(os.getpid())
    _update_vm_registry(args, vm_path, pid_str)
    logger.info("Orchestrator reclaimed pool VM (for dep reuse): %s", vm_path)


def _take_subtask_snapshot(
    args: argparse.Namespace, vm_path: str, sid: str
) -> str:
    """Snapshot a VM right before launching a subtask. Returns the snapshot name."""
    snap_name = f"macu_subtask_{sid}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    try:
        _create_vm_snapshot(
            args,
            vm_path=vm_path,
            snapshot_name=snap_name,
            label=f"{sid}:pre-launch-snapshot",
            timeout=300,
        )
        return snap_name
    except RuntimeError as exc:
        if "state of the virtual machine has not changed" in str(exc):
            existing = _list_vm_snapshots(args, vm_path, label=f"{sid}:list-existing")
            if existing:
                reused = existing[-1]
                logger.info(
                    "[%s] VM already has snapshot %s; reusing it instead of creating a new one",
                    sid, reused,
                )
                return reused
        raise


_APPTAINER_VMS_DIR = os.environ.get(
    "OSWORLD_APPTAINER_VMS_DIR",
    os.path.abspath("osworld_vms"),
)
_APPTAINER_OVERLAY_DIR = os.environ.get(
    "OSWORLD_APPTAINER_OVERLAY_DIR",
    os.path.abspath("apptainer/overlays"),
)
_APPTAINER_SIF_PATH = os.environ.get(
    "OSWORLD_APPTAINER_SIF",
    os.path.abspath("apptainer/images/osworld-docker.sif"),
)
_APPTAINER_BASE_IMAGE: Optional[str] = None


def _apptainer_resolve_base_image() -> str:
    global _APPTAINER_BASE_IMAGE
    if _APPTAINER_BASE_IMAGE is not None:
        return _APPTAINER_BASE_IMAGE
    base = os.path.join(_APPTAINER_VMS_DIR, "Ubuntu.qcow2")
    if not os.path.exists(base):
        raise FileNotFoundError(f"Apptainer base image not found: {base}")
    _APPTAINER_BASE_IMAGE = os.path.abspath(base)
    return _APPTAINER_BASE_IMAGE


def _apptainer_create_overlay_from(backing: str, label: str = "") -> str:
    """Create a qcow2 overlay backed by *backing*."""
    backing_abs = os.path.abspath(backing)
    os.makedirs(_APPTAINER_OVERLAY_DIR, exist_ok=True)
    overlay_name = f"overlay_{int(time.time())}_{uuid.uuid4().hex[:8]}.qcow2"
    overlay_path = os.path.join(_APPTAINER_OVERLAY_DIR, overlay_name)
    bind_dirs = sorted({_APPTAINER_OVERLAY_DIR, os.path.dirname(backing_abs)})
    bind_args = []
    for directory in bind_dirs:
        bind_args += ["--bind", f"{directory}:{directory}"]
    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                *bind_args,
                _APPTAINER_SIF_PATH,
                "qemu-img",
                "create",
                "-f",
                "qcow2",
                "-b",
                backing_abs,
                "-F",
                "qcow2",
                overlay_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to create apptainer overlay from {backing} for [{label}]: "
            f"{exc.stderr.decode(errors='ignore')}"
        )
    logger.info("Created apptainer overlay %s (backing %s) for [%s]", overlay_path, backing_abs, label)
    return overlay_path


def _apptainer_create_overlay(label: str = "") -> str:
    """Create a fresh qcow2 overlay backed by the shared base image."""
    return _apptainer_create_overlay_from(_apptainer_resolve_base_image(), label)


def _apptainer_save_vm_state(overlay_path: str, label: str) -> Optional[str]:
    """Save full VM state (CPU + RAM + devices) via QMP migration."""
    if not overlay_path:
        return None
    overlay_abs = os.path.abspath(overlay_path)
    sidecar_path = overlay_abs + ".conn.json"
    if not os.path.exists(sidecar_path):
        logger.warning("[%s] No sidecar for %s, cannot save VM state", label, overlay_abs)
        return None

    import json as _json_mod

    with open(sidecar_path) as fh:
        data = _json_mod.load(fh)

    instance_dir = data.get("instance_dir")
    qmp_sock = data.get("qmp_sock")
    if not qmp_sock and instance_dir:
        qmp_sock = os.path.join(instance_dir, "qmp.sock")
    if not qmp_sock or not os.path.exists(qmp_sock):
        logger.warning("[%s] QMP socket not found for %s (tried %s)", label, overlay_abs, qmp_sock)
        return None

    vmstate_path = overlay_abs + ".vmstate"

    try:
        import socket as _socket

        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect(qmp_sock)
        buf = b""

        def _qmp_recv():
            nonlocal buf
            while True:
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    msg = _json_mod.loads(line)
                    if "event" not in msg:
                        return msg
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("QMP closed")
                buf += chunk

        def _qmp_exec(cmd, **kwargs):
            payload = {"execute": cmd}
            if kwargs:
                payload["arguments"] = kwargs
            sock.sendall(_json_mod.dumps(payload).encode() + b"\n")
            return _qmp_recv()

        _qmp_recv()
        _qmp_exec("qmp_capabilities")
        _qmp_exec("stop")
        logger.info("[%s] VM paused; saving state to %s", label, vmstate_path)
        _qmp_exec("migrate", uri=f"exec:cat > {vmstate_path}")

        for _ in range(240):
            resp = _qmp_exec("query-migrate")
            status = resp.get("return", {}).get("status")
            if status == "completed":
                logger.info("[%s] VM state saved to %s", label, vmstate_path)
                sock.close()
                return vmstate_path
            if status == "failed":
                err = resp.get("return", {}).get("error-desc", "unknown")
                logger.error("[%s] VM state migration failed: %s", label, err)
                sock.close()
                return None
            time.sleep(0.5)

        logger.error("[%s] VM state migration timed out", label)
        sock.close()
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Failed to save VM state for %s: %s", label, overlay_path, exc)
        return None


def _apptainer_kill_qemu_for_overlay(overlay_path: str, label: str) -> None:
    """Kill any qemu process holding *overlay_path* open."""
    if not overlay_path:
        return
    overlay_abs = os.path.abspath(overlay_path)
    try:
        result = subprocess.run(
            ["fuser", overlay_abs],
            capture_output=True,
            timeout=5,
        )
        pids = result.stdout.decode().split()
        killed_any = False
        for pid_str in pids:
            pid_str = pid_str.strip()
            if not pid_str.isdigit():
                continue
            pid_int = int(pid_str)
            try:
                cmdline = Path(f"/proc/{pid_int}/cmdline").read_bytes()
                if b"qemu" not in cmdline:
                    logger.debug("[%s] PID %d holds overlay but is not qemu, skipping", label, pid_int)
                    continue
            except OSError:
                continue
            logger.info(
                "[%s] Stopping qemu process %d for overlay %s",
                label, pid_int, overlay_abs,
            )
            try:
                os.kill(pid_int, 15)
                for _ in range(10):
                    time.sleep(0.5)
                    os.kill(pid_int, 0)
                os.kill(pid_int, 9)
            except OSError:
                pass
            killed_any = True
        if not killed_any:
            logger.info("[%s] No qemu process found holding overlay %s", label, overlay_abs)
        else:
            logger.info("[%s] Qemu killed; overlay %s is now free", label, overlay_abs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Failed to kill qemu for overlay %s: %s", label, overlay_abs, exc)


def _take_initial_vm_screenshot(
    canonical_config: dict,
    task_id: str,
    args: argparse.Namespace,
    cua_args: list[str],
    run_dir: Path,
) -> tuple[Optional[Path], Optional[str]]:
    """Boot a scout VM with the canonical OSWorld config and take a screenshot."""
    provider_name = "vmware"
    headless = False
    wait_seconds = 90.0
    for i, arg in enumerate(cua_args):
        if arg == "--provider_name" and i + 1 < len(cua_args):
            provider_name = cua_args[i + 1]
        if arg == "--headless":
            headless = True
        if arg == "--env_ready_wait_seconds" and i + 1 < len(cua_args):
            wait_seconds = float(cua_args[i + 1])
    use_apptainer = provider_name == "apptainer"

    if use_apptainer:
        vm_path = _apptainer_create_overlay(f"scout_{task_id}")
    else:
        vm_path = _orchestrator_alloc_vm(args)

    screenshot_output = run_dir / "initial_screenshot.png"
    logger.info(
        "Taking initial VM screenshot for task %s (vm: %s, wait: %.0fs)...",
        task_id,
        vm_path,
        wait_seconds,
    )
    setup_ok = False
    try:
        setup_ok = take_vm_screenshot(
            osworld_root=args.osworld_root,
            provider_name=provider_name,
            path_to_vm=vm_path,
            task_config=canonical_config,
            output=screenshot_output,
            wait_seconds=wait_seconds,
            headless=headless,
        )
        if setup_ok:
            logger.info("Initial VM screenshot capture succeeded for task %s", task_id)
        else:
            logger.warning("Initial VM screenshot capture failed for task %s", task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Initial VM screenshot capture raised for task %s: %s", task_id, exc)

    reusable_overlay: Optional[str] = None
    if use_apptainer:
        _apptainer_kill_qemu_for_overlay(vm_path, f"scout_{task_id}")
        if setup_ok:
            reusable_overlay = vm_path
            logger.info("Scout overlay kept for reuse: %s", vm_path)
        else:
            logger.warning(
                "Scout setup failed; discarding overlay %s "
                "(first CUA subtask will run its own setup)",
                vm_path,
            )
            try:
                os.remove(vm_path)
            except OSError:
                pass
    else:
        try:
            _orchestrator_release_vm(args, vm_path)
        except Exception:  # noqa: BLE001
            pass

    if screenshot_output.exists():
        logger.info("Initial VM screenshot saved: %s", screenshot_output)
        return screenshot_output, reusable_overlay
    logger.warning("No initial VM screenshot produced for task %s", task_id)
    return None, reusable_overlay


def read_vm_info_from_dir(subtask_dir: Path, label: str) -> Optional[dict]:
    """Read a branch/subtask ``vm_info.json`` from the given directory."""
    info_path = subtask_dir / "vm_info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("vm_path"):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read vm_info for [%s]: %s", label, exc)
    return None


def reset_vm_pool(args: argparse.Namespace, run_dir: Optional[Path] = None) -> None:
    """Revert every VM in the OSWorld pool to its snapshot."""
    log_path = (
        (run_dir / "reset_vm_pool.log").resolve()
        if run_dir is not None
        else (Path(args.result_dir) / "reset_vm_pool.log").resolve()
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    reset_logger = logging.getLogger("macu.reset_vm_pool")
    reset_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    reset_logger.addHandler(file_handler)

    registry_path, _ = _vmware_registry_paths(args)
    logger.info(
        "Resetting VM pool to snapshot %r (registry=%s)",
        args.reset_pool_snapshot,
        registry_path,
    )
    try:
        num_ok, num_fail = revert_vm_pool(
            registry_path=registry_path,
            snapshot_name=args.reset_pool_snapshot,
            timeout=args.reset_pool_timeout,
            log=reset_logger,
            cwd=args.osworld_root,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VM pool reset failed: %s (see %s)", exc, log_path)
        return
    finally:
        reset_logger.removeHandler(file_handler)
        file_handler.close()

    if num_fail:
        logger.warning(
            "VM pool reset completed with failures (reverted=%d failed=%d); see %s",
            num_ok,
            num_fail,
            log_path,
        )
    else:
        logger.info("VM pool reset complete.")


__all__ = [
    "VMWARE_LOCK_FILENAME",
    "VMWARE_REGISTRY_FILENAME",
    "_APPTAINER_BASE_IMAGE",
    "_APPTAINER_OVERLAY_DIR",
    "_APPTAINER_SIF_PATH",
    "_APPTAINER_VMS_DIR",
    "_apptainer_create_overlay",
    "_apptainer_create_overlay_from",
    "_apptainer_kill_qemu_for_overlay",
    "_apptainer_resolve_base_image",
    "_apptainer_save_vm_state",
    "_clone_vm_from_source",
    "_create_vm_snapshot",
    "_delete_vm_tree",
    "_mark_all_free",
    "_list_vm_snapshots",
    "_orchestrator_alloc_vm",
    "_orchestrator_reclaim_vm",
    "_orchestrator_release_vm",
    "_read_registry_paths",
    "_registry_path_to_absolute",
    "_run_vmrun",
    "_stop_vm_if_needed",
    "_take_initial_vm_screenshot",
    "_take_subtask_snapshot",
    "_update_vm_registry",
    "_vmrun_type_args",
    "_vmware_registry_paths",
    "read_vm_info_from_dir",
    "reset_vm_pool",
    "revert_vm_pool",
    "take_vm_screenshot",
]
