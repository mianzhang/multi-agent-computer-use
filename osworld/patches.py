"""Monkey-patches for OSWorld classes.

The canonical copies of the patched files live in ``osworld/desktop_env/``
for version control. This module applies the changes at runtime after
OSWorld's own modules have been imported via sys.path.

It does two things:

1. Patches ``VMwareVMManager`` / ``DesktopEnv`` with ``release_vm()`` /
   ``release()`` so VMs can be released back into a pool instead of being
   stopped.

2. Registers an out-of-tree ``apptainer`` provider (whose code lives at
   ``osworld/desktop_env/providers/apptainer/`` *inside this repo*, not in
   the OSWorld checkout) so that ``DesktopEnv(provider_name="apptainer")``
   works on hosts without docker/vmware. We use ``importlib`` to load the
   in-repo files, monkey-patch ``create_vm_manager_and_provider`` to know
   about ``"apptainer"``, and wrap ``DesktopEnv.__init__`` with a sentinel
   trick so the upstream provider-name validation set passes without us
   touching OSWorld code.
"""

import importlib.util
import json
import logging
import os
import sys
import time

import requests

logger = logging.getLogger("osworld.patches")

_HERE = os.path.dirname(os.path.abspath(__file__))
_APPTAINER_PKG_DIR = os.path.join(_HERE, "desktop_env", "providers", "apptainer")


# ---------------------------------------------------------------------------
# VM-pool reuse patches
# ---------------------------------------------------------------------------

def _release_vm_impl(self, vm_path):
    logger.info("Releasing VM back to pool: %s", vm_path)
    with self.lock:
        new_lines = []
        with open(self.registry_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                registered_vm_path, _ = line.strip().split('|')
                if registered_vm_path == vm_path:
                    new_lines.append(f'{registered_vm_path}|free\n')
                else:
                    new_lines.append(line)
        with open(self.registry_path, 'w') as file:
            file.writelines(new_lines)


def _release_vm(self, vm_path, lock_needed=True):
    if lock_needed:
        with self.lock:
            _release_vm_impl(self, vm_path)
    else:
        _release_vm_impl(self, vm_path)


def _desktop_env_release(self):
    """Release VM back to pool without stopping it."""
    if hasattr(self.manager, 'release_vm'):
        self.manager.release_vm(self.path_to_vm)
    else:
        self.close()


# ---------------------------------------------------------------------------
# Guest network / resolution helpers
#
# Some OSWorld checkouts provide these methods on DesktopEnv directly.
# scripts/run_cua.py relies on them for mid-task network recovery, so when
# running against an upstream checkout that lacks them we re-inject the helpers
# as bound methods.
# ---------------------------------------------------------------------------


def _execute_shell_on_vm(self, cmd, timeout=30):
    url = f"http://{self.vm_ip}:{self.server_port}/execute"
    payload = json.dumps({"command": cmd, "shell": True})
    return requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=timeout,
    )


def _configure_dns(self):
    """Pin reliable public DNS servers and disable captive portal popup.

    Break the /etc/resolv.conf symlink so DNS is hardcoded and survives
    NetworkManager restarts / interface flaps that would otherwise wipe the
    resolvectl runtime state.
    """
    try:
        self._execute_shell_on_vm(
            "sudo chattr -i /etc/resolv.conf 2>/dev/null; "
            "sudo rm -f /etc/resolv.conf; "
            "printf 'nameserver 8.8.8.8\\nnameserver 1.1.1.1\\nnameserver 8.8.4.4\\n' "
            "| sudo tee /etc/resolv.conf > /dev/null; "
            "sudo chattr +i /etc/resolv.conf 2>/dev/null; "
            "IFACE=$(ip -o route get 8.8.8.8 2>/dev/null | awk '{print $5}' || echo ens33); "
            "resolvectl dns $IFACE 8.8.8.8 1.1.1.1 2>/dev/null; "
            "DISPLAY=:0 gsettings set org.gnome.system.proxy autoconfig-url '' 2>/dev/null; "
            "DISPLAY=:0 dbus-launch gsettings set org.gnome.desktop.privacy connectivity-checking-enabled false 2>/dev/null; "
            "true",
            timeout=15,
        )
        logger.info("Pinned guest DNS to 8.8.8.8 / 1.1.1.1 (immutable resolv.conf)")
    except Exception as e:
        logger.warning("Failed to configure guest DNS: %s", e)


def _restart_vm_networking(self):
    """Actively restart networking inside the VM to force DHCP renewal."""
    logger.info("Restarting guest networking to force DHCP renewal...")
    try:
        self._execute_shell_on_vm(
            "sudo nmcli networking off 2>/dev/null; sleep 2; sudo nmcli networking on 2>/dev/null; "
            "sudo dhclient -r 2>/dev/null; sudo dhclient 2>/dev/null; true",
            timeout=20,
        )
    except Exception as e:
        logger.warning("Guest networking restart failed: %s", e)


def _wait_for_network(self, max_wait=180):
    """Wait for network/DNS to be ready inside the VM after snapshot revert."""
    logger.info("Waiting for guest network connectivity (up to %ds)...", max_wait)
    restarted_networking = False
    for i in range(max_wait // 5):
        if i > 0 and i % 12 == 0 and not restarted_networking:
            restarted_networking = True
            self._restart_vm_networking()
        try:
            resp = self._execute_shell_on_vm(
                "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://www.google.com",
                timeout=10,
            )
            if resp.status_code == 200:
                output = resp.json().get("output", "").strip()
                if output.startswith("2") or output.startswith("3"):
                    logger.info("Guest network is ready (HTTP %s) after %ds.", output, i * 5)
                    self._configure_dns()
                    return
        except Exception:
            pass
        logger.info("Network not ready yet, retrying in 5s... (%d/%d)", i + 1, max_wait // 5)
        time.sleep(5)
    logger.warning("Network readiness check timed out after %ds, proceeding anyway.", max_wait)


def _enforce_screen_resolution(self):
    """Force the VM guest display to the desired resolution using xrandr
    and disable VMware Tools auto-resize so it doesn't revert mid-run.
    """
    desired = f"{self.screen_width}x{self.screen_height}"
    try:
        disable_cmd = (
            "vmware-toolbox-cmd config set resolutionKMS enable FALSE 2>/dev/null; "
            "vmware-toolbox-cmd config set resolutionSet enable FALSE 2>/dev/null; "
            "true"
        )
        self._execute_shell_on_vm(disable_cmd)

        xrandr_cmd = (
            f"export DISPLAY=:0; "
            f"OUTPUT=$(xrandr --current | grep ' connected' | head -1 | awk '{{print $1}}'); "
            f"xrandr --output $OUTPUT --mode {desired}"
        )
        resp = self._execute_shell_on_vm(xrandr_cmd)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("returncode") == 0:
                logger.info("Screen resolution set to %s", desired)
            else:
                logger.warning(
                    "xrandr returned non-zero: %s %s",
                    result.get("output", ""), result.get("error", ""),
                )
        else:
            logger.warning("Failed to set screen resolution (HTTP %d)", resp.status_code)
    except Exception as e:
        logger.warning("Could not enforce screen resolution: %s", e)


def _patch_desktop_env_network_helpers():
    from desktop_env.desktop_env import DesktopEnv

    for name, fn in (
        ("_execute_shell_on_vm", _execute_shell_on_vm),
        ("_configure_dns", _configure_dns),
        ("_restart_vm_networking", _restart_vm_networking),
        ("_wait_for_network", _wait_for_network),
        ("_enforce_screen_resolution", _enforce_screen_resolution),
    ):
        if not hasattr(DesktopEnv, name):
            setattr(DesktopEnv, name, fn)
            logger.info("Patched DesktopEnv with %s()", name)


def _patch_desktop_env_get_obs_for_resolution():
    """Wrap DesktopEnv._get_obs() to detect resolution drift and re-enforce.

    VMware Tools can renegotiate resolution mid-run, dropping the guest from
    the configured width/height to 1280x800. Qwen VL coordinates assume the
    configured resolution, so we re-run xrandr and retake the screenshot.
    """
    from desktop_env.desktop_env import DesktopEnv

    if getattr(DesktopEnv._get_obs, "_resolution_patched", False):
        return

    from PIL import Image
    import io

    _orig_get_obs = DesktopEnv._get_obs

    def _patched_get_obs(self):
        obs = _orig_get_obs(self)
        screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
        if screenshot:
            try:
                img = Image.open(io.BytesIO(screenshot))
                if img.size != (self.screen_width, self.screen_height):
                    logger.warning(
                        "Resolution drifted to %s, enforcing %dx%d",
                        img.size, self.screen_width, self.screen_height,
                    )
                    self._enforce_screen_resolution()
                    time.sleep(1)
                    obs["screenshot"] = self.controller.get_screenshot()
            except Exception as e:
                logger.warning("Resolution-drift check failed (non-fatal): %s", e)
        return obs

    _patched_get_obs._resolution_patched = True
    DesktopEnv._get_obs = _patched_get_obs
    logger.info("Patched DesktopEnv._get_obs() to re-enforce screen resolution on drift")


def _patch_desktop_env_reset_for_network():
    """Wrap DesktopEnv.reset() to call _wait_for_network() after snapshot revert.

    Upstream reset() doesn't wait for guest networking after reverting to a
    snapshot, so the first few task steps can fail on DNS lookups (chrome,
    multi_apps, anything hitting the internet). We detect the revert case by
    snapshotting ``is_environment_used`` before and after upstream reset — if
    it was True before (meaning a revert will happen) we call the wait helper.
    """
    from desktop_env.desktop_env import DesktopEnv

    if getattr(DesktopEnv.reset, "_network_wait_patched", False):
        return

    _orig_reset = DesktopEnv.reset

    def _patched_reset(self, task_config=None, seed=None, options=None):
        needed_wait = bool(getattr(self, "is_environment_used", False))
        obs = _orig_reset(self, task_config=task_config, seed=seed, options=options)
        if needed_wait:
            try:
                self._wait_for_network()
            except Exception as e:
                logger.warning("Post-reset network wait failed (non-fatal): %s", e)
        return obs

    _patched_reset._network_wait_patched = True
    DesktopEnv.reset = _patched_reset
    logger.info("Patched DesktopEnv.reset() to wait for guest network after revert")


# ---------------------------------------------------------------------------
# Out-of-tree apptainer provider registration
# ---------------------------------------------------------------------------


class _ApptainerSentinel(str):
    """A ``str`` subclass equal to ``"docker"`` so it satisfies upstream
    DesktopEnv's hard-coded provider-name validation set, while still being
    distinguishable by ``isinstance()`` so our patched factory can route it
    to the apptainer provider instead.
    """

    def __new__(cls):
        return super().__new__(cls, "docker")


def _load_in_repo_module(dotted_name, file_path):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {dotted_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _register_apptainer_provider():
    """Load apptainer provider/manager from the in-repo files and register
    them as ``desktop_env.providers.apptainer.{provider,manager}`` in
    ``sys.modules`` so future ``import`` statements resolve cleanly.

    Then monkey-patch ``desktop_env.providers.create_vm_manager_and_provider``
    to handle ``provider_name == "apptainer"``.
    """
    if "desktop_env.providers.apptainer.provider" in sys.modules:
        return  # already registered

    # Make sure the parent package is at least a placeholder so submodule
    # imports under desktop_env.providers.apptainer work.
    if "desktop_env.providers.apptainer" not in sys.modules:
        pkg_init = os.path.join(_APPTAINER_PKG_DIR, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "desktop_env.providers.apptainer",
            pkg_init,
            submodule_search_locations=[_APPTAINER_PKG_DIR],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load apptainer package init")
        pkg_mod = importlib.util.module_from_spec(spec)
        sys.modules["desktop_env.providers.apptainer"] = pkg_mod
        spec.loader.exec_module(pkg_mod)

    _load_in_repo_module(
        "desktop_env.providers.apptainer.manager",
        os.path.join(_APPTAINER_PKG_DIR, "manager.py"),
    )
    _load_in_repo_module(
        "desktop_env.providers.apptainer.provider",
        os.path.join(_APPTAINER_PKG_DIR, "provider.py"),
    )

    import desktop_env.providers as providers_pkg

    _orig_factory = providers_pkg.create_vm_manager_and_provider

    def _patched_factory(provider_name, region, use_proxy=False):
        # Recognise both the literal string and our sentinel (which == "docker"
        # for upstream's validation set but is identifiable here).
        if isinstance(provider_name, _ApptainerSentinel) or provider_name == "apptainer":
            from desktop_env.providers.apptainer.manager import ApptainerVMManager
            from desktop_env.providers.apptainer.provider import ApptainerProvider
            return ApptainerVMManager(), ApptainerProvider(region)
        return _orig_factory(provider_name, region, use_proxy=use_proxy)

    providers_pkg.create_vm_manager_and_provider = _patched_factory
    # Also rebind on the desktop_env module since it imported the symbol by
    # name (``from desktop_env.providers import create_vm_manager_and_provider``)
    # at import time.
    import desktop_env.desktop_env as de_mod
    de_mod.create_vm_manager_and_provider = _patched_factory

    logger.info(
        "Registered out-of-tree apptainer provider from %s",
        _APPTAINER_PKG_DIR,
    )


def _patch_desktop_env_init_for_apptainer():
    """Wrap ``DesktopEnv.__init__`` so ``provider_name="apptainer"`` is accepted.

    Upstream ``DesktopEnv.__init__`` validates ``provider_name`` against a
    hard-coded set literal and raises ``ValueError`` for anything outside
    ``{vmware, virtualbox, docker, aws, gcp, azure, aliyun, volcengine}``.
    We don't want to edit OSWorld, so when the caller asks for "apptainer"
    we substitute a ``str`` sentinel that *equals* ``"docker"`` (so the
    validation set passes), then restore ``self.provider_name = "apptainer"``
    afterwards. The patched factory above looks at ``isinstance(...)`` to
    route the sentinel to the apptainer manager / provider.
    """
    from desktop_env.desktop_env import DesktopEnv

    if getattr(DesktopEnv.__init__, "_apptainer_patched", False):
        return

    _orig_init = DesktopEnv.__init__

    def _patched_init(self, *args, **kwargs):
        # ``provider_name`` is the first positional arg in the signature.
        pn = kwargs.get("provider_name", None)
        if pn is None and len(args) >= 1:
            pn = args[0]

        if pn == "apptainer":
            sentinel = _ApptainerSentinel()
            if "provider_name" in kwargs:
                kwargs["provider_name"] = sentinel
            elif len(args) >= 1:
                args = (sentinel,) + tuple(args[1:])
            else:
                kwargs["provider_name"] = sentinel
            _orig_init(self, *args, **kwargs)
            self.provider_name = "apptainer"
        else:
            _orig_init(self, *args, **kwargs)

    _patched_init._apptainer_patched = True
    DesktopEnv.__init__ = _patched_init
    logger.info("Patched DesktopEnv.__init__ to accept provider_name='apptainer'")


def _patch_desktop_env_init_for_resolution():
    """Wrap DesktopEnv.__init__ to enforce screen resolution once the VM
    is up. Upstream doesn't run xrandr, so GNOME may fall back to 1280x800.
    """
    from desktop_env.desktop_env import DesktopEnv

    if getattr(DesktopEnv.__init__, "_resolution_patched", False):
        return

    _orig_init = DesktopEnv.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self._enforce_screen_resolution()
        except Exception as e:
            logger.warning("Initial screen-resolution enforce failed (non-fatal): %s", e)

    _patched_init._resolution_patched = True
    # Preserve the apptainer marker so we don't double-wrap.
    if getattr(_orig_init, "_apptainer_patched", False):
        _patched_init._apptainer_patched = True
    DesktopEnv.__init__ = _patched_init
    logger.info("Patched DesktopEnv.__init__ to enforce screen resolution")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Save-dialog handling for postconfig Ctrl+S
#
# Many OSWorld eval postconfigs do `pyautogui.hotkey('ctrl','s'); time.sleep(0.5)`
# to flush the document to disk before the evaluator pulls it. Two failure
# modes are common in LibreOffice (Writer/Calc/Impress):
#
# (a) The active edit (e.g. a placeholder text frame) hasn't been committed
#     to the document tree yet, so Ctrl+S writes the previous document state.
# (b) Saving a foreign format (.docx/.pptx/.xlsx) opens a modal "Use
#     Microsoft Format!" dialog. The 0.5 s sleep doesn't dismiss it, so the
#     file on disk stays unchanged.
#
# We rewrite any postconfig _execute_setup whose command body contains
# `pyautogui.hotkey('ctrl', 's')` to:
#   ESC (commit edit) -> sleep 0.3 -> Ctrl+S -> sleep 1.0 ->
#   ENTER (default = "keep current format") -> sleep 0.5
#
# This is idempotent: if a command already presses Esc/Enter we don't add
# a second one (we just look for the Ctrl+S marker; if found, replace the
# whole command).
# ---------------------------------------------------------------------------

_SAVE_DIALOG_HARDENED_SCRIPT = (
    "import pyautogui, time\n"
    "pyautogui.press('escape'); time.sleep(0.3)\n"
    "pyautogui.hotkey('ctrl', 's'); time.sleep(1.0)\n"
    "pyautogui.press('enter'); time.sleep(0.5)\n"
)


def _looks_like_ctrl_s(cmd):
    if not cmd:
        return False
    if isinstance(cmd, list):
        joined = " ".join(str(x) for x in cmd)
    else:
        joined = str(cmd)
    return "hotkey('ctrl', 's')" in joined or 'hotkey("ctrl", "s")' in joined


def _maybe_harden_save(cmd):
    """Return (replaced_cmd, replaced?) — wraps Ctrl+S with edit-commit + dialog dismiss."""
    if not _looks_like_ctrl_s(cmd):
        return cmd, False
    # If the command already contains an Enter press AND an Escape, leave it.
    joined = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else str(cmd)
    if "press('enter')" in joined and "press('escape')" in joined:
        return cmd, False
    # Replace with our hardened payload while preserving the original wrapper
    # shape (`["python", "-c", "..."]` is the typical form).
    if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("python", "python3") and cmd[1] in ("-c",):
        return [cmd[0], cmd[1], _SAVE_DIALOG_HARDENED_SCRIPT], True
    # For other shapes leave alone — we only know how to rewrite the standard form.
    return cmd, False


def _patch_setup_controller_save_dialog():
    """Wrap SetupController._execute_setup to harden Ctrl+S commands."""
    try:
        from desktop_env.controllers.setup import SetupController
    except Exception as exc:  # pragma: no cover
        logger.warning("setup_controller patch skipped: %s", exc)
        return
    if getattr(SetupController._execute_setup, "_macu_save_hardened", False):
        return
    orig = SetupController._execute_setup

    def wrapped(self, command, stdout="", stderr="", shell=False, until=None):
        new_cmd, replaced = _maybe_harden_save(command)
        if replaced:
            logger.info(
                "Hardening Ctrl+S postconfig: ESC -> Ctrl+S -> ENTER (was: %r)",
                command,
            )
        return orig(self, new_cmd, stdout=stdout, stderr=stderr, shell=shell, until=until)

    wrapped._macu_save_hardened = True
    SetupController._execute_setup = wrapped
    logger.info("Patched SetupController._execute_setup with save-dialog hardening")


def apply():
    """Apply all OSWorld monkey-patches.

    Idempotent: safe to call multiple times.
    """
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.vmware.manager import VMwareVMManager

    if not hasattr(VMwareVMManager, 'release_vm'):
        VMwareVMManager.release_vm = _release_vm
        logger.info("Patched VMwareVMManager with release_vm()")

    if not hasattr(DesktopEnv, 'release'):
        DesktopEnv.release = _desktop_env_release
        logger.info("Patched DesktopEnv with release()")

    _patch_desktop_env_network_helpers()
    _patch_desktop_env_reset_for_network()
    _patch_desktop_env_get_obs_for_resolution()
    _register_apptainer_provider()
    _patch_desktop_env_init_for_apptainer()
    _patch_desktop_env_init_for_resolution()
    _patch_setup_controller_save_dialog()
