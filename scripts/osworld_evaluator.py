#!/usr/bin/env python3
"""Run the OSWorld evaluator against an already-running VM.

Launched as a subprocess by ``run_macu_osworld.py`` after all CUA
subtasks have finished. Attaches to the VM path that the manager
designated as the "final aggregated VM" in ``final_results.json``,
runs ``env.evaluate()`` on it, and writes one or more ``result.txt``
files with the real OSWorld float score.

The CUA self-report is now written to ``completed.txt`` (not
``result.txt``), so ``result.txt`` is exclusively the evaluator score.

Design constraints:
* Do NOT call ``env.reset(task_config=...)``. For vmware,
  ``DesktopEnv.__init__`` sets ``is_environment_used=True``
  (``desktop_env.py:166-167``) and ``reset()`` would revert the
  snapshot (``desktop_env.py:391-392``), destroying the agent's work.
  Populate evaluator state manually via ``_set_task_info`` +
  ``setup_controller.reset_cache_dir``.
* Run with ``cwd=osworld_root`` (the orchestrator arranges this).
  That single choice anchors ``DesktopEnv.cache_dir_base="cache"``,
  the vmware manager registry path, and any relative
  ``final_vm_path`` strings.

CLI shape (invoked by the orchestrator):

    python scripts/osworld_evaluator.py \
        --task-config /abs/path/to/canonical.json \
        --final-vm-path /abs/or/rel/Ubuntu0.vmx \
        --result-path /abs/path/to/run_dir/result.txt \
        --result-path /abs/path/to/run_dir/pyautogui/.../result.txt \
        --log-path /abs/path/to/run_dir/osworld_evaluator.log \
        --provider-name vmware \
        --headless \
        --screen-width 1920 --screen-height 1080 \
        --client-password password \
        --submit-fail-before-evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Put the multi-agent-cua project root on sys.path so ``osworld.patches``
# resolves (the ``osworld`` helper package lives at
# ``<project_root>/osworld/``, NOT under osworld_root). We resolve this
# statically from __file__ so the import works regardless of the cwd the
# orchestrator launches us with (cwd is osworld_root, not project root).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _setup_file_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear any pre-existing handlers from imported modules.
    root.handlers = []
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[evaluator %(levelname)s] %(message)s"))
    root.addHandler(sh)
    return logging.getLogger("osworld_evaluator")


def _write_result(result_paths: list[str], score: float) -> None:
    """Write ``score`` to every path, creating parent dirs as needed.

    This is the only writer for ``result.txt`` in the evaluator.
    The CUA self-report is in ``completed.txt``; this writes the real
    evaluator score to ``result.txt``.
    """
    for path in result_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{score}\n")


def run_evaluator(
    task_config: dict,
    final_vm_path: str,
    result_paths: list[str],
    provider_name: str,
    headless: bool,
    screen_size: tuple[int, int],
    client_password: str,
    submit_fail_before_evaluate: bool,
    logger: logging.Logger,
) -> float:
    """Attach to the existing VM and run OSWorld's evaluator.

    On success, writes the float score to every path in ``result_paths``
    and returns it. Raises on failure (the caller writes ``0.0`` on
    exception).
    """
    vm_path = final_vm_path
    if not os.path.isabs(vm_path):
        vm_path = os.path.abspath(vm_path)
    if not os.path.exists(vm_path):
        raise FileNotFoundError(f"VM .vmx not found: {vm_path}")
    logger.info("Resolved final_vm_path -> %s", vm_path)

    # Make OSWorld importable. cwd=osworld_root lets ``desktop_env`` resolve
    # naturally; the multi-agent-cua project root was added to sys.path at
    # module load time (see top of file) so ``osworld.patches`` also
    # resolves. Apply the release() patch so we can hand the VM back to
    # the pool without stopping it.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from osworld.patches import apply as _apply_patches
    _apply_patches()
    from desktop_env.desktop_env import DesktopEnv

    snapshot = task_config.get("snapshot", "init_state")
    logger.info("Constructing DesktopEnv(path_to_vm=%s, snapshot=%s, provider=%s)",
                vm_path, snapshot, provider_name)

    env = DesktopEnv(
        path_to_vm=vm_path,
        provider_name=provider_name,
        snapshot_name=snapshot,
        action_space="pyautogui",
        screen_size=screen_size,
        headless=headless,
        os_type="Ubuntu",
        require_a11y_tree=False,
        enable_proxy=False,
        client_password=client_password,
    )

    try:
        # CRITICAL: do NOT call env.reset(task_config=...). For vmware
        # that would revert the snapshot and destroy the agent's work.
        # Populate evaluator state manually.
        env.is_environment_used = False
        env._set_task_info(task_config)
        env.setup_controller.reset_cache_dir(env.cache_dir)
        logger.info("Task info populated (task_id=%s, cache_dir=%s).",
                    getattr(env, "task_id", "?"), getattr(env, "cache_dir", "?"))
        if submit_fail_before_evaluate:
            logger.info("Submitting FAIL action before env.evaluate().")
            env.step("FAIL", 0)

        logger.info("Running env.evaluate()...")
        score = env.evaluate()
        logger.info("env.evaluate() returned: %s", score)
        _write_result(result_paths, float(score))
        return float(score)
    finally:
        try:
            env.close()
            logger.info("Stopped VM after evaluation.")
        except Exception as exc:  # noqa: BLE001 - non-fatal cleanup path
            logger.warning("env.close() failed (non-fatal): %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="OSWorld evaluator for MACU runs")
    parser.add_argument("--task-config", required=True,
                        help="Path to canonical OSWorld task config JSON")
    parser.add_argument("--final-vm-path", required=True,
                        help="VM path (absolute or relative to cwd, which the "
                             "orchestrator sets to osworld_root)")
    parser.add_argument("--result-path", action="append", required=True,
                        help="Path to write result.txt. May be given multiple times.")
    parser.add_argument("--log-path", required=True,
                        help="Log file for evaluator output + any traceback")
    parser.add_argument("--provider-name", default="vmware")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--client-password", default="password")
    parser.add_argument(
        "--submit-fail-before-evaluate",
        action="store_true",
        help="Submit a FAIL action to DesktopEnv before running env.evaluate().",
    )
    args = parser.parse_args()

    logger = _setup_file_logger(args.log_path)
    logger.info("osworld_evaluator starting; cwd=%s", os.getcwd())
    logger.info("args=%s", vars(args))

    try:
        with open(args.task_config, "r", encoding="utf-8") as f:
            task_config = json.load(f)
    except Exception as exc:
        logger.error("Failed to load task config %s: %s", args.task_config, exc)
        _write_result(args.result_path, 0.0)
        return 1

    if not args.final_vm_path or args.final_vm_path.lower() == "none":
        logger.warning("No final_vm_path provided; writing 0.0 as the score.")
        _write_result(args.result_path, 0.0)
        return 0

    try:
        run_evaluator(
            task_config=task_config,
            final_vm_path=args.final_vm_path,
            result_paths=args.result_path,
            provider_name=args.provider_name,
            headless=args.headless,
            screen_size=(args.screen_width, args.screen_height),
            client_password=args.client_password,
            submit_fail_before_evaluate=args.submit_fail_before_evaluate,
            logger=logger,
        )
        return 0
    except Exception:
        tb = traceback.format_exc()
        logger.error("Evaluator failed with exception:\n%s", tb)
        _write_result(args.result_path, 0.0)
        return 1


if __name__ == "__main__":
    sys.exit(main())
