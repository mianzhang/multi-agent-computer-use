from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import subprocess
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("macu")


def load_canonical_osworld_config(
    osworld_data_dir: str,
    domain: str,
    task_id: str,
) -> dict:
    """Load the canonical OSWorld task config for ``<domain>/<task_id>``."""
    path = Path(osworld_data_dir) / "examples" / domain / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical OSWorld config not found at {path}. "
            f"Verify --osworld-data-dir and that the dependency graph's "
            f"task_id / domain match an entry in the OSWorld examples tree."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("id") != task_id:
        raise ValueError(
            f"Canonical OSWorld config id mismatch at {path}: "
            f"graph task_id={task_id!r} vs config id={cfg.get('id')!r}"
        )
    return cfg


def write_subtask_osworld_example(
    config_base_dir: str,
    domain: str,
    subtask_id: str,
    canonical_config: dict,
    subtask_instruction: str,
    staged_files: Optional[list[dict]] = None,
    skip_canonical_setup: bool = False,
) -> Path:
    """Pre-write the per-subtask OSWorld example file on disk."""
    examples_dir = Path(config_base_dir) / "examples" / domain
    examples_dir.mkdir(parents=True, exist_ok=True)
    path = examples_dir / f"{subtask_id}.json"
    payload = copy.deepcopy(canonical_config)
    payload["instruction"] = subtask_instruction
    if skip_canonical_setup:
        payload["config"] = []
    if staged_files:
        upload_entry = {
            "type": "upload_file",
            "parameters": {
                "files": [
                    {"local_path": entry["local_path"], "path": entry["guest_path"]}
                    for entry in staged_files
                ],
            },
        }
        if "config" not in payload or not isinstance(payload["config"], list):
            payload["config"] = []
        payload["config"].append(upload_entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _write_chrome_example_with_uploads(
    config_base_dir: str,
    domain: str,
    subtask_id: str,
    instruction: str,
    website: str,
    staged_files: Optional[list[dict]] = None,
    skip_canonical_setup: bool = False,
) -> Path:
    """Pre-write a chrome-stub OSWorld example with optional uploads / no-setup."""
    examples_dir = Path(config_base_dir) / "examples" / domain
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_path = examples_dir / f"{subtask_id}.json"

    if skip_canonical_setup:
        config_list: list[dict] = []
    else:
        config_list = [
            {
                "type": "launch",
                "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]},
            },
            {
                "type": "launch",
                "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]},
            },
            {
                "type": "chrome_open_tabs",
                "parameters": {"urls_to_open": [website]},
            },
        ]

    if staged_files:
        config_list.append(
            {
                "type": "upload_file",
                "parameters": {
                    "files": [
                        {"local_path": entry["local_path"], "path": entry["guest_path"]}
                        for entry in staged_files
                    ],
                },
            }
        )

    payload = {
        "id": subtask_id,
        "snapshot": "chrome",
        "instruction": instruction,
        "source": website,
        "config": config_list,
        "trajectory": "trajectories/",
        "related_apps": [] if skip_canonical_setup else ["chrome"],
        "evaluator": {"func": "infeasible"},
        "proxy": False,
        "fixed_ip": False,
        "possibility_of_env_change": "high",
        "metadata": {
            "domain": domain,
            "website": website,
            "no_initial_setup": skip_canonical_setup,
        },
    }
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return example_path


def run_osworld_evaluator_subprocess(
    run_dir: Path,
    task_id: str,
    domain: str,
    canonical_config: dict,
    final_vm_path: Optional[str],
    cua_model: str,
    cua_args: list[str],
    args: argparse.Namespace,
) -> Optional[float]:
    """Launch ``scripts/osworld_evaluator.py`` and overwrite ``result.txt``.

    The evaluator subprocess runs with ``cwd=osworld_root`` so that
    DesktopEnv's relative ``cache_dir_base``, the vmware registry path,
    and any relative ``final_vm_path`` strings all resolve correctly.

    On success, writes the float score returned by ``env.evaluate()`` to
    both:
    1. ``<run_dir>/result.txt`` -- MACU-native, next to ``final_results.json``
    2. ``<run_dir>/pyautogui/screenshot/<model>/<domain>/<canonical_task_id>/result.txt``
       -- OSWorld convention so downstream ``get_result`` aggregation works.

    Failures (missing final_vm_path, subprocess error, timeout) still
    write ``0.0`` to both locations so downstream ``get_unfinished``
    heuristics treat the task as done.
    """
    model_for_path = cua_model or "unknown_model"
    osworld_result_dir = (
        run_dir / "pyautogui" / "screenshot" / model_for_path / domain / task_id
    )
    result_path_top = run_dir / "result.txt"
    result_path_osworld = osworld_result_dir / "result.txt"
    eval_log_path = run_dir / "osworld_evaluator.log"
    final_results_path = run_dir / "final_results.json"

    def _write_zero(reason: str) -> None:
        for p in (result_path_top, result_path_osworld):
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write("0.0\n")
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write(f"[orchestrator] Wrote 0.0 to result.txt because: {reason}\n")

    if not final_vm_path:
        logger.warning("No final_vm_path in final_results.json; writing 0.0.")
        _write_zero("final_vm_path is missing (no subtask VM was tracked)")
        return 0.0

    submit_fail_before_evaluate = False
    try:
        if final_results_path.exists():
            with open(final_results_path, "r", encoding="utf-8") as f:
                final_results = json.load(f)
            submit_fail_before_evaluate = bool(
                final_results.get(
                    "osworld_submit_fail_before_evaluate",
                    final_results.get("osworld_infeasible_verdict"),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read infeasible verdict from %s: %s", final_results_path, exc)

    # Write the canonical task config to a standalone file so the
    # evaluator subprocess can load it directly.
    task_config_path = run_dir / "evaluator_task_config.json"
    with open(task_config_path, "w", encoding="utf-8") as f:
        json.dump(canonical_config, f, indent=2, ensure_ascii=False)

    evaluator_script = str((PROJECT_ROOT / "scripts" / "osworld_evaluator.py").resolve())

    # Parse headless/screen/provider flags out of cua_args so the
    # evaluator attaches to the VM with consistent settings.
    headless = True
    screen_w, screen_h = 1920, 1080
    client_password = "password"
    provider_name = "vmware"
    i = 0
    while i < len(cua_args):
        tok = cua_args[i]
        if tok in ("--no-headless",):
            headless = False
            i += 1
            continue
        if tok in ("--headless",):
            headless = True
            i += 1
            continue
        if i + 1 < len(cua_args):
            nxt = cua_args[i + 1]
            if tok == "--screen_width":
                try:
                    screen_w = int(nxt)
                except ValueError:
                    pass
                i += 2
                continue
            if tok == "--screen_height":
                try:
                    screen_h = int(nxt)
                except ValueError:
                    pass
                i += 2
                continue
            if tok == "--client_password":
                client_password = nxt
                i += 2
                continue
            if tok == "--provider_name":
                provider_name = nxt
                i += 2
                continue
        i += 1

    cmd = [
        args.python,
        evaluator_script,
        "--task-config", str(task_config_path),
        "--final-vm-path", final_vm_path,
        "--result-path", str(result_path_top),
        "--result-path", str(result_path_osworld),
        "--log-path", str(eval_log_path),
        "--provider-name", provider_name,
        "--screen-width", str(screen_w),
        "--screen-height", str(screen_h),
        "--client-password", client_password,
    ]
    if headless:
        cmd.append("--headless")
    else:
        cmd.append("--no-headless")
    if submit_fail_before_evaluate:
        cmd.append("--submit-fail-before-evaluate")
        logger.info(
            "Final results request FAIL before evaluation; evaluator will submit FAIL before env.evaluate()."
        )

    logger.info("Launching OSWorld evaluator: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            cwd=args.osworld_root,
            timeout=getattr(args, "evaluator_timeout", 600),
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "OSWorld evaluator timed out after %ds",
            getattr(args, "evaluator_timeout", 600),
        )
        _write_zero(
            f"evaluator subprocess timeout after {getattr(args, 'evaluator_timeout', 600)}s"
        )
        return 0.0
    except Exception as exc:  # noqa: BLE001
        logger.error("OSWorld evaluator subprocess failed to launch: %s", exc)
        _write_zero(f"subprocess launch error: {exc}")
        return 0.0

    if completed.returncode != 0:
        logger.warning(
            "OSWorld evaluator returned non-zero exit code %d (see %s)",
            completed.returncode,
            eval_log_path,
        )
        if not result_path_top.exists() or not result_path_osworld.exists():
            _write_zero(
                f"evaluator returned {completed.returncode} and left "
                f"result.txt missing"
            )

    try:
        with open(result_path_top, "r", encoding="utf-8") as f:
            return float(f.read().strip() or "0.0")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read evaluator result.txt: %s", exc)
        return None


def _aggregate_success_rate(result_dir: Path) -> None:
    """Log the mean of all top-level ``result.txt`` files under ``result_dir``.

    Each task's score has already been written by the evaluator subprocess
    (possibly ``0.0`` on failure). This just averages them for the run log.
    """
    scores: list[float] = []
    for rt in sorted(result_dir.glob("*/result.txt")):
        try:
            scores.append(float(rt.read_text(encoding="utf-8").strip() or "0.0"))
        except Exception:  # noqa: BLE001
            continue
    if scores:
        mean = sum(scores) / len(scores)
        logger.info("OSWorld aggregate success rate: %.4f (%d task(s))", mean, len(scores))
    else:
        logger.info("No result.txt files found under %s for aggregation.", result_dir)
