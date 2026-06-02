#!/usr/bin/env python3
"""Run online-mind2web tasks in OSWorld container with Qwen or GPT-5.4 CUA.

Supports multiple CUA providers via --cua_provider:
  - qwen: Qwen CUA agent (default)
  - openai: GPT-5.4 / GPT-5.4-mini agent (OpenAI CUA responses API)

Example (Qwen):
    python scripts/run_cua.py \
        --cua_provider qwen \
        --headless \
        --model Qwen/Qwen3.6-27B \
        --provider_name vmware \
        --num_envs 4 \
        --max_steps 100 \
        --sleep_after_execution 5.0 \
        --mind2web_tasks_path data/journeys_cua_final/tasks_journeys_cua_final.json \
        --test_config_base_dir data/journeys_cua_final \
        --test_all_meta_path data/journeys_cua_final/test_journeys_cua_final.json \
        --result_dir test_runs_journeys_cua_qwen35_4b \
        --start_idx 0 --end_idx 1

Example (OpenAI):
    python scripts/run_cua.py \
        --cua_provider openai \
        --headless \
        --model gpt-5.4-mini \
        --provider_name vmware \
        --num_envs 4 \
        --max_steps 100 \
        --sleep_after_execution 5.0 \
        --mind2web_tasks_path data/journeys_cua_final/tasks_journeys_cua_final.json \
        --test_config_base_dir data/journeys_cua_final \
        --test_all_meta_path data/journeys_cua_final/test_journeys_cua_final.json \
        --result_dir test_runs_journeys_cua_gpt54 \
        --start_idx 0 --end_idx 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process, current_process
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_OSWORLD_ROOT = Path(os.environ.get("OSWORLD_ROOT", "OSWorld"))
if str(DEFAULT_OSWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_OSWORLD_ROOT))

DesktopEnv = None
Qwen35VLAgent = None
GPT54Agent = None
log_task_completion = None
log_task_error = None

processes: List[Process] = []
is_terminating = False
FILE_SCAN_ROOTS = "/home/user/Desktop,/home/user/Documents,/home/user/Downloads,/tmp"
FILE_DOWNLOAD_MAX_BYTES = 500_000_000


def _file_scan_roots() -> List[str]:
    return [p.strip() for p in FILE_SCAN_ROOTS.split(",") if p.strip()]


def _macu_file_management_enabled(args: argparse.Namespace) -> bool:
    return bool(args.vm_info_path)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online-mind2web inside OSWorld container (Qwen or GPT-5.4)."
    )

    # CUA provider selection
    parser.add_argument(
        "--cua_provider",
        type=str,
        choices=["qwen", "openai"],
        default="qwen",
        help="CUA agent provider: qwen or openai (GPT-5.4).",
    )

    # Paths / dataset generation
    parser.add_argument("--osworld_root", type=str, default=str(DEFAULT_OSWORLD_ROOT))
    parser.add_argument(
        "--mind2web_tasks_path",
        type=str,
        default="data/online_mind2web_tasks.json",
        help="Path to online-mind2web task list (task_id/confirmed_task/website).",
    )
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="data/online_mind2web.json",
        help="OSWorld-format meta file to write/read: {domain: [task_ids...]}",
    )
    parser.add_argument(
        "--test_config_base_dir",
        type=str,
        default="data/online_mind2web_osworld",
        help="Directory where generated OSWorld examples are stored under examples/<domain>/<task_id>.json",
    )
    parser.add_argument(
        "--mind2web_domain",
        type=str,
        default="mind2web_chrome",
        help="Domain key used in OSWorld meta JSON.",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument(
        "--overwrite_generated_examples",
        action="store_true",
        help="Overwrite generated example JSON files if they already exist.",
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--vm_info_path",
        type=str,
        default=None,
        help="If set, write {\"vm_path\": <provider VM path>} to this file once the "
             "DesktopEnv is up. Used by run_macu.py to discover which VM each subtask "
             "ended up using so it can later designate a final aggregated VM.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--action_space",
        type=str,
        choices=["pyautogui"],
        default="pyautogui",
        help="Action type",
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument(
        "--provider_name",
        type=str,
        default="vmware",
        choices=["aws", "virtualbox", "vmware", "docker", "azure", "aliyun", "apptainer"],
        help="Provider name",
    )
    parser.add_argument("--client_password", type=str, default="", help="Client password")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--env_ready_wait_seconds", type=float, default=60.0)
    parser.add_argument("--post_task_wait_seconds", type=float, default=10.0)
    # Agent config (shared)
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--model", type=str, default=None, help="Model name (default depends on --cua_provider)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default depends on --cua_provider: 0.6 for qwen, 1.0 for openai)")
    parser.add_argument("--top_p", type=float, default=None, help="Sampling top_p (default depends on --cua_provider: 0.95 for qwen, 0.9 for openai)")
    parser.add_argument("--top_k", type=int, default=None, help="Sampling top_k (qwen only)")
    parser.add_argument("--min_p", type=float, default=None, help="Sampling min_p (qwen only)")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Presence penalty (qwen/openai)")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (qwen only)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max output tokens (default depends on --cua_provider)")
    parser.add_argument("--stop_token", type=str, default=None)

    # Agent config (qwen-specific)
    parser.add_argument(
        "--coord",
        type=str,
        choices=["absolute", "relative"],
        default="relative",
        help="Coordinate system for agent outputs (qwen only)",
    )
    parser.add_argument(
        "--add_thought_prefix",
        action="store_true",
        help="Add thought prefix to the response (qwen only)",
    )
    parser.add_argument("--image_max", type=int, default=20, help="Max screenshots before folding (qwen only)")
    parser.add_argument("--fold_size", type=int, default=10, help="Number of screenshots to fold at once (qwen only)")
    parser.add_argument("--history_n", type=int, default=100, help="Number of prior steps to include in conversation history (qwen only)")
    parser.add_argument("--keep_reasoning", action="store_true", help="Keep thinking/reasoning tokens in conversation history (qwen only)")
    parser.add_argument("--preserve_reasoning_content", action="store_true", help="Preserve raw reasoning content from vllm in step trajectory (qwen only)")
    parser.add_argument("--disable_native_tool_calling", action="store_true", help="Disable OpenAI-compatible native tool calls for Qwen and use XML prompting only")
    parser.add_argument("--max_tool_calls_per_turn", type=int, default=10, help="Maximum Qwen computer_use tool calls to execute from one model turn")

    # Agent config (openai-specific)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["none", "low", "medium", "high", "xhigh"],
        default="xhigh",
        help="Reasoning effort level (openai only)",
    )

    # MACU orchestrator integration
    parser.add_argument(
        "--enable_manager_followup",
        action="store_true",
        default=False,
        help="Wait for manager review after CUA completion (used by run_macu.py).",
    )
    # Run selection / logging
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--specific_task_id",
        type=str,
        default=None,
        help="Run only this task id after dataset generation/filtering.",
    )
    parser.add_argument("--result_dir", type=str, default=None, help="Output directory (default depends on --cua_provider)")
    parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument("--region", type=str, default="us-east-1")

    args = parser.parse_args()

    # Apply provider-specific defaults
    if args.model is None:
        args.model = "Qwen/Qwen3.6-27B" if args.cua_provider == "qwen" else "gpt-5.4"
    if args.temperature is None:
        # Qwen thinking-mode agent recommendation:
        # temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        # presence_penalty=0.0, repetition_penalty=1.0.
        args.temperature = 0.6 if args.cua_provider == "qwen" else 1.0
    if args.top_p is None:
        args.top_p = 0.95 if args.cua_provider == "qwen" else 0.9
    if args.top_k is None:
        args.top_k = 20 if args.cua_provider == "qwen" else None
    if args.min_p is None:
        args.min_p = 0.0 if args.cua_provider == "qwen" else None
    if args.presence_penalty is None:
        args.presence_penalty = 0.0
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.0 if args.cua_provider == "qwen" else None
    if args.max_tokens is None:
        args.max_tokens = 65536 if args.cua_provider == "qwen" else 4096
    if args.result_dir is None:
        args.result_dir = (
            "runs/mind2web_container_qwen35vl"
            if args.cua_provider == "qwen"
            else "runs/mind2web_container_gpt54"
        )

    return args


args = config()


def _resolve_path(value: str) -> str:
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((Path.cwd() / p).resolve())


args.osworld_root = _resolve_path(args.osworld_root)
args.mind2web_tasks_path = _resolve_path(args.mind2web_tasks_path)
args.test_all_meta_path = _resolve_path(args.test_all_meta_path)
args.test_config_base_dir = _resolve_path(args.test_config_base_dir)
args.result_dir = _resolve_path(args.result_dir)

if args.osworld_root not in sys.path:
    sys.path.insert(0, args.osworld_root)

if os.path.exists(".env"):
    from dotenv import load_dotenv

    load_dotenv()


def load_osworld_dependencies() -> None:
    global DesktopEnv, Qwen35VLAgent, GPT54Agent, log_task_completion, log_task_error
    from desktop_env.desktop_env import DesktopEnv as _DesktopEnv

    try:
        from lib_results_logger import (
            log_task_completion as _log_task_completion,
            log_task_error as _log_task_error,
        )
    except ModuleNotFoundError:
        fallback_logger = logging.getLogger("desktopenv.results_logger")

        def _log_task_completion(example, result, example_result_dir, _args):
            fallback_logger.info(
                "lib_results_logger unavailable; task %s completed with result %.2f in %s",
                example.get("id"),
                result,
                example_result_dir,
            )

        def _log_task_error(example, error, example_result_dir, _args):
            fallback_logger.error(
                "lib_results_logger unavailable; task %s errored in %s: %s",
                example.get("id"),
                example_result_dir,
                error,
            )

    DesktopEnv = _DesktopEnv
    log_task_completion = _log_task_completion
    log_task_error = _log_task_error

    if args.cua_provider == "qwen":
        from osworld.mm_agents.qwen35vl_agent import Qwen35VLAgent as _Qwen35VLAgent
        Qwen35VLAgent = _Qwen35VLAgent
    else:
        from osworld.mm_agents.gpt54_agent import GPT54Agent as _GPT54Agent
        GPT54Agent = _GPT54Agent


load_osworld_dependencies()

# Apply VM-reuse patches to OSWorld classes
from osworld.patches import apply as _apply_patches
_apply_patches()

if not args.model or args.model.strip() == "":
    print("ERROR: Model must be specified. Use --model <model_name>")
    sys.exit(1)

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)
datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

logs_dir = Path(args.result_dir).parent / f"logs_mind2web_container_{args.cua_provider}"
logs_dir.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(
    logs_dir / f"normal-{datetime_str}.log", encoding="utf-8"
)
debug_handler = logging.FileHandler(
    logs_dir / f"debug-{datetime_str}.log", encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt=(
        "\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s "
        "\x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] "
        "\x1b[0m%(message)s"
    )
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")


def setup_logger(example: Dict, example_result_dir: str):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger


def _snapshot_guest_files(controller, scan_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Walk *scan_paths* inside the guest VM and return a path -> {size, mtime} dict.

    Used by MACU to capture pre/post-task file state. Runs a single python
    snippet inside the guest via PythonController.execute_python_command so it
    works for any provider that has the OSWorld in-VM HTTP server.

    Returns an empty dict on any error so the caller can fall back gracefully.
    """
    if not scan_paths:
        return {}

    # Build a self-contained python snippet for the guest. The snippet emits
    # JSON between two unique sentinels so we can robustly parse it out of any
    # surrounding stdout chatter from the executor.
    snippet = (
        "import os, json\n"
        "paths = " + json.dumps(scan_paths) + "\n"
        "out = {}\n"
        "for root_path in paths:\n"
        "    if not os.path.isdir(root_path):\n"
        "        continue\n"
        "    for dirpath, _dirnames, filenames in os.walk(root_path):\n"
        "        for fname in filenames:\n"
        "            full = os.path.join(dirpath, fname)\n"
        "            try:\n"
        "                st = os.stat(full)\n"
        "            except OSError:\n"
        "                continue\n"
        "            out[full] = {'size': st.st_size, 'mtime': st.st_mtime}\n"
        "print('__GUEST_SNAPSHOT_BEGIN__' + json.dumps(out) + '__GUEST_SNAPSHOT_END__')\n"
    )

    try:
        result = controller.execute_python_command(snippet)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guest file snapshot: execute_python_command raised: %s", exc)
        return {}

    if not isinstance(result, dict):
        logger.warning("Guest file snapshot: unexpected result shape %r", type(result).__name__)
        return {}

    stdout = result.get("output") or result.get("stdout") or ""
    if not isinstance(stdout, str):
        logger.warning("Guest file snapshot: stdout is not a string (%r)", type(stdout).__name__)
        return {}

    begin_marker = "__GUEST_SNAPSHOT_BEGIN__"
    end_marker = "__GUEST_SNAPSHOT_END__"
    bi = stdout.find(begin_marker)
    ei = stdout.find(end_marker, bi + len(begin_marker)) if bi != -1 else -1
    if bi == -1 or ei == -1:
        logger.warning("Guest file snapshot: sentinels not found in stdout (len=%d)", len(stdout))
        return {}

    payload = stdout[bi + len(begin_marker):ei]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Guest file snapshot: JSON parse failed: %s", exc)
        return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def _write_file_snapshot(env, example_result_dir: str, scan_paths: List[str], label: str) -> None:
    """Capture a guest file snapshot and write it to ``files_<label>.json``.

    Best-effort: failures are logged and swallowed so the agent loop is never
    blocked by a snapshot error.
    """
    try:
        snap = _snapshot_guest_files(env.controller, scan_paths)
    except Exception as exc:  # noqa: BLE001
        logger.warning("File snapshot (%s) failed: %s", label, exc)
        return
    out_path = os.path.join(example_result_dir, f"files_{label}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snap, f)
        logger.info("Wrote %s with %d entries", out_path, len(snap))
    except OSError as exc:
        logger.warning("Failed to write %s: %s", out_path, exc)


def is_done_action(action) -> bool:
    if isinstance(action, str):
        return action.strip().upper() == "DONE"
    if isinstance(action, dict):
        return str(action.get("action_type", "")).upper() == "DONE"
    return False


def is_fail_action(action) -> bool:
    if isinstance(action, str):
        return action.strip().upper() == "FAIL"
    if isinstance(action, dict):
        return str(action.get("action_type", "")).upper() == "FAIL"
    return False


def build_example(task: Dict, domain: str) -> Dict:
    task_id = str(task.get("task_id") or task.get("id") or "").strip()
    instruction = str(task.get("confirmed_task") or task.get("task") or "").strip()
    website = str(task.get("website") or task.get("start_url") or "").strip()
    no_initial_setup = bool(task.get("no_initial_setup") or task.get("blank_vm"))
    if not task_id or not instruction or (not website and not no_initial_setup):
        raise ValueError(f"Invalid mind2web task: {task}")

    if no_initial_setup:
        config_list: List[Dict] = []
        related_apps: List[str] = []
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
        related_apps = ["chrome"]

    return {
        "id": task_id,
        "snapshot": "chrome",
        "instruction": instruction,
        "source": website,
        "config": config_list,
        "trajectory": "trajectories/",
        "related_apps": related_apps,
        "evaluator": {"func": "infeasible"},
        "proxy": False,
        "fixed_ip": False,
        "possibility_of_env_change": "high",
        "metadata": {
            "domain": domain,
            "website": website,
            "no_initial_setup": no_initial_setup,
            "reference_length": task.get("reference_length"),
            "level": task.get("level"),
        },
    }


def prepare_mind2web_osworld_data(args: argparse.Namespace) -> Dict[str, List[str]]:
    tasks_path = Path(args.mind2web_tasks_path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"mind2web tasks file not found: {tasks_path}")

    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("tasks")
    if not isinstance(raw, list):
        raise ValueError("mind2web task source must be a list or {'tasks': [...]} format")

    total = len(raw)
    if total == 0:
        raise ValueError("mind2web task source is empty")

    start_idx = max(0, int(args.start_idx))
    if args.end_idx is None:
        end_idx = total - 1
    else:
        end_idx = min(max(0, int(args.end_idx)), total - 1)

    if start_idx >= total or end_idx < start_idx:
        raise ValueError(
            f"Invalid selection window start_idx={args.start_idx}, end_idx={args.end_idx}, total={total}"
        )

    selected = raw[start_idx : end_idx + 1]
    logger.info(
        "Preparing mind2web tasks [%d:%d] (count=%d, total=%d)",
        start_idx,
        end_idx,
        len(selected),
        total,
    )

    domain = args.mind2web_domain
    examples_dir = Path(args.test_config_base_dir) / "examples" / domain
    examples_dir.mkdir(parents=True, exist_ok=True)

    task_ids: List[str] = []
    for item in selected:
        example = build_example(item, domain)
        task_id = str(example["id"])
        task_ids.append(task_id)
        example_path = examples_dir / f"{task_id}.json"
        if example_path.exists() and not args.overwrite_generated_examples:
            continue
        example_path.write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {domain: task_ids}
    meta_path = Path(args.test_all_meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote OSWorld meta file: %s", meta_path)
    logger.info("Generated OSWorld examples under: %s", examples_dir)
    return meta


def _check_vm_network(env):
    """Quick check: does the VM have working internet right now?"""
    try:
        resp = env._execute_shell_on_vm(
            "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://www.google.com",
            timeout=10,
        )
        if resp.status_code == 200:
            output = resp.json().get("output", "").strip()
            return output.startswith("2") or output.startswith("3")
    except Exception:
        pass
    return False


def _recover_vm_network(env):
    """Restart networking + re-pin DNS.  Returns True if network came back."""
    logger.warning("Network down, attempting recovery...")
    env._restart_vm_networking()
    time.sleep(5)
    env._configure_dns()
    time.sleep(2)
    ok = _check_vm_network(env)
    if ok:
        logger.info("Network recovered after restart + DNS re-pin.")
    else:
        logger.warning("Network still down after recovery attempt.")
    return ok


def _reload_chrome_url(env, url, timeout=60):
    """Navigate Chrome to the given URL via Playwright CDP after network recovery."""
    from playwright.sync_api import sync_playwright

    host = env.vm_ip
    port = getattr(env, "chromium_port", 9222)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://{host}:{port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, timeout=timeout * 1000)
        logger.info("Reloaded Chrome to: %s", url)


def _post_reset_network_check(env, example, args):
    """Post-reset network verification shared by both providers."""
    if not _check_vm_network(env):
        logger.warning("Network still down after reset + wait, reloading Chrome URL...")
        url = example.get("source", "")
        if url:
            try:
                _reload_chrome_url(env, url)
            except Exception as reload_err:
                logger.warning("Chrome CDP reload failed: %s, trying xdotool F5", reload_err)
                try:
                    env._execute_shell_on_vm("DISPLAY=:0 xdotool key F5", timeout=10)
                except Exception:
                    pass
            time.sleep(10)


# ---------------------------------------------------------------------------
# Manager followup: file-based IPC with the MACU orchestrator
# ---------------------------------------------------------------------------

def _wait_for_manager_decision(signal_dir: str, timeout: int = 300) -> dict | None:
    """Signal the orchestrator that the CUA is ready for review, then wait.

    Writes an empty ``ready_for_review`` file to *signal_dir* and polls for
    ``manager_decision.json``.  Returns the parsed decision dict or ``None``
    on timeout.
    """
    ready_path = os.path.join(signal_dir, "ready_for_review")
    decision_path = os.path.join(signal_dir, "manager_decision.json")

    # Signal readiness
    with open(ready_path, "w") as f:
        f.write("")
    logger.info("Wrote ready_for_review signal to %s", ready_path)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(decision_path):
            try:
                with open(decision_path, "r") as f:
                    decision = json.load(f)
                logger.info("Received manager decision: %s", decision)
                return decision
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read manager decision, retrying: %s", exc)
        time.sleep(2)

    logger.warning("Timed out waiting for manager decision after %ds", timeout)
    return None


def _run_filemgmt_ipc(env, signal_dir: str, example_result_dir: str, max_bytes: int, timeout: int = 300) -> None:
    """File-management IPC: signal the orchestrator, wait for copy actions,
    then download each requested guest file via the still-live VM controller.

    This must run before subprocess cleanup so downloads happen while
    ``env.controller.get_file()`` still works.

    Writes:
      * ``<signal_dir>/ready_for_filemgmt``              — signal file
      * ``<example_result_dir>/guest_files_dump/<name>`` — downloaded bytes
      * ``<signal_dir>/filemgmt_downloads.json``         — mapping of
         guest_path → host local path, read by the orchestrator post-exit.

    Reads:
      * ``<signal_dir>/filemgmt_decision.json`` — {"actions": [...]} from
         the orchestrator. Each action must have ``src_path`` (an absolute
         guest path). Any additional fields (e.g. ``archive_name``) are
         carried through unchanged — the orchestrator applies them after
         this subprocess exits.

    Best-effort: failures are logged and swallowed. A timeout or malformed
    response still leaves the subprocess free to exit cleanly.
    """
    ready_path = os.path.join(signal_dir, "ready_for_filemgmt")
    decision_path = os.path.join(signal_dir, "filemgmt_decision.json")
    downloads_manifest_path = os.path.join(signal_dir, "filemgmt_downloads.json")
    dump_dir = os.path.join(example_result_dir, "guest_files_dump")

    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError as exc:
        logger.warning("filemgmt IPC: failed to create dump dir %s: %s", dump_dir, exc)
        return

    try:
        with open(ready_path, "w") as f:
            f.write("")
    except OSError as exc:
        logger.warning("filemgmt IPC: failed to write ready signal %s: %s", ready_path, exc)
        return
    logger.info("Wrote ready_for_filemgmt signal to %s", ready_path)

    deadline = time.time() + timeout
    decision = None
    while time.time() < deadline:
        if os.path.exists(decision_path):
            try:
                with open(decision_path, "r") as f:
                    decision = json.load(f)
                break
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("filemgmt IPC: failed to read decision, retrying: %s", exc)
        time.sleep(2)

    if decision is None:
        logger.warning("filemgmt IPC: timed out waiting for decision after %ds", timeout)
        return

    actions = decision.get("actions") if isinstance(decision, dict) else None
    if not isinstance(actions, list) or not actions:
        logger.info("filemgmt IPC: no copy actions requested")
        try:
            with open(downloads_manifest_path, "w") as f:
                json.dump({}, f)
        except OSError:
            pass
        return

    # Collect unique guest paths to download (an action may target the same
    # source file twice, e.g. archive AND stage).
    unique_paths: list[str] = []
    seen: set = set()
    for a in actions:
        if not isinstance(a, dict):
            continue
        p = a.get("src_path")
        if isinstance(p, str) and p and p not in seen:
            seen.add(p)
            unique_paths.append(p)

    downloads: dict = {}
    for guest_path in unique_paths:
        base = os.path.basename(guest_path) or "file"
        # Disambiguate with a short hash of the full guest path in case two
        # src paths have the same basename.
        tag = f"{abs(hash(guest_path)) & 0xFFFF:04x}"
        local_path = os.path.join(dump_dir, f"{tag}_{base}")
        try:
            blob = env.controller.get_file(guest_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("filemgmt IPC: get_file(%s) raised: %s", guest_path, exc)
            continue
        if blob is None:
            logger.warning("filemgmt IPC: get_file(%s) returned None", guest_path)
            continue
        if len(blob) > max_bytes:
            logger.warning(
                "filemgmt IPC: %s is %d bytes (> max_bytes=%d), skipping",
                guest_path, len(blob), max_bytes,
            )
            continue
        try:
            with open(local_path, "wb") as f:
                f.write(blob)
        except OSError as exc:
            logger.warning("filemgmt IPC: write %s failed: %s", local_path, exc)
            continue
        downloads[guest_path] = local_path
        logger.info("filemgmt IPC: downloaded %s -> %s (%d bytes)",
                    guest_path, local_path, len(blob))

    try:
        with open(downloads_manifest_path, "w") as f:
            json.dump(downloads, f)
        logger.info("filemgmt IPC: wrote manifest %s with %d entries",
                    downloads_manifest_path, len(downloads))
    except OSError as exc:
        logger.warning("filemgmt IPC: failed to write manifest: %s", exc)


# ---------------------------------------------------------------------------
# Qwen agent: post-completion summary extraction
# ---------------------------------------------------------------------------

def _qwen_inference_snapshot(agent) -> tuple[float, int]:
    return (
        float(getattr(agent, "inference_time_total", 0.0) or 0.0),
        len(getattr(agent, "inference_intervals", []) or []),
    )


def _qwen_model_usage_since(agent, snapshot: tuple[float, int]) -> dict:
    started_total, started_count = snapshot
    intervals = list(getattr(agent, "inference_intervals", []) or [])[started_count:]
    inference_seconds = max(
        0.0,
        float(getattr(agent, "inference_time_total", 0.0) or 0.0) - started_total,
    )
    return {
        "model_time": inference_seconds,
        "inference_seconds": inference_seconds,
        "inference_intervals": intervals,
    }


def _extract_qwen_summary(
    agent,
    instruction: str,
    step_idx: int,
    example_result_dir: str,
    custom_prompt: str | None = None,
    num_past_screenshots: int | None = 5,
) -> bool:
    """Ask the Qwen agent to summarize its findings after task completion.

    Rebuilds the conversation with all textual turns and agent responses,
    includes the final assistant response, and appends a user message asking
    for a plain-text summary. By default only the last 5 screenshots are
    included as images; pass ``num_past_screenshots=None`` to include every
    screenshot.

    If *custom_prompt* is provided it replaces the default summary prompt.
    """
    from datetime import datetime as _dt
    total_steps = len(agent.screenshots)
    if total_steps == 0:
        return False

    # Use a text-only system prompt for summary extraction.  The original
    # tool-heavy prompt (with tool definitions and "Output exactly: Action,
    # <tool_call>") caused the model to emit <tool_call> blocks instead of
    # plain text ~30% of the time.
    system_prompt = (
        "You are a helpful assistant. You previously used a computer to "
        "complete a task. Now answer the user's follow-up question using "
        "only the information you gathered during the task.\n\n"
        "Respond with plain text only. Do NOT output any <tool_call> "
        "blocks, function calls, or action descriptions.\n\n"
        f"The current date is {_dt.today().strftime('%A, %B %d, %Y')}."
    )

    # ---- rebuild conversation, capping only screenshot image payloads ----
    start_step = 1
    if num_past_screenshots is None:
        first_screenshot_step = 1
    else:
        if num_past_screenshots < 1:
            raise ValueError("num_past_screenshots must be positive or None")
        first_screenshot_step = max(1, total_steps - num_past_screenshots + 1)

    action_history = [
        f"Step {i + 1}: {agent.actions[i]}"
        for i in range(len(agent.actions))
    ]
    action_history_str = "\n".join(action_history) if action_history else "None"

    instruction_prompt = (
        f"\nThe following is the task instruction and full action history "
        f"from the computer-use trajectory.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Full action history:\n"
        f"{action_history_str}"
    )

    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = agent._should_collapse_step(step_num)
        include_screenshot = step_num >= first_screenshot_step

        # — user turn (screenshot + instruction on first turn) —
        if not include_screenshot:
            parts = [{"type": "text", "text": "[Screenshot omitted from summary context.]"}]
            if is_first_turn:
                user_content = [
                    {"type": "text", "text": instruction_prompt},
                    *parts,
                ]
            else:
                user_content = agent._wrap_tool_response(parts)
        elif is_collapsed:
            parts = [{"type": "text", "text": agent.collapse_text}]
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = agent._wrap_tool_response(parts)
        else:
            img_url = f"data:image/png;base64,{agent.screenshots[step_num - 1]}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = agent._wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )
        messages.append({"role": "user", "content": user_content})

        # — assistant turn: include ALL responses (predict() skips the last
        #   one because it's generating it; here we want the full history) —
        if (step_num - 1) < len(agent.responses):
            # messages.append(agent._assistant_message_for_step(step_num - 1))
            response_idx = step_num - 1
            messages.append(agent._assistant_message_for_step(response_idx))
            messages.extend(agent._tool_response_messages_for_step(response_idx))

    # ---- append summary request ----
    if custom_prompt is not None:
        summary_prompt = custom_prompt
    else:
        summary_prompt = (
            "You have completed the task above. Now provide a COMPLETE and DETAILED "
            "summary of ALL the information you found during the task. Include every "
            "specific data point: names, numbers, prices, rates, dates, availability, "
            "URLs, etc.  If some information was not available or a page was sold out, "
            "state that clearly.\n\n"
            "Respond with ONLY the summary text. Do NOT call any tools or functions."
        )
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": summary_prompt}],
    })

    inference_snapshot = _qwen_inference_snapshot(agent)
    try:
        summary = agent.call_llm(
            {
                "model": agent.model,
                "messages": messages,
                "max_tokens": agent.max_tokens,
                "temperature": 0.0,
                "top_p": 0.9,
            },
            agent.model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Summary extraction failed after retries; keeping non-summary response: %s",
            exc,
        )
        return False
    model_usage = _qwen_model_usage_since(agent, inference_snapshot)

    if summary and summary.strip():
        logger.info("Summary extraction response: %s", summary)
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": "SUMMARY",
                        "response": summary,
                        "reward": 0,
                        "done": True,
                        "info": {"summary_extraction": True},
                        "screenshot_file": "",
                        "model_usage": model_usage,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
        return True
    else:
        logger.warning("Summary extraction returned empty response")
        return False


# ---------------------------------------------------------------------------
# Qwen agent: inject a followup user turn and predict
# ---------------------------------------------------------------------------

def _predict_with_followup_qwen(
    agent,
    instruction: str,
    followup_message: str,
    obs: dict,
) -> tuple[str, list[str]]:
    """Like ``agent.predict()`` but appends *followup_message* as a new user
    turn after the last assistant response, instead of embedding it in the
    first (instruction) turn.

    The conversation is rebuilt from agent state (screenshots / responses /
    actions) exactly as ``predict()`` does, the agent's own tool-enabled
    system prompt is reconstructed, and ``agent.parse_response()`` is used
    for action extraction — so no internal logic is duplicated.

    Agent state (screenshots, responses, actions) is updated the same way
    ``predict()`` would, so subsequent ``predict()`` calls continue
    seamlessly.
    """
    import base64
    from io import BytesIO
    from PIL import Image

    # ---- process screenshot (same as predict()) ----
    from osworld.mm_agents.qwen35vl_agent import process_image

    screenshot_bytes = obs["screenshot"]
    original_img = Image.open(BytesIO(screenshot_bytes))
    original_width, original_height = original_img.size

    processed_b64 = process_image(
        screenshot_bytes,
        min_pixels=agent.min_pixels,
        max_pixels=agent.max_pixels,
    )
    processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
    processed_width, processed_height = processed_img.size

    agent.screenshots.append(processed_b64)
    total_steps = len(agent.screenshots)
    agent._update_folding_state(total_steps)

    start_step = max(1, total_steps - agent.history_n)

    # ---- build system prompt (same as predict()) ----
    system_prompt, tools_def = agent.build_tool_prompt(
        processed_width=processed_width,
        processed_height=processed_height,
    )

    # ---- build instruction prompt (same as predict()) ----
    previous_actions = [
        f"Step {i + 1}: {agent.actions[i]}"
        for i in range(0, min(start_step - 1, len(agent.actions)))
    ]
    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    instruction_prompt = (
        f"\nPlease generate the next move according to the UI screenshot, "
        f"instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n"
        f"{previous_actions_str}"
    )

    # ---- build messages (same as predict(), but include all responses) ----
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = agent._should_collapse_step(step_num)

        if is_collapsed:
            parts = [{"type": "text", "text": agent.collapse_text}]
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = agent._wrap_tool_response(parts)
        else:
            img_url = f"data:image/png;base64,{agent.screenshots[step_num - 1]}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = agent._wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )
        messages.append({"role": "user", "content": user_content})

        # Include ALL assistant responses (including the last one)
        if (step_num - 1) < len(agent.responses):
            # messages.append(agent._assistant_message_for_step(step_num - 1))
            response_idx = step_num - 1
            messages.append(agent._assistant_message_for_step(response_idx))
            messages.extend(agent._tool_response_messages_for_step(response_idx))

    # ---- append followup as a NEW user turn ----
    img_url = f"data:image/png;base64,{agent.screenshots[-1]}"
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": followup_message},
        ],
    })

    # ---- call LLM ----
    response = agent.call_llm(agent._request_payload(messages, tools_def), agent.model)

    logger.info("Followup predict response: %s", response)
    agent.record_response(response or "")

    # ---- parse actions (reuse agent's own parser) ----
    low_level_instruction, pyautogui_code = agent.parse_response(
        response or "",
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
    )

    logger.info("Followup low-level instruction: %s", low_level_instruction)
    logger.info("Followup pyautogui code: %s", pyautogui_code)
    agent.actions.append(low_level_instruction)

    return response or "", pyautogui_code


# ---------------------------------------------------------------------------
# Qwen agent: run_single_example
# ---------------------------------------------------------------------------

def _predict_qwen_with_retries(
    agent,
    instruction: str,
    obs: dict,
    step_idx: int,
    *,
    call_fn=None,
    log_prefix: str = "",
) -> tuple[str, list[str]]:
    """Run one Qwen prediction, retrying no-action responses without
    polluting the agent's conversation state.

    ``agent.predict()`` records screenshots/actions/responses as it goes. If
    parsing yields no executable action, the runner should not leave that
    failed attempt in history before trying a different sampling temperature.
    """
    state_before_predict = agent.snapshot_state()

    def do_predict():
        if call_fn is not None:
            return call_fn()
        return agent.predict(instruction, obs)

    try:
        response, actions = do_predict()
    except Exception:
        agent.restore_state(state_before_predict)
        raise

    if actions:
        return response, actions

    retry_temps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    original_temp = agent.temperature
    try:
        for retry_idx, temp in enumerate(retry_temps):
            logger.warning(
                "%sNo actions returned at step %d, retry %d/%d with temperature=%.1f",
                log_prefix,
                step_idx + 1,
                retry_idx + 1,
                len(retry_temps),
                temp,
            )
            agent.restore_state(state_before_predict)
            agent.temperature = temp
            try:
                response, actions = do_predict()
            except Exception as exc:
                logger.warning(
                    "%spredict retry failed at step %d (%s)",
                    log_prefix,
                    step_idx + 1,
                    exc,
                )
                response, actions = "", []
            if actions:
                return response, actions
        agent.restore_state(state_before_predict)
        return response, actions
    finally:
        agent.temperature = original_temp


def _run_single_example_qwen(
    agent,
    env,
    example: Dict,
    max_steps: int,
    instruction: str,
    args: argparse.Namespace,
    example_result_dir: str,
    scores: list,
) -> None:
    runtime_logger = setup_logger(example, example_result_dir)

    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    time.sleep(args.env_ready_wait_seconds)
    _post_reset_network_check(env, example, args)

    # MACU file management: take a pre-task snapshot of the guest filesystem
    # so the orchestrator can later compute a diff against the post-task state.
    if _macu_file_management_enabled(args):
        _write_file_snapshot(env, example_result_dir, _file_scan_roots(), "pre")

    obs = env._get_obs()
    done = False
    step_idx = 0
    final_action = None

    env.controller.start_recording()
    try:
        while not done and step_idx < max_steps:
            if step_idx > 0 and step_idx % 10 == 0:
                if not _check_vm_network(env):
                    _recover_vm_network(env)

            # Capture a fresh pre-action observation so the agent reasons
            # over the *current* VM state, and persist it for analysis /
            # visualization.
            # obs = env._get_obs()
            # if obs.get("screenshot") is None:
            #     logger.warning("Step %d: no screenshot available, skipping predict", step_idx + 1)
            #     step_idx += 1
            #     continue

            # before_action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            # before_screenshot_name = f"step_{step_idx + 1}_{before_action_timestamp}_before.png"
            # with open(os.path.join(example_result_dir, before_screenshot_name), "wb") as handle:
            #     handle.write(obs["screenshot"])

            inference_snapshot = _qwen_inference_snapshot(agent)
            try:
                response, actions = _predict_qwen_with_retries(
                    agent, instruction, obs, step_idx
                )
            except Exception as _pred_exc:
                logger.warning("Step %d: agent.predict failed (%s), skipping step", step_idx + 1, _pred_exc)
                step_idx += 1
                continue
            model_usage = _qwen_model_usage_since(agent, inference_snapshot)
            reasoning = agent.reasonings[-1] if getattr(agent, "reasonings", None) else ""
            if not actions:
                logger.warning("No actions after all retries at step %d, stopping.", step_idx + 1)
                break

            for action in actions:
                final_action = action
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                logger.info("Step %d: %s", step_idx + 1, action)

                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)

                # Retry screenshot capture if it failed
                if obs.get("screenshot") is None:
                    for _ss_retry in range(1, 4):
                        logger.warning(
                            "Step %d: screenshot was None, retry %d/3",
                            step_idx + 1, _ss_retry,
                        )
                        time.sleep(2)
                        try:
                            obs["screenshot"] = env.controller.get_screenshot()
                        except Exception as _ss_exc:
                            logger.warning("Screenshot retry %d failed: %s", _ss_retry, _ss_exc)
                        if obs.get("screenshot") is not None:
                            break

                screenshot_name = f"step_{step_idx + 1}_{action_timestamp}.png"
                if obs.get("screenshot") is not None:
                    with open(os.path.join(example_result_dir, screenshot_name), "wb") as handle:
                        handle.write(obs["screenshot"])
                else:
                    logger.warning("Step %d: screenshot still None after retries, skipping save", step_idx + 1)

                with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step_num": step_idx + 1,
                                "action_timestamp": action_timestamp,
                                "action": action,
                                "response": response,
                                "reasoning": reasoning,
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "screenshot_file": screenshot_name,
                                "model_usage": model_usage,
                                # "before_screenshot": before_screenshot_name,
                            },
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")

                if done:
                    break
            step_idx += 1

        # --- Post-completion summary extraction for less capable models ---
        # After the agent finishes (terminate or max_steps), make one final
        # LLM call asking it to report all findings in plain text.  This
        # ensures the last traj.jsonl entry contains the actual information
        # the task requested, which is then picked up by extract_cua_result.
        try:
            _extract_qwen_summary(agent, instruction, step_idx, example_result_dir)
        except Exception as exc:
            logger.warning("Summary extraction failed: %s", exc)

        # --- Manager followup (only when launched by MACU orchestrator) ---
        if getattr(args, "enable_manager_followup", False):
            signal_dir = os.path.dirname(args.result_dir)
            decision = _wait_for_manager_decision(signal_dir)

            followup_type = decision.get("type") if decision else None

            # If no steps remain, downgrade "continue" to "summary"
            if followup_type == "continue" and step_idx >= max_steps:
                logger.info("No steps remaining, downgrading continue to summary")
                followup_type = "summary"

            if followup_type == "continue":
                followup_msg = decision["message"]
                followup_user_msg = (
                    f"The manager has reviewed your work and needs additional "
                    f"information:\n{followup_msg}\n\n"
                    f"Please continue working on the computer to find this."
                )
                logger.info("Manager requested continue followup for Qwen agent")
                obs = env._get_obs()
                done = False
                is_first_followup_step = True
                while not done and step_idx < max_steps:
                    if step_idx > 0 and step_idx % 10 == 0:
                        if not _check_vm_network(env):
                            _recover_vm_network(env)

                    # obs = env._get_obs()
                    # before_screenshot_name = ""
                    # if obs.get("screenshot") is not None:
                    #     before_action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                    #     before_screenshot_name = f"step_{step_idx + 1}_{before_action_timestamp}_before.png"
                    #     with open(os.path.join(example_result_dir, before_screenshot_name), "wb") as handle:
                    #         handle.write(obs["screenshot"])

                    if is_first_followup_step:
                        # First step: inject followup as a new user turn
                        # after the conversation history (not in the
                        # instruction turn).
                        inference_snapshot = _qwen_inference_snapshot(agent)
                        response, actions = _predict_qwen_with_retries(
                            agent,
                            instruction,
                            obs,
                            step_idx,
                            call_fn=lambda: _predict_with_followup_qwen(
                                agent, instruction, followup_user_msg, obs,
                            ),
                            log_prefix="Followup: ",
                        )
                        is_first_followup_step = False
                    else:
                        inference_snapshot = _qwen_inference_snapshot(agent)
                        response, actions = _predict_qwen_with_retries(
                            agent,
                            instruction,
                            obs,
                            step_idx,
                            log_prefix="Followup: ",
                        )
                    model_usage = _qwen_model_usage_since(agent, inference_snapshot)
                    reasoning = agent.reasonings[-1] if getattr(agent, "reasonings", None) else ""
                    if not actions:
                        logger.warning("Followup: no actions after retries at step %d, stopping.", step_idx + 1)
                        break

                    for action in actions:
                        final_action = action
                        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                        logger.info("Followup step %d: %s", step_idx + 1, action)

                        obs, reward, done, info = env.step(action, args.sleep_after_execution)
                        logger.info("Reward: %.2f", reward)
                        logger.info("Done: %s", done)

                        screenshot_name = f"step_{step_idx + 1}_{action_timestamp}.png"
                        with open(os.path.join(example_result_dir, screenshot_name), "wb") as handle:
                            handle.write(obs["screenshot"])

                        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "step_num": step_idx + 1,
                                        "action_timestamp": action_timestamp,
                                        "action": action,
                                        "response": response,
                                        "reasoning": reasoning,
                                        "reward": reward,
                                        "done": done,
                                        "info": info,
                                        "screenshot_file": screenshot_name,
                                        "model_usage": model_usage,
                                        # "before_screenshot": before_screenshot_name,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                            handle.write("\n")

                        if done:
                            break
                    step_idx += 1

                # Summary extraction for followup results
                try:
                    _extract_qwen_summary(agent, instruction, step_idx, example_result_dir)
                except Exception as exc:
                    logger.warning("Followup summary extraction failed: %s", exc)

            elif followup_type == "summary":
                logger.info("Manager requested summary followup for Qwen agent")
                summary_prompt = (
                    f"You have completed the task above. The manager needs the "
                    f"following additional information from your browsing session:\n\n"
                    f"{decision['message']}\n\n"
                    f"Review your actions and provide a detailed response with every "
                    f"specific data point. Respond with ONLY the answer text. "
                    f"Do NOT call any tools or functions."
                )
                try:
                    _extract_qwen_summary(
                        agent, instruction, step_idx, example_result_dir,
                        custom_prompt=summary_prompt,
                    )
                except Exception as exc:
                    logger.warning("Followup summary extraction failed: %s", exc)

        time.sleep(args.post_task_wait_seconds)
        result = 1.0 if is_done_action(final_action) and not is_fail_action(final_action) else 0.0
        logger.info("Mind2web self-reported result: %.2f", result)
        scores.append(result)

        # MACU file management: take a post-task snapshot of the guest filesystem
        # so the orchestrator can diff against files_pre.json. Done after followup
        # finishes (so we capture any files the followup loop produced) but before
        # the process exits.
        if _macu_file_management_enabled(args):
            _write_file_snapshot(env, example_result_dir, _file_scan_roots(), "post")
            # Second IPC round-trip with the orchestrator: ask it which files
            # to copy, then download them via the STILL-LIVE VM controller.
            _run_filemgmt_ipc(env, os.path.dirname(args.result_dir), example_result_dir, FILE_DOWNLOAD_MAX_BYTES)

        with open(os.path.join(example_result_dir, "completed.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{result}\n")
        log_task_completion(example, result, example_result_dir, args)
    finally:
        try:
            env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
        except Exception as rec_err:
            logger.error("Failed to end recording: %s", rec_err)


# ---------------------------------------------------------------------------
# OpenAI (GPT-5.4) agent: post-completion summary extraction
# ---------------------------------------------------------------------------

def _extract_openai_summary(
    agent,
    env,
    step_idx: int,
    example_result_dir: str,
    summary_prompt: str,
) -> None:
    """One-shot text summary from the GPT-5.4 agent (no actions executed).

    Injects *summary_prompt* as a new user message via ``pending_input_items``
    and calls ``predict()`` once.  The agent's conversation history is preserved
    on the API side via ``previous_response_id``, so the model sees all prior
    actions and can answer from memory.
    """
    obs = env._get_obs()
    agent.pending_input_items = [{
        "role": "user",
        "content": [{"type": "input_text", "text": summary_prompt}],
    }]
    response, _actions = agent.predict("", obs)
    summary = response.get("response", "")
    if summary and summary.strip():
        logger.info("OpenAI summary extraction response: %s", summary)
        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": {},
                        "response": summary,
                        "reward": 0,
                        "done": True,
                        "info": {"summary_extraction": True},
                        "screenshot_file": "",
                        "state_correct": True,
                        "model_usage": response.get("model_usage", {}),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    else:
        logger.warning("OpenAI summary extraction returned empty response")


# ---------------------------------------------------------------------------
# OpenAI (GPT-5.4) agent: run_single_example
# ---------------------------------------------------------------------------

def _run_single_example_openai(
    agent,
    env,
    example: Dict,
    max_steps: int,
    instruction: str,
    args: argparse.Namespace,
    example_result_dir: str,
    scores: list,
) -> None:
    runtime_logger = setup_logger(example, example_result_dir)

    env.reset(task_config=example)
    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()

    time.sleep(args.env_ready_wait_seconds)
    _post_reset_network_check(env, example, args)

    # MACU file management: take a pre-task snapshot of the guest filesystem
    # so the orchestrator can later compute a diff against the post-task state.
    if _macu_file_management_enabled(args):
        _write_file_snapshot(env, example_result_dir, _file_scan_roots(), "pre")

    obs = env._get_obs()
    done = False
    step_idx = 0
    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")

    env.controller.start_recording()
    try:
        while not done and step_idx < max_steps:
            if step_idx > 0 and step_idx % 10 == 0:
                if not _check_vm_network(env):
                    _recover_vm_network(env)

            response, actions = agent.predict(instruction, obs)
            logger.info("Agent response: %s", response.get("response", ""))
            logger.info("Agent state_correct: %s", response.get("state_correct", False))
            logger.info("Agent model_usage: %s", response.get("model_usage", {}))

            # Check if computer_call exists; if not, this is the final response
            if not response.get("computer_call_present", False):
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step_num": step_idx + 1,
                                "action_timestamp": action_timestamp,
                                "response": response.get("response", ""),
                                "state_correct": response.get("state_correct", False),
                                "model_usage": response.get("model_usage", {}),
                                "action": {},
                                "reward": 0,
                                "done": True,
                                "info": {},
                                "screenshot_file": "",
                            },
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")
                logger.info("No computer_call in response, treating as final response: %s", response.get("response", ""))
                break

            done = not response.get("state_correct", False)
            if not actions:
                logger.warning("No actions returned at step %d", step_idx + 1)
                break

            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                logger.info("Step %d: %s", step_idx + 1, action)
                obs, reward, done, info, step_info = agent.step(action)

                if not done and not response.get("state_correct", False):
                    done = True

                controller_result = info.get("controller_result", {}) if isinstance(info, dict) else {}
                if controller_result:
                    logger.info("Controller returncode: %s", controller_result.get("returncode"))
                    controller_error = str(controller_result.get("error") or "").strip()
                    controller_output = str(controller_result.get("output") or "").strip()
                    if controller_error:
                        logger.warning("Controller stderr: %s", controller_error)
                    elif controller_output:
                        logger.info("Controller stdout: %s", controller_output)

                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)

                screenshot_name = f"step_{step_idx + 1}_{action_timestamp}.png"
                if obs.get("screenshot") is not None:
                    with open(os.path.join(example_result_dir, screenshot_name), "wb") as handle:
                        handle.write(obs["screenshot"])
                else:
                    logger.warning("Screenshot is None at step %d, skipping save", step_idx + 1)

                action_for_log = dict(action)
                if "pending_checks" in action_for_log:
                    action_for_log["pending_checks"] = "<omitted>"

                with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step_num": step_idx + 1,
                                "action_timestamp": action_timestamp,
                                "response": response.get("response", ""),
                                "state_correct": response.get("state_correct", False),
                                "model_usage": response.get("model_usage", {}),
                                "action": action_for_log,
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "step_info": step_info,
                                "screenshot_file": screenshot_name,
                            },
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")

                if done:
                    break
            step_idx += 1

        # --- Manager followup (only when launched by MACU orchestrator) ---
        if getattr(args, "enable_manager_followup", False):
            signal_dir = os.path.dirname(args.result_dir)
            decision = _wait_for_manager_decision(signal_dir)

            followup_type = decision.get("type") if decision else None

            # If no steps remain, downgrade "continue" to "summary"
            if followup_type == "continue" and step_idx >= max_steps:
                logger.info("No steps remaining, downgrading continue to summary")
                followup_type = "summary"

            if followup_type == "continue":
                followup_msg = decision["message"]
                logger.info("Manager requested continue followup for OpenAI agent")
                agent.pending_input_items = [{
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            f"The manager needs additional information: {followup_msg}. "
                            f"Please continue working on the computer to find this."
                        ),
                    }],
                }]
                obs = env._get_obs()
                done = False
                while not done and step_idx < max_steps:
                    if step_idx > 0 and step_idx % 10 == 0:
                        if not _check_vm_network(env):
                            _recover_vm_network(env)

                    response, actions = agent.predict(instruction, obs)
                    logger.info("Followup response: %s", response.get("response", ""))

                    if not response.get("computer_call_present", False):
                        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "step_num": step_idx + 1,
                                        "action_timestamp": action_timestamp,
                                        "response": response.get("response", ""),
                                        "state_correct": response.get("state_correct", False),
                                        "model_usage": response.get("model_usage", {}),
                                        "action": {},
                                        "reward": 0,
                                        "done": True,
                                        "info": {},
                                        "screenshot_file": "",
                                    },
                                    ensure_ascii=False,
                                )
                            )
                            handle.write("\n")
                        logger.info("Followup: final response (no computer_call): %s", response.get("response", ""))
                        break

                    done = not response.get("state_correct", False)
                    if not actions:
                        logger.warning("Followup: no actions at step %d", step_idx + 1)
                        break

                    for action in actions:
                        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                        logger.info("Followup step %d: %s", step_idx + 1, action)
                        obs, reward, done, info, step_info = agent.step(action)

                        if not done and not response.get("state_correct", False):
                            done = True

                        controller_result = info.get("controller_result", {}) if isinstance(info, dict) else {}
                        if controller_result:
                            logger.info("Controller returncode: %s", controller_result.get("returncode"))

                        logger.info("Reward: %.2f  Done: %s", reward, done)

                        screenshot_name = f"step_{step_idx + 1}_{action_timestamp}.png"
                        if obs.get("screenshot") is not None:
                            with open(os.path.join(example_result_dir, screenshot_name), "wb") as handle:
                                handle.write(obs["screenshot"])

                        action_for_log = dict(action)
                        if "pending_checks" in action_for_log:
                            action_for_log["pending_checks"] = "<omitted>"

                        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "step_num": step_idx + 1,
                                        "action_timestamp": action_timestamp,
                                        "response": response.get("response", ""),
                                        "state_correct": response.get("state_correct", False),
                                        "model_usage": response.get("model_usage", {}),
                                        "action": action_for_log,
                                        "reward": reward,
                                        "done": done,
                                        "info": info,
                                        "step_info": step_info,
                                        "screenshot_file": screenshot_name,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                            handle.write("\n")

                        if done:
                            break
                    step_idx += 1

            elif followup_type == "summary":
                logger.info("Manager requested summary followup for OpenAI agent")
                summary_prompt = (
                    f"The manager needs the following additional information from "
                    f"your session:\n\n{decision['message']}\n\n"
                    f"Provide a detailed response. Do not take any computer actions."
                )
                try:
                    _extract_openai_summary(
                        agent, env, step_idx, example_result_dir, summary_prompt,
                    )
                except Exception as exc:
                    logger.warning("OpenAI followup summary extraction failed: %s", exc)

        time.sleep(args.post_task_wait_seconds)
        result = 1.0
        logger.info("Mind2web result: %.2f", result)
        scores.append(result)

        # MACU file management: take a post-task snapshot of the guest filesystem
        # so the orchestrator can diff against files_pre.json. Done after followup
        # finishes (so we capture any files the followup loop produced) but before
        # the process exits.
        if _macu_file_management_enabled(args):
            _write_file_snapshot(env, example_result_dir, _file_scan_roots(), "post")
            # Second IPC round-trip with the orchestrator: ask it which files
            # to copy, then download them via the STILL-LIVE VM controller.
            _run_filemgmt_ipc(env, os.path.dirname(args.result_dir), example_result_dir, FILE_DOWNLOAD_MAX_BYTES)

        with open(os.path.join(example_result_dir, "completed.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{result}\n")
        log_task_completion(example, result, example_result_dir, args)
    finally:
        try:
            env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
        except Exception as rec_err:
            logger.error("Failed to end recording: %s", rec_err)


# ---------------------------------------------------------------------------
# Dispatch: run_single_example_mind2web
# ---------------------------------------------------------------------------

def run_single_example_mind2web(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    if args.cua_provider == "qwen":
        _run_single_example_qwen(agent, env, example, max_steps, instruction, args, example_result_dir, scores)
    else:
        _run_single_example_openai(agent, env, example, max_steps, instruction, args, example_result_dir, scores)


# ---------------------------------------------------------------------------
# Task distribution and worker processes
# ---------------------------------------------------------------------------

def distribute_tasks(test_all_meta: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    all_tasks: List[Tuple[str, str]] = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def _create_agent(env, args):
    """Create the appropriate agent based on --cua_provider."""
    if args.cua_provider == "qwen":
        return Qwen35VLAgent(
            history_n=args.history_n,
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            action_space=args.action_space,
            observation_type=args.observation_type,
            coordinate_type=args.coord,
            add_thought_prefix=args.add_thought_prefix,
            image_max=args.image_max,
            fold_size=args.fold_size,
            keep_reasoning=args.keep_reasoning,
            preserve_reasoning_content=args.preserve_reasoning_content,
            use_native_tool_calling=not args.disable_native_tool_calling,
            max_tool_calls_per_turn=args.max_tool_calls_per_turn,
        )
    else:
        return GPT54Agent(
            env=env,
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            max_trajectory_length=args.max_trajectory_length,
            client_password=args.client_password,
            provider_name=args.provider_name,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            sleep_after_execution=args.sleep_after_execution,
            reasoning_effort=args.reasoning_effort,
        )


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list):
    env = None
    try:
        region = args.region
        screen_size = (args.screen_width, args.screen_height)
        snapshot_name = "init_state"
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP

            snapshot_name = IMAGE_ID_MAP[region].get(screen_size, IMAGE_ID_MAP[region][(1920, 1080)])

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=region,
            snapshot_name=snapshot_name,
            screen_size=screen_size,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=False,
            client_password=args.client_password,
        )

        # Skip snapshot revert on first reset() -- keep the VM state from the
        # previous worker so the new task continues where the last one left off.
        env.is_environment_used = False

        # Record which VM this subprocess ended up attached to so the
        # orchestrator can designate one as the final aggregated VM.
        if args.vm_info_path:
            try:
                vm_path = getattr(env, "path_to_vm", None)
                # Resolve to absolute so downstream consumers (e.g. OSWorld
                # reward scripts) running from a different cwd can use the path.
                if vm_path:
                    vm_path = os.path.abspath(vm_path)
                os.makedirs(os.path.dirname(os.path.abspath(args.vm_info_path)), exist_ok=True)
                vm_info_payload = {
                    "vm_path": vm_path,
                    "provider": args.provider_name,
                    # Expose vm_ip and server_port so the MACU orchestrator can
                    # talk directly to the OSWorld in-VM HTTP server (used by the
                    # file-management feature for downloading guest files).
                    "vm_ip": getattr(env, "vm_ip", None),
                    "server_port": getattr(env, "server_port", 5000),
                }
                with open(args.vm_info_path, "w") as vm_info_f:
                    json.dump(vm_info_payload, vm_info_f)
                logger.info("Wrote VM info (%s) to %s", vm_path, args.vm_info_path)
            except Exception as vm_info_err:
                logger.warning("Failed to write vm_info to %s: %s", args.vm_info_path, vm_info_err)

        agent = _create_agent(env, args)
        logger.info("Process %s started (cua_provider=%s).", current_process().name, args.cua_provider)

        while True:
            try:
                domain, example_id = task_queue.get(timeout=5)
            except Exception:
                break

            config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
            example_result_dir = os.path.join(
                args.result_dir, args.action_space, args.observation_type, args.model, domain, example_id
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                with open(config_file, "r", encoding="utf-8") as handle:
                    example = json.load(handle)
                logger.info("[%s][Domain]: %s", current_process().name, domain)
                logger.info("[%s][Example ID]: %s", current_process().name, example_id)
                logger.info("[%s][Instruction]: %s", current_process().name, example["instruction"])

                run_single_example_mind2web(
                    agent,
                    env,
                    example,
                    args.max_steps,
                    example["instruction"],
                    args,
                    example_result_dir,
                    shared_scores,
                )
            except Exception as err:
                import traceback

                logger.error("Exception in %s %s/%s: %s", current_process().name, domain, example_id, err)
                logger.error(traceback.format_exc())
                try:
                    log_task_error({"id": example_id}, str(err), example_result_dir, args)
                except Exception as log_err:
                    logger.error("Failed to log error: %s", log_err)
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"Error": f"{domain}/{example_id} - {err}"}))
                    handle.write("\n")

    except Exception as err:
        import traceback

        logger.error("Process-level error in %s: %s", current_process().name, err)
        logger.error(traceback.format_exc())
    finally:
        logger.info("%s cleaning up environment...", current_process().name)
        if env is not None:
            try:
                env.release()
                logger.info("%s environment released (VM still running)", current_process().name)
            except Exception as close_err:
                logger.error("%s error during cleanup: %s", current_process().name, close_err)


def signal_handler(signum, _frame):
    global is_terminating, processes
    if is_terminating:
        return
    is_terminating = True
    logger.info("Received signal %s, shutting down...", signum)

    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception as err:
                logger.error("Error terminating process %s: %s", p.name, err)
    time.sleep(1)
    for p in processes:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception as err:
                logger.error("Error killing process %s: %s", p.name, err)
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: Dict[str, List[str]]) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info("Total tasks: %d", len(all_tasks))

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)

        processes = []
        for i in range(max(1, int(args.num_envs))):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"EnvProcess-{i + 1}",
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info("Started process %s with PID %s", p.name, p.pid)

        while True:
            alive_count = 0
            for idx, p in enumerate(processes):
                if not p.is_alive() and not task_queue.empty():
                    logger.warning("Process %s died, restarting...", p.name)
                    new_p = Process(
                        target=run_env_tasks,
                        args=(task_queue, args, shared_scores),
                        name=f"EnvProcess-Restart-{idx + 1}",
                    )
                    new_p.daemon = True
                    new_p.start()
                    processes[idx] = new_p
                elif p.is_alive():
                    alive_count += 1

            if task_queue.empty():
                logger.info("All tasks finished.")
                break
            if alive_count == 0:
                logger.error("All processes died, exiting.")
                break
            time.sleep(5)

        for p in processes:
            p.join()

        scores = list(shared_scores)
        avg = sum(scores) / len(scores) if scores else 0.0
        logger.info("Average self-reported completion score: %.4f", avg)


def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    dir_contents = os.listdir(example_path)
                    if "completed.txt" not in dir_contents and "result.txt" not in dir_contents:
                        for file_name in dir_contents:
                            os.remove(os.path.join(example_path, file_name))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json
    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]
    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result: List[float] = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if os.path.isdir(example_path):
                dir_contents = os.listdir(example_path)
                completed_file = None
                if "completed.txt" in dir_contents:
                    completed_file = "completed.txt"
                elif "result.txt" in dir_contents:
                    completed_file = "result.txt"
                if completed_file:
                    try:
                        with open(os.path.join(example_path, completed_file), "r", encoding="utf-8") as handle:
                            all_result.append(float(handle.read()))
                    except Exception:
                        all_result.append(0.0)
    if not all_result:
        print("New experiment, no result yet.")
        return None
    print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
    return all_result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _exit_code = 0
    try:
        generated_meta = prepare_mind2web_osworld_data(args)

        if args.specific_task_id:
            found = False
            filtered = {}
            for domain, ids in generated_meta.items():
                if args.specific_task_id in ids:
                    filtered[domain] = [args.specific_task_id]
                    found = True
                    break
            if not found:
                logger.error("specific_task_id not found: %s", args.specific_task_id)
                sys.exit(1)
            generated_meta = filtered
        elif args.domain != "all":
            if args.domain not in generated_meta:
                logger.error("domain not found: %s", args.domain)
                sys.exit(1)
            generated_meta = {args.domain: generated_meta[args.domain]}

        # Save args for reproducibility
        path_to_args = os.path.join(
            args.result_dir, args.action_space, args.observation_type, args.model, "args.json"
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)

        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            generated_meta,
        )
        left_info = "\n".join(f"{domain}: {len(ids)}" for domain, ids in test_file_list.items())
        logger.info("Left tasks:\n%s", left_info)

        get_result(args.action_space, args.model, args.observation_type, args.result_dir)
        test(args, test_file_list)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        _exit_code = 130
    except SystemExit as exc:
        _exit_code = exc.code if isinstance(exc.code, int) else 1
    except Exception as err:
        logger.error("Unexpected error in main process: %s", err, exc_info=True)
        signal_handler(signal.SIGTERM, None)
        _exit_code = 1
    finally:
        logger.info("Main process final cleanup...")
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        # Force-exit to avoid hanging on multiprocessing's atexit handler
        # which tries to join all active child processes. Releasing VMs keeps
        # QEMU alive under an apptainer container, and those grandchild
        # processes prevent a clean interpreter shutdown.
        # Flush handlers first so logs are not lost.
        logging.shutdown()
        os._exit(_exit_code)
