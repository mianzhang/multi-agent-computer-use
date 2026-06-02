#!/usr/bin/env python3
"""MACU research runner.

The script intentionally stays small: it exposes the CLI entrypoint and the
research-facing runner class, while bulky process, VM, file, graph, and result
plumbing lives in ``utils`` modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import utils.macu_runtime as _runtime  # noqa: E402
from utils.macu_runtime import (  # noqa: E402
    CUAProcess,
    _signal_handler,
)
from utils.graph_utils import (  # noqa: E402
    _append_replan_log,
    apply_replan_decision,
    build_merged_final_traj,
    persist_graph_snapshot,
)
from utils.manager_utils import (  # noqa: E402
    ask_manager_for_followup,
    ask_manager_for_replanning,
)
from utils.utils import extract_cua_result_from_dir  # noqa: E402
from utils.vm_utils import reset_vm_pool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
_RUNTIME_RUNNER = _runtime._RunSingleGraphRunner
_RUNTIME_RUN_WITH_TASKS_FILE = _runtime._run_with_tasks_file

DEFAULT_TASK_TIMEOUT_SECONDS = 1.5 * 3600
INFEASIBLE_VERDICT_PATTERNS = (
    re.compile(r"\binfeasible\b", re.IGNORECASE),
    re.compile(r"\bunfeasible\b", re.IGNORECASE),
    re.compile(r"\bimpossible\b", re.IGNORECASE),
    re.compile(r"\bnot\s+feasible\b", re.IGNORECASE),
    re.compile(r"\bnot\s+possible\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+be\s+(?:done|completed|performed|fulfilled|satisfied)\b", re.IGNORECASE),
)


def _contains_infeasible_verdict(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in INFEASIBLE_VERDICT_PATTERNS)


def _action_is_fail(action) -> bool:
    if isinstance(action, str):
        return action.strip().upper() == "FAIL"
    if isinstance(action, dict):
        for key in ("action_type", "action", "command"):
            if str(action.get(key, "")).strip().upper() == "FAIL":
                return True
        return False
    return False


def _subtask_has_fail_action(subtask_dir: Path) -> bool:
    for traj_path in subtask_dir.rglob("traj.jsonl"):
        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if _action_is_fail(entry.get("action")):
                        return True
        except OSError:
            continue
    return False


def _raw_task_id(task_text: str) -> str:
    """Build a stable filesystem-safe task id for a user-supplied task string."""
    slug = re.sub(r"[^a-z0-9]+", "-", task_text.lower()).strip("-")
    slug = slug[:48].strip("-") or "task"
    digest = hashlib.sha1(task_text.encode("utf-8")).hexdigest()[:8]
    return f"user_task_{slug}_{digest}"


def _write_raw_task_file(args: argparse.Namespace, task_text: str) -> str:
    """Materialize a raw task string as a one-item generic CUA task file."""
    task_id = args.task_id or _raw_task_id(task_text)
    args.task_id = task_id
    tasks_dir = Path(args.result_dir) / "_input_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = tasks_dir / f"{task_id}.json"
    payload = [
        {
            "task_id": task_id,
            "confirmed_task": task_text,
            "website": "",
            "no_initial_setup": True,
        }
    ]
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return str(tasks_file.resolve())


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse MACU runner flags and split out CUA subprocess passthrough args."""
    parser = argparse.ArgumentParser(
        description="MACU: Asynchronous Multi-Agent CUA orchestrator",
    )
    parser.add_argument(
        "tasks_file",
        nargs="?",
        type=str,
        help=(
            "Path to a CUA tasks JSON list or OSWorld domain-to-UUID task map. "
            "Omit when using --task."
        ),
    )
    parser.add_argument(
        "--task",
        dest="raw_task",
        type=str,
        default=None,
        help=(
            "Run one user-defined task string. MACU writes a one-item task file "
            "under --result-dir/_input_tasks and launches workers with no initial "
            "app setup."
        ),
    )
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--graph-gen-max-tokens", type=int, default=16384)
    parser.add_argument("--result-dir", type=str, default="runs_macu")
    parser.add_argument("--cua-script", type=str, default="scripts/run_cua.py")
    parser.add_argument(
        "--osworld-root",
        type=str,
        default=None,
        help="Path to the local OSWorld checkout. Required for OSWorld-backed runs.",
    )
    parser.add_argument("--cua-provider", type=str, default="qwen", choices=["qwen", "openai"])
    parser.add_argument(
        "--manager-provider",
        type=str,
        default="anthropic",
        choices=["openai", "anthropic", "huggingface", "google"],
    )
    parser.add_argument("--manager-model", type=str, default="claude-opus-4-6")
    parser.add_argument("--manager-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--manager-reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--manager-screenshots", type=int, default=2)
    parser.add_argument("--replan-max-screenshots", type=int, default=12)
    parser.add_argument(
        "--max-subtask-timeout",
        "--max-wave-timeout",
        dest="max_subtask_timeout",
        type=int,
        default=3600,
    )
    parser.add_argument("--website", type=str, default="https://www.google.com")
    parser.add_argument("--domain", type=str, default="macu")
    parser.add_argument(
        "--no-initial-setup",
        action="store_true",
        default=False,
        help=(
            "For generic CUA tasks, launch from the VM's clean desktop without "
            "pre-opening Chrome or any benchmark preset. This is implied by --task."
        ),
    )
    parser.add_argument("--max-parallelism", type=int, default=4)
    parser.add_argument("--max-replans", type=int, default=10)
    parser.add_argument("--spare-capacity-min-steps", type=int, default=20)
    parser.add_argument(
        "--osworld-data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "osworld" / "evaluation_examples"),
    )
    parser.add_argument(
        "--no-manager", action="store_true", default=False,
        help=(
            "Run without a manager: execute the initial task description"
            "and skip all manager followup and replan hooks."
        ),
    )
    parser.add_argument("--evaluator-timeout", type=int, default=600)
    parser.add_argument(
        "--max-task-timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help="Maximum wall-clock seconds for one claimed task before it is cancelled.",
    )

    # CUA flags that MACU parses itself and reinjects into each subprocess.
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--post_task_wait_seconds", type=float, default=60.0)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--reset-pool-snapshot", default="init_state")
    parser.add_argument("--reset-pool-timeout", type=int, default=600)

    argv = sys.argv[1:]
    if "--" in argv:
        sep_idx = argv.index("--")
        main_argv = argv[:sep_idx]
        cua_argv = argv[sep_idx + 1:]
    else:
        main_argv, cua_argv = argv, []
    args, extra = parser.parse_known_args(main_argv)
    cua_args = [a for a in extra + cua_argv if a.strip()]

    forwarded_flags: dict[str, tuple[str, type]] = {
        "--sleep_after_execution": ("sleep_after_execution", float),
        "--max_steps": ("max_steps", int),
        "--post_task_wait_seconds": ("post_task_wait_seconds", float),
        "--reasoning_effort": ("reasoning_effort", str),
    }
    filtered_cua_args: list[str] = []
    i = 0
    while i < len(cua_args):
        if cua_args[i] in forwarded_flags and i + 1 < len(cua_args):
            attr, conv = forwarded_flags[cua_args[i]]
            if getattr(args, attr) is None:
                setattr(args, attr, conv(cua_args[i + 1]))
            i += 2
        else:
            filtered_cua_args.append(cua_args[i])
            i += 1
    cua_args = filtered_cua_args

    args.cua_script = str(Path(args.cua_script).resolve())
    args.python = sys.executable
    args.result_dir = str(Path(args.result_dir).resolve())
    args.osworld_data_dir = str(Path(args.osworld_data_dir).resolve())

    if args.raw_task is not None:
        task_text = args.raw_task.strip()
        if not task_text:
            parser.error("--task must not be empty")
        if args.tasks_file is not None:
            parser.error("provide either a tasks_file or --task, not both")
        args.no_initial_setup = True
        args.tasks_file = _write_raw_task_file(args, task_text)
    elif args.tasks_file is None:
        parser.error("the following arguments are required: tasks_file or --task")
    else:
        args.tasks_file = str(Path(args.tasks_file).resolve())

    if not Path(args.tasks_file).is_file():
        raise SystemExit(f"Tasks file does not exist or is not a file: {args.tasks_file}")
    with open(args.tasks_file, encoding="utf-8") as f:
        probe = json.load(f)
    if isinstance(probe, list):
        if probe and "confirmed_task" not in probe[0]:
            raise SystemExit(
                "Tasks file is a JSON list but first element lacks 'confirmed_task' key. "
                "Expected a CUA tasks file or an OSWorld tasks file."
            )
    elif isinstance(probe, dict):
        if not probe or not all(isinstance(v, list) for v in probe.values()):
            raise SystemExit(
                "Tasks file is a JSON dict but values are not all lists. "
                "Expected an OSWorld tasks file like {\"chrome\": [\"uuid1\"]}."
            )
    else:
        raise SystemExit(
            "Tasks file must be a JSON list (CUA tasks) or JSON dict "
            f"(OSWorld domain->UUID map), got {type(probe).__name__}."
        )

    if args.osworld_root is None:
        parser.error("--osworld-root is required")
    args.osworld_root = str(Path(args.osworld_root).resolve())

    return args, cua_args


class _RunSingleGraphRunner(_RUNTIME_RUNNER):
    """One MACU run: schedule CUA workers, ask the manager, and replan."""

    def run(self) -> None:
        """Run graph setup, the MACU scheduler loop, final aggregation, and cleanup."""
        self._task_started_at = time.time()
        self._task_timeout_handled = False
        if not hasattr(self.args, "max_task_timeout"):
            self.args.max_task_timeout = DEFAULT_TASK_TIMEOUT_SECONDS

        self._prepare_graph()
        self._initialize_runtime_state()
        if not self._maybe_handle_task_timeout():
            self._run_event_loop()
        self._run_final_aggregation_after_task_timeout()

        self._drain_pending_filemgmt()
        self._cleanup_vmware_variant_clones()
        self._cleanup_vmware_init_from_clones()
        self._cleanup_vmware_frozen_initial_clones()
        final_response, final_output, total_cua_cost, total_manager_cost, total_cost = (
            self._finalize_results()
        )
        self._cleanup_unused_scout_overlay()
        self._cleanup_apptainer_overlays()
        build_merged_final_traj(
            run_dir=self.run_dir,
            task_id=self.task_id,
            original_task_text=self.original_task_text,
            all_subtasks=self.all_subtasks,
            graph=self.graph,
            results=self.results,
            completion_order=self.completion_order,
            final_response=final_response,
            no_manager=self.no_manager,
        )
        self._print_final_response(
            final_response,
            final_output,
            total_cua_cost,
            total_manager_cost,
            total_cost,
        )
        if not self.use_apptainer:
            reset_vm_pool(self.args, run_dir=self.run_dir)

    def _task_elapsed_seconds(self) -> float:
        started_at = (
            getattr(self, "_task_started_at", None)
            or self.overall_start
            or time.time()
        )
        return time.time() - started_at

    def _task_timeout_seconds(self) -> float:
        return float(getattr(self.args, "max_task_timeout", DEFAULT_TASK_TIMEOUT_SECONDS))

    def _task_timeout_response(self, elapsed: float) -> str:
        return (
            f"[ERROR] Task timed out after {elapsed:.0f}s "
            f"(limit {self._task_timeout_seconds():.0f}s)"
        )

    def _aggregation_id(self) -> str | None:
        aggregation = self.graph.get("dependency_graph", {}).get("aggregation") or {}
        return aggregation.get("id")

    def _maybe_handle_task_timeout(self) -> bool:
        timeout_seconds = self._task_timeout_seconds()
        elapsed = self._task_elapsed_seconds()
        if elapsed <= timeout_seconds:
            return False
        self._handle_task_timeout(elapsed)
        return True

    def _handle_task_timeout(self, elapsed: float) -> None:
        if getattr(self, "_task_timeout_handled", False):
            return
        self._task_timeout_handled = True
        self.task_timeout_elapsed_seconds = elapsed
        response = self._task_timeout_response(elapsed)
        _runtime.logger.warning(
            "Task %s timed out after %.0fs (limit %.0fs); cancelling %d running subtask(s).",
            self.task_id,
            elapsed,
            self._task_timeout_seconds(),
            len(self.running),
        )

        running_dirs: dict[str, Path] = {}
        for sid, (cua_proc, started_at) in list(self.running.items()):
            running_dirs[sid] = Path(cua_proc.subtask_dir)
            subtask_elapsed = time.time() - started_at
            try:
                cua_proc.proc.terminate()
                try:
                    cua_proc.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    cua_proc.proc.kill()
            except Exception as exc:  # noqa: BLE001
                _runtime.logger.warning("task-timeout cancel [%s]: terminate raised %s", sid, exc)
            self._pop_running_subtask(sid)
            self._save_apptainer_state(sid)
            meta = self.meta_by_sid.get(sid)
            if meta is not None:
                try:
                    _partial_response, partial_usage = extract_cua_result_from_dir(
                        cua_proc.subtask_dir,
                        sid,
                        self.cua_model,
                    )
                except Exception as exc:  # noqa: BLE001
                    _runtime.logger.warning(
                        "task-timeout partial usage [%s]: failed to read trajectory: %s",
                        sid,
                        exc,
                    )
                    partial_usage = _runtime.UsageInfo(model=self.cua_model)
                meta["state"] = "timed_out"
                meta["ended_at"] = time.time()
                meta["duration_seconds"] = subtask_elapsed
                meta["response"] = response
                meta["usage"] = meta.get("usage") or partial_usage
                meta["screenshot_path"] = (
                    meta.get("screenshot_path")
                    or _runtime.extract_latest_screenshot_from_dir(cua_proc.subtask_dir)
                )
                info = self._ensure_vm_info(sid)
                if info is not None:
                    self.vm_info_by_sid[sid] = info

        self.final_vm_choice.clear()
        aggregation_id = self._aggregation_id()
        for sid in sorted(self.all_subtask_ids - self.completed):
            st = self.st_by_id.get(sid, {})
            usage = (
                self.meta_by_sid.get(sid, {}).get("usage")
                or _runtime.UsageInfo(
                    model=getattr(self.args, "manager_model", None)
                    if st.get("agent_type") == "manager"
                    else self.cua_model
                )
            )
            if sid == aggregation_id and st.get("agent_type") == "manager" and not self.no_manager:
                continue
            self.cancelled.add(sid)
            self.results[sid] = response
            if st.get("agent_type") == "manager":
                self.cost_tracker.setdefault(
                    sid,
                    {
                        "agent_type": "manager",
                        "model": getattr(self.args, "manager_model", None),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "inference_seconds": 0.0,
                    },
                )
            else:
                self.cost_tracker.setdefault(
                    sid,
                    {
                        "agent_type": "cua",
                        "model": usage.model or self.cua_model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cost_usd": usage.cost_usd,
                        "inference_seconds": round(
                            _runtime._usage_inference_seconds(usage),
                            3,
                        ),
                    },
                )
            self.task_timings.setdefault(
                sid,
                self._task_timing_payload(
                    (
                        self._logical_duration_seconds(sid)
                        if sid in self.logical_start_times
                        else 0.0
                    ),
                    usage,
                    extra_inference_seconds=self._blocking_manager_inference_seconds(sid),
                ),
            )
            if self._blocking_manager_inference_seconds(sid) > 0:
                self.task_timings[sid]["blocking_manager_inference_seconds"] = round(
                    self._blocking_manager_inference_seconds(sid),
                    1,
                )
            if sid in running_dirs:
                self.completion_order.append((sid, time.time(), running_dirs[sid]))

        self._persist_task_timeout_sentinel(response, elapsed)

    def _run_final_aggregation_after_task_timeout(self) -> None:
        if not getattr(self, "_task_timeout_handled", False) or self.no_manager:
            return
        aggregation_id = self._aggregation_id()
        if not aggregation_id:
            return
        if aggregation_id in self.completed or aggregation_id in self.cancelled:
            return
        aggregation = self.st_by_id.get(aggregation_id)
        if aggregation is None or aggregation.get("agent_type") != "manager":
            return

        _runtime.logger.info(
            "Task %s timed out; running final aggregation manager [%s] with timeout sentinels.",
            self.task_id,
            aggregation_id,
        )
        self._process_manager_subtasks([aggregation])

    def _persist_task_timeout_sentinel(self, response: str, elapsed: float) -> None:
        total_cost = sum(info.get("cost_usd", 0.0) for info in self.cost_tracker.values())
        final_output = self._build_final_output(response, elapsed, total_cost)
        final_output["status"] = "timed_out"
        final_output["timed_out"] = True
        final_output["task_timeout_seconds"] = self._task_timeout_seconds()
        final_output["task_timeout_elapsed_seconds"] = round(elapsed, 1)
        self._persist_final_results(final_output)
        if self.is_osworld:
            self._write_osworld_timeout_result()

    def _write_osworld_timeout_result(self) -> None:
        result_path = self.run_dir / "result.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("0.0\n")
        eval_log_path = self.run_dir / "osworld_evaluator.log"
        with open(eval_log_path, "a", encoding="utf-8") as f:
            f.write("[orchestrator] Wrote 0.0 to result.txt because: task timed out\n")

    def _final_response(self) -> str:
        if getattr(self, "_task_timeout_handled", False):
            aggregation_id = self._aggregation_id()
            if not self.no_manager and aggregation_id in self.results:
                return super()._final_response()
            elapsed = getattr(self, "task_timeout_elapsed_seconds", None)
            if elapsed is None:
                elapsed = self._task_elapsed_seconds()
            return self._task_timeout_response(elapsed)
        return super()._final_response()

    def _add_task_timeout_fields(self, final_output: dict) -> dict:
        if getattr(self, "_task_timeout_handled", False):
            elapsed = (
                getattr(self, "task_timeout_elapsed_seconds", None)
                or self._task_elapsed_seconds()
            )
            final_output["status"] = "timed_out"
            final_output["timed_out"] = True
            final_output["task_timeout_seconds"] = self._task_timeout_seconds()
            final_output["task_timeout_elapsed_seconds"] = round(elapsed, 1)
        return final_output

    def _persist_final_results(self, final_output: dict) -> Path:
        output_path = super()._persist_final_results(final_output)
        if final_output.get("timed_out"):
            alias_path = self.run_dir / "final_result.json"
            with open(alias_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
        return output_path

    def _maybe_auto_assign_final_vm(self) -> None:
        if getattr(self, "_task_timeout_handled", False):
            return
        super()._maybe_auto_assign_final_vm()

    def _run_event_loop(self) -> None:
        """Drive the dependency scheduler until every subtask is complete or cancelled."""
        while self._remaining_sids():
            if self._maybe_handle_task_timeout():
                return
            made_progress = self._poll_running_subtasks()
            if self._maybe_handle_task_timeout():
                return
            ready = self._ready_subtasks()
            made_progress = self._process_manager_subtasks(ready) or made_progress
            if self._maybe_handle_task_timeout():
                return
            launched_any = self._launch_ready_cua_subtasks(ready)
            if self._maybe_handle_task_timeout():
                return
            made_progress = launched_any or made_progress
            made_progress = self._maybe_spare_capacity_replan(launched_any) or made_progress
            if self._maybe_handle_task_timeout():
                return
            if made_progress:
                continue
            if self.running:
                time.sleep(1)
                continue
            raise ValueError(
                f"Scheduler deadlock: remaining={self._remaining_sids()}, no tasks running or ready"
            )

    def _poll_running_subtasks(self) -> bool:
        """Check active CUA processes and trigger followup, file review, timeout, or completion."""
        made_progress = False
        for sid in list(self.running):
            entry = self.running.get(sid)
            if entry is None:
                continue
            cua_proc, start_time = entry
            elapsed = time.time() - start_time
            ret = cua_proc.proc.poll()
            if ret is not None:
                self._handle_finished_subtask(sid, elapsed, ret)
                made_progress = True
                continue

            self._refresh_running_subtask(sid, cua_proc)
            made_progress = self._maybe_handle_manager_followup(sid, cua_proc) or made_progress
            made_progress = self._maybe_handle_file_management(sid, cua_proc) or made_progress
            if elapsed > self.args.max_subtask_timeout:
                self._handle_timed_out_subtask(sid, cua_proc, elapsed)
                made_progress = True
        return made_progress

    def _handle_finished_subtask(self, sid: str, elapsed: float, ret_code: int) -> None:
        """Record a completed CUA run and let the manager replan from the new evidence."""
        self._pop_running_subtask(sid)
        self._record_completion(sid, ret_code, elapsed)
        self._save_apptainer_state(sid)
        self._materialize_logical_completion(sid)
        self._maybe_replan_after_completion(sid)

    def _maybe_replan_after_completion(self, sid: str) -> bool:
        """Ask the manager to revise pending work after a CUA subtask finishes."""
        if self.no_manager or sid not in self.completed:
            return False
        if sid == "final_aggregation":
            return False
        if self.st_by_id.get(sid, {}).get("agent_type", "cua") == "manager":
            return False
        spare_slots = (
            self.max_par - len(self.running)
            if self.max_par is not None
            else len(self.all_subtask_ids)
        )
        return self._run_replan(sid, "completed", max(0, spare_slots))

    def _maybe_handle_manager_followup(self, sid: str, cua_proc: CUAProcess) -> bool:
        """Let the manager request a live followup when a CUA worker pauses for review."""
        if self.no_manager or sid in self.reviewed_sids:
            return False
        ready_path = Path(cua_proc.subtask_dir) / "ready_for_review"
        if not ready_path.exists():
            return False

        initial_text, _ = extract_cua_result_from_dir(cua_proc.subtask_dir, sid, self.cua_model)
        subtask_dict = self.st_by_id.get(sid, {"id": sid, "instruction": ""})
        try:
            needs_fu, fu_type, fu_msg, fu_usage = ask_manager_for_followup(
                subtask_dict,
                initial_text,
                self.original_task_text,
                self.args,
                self.run_dir,
            )
            self.manager_usages.append(fu_usage)
            self._add_blocking_manager_inference(sid, fu_usage)
        except Exception as exc:  # noqa: BLE001
            _runtime.logger.warning("Manager followup call failed for [%s]: %s", sid, exc)
            needs_fu, fu_type, fu_msg = False, None, None

        decision = {"type": fu_type, "message": fu_msg} if needs_fu and fu_msg else {"type": "done"}
        with open(Path(cua_proc.subtask_dir) / "manager_decision.json", "w") as f:
            json.dump(decision, f)
        self.reviewed_sids.add(sid)
        _runtime.logger.info("Manager decision for [%s]: %s", sid, decision["type"])
        return True

    def _run_replan(self, focus_sid, reason: str, spare_slots: int) -> bool:
        """Call the replan manager and apply accepted edits to pending graph nodes."""
        if self.no_manager:
            return False
        if self.num_applied_replans >= self.args.max_replans:
            _runtime.logger.info(
                "Replan budget exhausted; skipping manager call (trigger %s).",
                focus_sid or reason,
            )
            return False

        self.replan_count += 1
        try:
            usage_key = f"__replan_{self.replan_count:03d}_{focus_sid or reason}"
            decision, usage = ask_manager_for_replanning(
                focus_sid=focus_sid,
                all_subtasks=self.all_subtasks,
                results=self.results,
                completed=self.completed,
                running_logicals=self._running_logical_sids(),
                original_task_text=self.original_task_text,
                args=self.args,
                run_dir=self.run_dir,
                iteration=self.replan_count,
                num_applied=self.num_applied_replans,
                spare_slots=spare_slots,
                archive_pool=self.archive_pool,
                reason=reason,
                cancelled=self.cancelled,
                focus_step_count=self.meta_by_sid.get(focus_sid, {}).get("step_count") if focus_sid else None,
                focus_max_steps=getattr(self.args, "max_steps", None),
            )
            self.manager_usages.append(usage)
            self.cost_tracker[usage_key] = {
                "agent_type": "manager",
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
                "inference_seconds": round(
                    _runtime._usage_inference_seconds(usage),
                    3,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            _runtime.logger.warning(
                "Replanning hook raised (trigger %s, iter %d): %s",
                focus_sid or reason,
                self.replan_count,
                exc,
            )
            decision = None

        if decision is None:
            _append_replan_log(self.run_dir, self.replan_count, focus_sid, None, [], False, reason)
            return False

        applied, errors, cancelled_sids = apply_replan_decision(
            decision,
            self.all_subtasks,
            self.st_by_id,
            self.all_subtask_ids,
            self.downstream_counts,
            self.completed,
            self._running_logical_sids(),
            self.task_id,
        )
        label = focus_sid or reason
        if not applied:
            _runtime.logger.warning("Replan (trigger %s) rejected by validator: %s", label, errors)
            _append_replan_log(self.run_dir, self.replan_count, focus_sid, decision, errors, False, reason)
            return False

        if any(decision.get(k) for k in ("add", "remove", "modify", "cancel")):
            self.num_applied_replans += 1
        for sid in cancelled_sids:
            self._cancel_running_subtask(sid)
        _runtime.logger.info(
            "Replan (trigger %s) applied: +%d/-%d/~%d/cancel=%d (reason: %s)",
            label,
            len(decision.get("add") or []),
            len(decision.get("remove") or []),
            len(decision.get("modify") or {}),
            len(cancelled_sids),
            decision.get("reasoning", ""),
        )
        persist_graph_snapshot(
            self.run_dir,
            self.all_subtasks,
            self.task_id,
            self.original_task_text,
            iteration=self.replan_count,
            label=f"after_{label}",
            decision=decision,
        )
        _append_replan_log(self.run_dir, self.replan_count, focus_sid, decision, [], True, reason)
        return True

    def _maybe_spare_capacity_replan(self, launched_any: bool) -> bool:
        """Use idle worker capacity to ask for exploratory branches from an active CUA state."""
        min_steps = max(1, int(getattr(self.args, "spare_capacity_min_steps", 5)))
        if (
            self.no_manager
            or launched_any
            or self.max_par is None
            or not self.running
            or self.max_par - len(self.running) <= 0
        ):
            return False
        ready_running = {
            sid
            for sid in self.running
            if self.meta_by_sid.get(sid, {}).get("step_count", 0) >= min_steps
        }
        unfired = ready_running - self.spare_capacity_fired_sids
        if not unfired:
            return False
        focus_sid = max(unfired, key=lambda sid: (self.downstream_counts.get(sid, 0), sid))
        self.spare_capacity_fired_sids.add(focus_sid)
        return self._run_replan(focus_sid, "spare_capacity", self.max_par - len(self.running))

    def _build_final_output(self, final_response: str, total_duration: float, total_cost: float) -> dict:
        """Add OSWorld evaluator hints derived from the manager's final verdict."""
        final_output = self._add_task_timeout_fields(
            super()._build_final_output(final_response, total_duration, total_cost)
        )
        if self.is_osworld:
            infeasible_verdict = _contains_infeasible_verdict(final_response)
            final_vm_sid = self.final_vm_choice.get("subtask_id")
            final_vm_had_fail_action = (
                _subtask_has_fail_action(self.run_dir / final_vm_sid)
                if final_vm_sid
                else False
            )
            final_output["osworld_infeasible_verdict"] = infeasible_verdict
            final_output["osworld_final_vm_had_fail_action"] = final_vm_had_fail_action
            final_output["osworld_submit_fail_before_evaluate"] = (
                infeasible_verdict or final_vm_had_fail_action
            )
        return final_output


def _run_with_tasks_file(args, cua_args: list[str]) -> None:
    """Run every task in a task file using the compact runner above."""
    original_runner = _runtime._RunSingleGraphRunner
    _runtime._RunSingleGraphRunner = _RunSingleGraphRunner
    try:
        _RUNTIME_RUN_WITH_TASKS_FILE(args, cua_args)
    finally:
        _runtime._RunSingleGraphRunner = original_runner


def main() -> None:
    args, cua_args = parse_args()
    _run_with_tasks_file(args, cua_args)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main()
