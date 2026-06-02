#!/usr/bin/env python3
"""Runtime support for ``run_macu.py``.

This module holds the CLI parsing, subprocess launch, VM bookkeeping, cleanup,
batch task claiming, and result persistence details that would otherwise swamp
the research loop in the top-level runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.graph_utils import (
    build_merged_final_traj,
    compute_downstream_count,
    generate_validated_dependency_graph,
    get_all_subtasks,
    persist_graph_snapshot,
    validate_subtask_ids,
)
from utils.benchmark_utils import (
    _aggregate_success_rate,
    _write_chrome_example_with_uploads,
    load_canonical_osworld_config,
    run_osworld_evaluator_subprocess,
    write_subtask_osworld_example,
)
from utils.file_management_utils import (
    apply_filemgmt_actions,
    ask_manager_for_file_management as ask_manager_for_file_management_impl,
    compute_file_diff,
    read_filemgmt_downloads,
    resolve_input_files,
    write_filemgmt_decision,
)
from utils.manager_utils import (
    call_manager_for_subtasks,
    save_manager_prompt,
    save_manager_response,
)
from utils.utils import (
    count_subtask_steps,
    extract_cua_result_from_dir,
    extract_latest_screenshot_from_dir,
    load_cua_tasks,
    load_osworld_tasks as load_osworld_tasks_for_gen,
)
from utils.llm import call_llm, load_prompt, parse_json_response, render_prompt, UsageInfo
from utils.vm_utils import (
    _apptainer_create_overlay,
    _apptainer_create_overlay_from,
    _apptainer_kill_qemu_for_overlay,
    _apptainer_save_vm_state,
    _clone_vm_from_source,
    _delete_vm_tree,
    _orchestrator_alloc_vm,
    _orchestrator_reclaim_vm,
    _orchestrator_release_vm,
    _registry_path_to_absolute,
    _stop_vm_if_needed,
    _take_initial_vm_screenshot,
    _take_subtask_snapshot,
    read_vm_info_from_dir,
    reset_vm_pool,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("macu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "aggregation.yaml"
MANAGER_GRAPH_PROMPT_PATH = PROJECT_ROOT / "prompts" / "manager_cua_v2.yaml"
REPLAN_PROMPT_PATH = PROJECT_ROOT / "prompts" / "replan.yaml"
FILE_MGMT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "file_management.yaml"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CUAProcess:
    proc: subprocess.Popen
    subtask_id: str
    log_file: object  # open file handle
    subtask_dir: Path
    result_dir: str
    instruction: str = ""


@dataclass
class _PreparedLaunch:
    st: dict
    sid: str
    instruction: str
    staged_files: Optional[list[dict]] = None
    vm_path: Optional[str] = None
    is_variant: bool = False
    temperature_override: Optional[float] = None
    skip_canonical_setup: bool = False


# Global list for signal handler cleanup
_child_processes: list[subprocess.Popen] = []


def _round_seconds(value: float, digits: int = 1) -> float:
    return round(max(0.0, float(value or 0.0)), digits)


def _usage_inference_seconds(usage: Optional[UsageInfo]) -> float:
    if usage is None:
        return 0.0
    return max(0.0, float(getattr(usage, "inference_seconds", 0.0) or 0.0))


def _usage_inference_intervals(usage: Optional[UsageInfo]) -> list[tuple[float, float]]:
    if usage is None:
        return []
    intervals: list[tuple[float, float]] = []
    for item in getattr(usage, "inference_intervals", []) or []:
        if isinstance(item, dict):
            started_at = item.get("start")
            ended_at = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            started_at, ended_at = item[0], item[1]
        else:
            continue
        try:
            started = float(started_at)
            ended = float(ended_at)
        except (TypeError, ValueError):
            continue
        if ended >= started > 0:
            intervals.append((started, ended))
    return intervals


def _merged_interval_seconds(
    intervals: list[tuple[float, float]],
    *,
    window_start: float,
    window_end: float,
) -> float:
    clipped: list[tuple[float, float]] = []
    for started_at, ended_at in intervals:
        started = max(started_at, window_start)
        ended = min(ended_at, window_end)
        if ended > started:
            clipped.append((started, ended))
    if not clipped:
        return 0.0

    clipped.sort()
    total = 0.0
    cur_start, cur_end = clipped[0]
    for started, ended in clipped[1:]:
        if started <= cur_end:
            cur_end = max(cur_end, ended)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = started, ended
    total += cur_end - cur_start
    return total


def launch_cua_subtask(
    subtask: dict,
    instruction: str,
    run_dir: Path,
    args: argparse.Namespace,
    cua_args: list[str],
    staged_files: Optional[list[dict]] = None,
    vm_path_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    skip_canonical_setup: bool = False,
    domain: Optional[str] = None,
    canonical_config: Optional[dict] = None,
    website_override: Optional[str] = None,
) -> CUAProcess:
    """Create temp task files and launch CUA subprocess for a subtask.

    When ``canonical_config`` is provided (OSWorld mode), the helper pre-writes
    a deep copy of the canonical OSWorld config with the instruction replaced.
    Otherwise (generic CUA mode), if ``staged_files`` or ``skip_canonical_setup``
    is set, it pre-writes a chrome-stub example.

    Variant subtasks (nodes with ``variant_of`` set) use ``vm_path_override``
    to attach to a pre-cloned VM, and ``temperature_override=1.0`` for
    diverse exploration vs. the original attempt.

    When ``skip_canonical_setup=True`` (dependents reusing a predecessor's
    working VM, or variants inheriting their source's flag), we pre-write
    an explicit empty-config example file and drop
    ``--overwrite_generated_examples`` so OSWorld's ``SetupController``
    runs no setup actions — the VM's seeded state is preserved into the
    subtask's run.
    """
    logical_subtask_id = subtask["id"]
    subtask_dir = run_dir / logical_subtask_id
    subtask_dir.mkdir(parents=True, exist_ok=True)

    effective_website = args.website if website_override is None else website_override

    # Write task JSON (use absolute paths so they work from any cwd)
    task_path = subtask_dir.resolve() / "task.json"
    task_data = [{
        "task_id": logical_subtask_id,
        "confirmed_task": instruction,
        "website": effective_website,
    }]
    if skip_canonical_setup:
        task_data[0]["no_initial_setup"] = True
    with open(task_path, "w") as f:
        json.dump(task_data, f, indent=2)

    config_base_dir = str((subtask_dir / "config").resolve())
    meta_path = str((subtask_dir / "meta.json").resolve())
    result_dir = str((subtask_dir / "results").resolve())

    effective_domain = domain if domain else args.domain

    pre_wrote_example = False
    if canonical_config is not None and domain is not None:
        # OSWorld mode: pre-write canonical config-based example
        example_path = write_subtask_osworld_example(
            config_base_dir=config_base_dir,
            domain=domain,
            subtask_id=logical_subtask_id,
            canonical_config=canonical_config,
            subtask_instruction=instruction,
            staged_files=staged_files,
            skip_canonical_setup=skip_canonical_setup,
        )
        pre_wrote_example = True
        setup_note = " [canonical-setup skipped]" if skip_canonical_setup else ""
        if staged_files:
            logger.info(
                "Pre-wrote OSWorld example for [%s] at %s with %d uploaded-file(s)%s",
                logical_subtask_id, example_path, len(staged_files), setup_note,
            )
        else:
            logger.info("Pre-wrote OSWorld example for [%s] at %s%s",
                        logical_subtask_id, example_path, setup_note)
    elif staged_files or skip_canonical_setup:
        # Generic CUA mode: chrome stub with uploads / no-setup
        example_path = _write_chrome_example_with_uploads(
            config_base_dir=config_base_dir,
            domain=effective_domain,
            subtask_id=logical_subtask_id,
            instruction=instruction,
            website=effective_website,
            staged_files=staged_files,
            skip_canonical_setup=skip_canonical_setup,
        )
        pre_wrote_example = True
        setup_note = " [canonical-setup skipped]" if skip_canonical_setup else ""
        upload_note = (
            f" with {len(staged_files)} uploaded-file(s)" if staged_files else ""
        )
        logger.info(
            "Pre-wrote chrome example for [%s] at %s%s%s",
            logical_subtask_id, example_path, upload_note, setup_note,
        )

    cmd = [
        args.python,
        args.cua_script,
        "--osworld_root", args.osworld_root,
        "--cua_provider", args.cua_provider,
        "--mind2web_tasks_path", str(task_path),
        "--test_config_base_dir", config_base_dir,
        "--test_all_meta_path", meta_path,
        "--result_dir", result_dir,
        "--mind2web_domain", effective_domain,
        "--start_idx", "0",
        "--end_idx", "0",
        "--num_envs", "1",
    ]
    # Drop --overwrite_generated_examples whenever we pre-wrote an explicit
    # example file (staged_files, skip_canonical_setup, or OSWorld canonical
    # config); in all other cases keep the original auto-generation behavior.
    if not pre_wrote_example:
        cmd.append("--overwrite_generated_examples")
    # Inject explicitly-parsed CUA flags before the passthrough extras,
    # so they are guaranteed to reach the subprocess even if cua_args
    # passthrough silently drops them.
    _FORWARDED_CUA_FLAGS = {
        "sleep_after_execution": args.sleep_after_execution,
        "max_steps": args.max_steps,
        "post_task_wait_seconds": args.post_task_wait_seconds,
        "reasoning_effort": args.reasoning_effort,
    }
    for flag_name, value in _FORWARDED_CUA_FLAGS.items():
        if value is not None:
            cmd.extend([f"--{flag_name}", str(value)])
    if temperature_override is not None:
        cmd.extend(["--temperature", str(temperature_override)])
    if vm_path_override:
        cmd.extend(["--path_to_vm", str(Path(vm_path_override).resolve())])
    no_manager = getattr(args, "no_manager", False)
    if not no_manager:
        cmd.append("--enable_manager_followup")
    vm_info_path = str((subtask_dir / "vm_info.json").resolve())
    cmd.extend(["--vm_info_path", vm_info_path])
    cmd.extend(cua_args)

    log_path = subtask_dir / "subprocess.log"
    log_file = open(log_path, "w")

    logger.info("Launching CUA subprocess for [%s]: %s", logical_subtask_id, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=args.osworld_root,
    )
    _child_processes.append(proc)

    return CUAProcess(
        proc=proc,
        subtask_id=logical_subtask_id,
        log_file=log_file,
        subtask_dir=subtask_dir,
        result_dir=result_dir,
        instruction=instruction,
    )

def ask_manager_for_file_management(
    subtask: dict,
    file_diff: dict[str, list[str]],
    archive_pool: dict[str, dict],
    original_task_text: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[dict], UsageInfo]:
    """Proxy to the file-management utility implementation with injected helpers."""
    return ask_manager_for_file_management_impl(
        subtask=subtask,
        file_diff=file_diff,
        archive_pool=archive_pool,
        original_task_text=original_task_text,
        args=args,
        run_dir=run_dir,
        prompt_yaml_path=FILE_MGMT_PROMPT_PATH,
        save_manager_prompt_fn=save_manager_prompt,
        save_manager_response_fn=save_manager_response,
        load_prompt_fn=load_prompt,
        render_prompt_fn=render_prompt,
        call_llm_fn=call_llm,
        parse_json_response_fn=parse_json_response,
    )

class _RunSingleGraphRunner:
    def __init__(
        self,
        graph_path: str,
        args: argparse.Namespace,
        cua_args: list[str],
        task_metadata: Optional[dict] = None,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.args = args
        self.cua_args = cua_args
        if not hasattr(self.args, "max_subtask_timeout") and hasattr(self.args, "max_wave_timeout"):
            self.args.max_subtask_timeout = self.args.max_wave_timeout

        metadata = self._load_task_metadata(task_metadata)
        self.task_id = metadata.get("task_id") or self.graph_path.parent.name
        self.original_task_text = (
            metadata.get("task_text")
            or metadata.get("instruction")
            or metadata.get("confirmed_task")
        )
        if not self.original_task_text:
            raise ValueError(
                f"Task metadata for {graph_path} has neither 'task_text' nor 'instruction'"
            )

        self.osworld_domain = metadata.get("domain")
        self.is_osworld = bool(self.osworld_domain)
        self.task_website = metadata.get("website")
        self.no_initial_setup = (
            not self.is_osworld
            and bool(
                metadata.get("no_initial_setup")
                or metadata.get("blank_vm")
                or getattr(args, "no_initial_setup", False)
            )
        )
        self.no_manager = getattr(args, "no_manager", False)
        self.run_dir = Path(args.result_dir) / self.task_id
        self.canonical_config = None
        if self.is_osworld:
            self.canonical_config = load_canonical_osworld_config(
                osworld_data_dir=args.osworld_data_dir,
                domain=self.osworld_domain,
                task_id=self.task_id,
            )

        self.cua_model = self._extract_cua_arg("--model", "")
        self.use_apptainer = self._extract_cua_arg("--provider_name", "vmware") == "apptainer"
        self.scout_overlay_path: Optional[str] = None
        self.graph: dict = {}

        self.all_subtasks: list[dict] = []
        self.results: dict[str, str] = {}
        self.cost_tracker: dict[str, dict] = {}
        self.manager_usages: list[UsageInfo] = []
        self.task_timings: dict[str, dict] = {}
        self.st_by_id: dict[str, dict] = {}
        self.vm_info_by_sid: dict[str, dict] = {}
        self.final_vm_choice: dict = {}
        self.archive_pool: dict[str, dict] = {}
        self.reviewed_sids: set[str] = set()
        self.filemgmt_reviewed_sids: set[str] = set()
        self.pending_filemgmt_actions: dict[str, list[dict]] = {}
        self.meta_by_sid: dict[str, dict] = {}
        self.logical_start_times: dict[str, float] = {}
        self.subtask_vm_path: dict[str, str] = {}
        self.initial_state_clone: dict[str, str] = {}
        self.vm_reused_by: dict[str, str] = {}
        self.overlay_parent: dict[str, str] = {}
        self.skip_setup_by_sid: dict[str, bool] = {}
        self.vmstate_saved: set[str] = set()
        self.completion_order: list[tuple[str, float, Optional[Path]]] = []
        self.downstream_counts: dict[str, int] = {}
        self.all_subtask_ids: set[str] = set()
        self.completed: set[str] = set()
        self.cancelled: set[str] = set()
        self.running: dict[str, tuple[CUAProcess, float]] = {}
        self.max_par = args.max_parallelism
        self.replan_count = 0
        self.num_applied_replans = 0
        self.spare_capacity_fired_sids: set[str] = set()
        self.overall_start = 0.0

    def _load_task_metadata(self, task_metadata: Optional[dict]) -> dict:
        """Load only task identity fields; dependency graphs are always regenerated."""
        if task_metadata is not None:
            return dict(task_metadata)
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Task metadata file not found: {self.graph_path}")
        with open(self.graph_path, encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> None:
        self._prepare_graph()

        self._initialize_runtime_state()
        self._run_event_loop()
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

    def _extract_cua_arg(self, flag: str, default: str) -> str:
        for i, arg in enumerate(self.cua_args):
            if arg == flag and i + 1 < len(self.cua_args):
                return self.cua_args[i + 1]
        return default

    def _prepare_graph(self) -> None:
        extra_fields: dict[str, object] = {}
        if self.is_osworld:
            extra_fields.update(
                {
                    "domain": self.osworld_domain,
                    "instruction": self.original_task_text,
                }
            )
        else:
            extra_fields["task_text"] = self.original_task_text
            if self.no_initial_setup:
                extra_fields["no_initial_setup"] = True
            if self.task_website:
                extra_fields["website"] = self.task_website

        if self.no_manager:
            logger.info(
                "Building single-node graph for %s (--no-manager, skipping LLM call)",
                self.task_id,
            )
            graph = {
                "task_analysis": "Vanilla baseline: single CUA subtask with the raw instruction.",
                "subtasks": [
                    {
                        "id": "subtask_1",
                        "agent_type": "cua",
                        "description": self.original_task_text,
                        "dependencies": [],
                        "instruction": self.original_task_text,
                        "outputs": [],
                        "input_files": [],
                    }
                ],
            }
        else:
            initial_screenshot = None
            if self.canonical_config is not None:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                initial_screenshot, self.scout_overlay_path = _take_initial_vm_screenshot(
                    canonical_config=self.canonical_config,
                    task_id=self.task_id,
                    args=self.args,
                    cua_args=self.cua_args,
                    run_dir=self.run_dir,
                )
            if initial_screenshot is not None:
                logger.info(
                    "Generating dependency graph for %s with initial VM screenshot...",
                    self.task_id,
                )
            else:
                logger.info(
                    "Generating dependency graph for %s without initial VM screenshot...",
                    self.task_id,
                )
            graph, gen_usage, gen_errors = generate_validated_dependency_graph(
                task_text=self.original_task_text,
                prompt_yaml_path=MANAGER_GRAPH_PROMPT_PATH,
                provider=self.args.manager_provider,
                model=self.args.manager_model,
                max_tokens=getattr(self.args, "graph_gen_max_tokens", 16384),
                image_path=initial_screenshot,
                reasoning_effort=getattr(self.args, "manager_reasoning_effort", None),
            )
            if "parse_error" in graph:
                raise RuntimeError(
                    f"Failed to parse dependency graph for task {self.task_id}: {graph['parse_error']}"
                )
            if gen_errors:
                logger.warning(
                    "Validation errors for task %s (after retries): %s",
                    self.task_id,
                    gen_errors,
                )
            logger.info(
                "Graph generated for %s: %d subtasks, cost $%.4f "
                "(%d in / %d out tokens, %s)",
                self.task_id,
                len(graph.get("subtasks", [])),
                gen_usage.cost_usd,
                gen_usage.input_tokens,
                gen_usage.output_tokens,
                gen_usage.model,
            )

        self.graph = {"task_id": self.task_id, **extra_fields, "dependency_graph": graph}
        validate_subtask_ids(self.graph)
        self.all_subtasks = get_all_subtasks(self.graph, no_manager=self.no_manager)

    def _initialize_runtime_state(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "dependency_graph.json", "w") as f:
            json.dump(self.graph, f, indent=2)

        for st in self.all_subtasks:
            st["_task_id"] = self.task_id

        self.st_by_id = {s["id"]: s for s in self.all_subtasks}
        self.downstream_counts = compute_downstream_count(self.all_subtasks)
        self.all_subtask_ids = {s["id"] for s in self.all_subtasks}
        logger.info(
            "Starting MACU for task %s (%d subtasks, max_parallelism=%s)",
            self.task_id,
            len(self.all_subtasks),
            self.args.max_parallelism or "unlimited",
        )
        logger.info("Original task: %s", self.original_task_text[:500])
        persist_graph_snapshot(
            self.run_dir,
            self.all_subtasks,
            self.task_id,
            self.original_task_text,
            iteration=0,
            label="initial",
            decision=None,
        )
        self.overall_start = time.time()

    def _remaining_sids(self) -> set[str]:
        return self.all_subtask_ids - self.completed - self.cancelled

    def _running_logical_sids(self) -> set[str]:
        return set(self.running)

    def _logical_duration_seconds(self, sid: str) -> float:
        return round(time.time() - self.logical_start_times.get(sid, time.time()), 1)

    def _task_timing_payload(
        self,
        duration_seconds: float,
        usage: Optional[UsageInfo] = None,
        *,
        extra_inference_seconds: float = 0.0,
    ) -> dict[str, float]:
        inference_seconds = _usage_inference_seconds(usage) + max(
            0.0,
            float(extra_inference_seconds or 0.0),
        )
        return {
            "duration_seconds": _round_seconds(duration_seconds),
            "inference_seconds": _round_seconds(inference_seconds),
            "duration_without_inference_seconds": _round_seconds(
                max(0.0, duration_seconds - inference_seconds)
            ),
        }

    def _add_blocking_manager_inference(self, sid: str, usage: UsageInfo) -> None:
        meta = self.meta_by_sid.get(sid)
        if meta is None:
            return
        meta["blocking_manager_inference_seconds"] = (
            float(meta.get("blocking_manager_inference_seconds", 0.0) or 0.0)
            + _usage_inference_seconds(usage)
        )

    def _blocking_manager_inference_seconds(self, sid: str) -> float:
        meta = self.meta_by_sid.get(sid) or {}
        return max(0.0, float(meta.get("blocking_manager_inference_seconds", 0.0) or 0.0))

    def _all_usage_infos(self) -> list[UsageInfo]:
        usages: list[UsageInfo] = []
        for meta in self.meta_by_sid.values():
            usage = meta.get("usage")
            if isinstance(usage, UsageInfo):
                usages.append(usage)
        usages.extend(self.manager_usages)
        return usages

    def _total_model_call_seconds(self) -> float:
        return sum(_usage_inference_seconds(usage) for usage in self._all_usage_infos())

    def _total_inference_wall_seconds(self, total_duration: float) -> float:
        intervals: list[tuple[float, float]] = []
        unplaced_inference_seconds = 0.0
        for usage in self._all_usage_infos():
            usage_intervals = _usage_inference_intervals(usage)
            if usage_intervals:
                intervals.extend(usage_intervals)
            else:
                unplaced_inference_seconds += _usage_inference_seconds(usage)
        if not intervals:
            return min(total_duration, self._total_model_call_seconds())
        window_start = self.overall_start
        window_end = self.overall_start + total_duration
        placed_wall_seconds = _merged_interval_seconds(
            intervals,
            window_start=window_start,
            window_end=window_end,
        )
        return min(total_duration, placed_wall_seconds + unplaced_inference_seconds)

    def _pop_running_subtask(self, sid: str) -> Optional[tuple[CUAProcess, float]]:
        entry = self.running.pop(sid, None)
        if entry is None:
            return None
        cua_proc, started_at = entry
        try:
            cua_proc.log_file.close()
        except Exception:  # noqa: BLE001
            pass
        return cua_proc, started_at

    def _save_apptainer_state(self, sid: str) -> None:
        if not self.use_apptainer or not self.subtask_vm_path.get(sid):
            return
        vmstate_path = _apptainer_save_vm_state(self.subtask_vm_path[sid], label=sid)
        if vmstate_path is not None:
            self.vmstate_saved.add(sid)
        _apptainer_kill_qemu_for_overlay(self.subtask_vm_path[sid], label=sid)

    def _cancel_running_subtask(self, sid: str) -> None:
        entry = self._pop_running_subtask(sid)
        if entry is None:
            logger.warning(
                "cancel [%s]: not in running dict (already exited or never started?)",
                sid,
            )
            return
        cua_proc, started_at = entry
        elapsed = time.time() - started_at
        logger.info(
            "Cancelling CUA [%s] after %.0fs (manager requested, terminating subprocess)",
            sid,
            elapsed,
        )
        try:
            cua_proc.proc.terminate()
            try:
                cua_proc.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cua_proc.proc.kill()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel [%s]: terminate raised %s", sid, exc)
        if self.use_apptainer and self.subtask_vm_path.get(sid):
            _apptainer_kill_qemu_for_overlay(self.subtask_vm_path[sid], label=sid)
        sentinel = "[CANCELLED] Manager requested cancellation during replan"
        self.cancelled.add(sid)
        self.results[sid] = sentinel
        meta = self.meta_by_sid.get(sid)
        if meta is not None:
            meta["state"] = "cancelled"
            meta["ended_at"] = time.time()
            meta["duration_seconds"] = elapsed
            meta["response"] = sentinel

    def _ensure_vm_info(self, sid: str, *, force_refresh: bool = False) -> Optional[dict]:
        meta = self.meta_by_sid.get(sid)
        if not meta:
            return None
        existing = meta.get("vm_info")
        if existing and existing.get("vm_path") and not force_refresh:
            return existing
        proc = meta.get("proc")
        if proc is None:
            return existing
        info = read_vm_info_from_dir(proc.subtask_dir, sid)
        if info:
            if (
                existing
                and existing.get("vm_path")
                and info.get("vm_path")
                and existing.get("vm_path") != info.get("vm_path")
            ):
                logger.info(
                    "CUA [%s] refreshed VM info at completion: %s -> %s",
                    sid,
                    existing.get("vm_path"),
                    info.get("vm_path"),
                )
            meta["vm_info"] = info
            return info
        return existing

    def _record_completion(self, sid: str, ret_code: int, elapsed: float) -> None:
        meta = self.meta_by_sid[sid]
        cua_proc = meta["proc"]
        meta["state"] = "finished"
        meta["ended_at"] = time.time()
        meta["duration_seconds"] = elapsed
        if ret_code != 0:
            meta["response"] = f"[ERROR] Process exited with code {ret_code}"
            meta["usage"] = UsageInfo(model=self.cua_model)
            logger.error("CUA [%s] failed with code %d", sid, ret_code)
            return

        response_text, usage = extract_cua_result_from_dir(
            cua_proc.subtask_dir,
            sid,
            self.cua_model,
        )
        meta["response"] = response_text
        meta["usage"] = usage
        meta["screenshot_path"] = extract_latest_screenshot_from_dir(cua_proc.subtask_dir)
        info = self._ensure_vm_info(sid, force_refresh=True)
        if info is not None:
            logger.info("CUA [%s] used VM: %s", sid, info.get("vm_path"))
        else:
            logger.warning("CUA [%s] complete but no vm_info.json found", sid)
        logger.info(
            "CUA [%s] complete (%d chars, $%.4f, %.1fs wall, %.1fs inference, %.1fs non-inference)",
            sid,
            len(meta["response"]),
            usage.cost_usd,
            elapsed,
            _usage_inference_seconds(usage),
            max(0.0, elapsed - _usage_inference_seconds(usage)),
        )

    def _apply_pending_filemgmt(self, sid: str, subtask_dir: Path) -> None:
        actions = self.pending_filemgmt_actions.pop(sid, None)
        if not actions:
            return
        try:
            downloads = read_filemgmt_downloads(subtask_dir)
            manifest = apply_filemgmt_actions(
                actions=actions,
                downloads=downloads,
                run_dir=self.run_dir,
                source_sid=sid,
            )
            for entry in manifest.get("archived", []):
                name = entry.get("archive_name")
                if not name:
                    continue
                self.archive_pool[name] = {
                    "host_path": entry.get("host_path"),
                    "source_sid": entry.get("source_sid", sid),
                    "guest_path": entry.get("guest_path"),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("File management apply [%s] raised: %s", sid, exc)

    def _materialize_logical_completion(self, sid: str, *, apply_filemgmt: bool = True) -> None:
        meta = self.meta_by_sid[sid]
        cua_proc = meta["proc"]
        usage = meta.get("usage") or UsageInfo(model=self.cua_model)
        self.results[sid] = meta.get("response", "")
        self.cost_tracker[sid] = {
            "agent_type": "cua",
            "model": usage.model or self.cua_model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "inference_seconds": _round_seconds(_usage_inference_seconds(usage), 3),
        }
        self.task_timings[sid] = self._task_timing_payload(
            self._logical_duration_seconds(sid),
            usage,
            extra_inference_seconds=self._blocking_manager_inference_seconds(sid),
        )
        if self._blocking_manager_inference_seconds(sid) > 0:
            self.task_timings[sid]["blocking_manager_inference_seconds"] = _round_seconds(
                self._blocking_manager_inference_seconds(sid)
            )
        info = meta.get("vm_info")
        if info is not None:
            self.vm_info_by_sid[sid] = info
        if apply_filemgmt:
            self._apply_pending_filemgmt(sid, Path(cua_proc.subtask_dir))
        self.completed.add(sid)
        self.completion_order.append((sid, time.time(), Path(cua_proc.subtask_dir)))

    def _run_event_loop(self) -> None:
        raise NotImplementedError("_RunSingleGraphRunner subclasses must implement _run_event_loop")

    def _handle_finished_subtask(self, sid: str, elapsed: float, ret_code: int) -> None:
        raise NotImplementedError(
            "_RunSingleGraphRunner subclasses must implement _handle_finished_subtask"
        )

    def _refresh_running_subtask(self, sid: str, cua_proc: CUAProcess) -> None:
        meta = self.meta_by_sid[sid]
        meta["screenshot_path"] = (
            extract_latest_screenshot_from_dir(cua_proc.subtask_dir)
            or meta.get("screenshot_path")
        )
        meta["step_count"] = count_subtask_steps(cua_proc.subtask_dir)
        self._ensure_vm_info(sid)

    def _maybe_handle_file_management(self, sid: str, cua_proc: CUAProcess) -> bool:
        if sid in self.filemgmt_reviewed_sids:
            return False
        fm_ready_path = Path(cua_proc.subtask_dir) / "ready_for_filemgmt"
        if not fm_ready_path.exists():
            return False

        actions: list[dict] = []
        try:
            pre_candidates = list(cua_proc.subtask_dir.rglob("files_pre.json"))
            post_candidates = list(cua_proc.subtask_dir.rglob("files_post.json"))
            pre_path = pre_candidates[0] if pre_candidates else cua_proc.subtask_dir / "files_pre.json"
            post_path = post_candidates[0] if post_candidates else cua_proc.subtask_dir / "files_post.json"
            diff = compute_file_diff(pre_path, post_path)
            n_changes = sum(len(diff.get(k, [])) for k in ("added", "modified"))
            if n_changes == 0:
                logger.info("File management [%s]: no added/modified files in diff", sid)
            elif not self.no_manager:
                logger.info(
                    "File management [%s]: %d added, %d modified, %d deleted",
                    sid,
                    len(diff.get("added", [])),
                    len(diff.get("modified", [])),
                    len(diff.get("deleted", [])),
                )
                actions, fm_usage = ask_manager_for_file_management(
                    self.st_by_id.get(sid, {"id": sid, "instruction": ""}),
                    diff,
                    self.archive_pool,
                    self.original_task_text,
                    self.args,
                    self.run_dir,
                )
                self.manager_usages.append(fm_usage)
                self._add_blocking_manager_inference(sid, fm_usage)
            else:
                logger.info(
                    "File management [%s]: no-manager mode; acknowledging %d change(s) with no actions",
                    sid,
                    n_changes,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("File management decision [%s] raised: %s", sid, exc)
            actions = []

        write_filemgmt_decision(Path(cua_proc.subtask_dir), actions)
        self.pending_filemgmt_actions[sid] = actions
        self.filemgmt_reviewed_sids.add(sid)
        return True

    def _handle_timed_out_subtask(self, sid: str, cua_proc: CUAProcess, elapsed: float) -> None:
        logger.warning("CUA [%s] timed out after %.0fs, terminating...", sid, elapsed)
        cua_proc.proc.terminate()
        try:
            cua_proc.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cua_proc.proc.kill()
        self._pop_running_subtask(sid)
        self._save_apptainer_state(sid)
        meta = self.meta_by_sid[sid]
        meta["state"] = "finished"
        meta["ended_at"] = time.time()
        meta["duration_seconds"] = elapsed
        meta["response"] = "[ERROR] Process timed out"
        try:
            _partial_response, usage = extract_cua_result_from_dir(
                cua_proc.subtask_dir,
                sid,
                self.cua_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUA [%s] timed out; failed to read partial usage: %s", sid, exc)
            usage = UsageInfo(model=self.cua_model)
        meta["usage"] = usage
        meta["screenshot_path"] = (
            meta.get("screenshot_path")
            or extract_latest_screenshot_from_dir(cua_proc.subtask_dir)
        )
        self._ensure_vm_info(sid, force_refresh=True)
        self._materialize_logical_completion(sid, apply_filemgmt=False)
        self._maybe_replan_after_completion(sid)

    def _ready_subtasks(self) -> list[dict]:
        ready: list[dict] = []
        dep_done = self.completed | self.cancelled
        running_logicals = self._running_logical_sids()
        for st in self.all_subtasks:
            sid = st["id"]
            if sid in self.completed or sid in self.cancelled or sid in running_logicals:
                continue
            if set(st.get("dependencies", [])).issubset(dep_done):
                ready.append(st)
        ready.sort(
            key=lambda s: (self.downstream_counts[s["id"]], len(s.get("dependencies", []))),
            reverse=True,
        )
        return ready

    def _process_manager_subtasks(self, ready: list[dict]) -> bool:
        mgr_ready = [s for s in ready if s.get("agent_type") == "manager"]
        if not mgr_ready:
            return False

        mgr_with_deps = [s for s in mgr_ready if s.get("dependencies")]
        mgr_without_deps = [s for s in mgr_ready if not s.get("dependencies")]
        for st in mgr_without_deps:
            self.results[st["id"]] = st["instruction"]
            self.completed.add(st["id"])
            self.task_timings[st["id"]] = self._task_timing_payload(0.0)
            self.completion_order.append((st["id"], time.time(), None))

        if mgr_with_deps:
            refined, usages = call_manager_for_subtasks(
                mgr_with_deps,
                self.results,
                self.original_task_text,
                self.args,
                self.run_dir,
                vm_info_by_sid=self.vm_info_by_sid,
                final_vm_choice=self.final_vm_choice,
                archive_pool=self.archive_pool,
                meta_by_sid=self.meta_by_sid,
            )
            self.manager_usages.extend(usages)
            usage_by_sid = {
                st["id"]: usage
                for st, usage in zip(mgr_with_deps, usages)
            }
            for st in mgr_with_deps:
                self.results[st["id"]] = refined.get(st["id"], "")
                self.completed.add(st["id"])
                usage = usage_by_sid.get(st["id"])
                if usage is not None:
                    self.cost_tracker[st["id"]] = {
                        "agent_type": "manager",
                        "model": usage.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cost_usd": usage.cost_usd,
                        "inference_seconds": _round_seconds(
                            _usage_inference_seconds(usage),
                            3,
                        ),
                    }
                duration_seconds = _usage_inference_seconds(usage)
                self.task_timings[st["id"]] = self._task_timing_payload(
                    duration_seconds,
                    usage,
                )
                self.completion_order.append((st["id"], time.time(), None))
                logger.info("Manager subtask [%s] complete", st["id"])
        return True

    def _launch_ready_cua_subtasks(self, ready: list[dict]) -> bool:
        cua_ready = [s for s in ready if s.get("agent_type") != "manager"]
        available_slots = (
            self.max_par - len(self.running)
            if self.max_par is not None
            else len(cua_ready)
        )
        to_launch = cua_ready[:max(0, available_slots)]
        if not to_launch:
            return False

        prepared_launches = [self._prepare_launch(st) for st in to_launch]
        website_override = self.task_website
        if self.no_initial_setup and website_override is None:
            website_override = ""
        for launch in prepared_launches:
            proc = launch_cua_subtask(
                launch.st,
                launch.instruction,
                self.run_dir,
                self.args,
                self.cua_args,
                staged_files=launch.staged_files,
                vm_path_override=launch.vm_path,
                temperature_override=launch.temperature_override,
                skip_canonical_setup=launch.skip_canonical_setup,
                domain=self.osworld_domain if self.is_osworld else None,
                canonical_config=self.canonical_config if self.is_osworld else None,
                website_override=website_override,
            )
            started_at = time.time()
            self.running[launch.sid] = (proc, started_at)
            self.meta_by_sid[launch.sid] = {
                "proc": proc,
                "instruction": launch.instruction,
                "state": "running",
                "vm_override_path": launch.vm_path,
                "is_variant": launch.is_variant,
                "response": "",
                "usage": None,
                "screenshot_path": None,
                "vm_info": None,
            }
            self.logical_start_times.setdefault(launch.sid, started_at)
            logger.info(
                "Launched CUA [%s] (%d/%s running)%s",
                launch.sid,
                len(self.running),
                self.max_par or "unlimited",
                "  [variant]" if launch.is_variant else "",
            )
        return True

    def _prepare_launch(self, st: dict) -> _PreparedLaunch:
        launch = _PreparedLaunch(
            st=st,
            sid=st["id"],
            instruction=st["instruction"],
            staged_files=resolve_input_files(st.get("input_files") or [], self.archive_pool) or None,
        )
        if self.no_initial_setup and not self.is_osworld:
            launch.skip_canonical_setup = True
        self._maybe_clone_from_init_from(launch)
        self._maybe_apply_variant(launch)
        if launch.vm_path is None and not launch.is_variant:
            self._maybe_reuse_dependency_vm(launch)
        self._maybe_reuse_scout_overlay(launch)
        self._maybe_allocate_fresh_vm(launch)
        self.skip_setup_by_sid[launch.sid] = launch.skip_canonical_setup
        if launch.vm_path is not None and not launch.is_variant and not self.no_manager:
            self._freeze_initial_state(launch)
        return launch

    def _maybe_clone_from_init_from(self, launch: _PreparedLaunch) -> None:
        init_from_sid = launch.st.get("init_from")
        if not init_from_sid or launch.vm_path is not None:
            return

        if self.use_apptainer:
            source_overlay = self.subtask_vm_path.get(init_from_sid)
            if not source_overlay:
                logger.warning(
                    "init_from [%s]: no overlay found for [%s]; falling through to other allocation paths",
                    launch.sid,
                    init_from_sid,
                )
                return
            try:
                fuser = subprocess.run(
                    ["fuser", source_overlay],
                    capture_output=True,
                    timeout=5,
                )
                fuser_pids = fuser.stdout.decode().strip()
                if fuser_pids:
                    raise RuntimeError(f"init_from overlay still held by PID(s): {fuser_pids}")
                launch.vm_path = _apptainer_create_overlay_from(
                    source_overlay,
                    label=f"{launch.sid}:init_from_{init_from_sid}",
                )
                self.overlay_parent[launch.vm_path] = source_overlay
                launch.skip_canonical_setup = True
                self.subtask_vm_path[launch.sid] = launch.vm_path
                if init_from_sid not in self.vmstate_saved:
                    logger.warning(
                        "init_from [%s] -> [%s]: VM state save was not confirmed; disk "
                        "state will be inherited but CPU/RAM may not restore",
                        launch.sid,
                        init_from_sid,
                    )
                logger.info(
                    "init_from [%s]: cloned from [%s]'s final overlay (canonical setup skipped): %s -> %s",
                    launch.sid,
                    init_from_sid,
                    source_overlay,
                    launch.vm_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "init_from [%s]: overlay clone from [%s] failed: %s; falling through to other allocation paths",
                    launch.sid,
                    init_from_sid,
                    exc,
                )
                launch.vm_path = None
            return

        source_vm = self.subtask_vm_path.get(init_from_sid)
        if not source_vm:
            logger.warning(
                "init_from [%s]: no VM path found for [%s]; falling through to other allocation paths",
                launch.sid,
                init_from_sid,
            )
            return
        try:
            _stop_vm_if_needed(self.args, source_vm, f"{launch.sid}:init_from_stop")
            clone_vmx = self.run_dir / "_init_from_vms" / launch.sid / f"{launch.sid}.vmx"
            _clone_vm_from_source(
                self.args,
                source_vm_path=source_vm,
                clone_vmx_path=clone_vmx,
                clone_name=launch.sid,
                label=f"{launch.sid}:init_from_{init_from_sid}",
            )
            launch.vm_path = str(clone_vmx.resolve())
            launch.skip_canonical_setup = True
            self.subtask_vm_path[launch.sid] = launch.vm_path
            logger.info(
                "init_from [%s]: cloned from [%s]'s VM -> %s (canonical setup skipped)",
                launch.sid,
                init_from_sid,
                launch.vm_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "init_from [%s]: clone from [%s] failed: %s; falling through to other allocation paths",
                launch.sid,
                init_from_sid,
                exc,
            )
            launch.vm_path = None

    def _maybe_apply_variant(self, launch: _PreparedLaunch) -> None:
        variant_of = launch.st.get("variant_of")
        if not variant_of:
            return
        if launch.vm_path is not None:
            launch.is_variant = True
            launch.temperature_override = 1.0
            logger.info(
                "Variant [%s] (variant_of=%s): VM already set by init_from, applying variant metadata only (temp=1.0)",
                launch.sid,
                variant_of,
            )
            return

        if self.use_apptainer:
            degraded_variant = False
            if variant_of in self.initial_state_clone:
                source = self.initial_state_clone[variant_of]
                try:
                    launch.vm_path = _apptainer_create_overlay_from(source, label=f"variant/{launch.sid}")
                    self.overlay_parent[launch.vm_path] = source
                    launch.is_variant = True
                    launch.temperature_override = 1.0
                    launch.skip_canonical_setup = self.skip_setup_by_sid.get(variant_of, False)
                    self.subtask_vm_path[launch.sid] = launch.vm_path
                    logger.info(
                        "Variant [%s] (variant_of=%s): cloned from frozen-initial overlay %s -> %s "
                        "(skip_canonical_setup=%s)",
                        launch.sid,
                        variant_of,
                        source,
                        launch.vm_path,
                        launch.skip_canonical_setup,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Apptainer variant overlay clone failed for [%s]: %s; falling back to DEGRADED fresh variant (different base state!)",
                        launch.sid,
                        exc,
                    )
                    launch.vm_path = None
                    degraded_variant = True
            if launch.vm_path is not None:
                return
            launch.is_variant = True
            launch.temperature_override = 1.0
            try:
                launch.vm_path = _apptainer_create_overlay(label=f"variant/{launch.sid}")
                launch.skip_canonical_setup = False
                self.subtask_vm_path[launch.sid] = launch.vm_path
                if degraded_variant:
                    launch.st["_variant_degraded"] = True
                    logger.warning(
                        "Variant [%s] (variant_of=%s): DEGRADED — running from fresh base instead of frozen-initial (clone failed): %s",
                        launch.sid,
                        variant_of,
                        launch.vm_path,
                    )
                else:
                    logger.info(
                        "Variant [%s] (variant_of=%s): fresh overlay (no frozen-initial available, canonical setup re-enabled): %s",
                        launch.sid,
                        variant_of,
                        launch.vm_path,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Apptainer variant overlay failed for [%s]: %s; variant will rely on CUA self-allocation",
                    launch.sid,
                    exc,
                )
            return

        if variant_of not in self.initial_state_clone:
            logger.warning(
                "Variant [%s] (variant_of=%s): no frozen-initial clone available for source; falling back to fresh allocation",
                launch.sid,
                variant_of,
            )
            return
        try:
            clone_vmx = self.run_dir / "_variant_vms" / launch.sid / f"{launch.sid}.vmx"
            _clone_vm_from_source(
                self.args,
                source_vm_path=self.initial_state_clone[variant_of],
                clone_vmx_path=clone_vmx,
                clone_name=launch.sid,
                label=f"{launch.sid}:variant_of_{variant_of}",
            )
            launch.vm_path = str(clone_vmx.resolve())
            launch.is_variant = True
            launch.temperature_override = 1.0
            launch.skip_canonical_setup = self.skip_setup_by_sid.get(variant_of, False)
            self.subtask_vm_path[launch.sid] = launch.vm_path
            logger.info(
                "Variant [%s]: cloned from frozen-initial of [%s] -> %s (skip_canonical_setup=%s)",
                launch.sid,
                variant_of,
                launch.vm_path,
                launch.skip_canonical_setup,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Variant [%s] (variant_of=%s): clone from frozen-initial failed: %s; falling back to fresh allocation",
                launch.sid,
                variant_of,
                exc,
            )
            launch.vm_path = None
            launch.is_variant = False

    def _maybe_reuse_dependency_vm(self, launch: _PreparedLaunch) -> None:
        reusable_dep = None
        for dep in launch.st.get("dependencies") or []:
            if (
                dep in self.subtask_vm_path
                and dep in self.completed
                and (self.use_apptainer or dep not in self.vm_reused_by)
                and self.st_by_id.get(dep, {}).get("agent_type", "cua") == "cua"
            ):
                reusable_dep = dep
                break
        if reusable_dep is None:
            return

        reuse_vm = self.subtask_vm_path[reusable_dep]
        if self.use_apptainer:
            try:
                fuser = subprocess.run(
                    ["fuser", reuse_vm],
                    capture_output=True,
                    timeout=5,
                )
                fuser_pids = fuser.stdout.decode().strip()
                if fuser_pids:
                    raise RuntimeError(f"overlay still held by PID(s): {fuser_pids}")
                logger.info(
                    "Dependent [%s]: fuser check passed for predecessor [%s] overlay %s (no holders)",
                    launch.sid,
                    reusable_dep,
                    reuse_vm,
                )
                launch.vm_path = _apptainer_create_overlay_from(
                    reuse_vm,
                    label=f"{launch.sid}:dep_{reusable_dep}",
                )
                self.overlay_parent[launch.vm_path] = reuse_vm
                launch.skip_canonical_setup = True
                self.subtask_vm_path[launch.sid] = launch.vm_path
                logger.info(
                    "Dependent [%s]: cloned predecessor [%s]'s overlay (state preserved, canonical setup skipped): %s -> %s",
                    launch.sid,
                    reusable_dep,
                    reuse_vm,
                    launch.vm_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Dependent [%s]: overlay clone from predecessor [%s] failed: %s; falling back to fresh allocation",
                    launch.sid,
                    reusable_dep,
                    exc,
                )
                launch.vm_path = None
            return

        try:
            _orchestrator_reclaim_vm(self.args, reuse_vm)
            launch.vm_path = reuse_vm
            launch.skip_canonical_setup = True
            self.vm_reused_by[reusable_dep] = launch.sid
            self.subtask_vm_path[launch.sid] = launch.vm_path
            logger.info(
                "Dependent [%s]: reusing predecessor [%s]'s VM (final state preserved, canonical setup skipped): %s",
                launch.sid,
                reusable_dep,
                launch.vm_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dependent [%s]: reclaim of predecessor [%s]'s VM failed: %s; falling back to fresh allocation",
                launch.sid,
                reusable_dep,
                exc,
            )
            launch.vm_path = None

    def _maybe_reuse_scout_overlay(self, launch: _PreparedLaunch) -> None:
        if launch.vm_path is not None or self.scout_overlay_path is None:
            return
        if launch.st.get("dependencies"):
            return
        launch.vm_path = self.scout_overlay_path
        launch.skip_canonical_setup = True
        self.subtask_vm_path[launch.sid] = launch.vm_path
        logger.info(
            "Reusing scout overlay for [%s] (canonical setup skipped): %s",
            launch.sid,
            launch.vm_path,
        )
        self.scout_overlay_path = None

    def _maybe_allocate_fresh_vm(self, launch: _PreparedLaunch) -> None:
        if launch.vm_path is not None:
            return
        if self.use_apptainer:
            try:
                launch.vm_path = _apptainer_create_overlay(label=launch.sid)
                self.subtask_vm_path[launch.sid] = launch.vm_path
                logger.info(
                    "Pre-allocated apptainer overlay for [%s]: %s",
                    launch.sid,
                    launch.vm_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not pre-allocate apptainer overlay for [%s]: %s; CUA subprocess will allocate its own",
                    launch.sid,
                    exc,
                )
            return

        vm_path: Optional[str] = None
        try:
            registry_vm_path = _orchestrator_alloc_vm(self.args)
            vm_path = str(_registry_path_to_absolute(self.args, registry_vm_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not pre-allocate VM for [%s]: %s; CUA subprocess will allocate its own",
                launch.sid,
                exc,
            )
        if vm_path is None:
            return
        try:
            _take_subtask_snapshot(self.args, vm_path, launch.sid)
            launch.vm_path = vm_path
            self.subtask_vm_path[launch.sid] = vm_path
            logger.info("Pre-launch snapshot for [%s]: %s", launch.sid, vm_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Pre-launch snapshot failed for [%s]: %s; releasing VM and falling back",
                launch.sid,
                exc,
            )
            _orchestrator_release_vm(self.args, vm_path)

    def _freeze_initial_state(self, launch: _PreparedLaunch) -> None:
        if self.use_apptainer:
            try:
                working_overlay = _apptainer_create_overlay_from(
                    launch.vm_path,
                    label=f"{launch.sid}:working",
                )
                self.overlay_parent[working_overlay] = launch.vm_path
                self.initial_state_clone[launch.sid] = launch.vm_path
                launch.vm_path = working_overlay
                self.subtask_vm_path[launch.sid] = working_overlay
                logger.info(
                    "Frozen-initial overlay for [%s]: %s (CUA runs on %s)",
                    launch.sid,
                    self.initial_state_clone[launch.sid],
                    working_overlay,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Frozen-initial overlay failed for [%s]: %s; future variants of this subtask will fall back",
                    launch.sid,
                    exc,
                )
            return

        try:
            _stop_vm_if_needed(self.args, launch.vm_path, f"{launch.sid}:pre-frozen-initial-stop")
            frozen_vmx = self.run_dir / "_frozen_initial" / launch.sid / f"{launch.sid}.vmx"
            _clone_vm_from_source(
                self.args,
                source_vm_path=launch.vm_path,
                clone_vmx_path=frozen_vmx,
                clone_name=f"{launch.sid}_initial",
                label=f"{launch.sid}:frozen_initial",
            )
            self.initial_state_clone[launch.sid] = str(frozen_vmx.resolve())
            logger.info(
                "Frozen-initial clone for [%s]: %s (source_vm=%s)",
                launch.sid,
                frozen_vmx,
                launch.vm_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Frozen-initial clone failed for [%s]: %s; future variants of this subtask will fall back",
                launch.sid,
                exc,
            )

    def _drain_pending_filemgmt(self) -> None:
        if not self.pending_filemgmt_actions or not self.running:
            return
        drain_sids = {sid for sid in self.running if sid in self.pending_filemgmt_actions}
        if not drain_sids:
            return
        logger.info(
            "Draining %d running subtask(s) with pending filemgmt actions...",
            len(drain_sids),
        )
        drain_deadline = time.time() + 120
        while drain_sids and time.time() < drain_deadline:
            drained_any = False
            for sid in list(drain_sids):
                entry = self.running.get(sid)
                if entry is None:
                    drain_sids.discard(sid)
                    continue
                cua_proc, start_time = entry
                ret = cua_proc.proc.poll()
                if ret is None:
                    continue
                self._pop_running_subtask(sid)
                drain_sids.discard(sid)
                drained_any = True
                self._record_completion(sid, ret, time.time() - start_time)
                self._apply_pending_filemgmt(sid, Path(cua_proc.subtask_dir))
            if not drained_any:
                time.sleep(0.5)
        if drain_sids:
            logger.warning(
                "Post-loop drain timed out; %d pending filemgmt action(s) not applied: %s",
                len(drain_sids),
                list(drain_sids),
            )

    def _cleanup_vmware_variant_clones(self) -> None:
        if self.use_apptainer:
            return
        for sid, meta in self.meta_by_sid.items():
            if not meta.get("is_variant"):
                continue
            clone_vmx = meta.get("vm_override_path")
            if not clone_vmx:
                continue
            _stop_vm_if_needed(self.args, clone_vmx, f"{sid}:variant-cleanup")
            _delete_vm_tree(clone_vmx, f"{sid}:variant-cleanup")

    def _cleanup_vmware_init_from_clones(self) -> None:
        if self.use_apptainer:
            return
        init_from_dir = self.run_dir / "_init_from_vms"
        if not init_from_dir.exists():
            return
        for sid_dir in init_from_dir.iterdir():
            if not sid_dir.is_dir():
                continue
            for vmx in sid_dir.glob("*.vmx"):
                _stop_vm_if_needed(self.args, str(vmx), f"{sid_dir.name}:init_from-cleanup")
                _delete_vm_tree(str(vmx), f"{sid_dir.name}:init_from-cleanup")

    def _cleanup_vmware_frozen_initial_clones(self) -> None:
        if self.use_apptainer:
            return
        for sid, frozen_vmx in self.initial_state_clone.items():
            _stop_vm_if_needed(self.args, frozen_vmx, f"{sid}:frozen-initial-cleanup")
            _delete_vm_tree(frozen_vmx, f"{sid}:frozen-initial-cleanup")

    def _cleanup_apptainer_overlays(self) -> None:
        if not self.use_apptainer:
            return

        our_overlays = {
            os.path.abspath(p)
            for p in list(self.subtask_vm_path.values()) + list(self.initial_state_clone.values())
            if p
        }
        killed_pids: set[int] = set()
        for sid_label, overlay_path in list(self.subtask_vm_path.items()) + [
            (f"frozen:{sid}", path) for sid, path in self.initial_state_clone.items()
        ]:
            if not overlay_path:
                continue
            try:
                result = subprocess.run(
                    ["fuser", overlay_path],
                    capture_output=True,
                    timeout=10,
                )
                for pid_str in result.stdout.decode().split():
                    if not pid_str.isdigit():
                        continue
                    pid_int = int(pid_str)
                    if pid_int in killed_pids:
                        continue
                    try:
                        cmdline = Path(f"/proc/{pid_int}/cmdline").read_bytes()
                        if b"qemu" not in cmdline:
                            continue
                    except OSError:
                        continue

                    owns_our_overlay = any(overlay.encode() in cmdline for overlay in our_overlays)
                    if not owns_our_overlay:
                        try:
                            for fd in Path(f"/proc/{pid_int}/fd").iterdir():
                                try:
                                    if os.readlink(str(fd)) in our_overlays:
                                        owns_our_overlay = True
                                        break
                                except OSError:
                                    continue
                        except OSError:
                            pass
                    if not owns_our_overlay:
                        continue

                    logger.info(
                        "[%s] Stopping qemu process %d using %s",
                        sid_label,
                        pid_int,
                        overlay_path,
                    )
                    try:
                        os.kill(pid_int, 15)
                        for _ in range(10):
                            time.sleep(0.5)
                            os.kill(pid_int, 0)
                        os.kill(pid_int, 9)
                    except OSError:
                        pass
                    killed_pids.add(pid_int)
            except Exception:  # noqa: BLE001
                pass

        all_overlays = {
            path for path in list(self.subtask_vm_path.values()) + list(self.initial_state_clone.values()) if path
        }
        children_of: dict[str, list[str]] = {path: [] for path in all_overlays}
        for child, parent in self.overlay_parent.items():
            if parent in children_of and child in all_overlays:
                children_of[parent].append(child)

        deleted: set[str] = set()
        remaining = set(all_overlays)
        while remaining:
            leaves = [path for path in remaining if all(c in deleted for c in children_of.get(path, []))]
            if not leaves:
                logger.warning(
                    "Overlay cleanup: %d overlay(s) have unresolved backing-chain order — deferring deletion: %s",
                    len(remaining),
                    [os.path.basename(path) for path in remaining],
                )
                gc_file = self.run_dir / "_deferred_overlay_gc.txt"
                try:
                    with open(gc_file, "a") as f:
                        for path in sorted(remaining):
                            f.write(path + "\n")
                    logger.info("Wrote %d deferred overlay path(s) to %s", len(remaining), gc_file)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to write deferred GC list: %s", exc)
                break

            failed_leaves: list[str] = []
            for leaf in leaves:
                if _delete_vm_tree(leaf, "overlay-cleanup"):
                    deleted.add(leaf)
                    remaining.discard(leaf)
                else:
                    failed_leaves.append(leaf)
            if not failed_leaves:
                continue

            defer = set(failed_leaves)
            for failed in failed_leaves:
                parent = self.overlay_parent.get(failed)
                while parent and parent in remaining:
                    defer.add(parent)
                    parent = self.overlay_parent.get(parent)
            remaining -= defer
            gc_file = self.run_dir / "_deferred_overlay_gc.txt"
            try:
                with open(gc_file, "a") as f:
                    for path in sorted(defer):
                        f.write(path + "\n")
            except Exception:  # noqa: BLE001
                pass
            logger.warning(
                "Overlay cleanup: %d overlay(s) could not be deleted (and their ancestors deferred): %s",
                len(defer),
                [os.path.basename(path) for path in defer],
            )

    def _write_summary(self, total_cua_cost: float, total_manager_cost: float, total_cost: float) -> None:
        summary = {
            "task_id": self.task_id,
            "subtask_costs": self.cost_tracker,
            "manager_calls": [
                {
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                    "inference_seconds": _round_seconds(
                        _usage_inference_seconds(usage),
                        3,
                    ),
                }
                for usage in self.manager_usages
            ],
            "total_cua_cost_usd": round(total_cua_cost, 6),
            "total_manager_cost_usd": round(total_manager_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_model_call_seconds": _round_seconds(self._total_model_call_seconds()),
        }
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Cost summary saved to %s", summary_path)

    def _maybe_auto_assign_final_vm(self) -> None:
        if not self.no_manager or self.final_vm_choice:
            return
        cua_subtasks = [s for s in self.all_subtasks if s.get("agent_type") != "manager"]
        if len(cua_subtasks) == 1 and cua_subtasks[0]["id"] in self.vm_info_by_sid:
            sid = cua_subtasks[0]["id"]
        elif self.vm_info_by_sid:
            sid = next(iter(self.vm_info_by_sid))
        else:
            return
        info = self.vm_info_by_sid[sid]
        self.final_vm_choice["subtask_id"] = sid
        self.final_vm_choice["vm_path"] = info.get("vm_path")
        logger.info(
            "--no-manager: auto-assigned final VM from subtask [%s]: %s",
            sid,
            self.final_vm_choice["vm_path"],
        )

    def _final_response(self) -> str:
        if self.no_manager:
            cua_subtasks = [s for s in self.all_subtasks if s.get("agent_type") != "manager"]
            return self.results.get(cua_subtasks[0]["id"], "") if cua_subtasks else ""
        agg = self.graph["dependency_graph"].get("aggregation", {})
        return self.results.get(agg.get("id", ""), "")

    def _build_final_output(self, final_response: str, total_duration: float, total_cost: float) -> dict:
        total_model_call_seconds = self._total_model_call_seconds()
        total_inference_wall_seconds = self._total_inference_wall_seconds(total_duration)
        final_output = {
            "task_id": self.task_id,
            "task_timings": self.task_timings,
            "subtask_results": {
                sid: {
                    "status": "error" if resp.startswith("[ERROR]") else "success",
                    "response": resp,
                }
                for sid, resp in self.results.items()
            },
            "final_response": final_response,
            "total_duration_seconds": round(total_duration, 1),
            "total_inference_seconds": _round_seconds(total_inference_wall_seconds),
            "total_inference_wall_seconds": _round_seconds(total_inference_wall_seconds),
            "total_model_call_seconds": _round_seconds(total_model_call_seconds),
            "total_duration_without_inference_seconds": _round_seconds(
                max(0.0, total_duration - total_inference_wall_seconds)
            ),
            "total_cost_usd": round(total_cost, 6),
        }
        if self.is_osworld:
            final_output["domain"] = self.osworld_domain
            final_output["instruction"] = self.original_task_text
        else:
            final_output["task_text"] = self.original_task_text
        final_output["subtask_vm_paths"] = {
            sid: info.get("vm_path")
            for sid, info in self.vm_info_by_sid.items()
        }
        final_output["final_vm_subtask_id"] = self.final_vm_choice.get("subtask_id")
        final_output["final_vm_path"] = self.final_vm_choice.get("vm_path")
        final_output["archive_pool"] = self.archive_pool
        if not self.no_manager:
            snapshot_count = 0
            snap_dir = self.run_dir / "graph_snapshots"
            if snap_dir.exists():
                snapshot_count = len(list(snap_dir.glob("dependency_graph_*.json")))
            final_output["replanning"] = {
                "enabled": True,
                "num_calls": self.replan_count,
                "num_applied": self.num_applied_replans,
                "snapshot_count": snapshot_count,
            }
        return final_output

    def _persist_final_results(self, final_output: dict) -> Path:
        output_path = self.run_dir / "final_results.json"
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        return output_path

    def _finalize_results(self) -> tuple[str, dict, float, float, float]:
        total_duration = time.time() - self.overall_start
        logger.info("All %d tasks complete in %.1fs", len(self.all_subtasks), total_duration)

        total_cua_cost = sum(
            info["cost_usd"]
            for info in self.cost_tracker.values()
            if info["agent_type"] == "cua"
        )
        total_manager_cost = sum(usage.cost_usd for usage in self.manager_usages)
        total_cost = total_cua_cost + total_manager_cost
        logger.info("=== Cost Summary ===")
        logger.info("CUA total:     $%.4f", total_cua_cost)
        logger.info("Manager total: $%.4f", total_manager_cost)
        logger.info("Overall total: $%.4f", total_cost)
        self._write_summary(total_cua_cost, total_manager_cost, total_cost)

        self._maybe_auto_assign_final_vm()
        final_response = self._final_response()
        final_output = self._build_final_output(final_response, total_duration, total_cost)
        output_path = self._persist_final_results(final_output)
        logger.info("Final results saved to %s", output_path)
        logger.info(
            "Total duration: %.1fs wall, %.1fs inference wall, %.1fs non-inference",
            final_output["total_duration_seconds"],
            final_output["total_inference_seconds"],
            final_output["total_duration_without_inference_seconds"],
        )

        if self.is_osworld:
            final_output["osworld_evaluator_score"] = run_osworld_evaluator_subprocess(
                run_dir=self.run_dir,
                task_id=self.task_id,
                domain=self.osworld_domain,
                canonical_config=self.canonical_config,
                final_vm_path=self.final_vm_choice.get("vm_path"),
                cua_model=self.cua_model,
                cua_args=self.cua_args,
                args=self.args,
            )
            self._persist_final_results(final_output)
            logger.info(
                "Re-saved final_results.json with osworld_evaluator_score=%s",
                final_output["osworld_evaluator_score"],
            )

        return final_response, final_output, total_cua_cost, total_manager_cost, total_cost

    def _cleanup_unused_scout_overlay(self) -> None:
        if self.scout_overlay_path is None:
            return
        try:
            os.remove(self.scout_overlay_path)
            logger.info("Removed unconsumed scout overlay %s", self.scout_overlay_path)
        except OSError:
            pass

    def _print_final_response(
        self,
        final_response: str,
        final_output: dict,
        total_cua_cost: float,
        total_manager_cost: float,
        total_cost: float,
    ) -> None:
        print(f"\n{'=' * 60}")
        print("FINAL RESPONSE")
        print(f"{'=' * 60}")
        print(final_response)
        print(
            f"\nTotal cost: ${total_cost:.4f} (CUA: ${total_cua_cost:.4f}, Manager: ${total_manager_cost:.4f})"
        )
        if self.is_osworld and final_output.get("osworld_evaluator_score") is not None:
            print(f"OSWorld evaluator score: {final_output['osworld_evaluator_score']}")
        print(f"{'=' * 60}")

_TASK_COMPLETION_MARKERS = ("final_results.json", "final_result.json")


def _task_has_completion_marker(task_dir: Path, *, osworld_mode: bool = False) -> bool:
    markers = _TASK_COMPLETION_MARKERS
    if osworld_mode:
        markers = (*markers, "result.txt")
    return any((task_dir / marker).exists() for marker in markers)


def _try_claim_task(result_dir: Path, task_id: str, worker_id: str) -> bool:
    """Atomically claim a task for this worker.

    Returns True if the claim was acquired, False if another worker already
    holds it. Uses O_EXCL file creation for atomicity on shared filesystems.
    Existing claims are never automatically reclaimed during a run; clean stale
    ``.claimed`` files explicitly before restarting workers.
    """
    task_dir = result_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    claim_path = task_dir / ".claimed"

    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as f:
        f.write(f"{worker_id}\n{time.time()}\n")
    return True


def _run_with_tasks_file(args: argparse.Namespace, cua_args: list[str]) -> None:
    """Generate graphs from a tasks file and run each task.

    Workers share a single result-dir and pick tasks via atomic claim files.
    Each task directory gets a ``.claimed`` sentinel created with O_EXCL so
    only one worker runs a given task. Existing claims are never auto-reclaimed;
    remove stale ``.claimed`` files manually before resuming a run. Completed
    tasks (those with a ``final_results.json`` or ``final_result.json``; plus
    ``result.txt`` for OSWorld mode) are skipped by all workers.

    Detects OSWorld vs generic CUA tasks by trying the OSWorld loader first
    (which produces dicts with a ``domain`` key). Falls back to ``load_cua_tasks``
    for generic CUA task files.
    """
    osworld_mode = False
    try:
        tasks = load_osworld_tasks_for_gen(
            tasks_path=args.tasks_file,
            examples_dir=args.osworld_data_dir,
        )
        if tasks and "domain" in tasks[0]:
            osworld_mode = True
    except Exception:  # noqa: BLE001
        tasks = None

    if not osworld_mode:
        tasks = load_cua_tasks(args.tasks_file)
        logger.info("Loaded %d CUA task(s) from %s", len(tasks), args.tasks_file)
    else:
        logger.info("Loaded %d OSWorld task(s) from %s", len(tasks), args.tasks_file)

    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            logger.error("Task ID %s not found in %s", args.task_id, args.tasks_file)
            sys.exit(1)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    total = len(tasks)
    processed = 0
    completed_skips = 0
    claim_skips = 0

    # Shuffle so concurrent workers don't all race for the same task_id.
    tasks = list(tasks)
    random.Random(1337).shuffle(tasks)

    for task in tasks:
        task_id = task["task_id"]
        task_dir = result_dir / task_id
        if _task_has_completion_marker(task_dir, osworld_mode=osworld_mode):
            completed_skips += 1
            continue

        if not _try_claim_task(result_dir, task_id, worker_id):
            claim_skips += 1
            continue

        # Re-check completion after claiming (another worker may have finished
        # between our check and our claim).
        if _task_has_completion_marker(task_dir, osworld_mode=osworld_mode):
            completed_skips += 1
            continue

        processed += 1
        if osworld_mode:
            logger.info("=== Task %d/%d (claimed): %s (%s) ===",
                        processed, total, task_id, task["domain"])
        else:
            logger.info("=== Task %d/%d (claimed): %s ===", processed, total, task_id)
        task_desc = task.get("confirmed_task") or task.get("instruction") or "(no description)"
        logger.info("Task description: %s", task_desc[:500])

        try:
            graph_path = result_dir / task_id / "dependency_graph.json"
            if osworld_mode:
                task_metadata = {
                    "task_id": task_id,
                    "domain": task["domain"],
                    "instruction": task["instruction"],
                }
            else:
                task_metadata = {
                    "task_id": task_id,
                    "task_text": task["confirmed_task"],
                }
                if "website" in task:
                    task_metadata["website"] = task["website"]
                if task.get("no_initial_setup") or task.get("blank_vm"):
                    task_metadata["no_initial_setup"] = True
            _RunSingleGraphRunner(str(graph_path), args, cua_args, task_metadata=task_metadata).run()
        except Exception:
            logger.exception("Failed on task %s, continuing to next", task_id)

    logger.info(
        "Worker %s done: processed=%d, skipped_completed=%d, skipped_claimed=%d, total=%d",
        worker_id, processed, completed_skips, claim_skips, total,
    )

    if osworld_mode:
        _aggregate_success_rate(Path(args.result_dir))


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, _frame):
    logger.info("Received signal %s, terminating child processes...", signum)
    for proc in _child_processes:
        if proc.poll() is None:
            proc.terminate()
    time.sleep(2)
    for proc in _child_processes:
        if proc.poll() is None:
            proc.kill()
    sys.exit(1)
