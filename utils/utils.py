from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from utils.llm import UsageInfo, compute_cost


logger = logging.getLogger("macu")


def load_cua_tasks(tasks_path: str | Path) -> list[dict]:
    """Load CUA tasks from a JSON file."""
    with open(tasks_path) as f:
        return json.load(f)


def load_osworld_tasks(
    tasks_path: str | Path,
    examples_dir: str | Path,
    domain_filter: str | None = None,
) -> list[dict]:
    """Load OSWorld tasks by resolving a domain-to-UUID mapping to per-task JSONs."""
    with open(tasks_path) as f:
        domain_map: dict[str, list[str]] = json.load(f)

    examples_root = Path(examples_dir) / "examples"
    tasks: list[dict] = []

    for domain, uuids in sorted(domain_map.items()):
        if domain_filter and domain != domain_filter:
            continue
        for uuid_str in uuids:
            task_file = examples_root / domain / f"{uuid_str}.json"
            if not task_file.exists():
                logger.warning("Task file not found: %s", task_file)
                continue
            with open(task_file) as f:
                raw = json.load(f)
            tasks.append(
                {
                    "task_id": raw["id"],
                    "domain": domain,
                    "instruction": raw["instruction"],
                    "related_apps": raw.get("related_apps", []),
                    "raw": raw,
                }
            )

    return tasks


def _clean_cua_response(text: str) -> str:
    """Strip extraneous whitespace and tool_call XML blocks from a CUA response."""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _coerce_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _extract_inference_interval(model_usage: dict) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    raw_intervals = model_usage.get("inference_intervals")
    if isinstance(raw_intervals, list):
        for item in raw_intervals:
            if isinstance(item, dict):
                started_at = _coerce_float(item.get("start"))
                ended_at = _coerce_float(item.get("end"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                started_at = _coerce_float(item[0])
                ended_at = _coerce_float(item[1])
            else:
                continue
            if ended_at >= started_at > 0:
                intervals.append((started_at, ended_at))

    started_at = _coerce_float(
        model_usage.get("inference_start_time") or model_usage.get("model_start_time")
    )
    ended_at = _coerce_float(
        model_usage.get("inference_end_time") or model_usage.get("model_end_time")
    )
    if ended_at >= started_at > 0:
        intervals.append((started_at, ended_at))
    return intervals


def extract_cua_result_from_dir(
    subtask_dir: Path,
    label: str,
    cua_model: str,
) -> tuple[str, UsageInfo]:
    """Parse ``traj.jsonl`` under *subtask_dir* and extract final response + usage."""
    traj_files = list(subtask_dir.rglob("traj.jsonl"))

    total_input = 0
    total_output = 0
    total_inference_seconds = 0.0
    inference_intervals: list[tuple[float, float]] = []

    if not traj_files:
        return (
            f"[ERROR] No trajectory file found for subtask {label}",
            UsageInfo(model=cua_model),
        )

    last_response = ""
    seen_steps = set()
    for traj_path in traj_files:
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resp = entry.get("response", "")
                if resp.strip():
                    last_response = resp
                step_num = entry.get("step_num")
                model_usage = entry.get("model_usage", {})
                if model_usage and step_num not in seen_steps:
                    seen_steps.add(step_num)
                    total_input += (
                        model_usage.get("prompt_tokens", 0)
                        or model_usage.get("input_tokens", 0)
                        or 0
                    )
                    total_output += (
                        model_usage.get("completion_tokens", 0)
                        or model_usage.get("output_tokens", 0)
                        or 0
                    )
                    inference_seconds = (
                        model_usage.get("inference_seconds")
                        or model_usage.get("model_time")
                        or model_usage.get("model_seconds")
                        or 0
                    )
                    total_inference_seconds += _coerce_float(inference_seconds)
                    inference_intervals.extend(_extract_inference_interval(model_usage))

    cost = compute_cost(cua_model, total_input, total_output)
    usage = UsageInfo(
        input_tokens=total_input,
        output_tokens=total_output,
        model=cua_model,
        cost_usd=cost,
        inference_seconds=total_inference_seconds,
        inference_intervals=inference_intervals,
    )

    if not last_response:
        return f"[ERROR] No response found in trajectory for subtask {label}", usage

    return _clean_cua_response(last_response), usage


def extract_latest_screenshot_from_dir(subtask_dir: Path) -> Optional[Path]:
    """Return the most recent screenshot in *subtask_dir*'s trajectory."""
    traj_files = list(subtask_dir.rglob("traj.jsonl"))
    if not traj_files:
        return None

    latest: Optional[Path] = None
    for traj_path in traj_files:
        traj_dir = traj_path.parent
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fname = entry.get("screenshot_file") or ""
                if fname:
                    candidate = traj_dir / fname
                    if candidate.exists():
                        latest = candidate
    return latest


def count_subtask_steps(subtask_dir: Path) -> int:
    """Return the number of CUA steps logged for a running subtask."""
    traj_files = list(subtask_dir.rglob("traj.jsonl"))
    if not traj_files:
        return 0
    count = 0
    for traj_path in traj_files:
        try:
            with open(traj_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("screenshot_file"):
                        count += 1
        except OSError:
            continue
    return count


def extract_last_k_screenshots_from_dir(subtask_dir: Path, k: int = 1) -> list[Path]:
    """Return the last *k* screenshots from *subtask_dir*'s trajectory."""
    traj_files = list(subtask_dir.rglob("traj.jsonl"))
    if not traj_files:
        return []

    all_screenshots: list[Path] = []
    for traj_path in traj_files:
        traj_dir = traj_path.parent
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fname = entry.get("screenshot_file") or ""
                if fname:
                    candidate = traj_dir / fname
                    if candidate.exists():
                        all_screenshots.append(candidate)

    return all_screenshots[-k:] if all_screenshots else []


def extract_last_k_screenshots(subtask_id: str, run_dir: Path, k: int = 1) -> list[Path]:
    """Return the last *k* screenshots from a subtask's trajectory."""
    return extract_last_k_screenshots_from_dir(run_dir / subtask_id, k=k)
