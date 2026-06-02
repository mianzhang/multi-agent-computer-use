from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

from utils.llm import UsageInfo, call_llm, load_prompt, parse_json_response, render_prompt


logger = logging.getLogger("macu")
graph_gen_logger = logging.getLogger("macu.graph_gen")

_REPLAN_ALLOWED_MODIFY_FIELDS = {
    "instruction",
    "dependencies",
    "description",
    "input_files",
    "init_from",
}


def generate_validated_dependency_graph(
    task_text: str,
    prompt_yaml_path: str | Path,
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    max_attempts: int = 3,
    image_path: str | Path | Sequence[str | Path] | None = None,
    reasoning_effort: str | None = None,
) -> tuple[dict, UsageInfo, list[str]]:
    """Generate a dependency graph with validation-driven retries."""
    prompt_config = load_prompt(str(prompt_yaml_path))
    system_prompt, user_prompt = render_prompt(prompt_config, task=task_text)

    total_usage = UsageInfo(input_tokens=0, output_tokens=0, model=model or "", cost_usd=0.0)
    graph: dict = {}
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        current_user_prompt = user_prompt
        if attempt > 1 and errors:
            correction = "\n\nYour previous dependency graph had the following validation errors:\n"
            correction += "".join(f"- {err}\n" for err in errors)
            correction += "\nPlease regenerate the complete dependency graph fixing all of the above issues."
            current_user_prompt = user_prompt + correction

        raw_response, usage = call_llm(
            system_prompt=system_prompt,
            user_prompt=current_user_prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            image_path=image_path,
            reasoning_effort=reasoning_effort,
        )
        total_usage = UsageInfo(
            input_tokens=total_usage.input_tokens + usage.input_tokens,
            output_tokens=total_usage.output_tokens + usage.output_tokens,
            model=usage.model,
            cost_usd=total_usage.cost_usd + usage.cost_usd,
        )

        try:
            graph = parse_json_response(raw_response)
        except json.JSONDecodeError as exc:
            errors = [f"JSON parse error: {exc}"]
            if attempt < max_attempts:
                graph_gen_logger.warning(
                    "Graph generation attempt %d/%d: JSON parse error, retrying",
                    attempt,
                    max_attempts,
                )
                continue
            graph = {"raw_response": raw_response, "parse_error": str(exc)}
            return graph, total_usage, errors

        errors = validate_dependency_graph(graph)
        if not errors:
            if attempt > 1:
                graph_gen_logger.info(
                    "Graph validation passed on attempt %d/%d",
                    attempt,
                    max_attempts,
                )
            return graph, total_usage, []

        graph_gen_logger.warning(
            "Graph generation attempt %d/%d validation errors: %s",
            attempt,
            max_attempts,
            errors,
        )

    return graph, total_usage, errors


def validate_dependency_graph(graph: dict) -> list[str]:
    """Validate a dependency graph dict and return a list of warnings/errors."""
    errors: list[str] = []
    subtasks = graph.get("subtasks", [])
    agg = graph.get("aggregation", {})

    ids = [st["id"] for st in subtasks]
    seen: set[str] = set()
    for sid in ids:
        if sid in seen:
            errors.append(f"Duplicate subtask id: '{sid}'")
        seen.add(sid)

    if agg and agg.get("id") in seen:
        errors.append(f"Aggregation id '{agg['id']}' duplicates a subtask id")

    if agg and agg.get("agent_type") != "manager":
        errors.append(f"Aggregation agent_type is '{agg.get('agent_type')}', expected 'manager'")

    for st in subtasks + ([agg] if agg else []):
        for dep in st.get("dependencies", []):
            if dep not in seen:
                errors.append(f"Subtask '{st['id']}' depends on unknown id '{dep}'")

    st_by_id = {st["id"]: st for st in subtasks}
    for st in subtasks:
        if "input_files" in st:
            ifiles = st.get("input_files")
            if not isinstance(ifiles, list) or not all(isinstance(x, str) for x in ifiles):
                errors.append(
                    f"Subtask '{st['id']}' input_files must be a list of strings"
                )
        if "variant_of" in st:
            variant_of = st.get("variant_of")
            if not isinstance(variant_of, str) or not variant_of:
                errors.append(
                    f"Subtask '{st['id']}' variant_of must be a non-empty string"
                )
                continue
            source = st_by_id.get(variant_of)
            if source is None:
                errors.append(
                    f"Subtask '{st['id']}' variant_of '{variant_of}' is not an existing subtask"
                )
                continue
            if source.get("agent_type", "cua") != "cua":
                errors.append(
                    f"Subtask '{st['id']}' variant_of '{variant_of}' is not a CUA subtask"
                )
        if "init_from" in st:
            init_from = st.get("init_from")
            if not isinstance(init_from, str) or not init_from:
                errors.append(
                    f"Subtask '{st['id']}' init_from must be a non-empty string"
                )
                continue
            if st.get("agent_type", "cua") != "cua":
                errors.append(
                    f"Subtask '{st['id']}' init_from requires agent_type 'cua'"
                )
                continue
            source = st_by_id.get(init_from)
            if source is None:
                errors.append(
                    f"Subtask '{st['id']}' init_from '{init_from}' is not an existing subtask"
                )
                continue
            if source.get("agent_type", "cua") != "cua":
                errors.append(
                    f"Subtask '{st['id']}' init_from '{init_from}' is not a CUA subtask"
                )
                continue
            if init_from not in (st.get("dependencies") or []):
                errors.append(
                    f"Subtask '{st['id']}' init_from '{init_from}' must also appear in dependencies"
                )

    cua_count = sum(1 for st in subtasks if st.get("agent_type") == "cua")
    if cua_count == 0:
        errors.append("Graph must contain at least one CUA subtask (agent_type='cua')")

    return errors


def validate_subtask_ids(graph: dict) -> None:
    """Raise ValueError if subtask IDs are not unique or collide with aggregation."""
    subtasks = graph["dependency_graph"].get("subtasks", [])
    agg = graph["dependency_graph"].get("aggregation")
    seen: dict[str, str] = {}
    for st in subtasks:
        sid = st["id"]
        if sid in seen:
            raise ValueError(
                f"Duplicate subtask id '{sid}' (first in {seen[sid]}, again in subtasks)"
            )
        seen[sid] = "subtasks"
    if agg:
        aid = agg["id"]
        if aid in seen:
            raise ValueError(
                f"Aggregation id '{aid}' duplicates subtask id. "
                f"The aggregation must use a unique id (e.g. 'final_aggregation')."
            )


def get_all_subtasks(graph: dict, no_manager: bool = False) -> list[dict]:
    """Return all subtasks including the aggregation step."""
    subtasks = list(graph["dependency_graph"]["subtasks"])
    if not no_manager:
        agg = graph["dependency_graph"].get("aggregation")
        if agg:
            subtasks.append(agg)
    return subtasks


def topological_sort_levels(all_subtasks: list[dict]) -> list[list[dict]]:
    """Group subtasks into topological levels via Kahn's algorithm.

    These levels are a static view of the dependency graph. The live
    orchestrator schedules ready subtasks individually rather than executing
    one full level at a time.
    """
    by_id = {s["id"]: s for s in all_subtasks}
    completed: set[str] = set()
    remaining = set(by_id.keys())
    levels: list[list[dict]] = []

    while remaining:
        level = []
        for sid in list(remaining):
            deps = set(by_id[sid].get("dependencies", []))
            if deps.issubset(completed):
                level.append(by_id[sid])

        if not level:
            unsatisfied = {sid: by_id[sid].get("dependencies", []) for sid in remaining}
            raise ValueError(f"Dependency cycle or missing dependency: {unsatisfied}")

        levels.append(level)
        for st in level:
            completed.add(st["id"])
            remaining.discard(st["id"])

    return levels

def compute_downstream_count(all_subtasks: list[dict]) -> dict[str, int]:
    """Count transitive downstream dependents for each subtask."""
    counts = {}
    for st in all_subtasks:
        sid = st["id"]
        visited: set[str] = set()
        queue = [sid]
        while queue:
            current = queue.pop(0)
            for other in all_subtasks:
                if current in other.get("dependencies", []) and other["id"] not in visited:
                    visited.add(other["id"])
                    queue.append(other["id"])
        counts[sid] = len(visited)
    return counts


def compute_aggregation_ancestors(
    all_subtasks: list[dict],
    aggregation_id: Optional[str],
) -> set[str]:
    """Return the set of subtask IDs the aggregation step transitively depends on."""
    by_id = {s["id"]: s for s in all_subtasks}
    if not aggregation_id or aggregation_id not in by_id:
        return set(by_id.keys())

    reachable: set[str] = set()
    queue = [aggregation_id]
    while queue:
        sid = queue.pop()
        if sid in reachable:
            continue
        reachable.add(sid)
        node = by_id.get(sid)
        if node is None:
            continue
        for dep in node.get("dependencies", []) or []:
            if dep not in reachable:
                queue.append(dep)
    return reachable


def build_merged_final_traj(
    run_dir: Path,
    task_id: str,
    original_task_text: str,
    all_subtasks: list[dict],
    graph: dict,
    results: dict,
    completion_order: list,
    final_response: str,
    no_manager: bool = False,
) -> Optional[Path]:
    """Write the merged ``final_traj/`` directory for a completed run.

    Only subtasks in the transitive ancestor closure of the aggregation step
    are included — irrelevant work (e.g. replans whose output was never
    consumed) is filtered so the judged trajectory reflects exactly the path
    that produced the final answer.

    ``completion_order`` entries are ``(sid, timestamp, winner_dir_or_None)``.
    When ``winner_dir`` is ``None`` the subtask directory under ``run_dir/sid``
    is searched via ``rglob("traj.jsonl")``. Manager subtasks produce no CUA
    trajectory and are injected as ``[Manager: ...]`` markers in
    ``traj_raw.jsonl`` only.

    Returns the written ``final_traj`` path, or ``None`` if there were no
    trajectory rows to emit.
    """
    st_by_id = {s["id"]: s for s in all_subtasks}
    if no_manager:
        cua_only = [s for s in all_subtasks if s.get("agent_type") != "manager"]
        relevant_ids: set[str] = {cua_only[0]["id"]} if cua_only else set()
    else:
        agg_id = (graph.get("dependency_graph", {}).get("aggregation") or {}).get("id")
        relevant_ids = compute_aggregation_ancestors(all_subtasks, agg_id)

    raw_steps: list[dict] = []
    agent_steps: list[dict] = []
    raw_counter = 0
    agent_counter = 0
    screenshot_copies: list[tuple[Path, str]] = []

    for sid, _ts, winner_dir in completion_order:
        if sid not in relevant_ids:
            continue
        st_info = st_by_id.get(sid)
        if st_info is None:
            continue
        agent_type = st_info.get("agent_type", "cua")

        if agent_type == "manager":
            response_text = results.get(sid, "")
            if response_text:
                raw_counter += 1
                raw_steps.append({
                    "step_num": raw_counter,
                    "action_timestamp": "",
                    "response": f"[Manager: {sid}]\n\n{response_text}",
                    "action": {"command": "manager_response", "input": {}},
                    "reward": 0,
                    "done": False,
                    "info": {},
                    "screenshot_file": "",
                })
            continue

        traj_dir = winner_dir
        if traj_dir is None:
            traj_dir = run_dir / sid
        traj_files = list(traj_dir.rglob("traj.jsonl"))
        if not traj_files:
            continue

        rows: list[dict] = []
        for traj_path in traj_files:
            with open(traj_path) as tf:
                for raw_line in tf:
                    raw_line = raw_line.strip()
                    if raw_line:
                        try:
                            rows.append(json.loads(raw_line))
                        except json.JSONDecodeError:
                            continue

        if not rows:
            continue

        traj_parent = traj_files[0].parent

        first_ts = ""
        for row in rows:
            first_ts = row.get("action_timestamp", "")
            if first_ts:
                break
        raw_counter += 1
        raw_steps.append({
            "step_num": raw_counter,
            "action_timestamp": first_ts,
            "response": f"[Agent: {sid} starting]",
            "action": {"command": "", "input": {}},
            "reward": 0,
            "done": False,
            "info": {},
            "screenshot_file": "",
        })

        for row in rows:
            raw_counter += 1
            agent_counter += 1

            new_row = dict(row)
            screenshot_file = row.get("screenshot_file", "")
            if screenshot_file:
                src_screenshot = traj_parent / screenshot_file
                if src_screenshot.exists():
                    dest_name = f"{sid}__{screenshot_file}"
                    screenshot_copies.append((src_screenshot, dest_name))
                    new_row["screenshot_file"] = dest_name
                else:
                    new_row["screenshot_file"] = ""

            raw_row = dict(new_row)
            raw_row["step_num"] = raw_counter
            raw_steps.append(raw_row)

            agent_row = dict(new_row)
            agent_row["step_num"] = agent_counter
            agent_steps.append(agent_row)

    if final_response:
        already_present = (
            raw_steps
            and final_response.strip() in (raw_steps[-1].get("response", "") or "").strip()
        )
        if not already_present:
            raw_counter += 1
            raw_steps.append({
                "step_num": raw_counter,
                "action_timestamp": "",
                "response": f"[Final aggregated response]\n\n{final_response}",
                "action": {"command": "final_response", "input": {}},
                "reward": 0,
                "done": True,
                "info": {},
                "screenshot_file": "",
            })

    if not (agent_steps or raw_steps):
        return None

    traj_dir_out = run_dir / "final_traj"
    traj_dir_out.mkdir(parents=True, exist_ok=True)

    if final_response:
        agent_counter += 1
        agent_steps.append({
            "step_num": agent_counter,
            "action_timestamp": "",
            "response": final_response,
            "action": {"command": "manager_response", "input": {}},
            "reward": 0,
            "done": False,
            "info": {},
            "screenshot_file": "",
        })
    with open(traj_dir_out / "traj.jsonl", "w") as mf:
        for step in agent_steps:
            mf.write(json.dumps(step, ensure_ascii=False) + "\n")

    with open(traj_dir_out / "traj_raw.jsonl", "w") as mf:
        for step in raw_steps:
            mf.write(json.dumps(step, ensure_ascii=False) + "\n")

    for src, dest_name in screenshot_copies:
        dest = traj_dir_out / dest_name
        try:
            shutil.copy2(str(src), str(dest))
        except OSError as exc:
            logger.warning("Failed to copy screenshot %s -> %s: %s", src, dest, exc)

    with open(traj_dir_out / "config.json", "w") as cf:
        json.dump({
            "task_id": task_id,
            "confirmed_task": original_task_text,
            "task": original_task_text,
        }, cf, indent=2, ensure_ascii=False)

    with open(traj_dir_out / "completed.txt", "w") as cf:
        cf.write("1.0\n")

    logger.info(
        "Merged trajectories in %s: traj.jsonl (%d agent steps), "
        "traj_raw.jsonl (%d annotated steps), %d screenshots",
        traj_dir_out, len(agent_steps), len(raw_steps), len(screenshot_copies),
    )
    return traj_dir_out


def _annotate_graph_for_replan(
    all_subtasks: list[dict],
    completed: set[str],
    running_logicals: set[str],
    cancelled: Optional[set[str]] = None,
) -> list[dict]:
    """Return a JSON-safe copy of the current graph with per-node status."""
    out = []
    cancelled = cancelled or set()
    for st in all_subtasks:
        sid = st["id"]
        status = (
            "completed"
            if sid in completed
            else "running"
            if sid in running_logicals
            else "cancelled"
            if sid in cancelled
            else "pending"
        )
        node = {
            "id": sid,
            "agent_type": st.get("agent_type", "cua"),
            "description": st.get("description", ""),
            "dependencies": list(st.get("dependencies", [])),
            "instruction": st.get("instruction", ""),
            "_status": status,
        }
        if "outputs" in st:
            node["outputs"] = list(st.get("outputs") or [])
        if st.get("variant_of"):
            node["variant_of"] = st["variant_of"]
        if st.get("init_from"):
            node["init_from"] = st["init_from"]
        if st.get("input_files"):
            node["input_files"] = list(st.get("input_files") or [])
        out.append(node)
    return out


def persist_graph_snapshot(
    run_dir: Path,
    all_subtasks: list[dict],
    task_id: str,
    task_text: str,
    iteration: int,
    label: str,
    decision: Optional[dict] = None,
) -> Path:
    """Persist a self-contained graph snapshot for debugging."""
    snap_dir = run_dir / "graph_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    filename = f"dependency_graph_{iteration:03d}_{safe_label}.json"
    path = snap_dir / filename

    subtasks_clean: list[dict] = []
    aggregation_clean: Optional[dict] = None
    for st in all_subtasks:
        node = {k: v for k, v in st.items() if not k.startswith("_")}
        if node.get("id") == "final_aggregation":
            aggregation_clean = node
        else:
            subtasks_clean.append(node)

    envelope = {
        "task_id": task_id,
        "task_text": task_text,
        "dependency_graph": {
            "subtasks": subtasks_clean,
            "aggregation": aggregation_clean,
        },
        "replan_iteration": iteration,
        "replan_label": label,
    }
    if decision is not None:
        envelope["replan_decision"] = decision
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)
    return path


def _append_replan_log(
    run_dir: Path,
    iteration: int,
    trigger_sid: Optional[str],
    decision: Optional[dict],
    errors: list[str],
    applied: bool,
    reason: str = "completed",
) -> None:
    """Append one JSON line to ``run_dir/replan_log.jsonl`` for audit."""
    entry = {
        "iteration": iteration,
        "trigger_sid": trigger_sid,
        "reason": reason,
        "timestamp": time.time(),
        "applied": applied,
        "errors": errors,
        "decision": decision,
    }
    log_path = run_dir / "replan_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def apply_replan_decision(
    decision: dict,
    all_subtasks: list[dict],
    st_by_id: dict[str, dict],
    all_subtask_ids: set[str],
    downstream_counts: dict[str, int],
    completed: set[str],
    running_logicals: set[str],
    task_id: str,
) -> tuple[bool, list[str], set[str]]:
    """Validate and apply a replan decision to the in-memory graph in place."""
    errors: list[str] = []
    frozen = completed | running_logicals
    remove_ids: set[str] = set(decision.get("remove") or [])
    add_list: list[dict] = list(decision.get("add") or [])
    modify: dict[str, dict] = dict(decision.get("modify") or {})
    cancel_ids: set[str] = set(decision.get("cancel") or [])

    for sid in cancel_ids:
        if sid not in running_logicals:
            errors.append(
                f"cancel: {sid!r} is not currently running (only running CUA "
                "subtasks may be cancelled)"
            )
            continue
        node = st_by_id.get(sid)
        if node is None:
            errors.append(f"cancel: unknown subtask id {sid!r}")
            continue
        if node.get("agent_type", "cua") != "cua":
            errors.append(f"cancel: {sid!r} is not a CUA subtask")

    for sid in remove_ids:
        if sid in frozen and sid not in cancel_ids:
            errors.append(f"remove: refusing to remove frozen subtask {sid!r}")
    for sid in modify:
        if sid in frozen:
            errors.append(f"modify: refusing to modify frozen subtask {sid!r}")

    for sid in remove_ids:
        if sid not in st_by_id:
            errors.append(f"remove: unknown subtask id {sid!r}")
        if sid == "final_aggregation":
            errors.append("remove: final_aggregation cannot be removed")

    for sid, patch in modify.items():
        if sid not in st_by_id:
            errors.append(f"modify: unknown subtask id {sid!r}")
            continue
        if not isinstance(patch, dict):
            errors.append(f"modify: value for {sid!r} must be an object")
            continue
        bad_keys = set(patch.keys()) - _REPLAN_ALLOWED_MODIFY_FIELDS
        if bad_keys:
            errors.append(f"modify: {sid!r} has disallowed fields {sorted(bad_keys)}")

    surviving_ids = set(all_subtask_ids) - remove_ids
    added_ids: list[str] = []
    cleaned_add: list[dict] = []
    for entry in add_list:
        if not isinstance(entry, dict):
            errors.append("add: entry is not an object")
            continue
        new_sid = entry.get("id")
        if not isinstance(new_sid, str) or not new_sid:
            errors.append("add: each entry must have a non-empty string id")
            continue
        if new_sid in surviving_ids:
            errors.append(f"add: id {new_sid!r} collides with an existing subtask")
            continue
        if new_sid in added_ids:
            errors.append(f"add: duplicate id {new_sid!r} in add list")
            continue
        agent_type = entry.get("agent_type")
        if agent_type not in ("cua", "manager"):
            errors.append(f"add: {new_sid!r} has invalid agent_type {agent_type!r}")
            continue
        instruction = entry.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"add: {new_sid!r} missing instruction")
            continue
        deps = entry.get("dependencies", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            errors.append(f"add: {new_sid!r} dependencies must be a list of strings")
            continue
        if new_sid in deps:
            errors.append(f"add: {new_sid!r} self-depends")
            continue
        if agent_type == "cua":
            outputs = entry.get("outputs", [])
            if not isinstance(outputs, list):
                errors.append(f"add: {new_sid!r} outputs must be a list")
                continue
        variant_of = entry.get("variant_of")
        if variant_of is not None:
            if not isinstance(variant_of, str) or not variant_of:
                errors.append(f"add: {new_sid!r} variant_of must be a non-empty string")
                continue
            if agent_type != "cua":
                errors.append(
                    f"add: {new_sid!r} variant_of requires agent_type 'cua', got {agent_type!r}"
                )
                continue
            source = st_by_id.get(variant_of)
            if source is None and variant_of in added_ids:
                source = next((e for e in cleaned_add if e["id"] == variant_of), None)
            if source is None:
                errors.append(
                    f"add: {new_sid!r} variant_of {variant_of!r} is not an existing subtask"
                )
                continue
            if source.get("agent_type", "cua") != "cua":
                errors.append(
                    f"add: {new_sid!r} variant_of {variant_of!r} is not a CUA subtask"
                )
                continue
        init_from = entry.get("init_from")
        if init_from is not None:
            if not isinstance(init_from, str) or not init_from:
                errors.append(f"add: {new_sid!r} init_from must be a non-empty string")
                continue
            if agent_type != "cua":
                errors.append(
                    f"add: {new_sid!r} init_from requires agent_type 'cua', got {agent_type!r}"
                )
                continue
            source = st_by_id.get(init_from)
            if source is None and init_from in added_ids:
                source = next((e for e in cleaned_add if e["id"] == init_from), None)
            if source is None:
                errors.append(
                    f"add: {new_sid!r} init_from {init_from!r} is not an existing subtask"
                )
                continue
            if source.get("agent_type", "cua") != "cua":
                errors.append(
                    f"add: {new_sid!r} init_from {init_from!r} is not a CUA subtask"
                )
                continue
            if init_from not in deps:
                errors.append(
                    f"add: {new_sid!r} init_from {init_from!r} must also appear in dependencies"
                )
                continue
        input_files = entry.get("input_files", [])
        if not isinstance(input_files, list) or not all(isinstance(x, str) for x in input_files):
            errors.append(f"add: {new_sid!r} input_files must be a list of strings")
            continue
        added_ids.append(new_sid)
        cleaned_entry = {
            "id": new_sid,
            "agent_type": agent_type,
            "description": entry.get("description", ""),
            "dependencies": list(deps),
            "instruction": instruction,
            "outputs": list(entry.get("outputs", [])) if agent_type == "cua" else [],
            "input_files": list(input_files),
            "_task_id": task_id,
        }
        if variant_of is not None:
            cleaned_entry["variant_of"] = variant_of
        if init_from is not None:
            cleaned_entry["init_from"] = init_from
        cleaned_add.append(cleaned_entry)

    for sid, patch in modify.items():
        if not isinstance(patch, dict):
            continue
        if "input_files" in patch:
            ifiles = patch.get("input_files") or []
            if not isinstance(ifiles, list) or not all(isinstance(x, str) for x in ifiles):
                errors.append(f"modify: {sid!r} input_files must be a list of strings")
        if "init_from" in patch:
            ifr_val = patch.get("init_from")
            if ifr_val is not None:
                if not isinstance(ifr_val, str) or not ifr_val:
                    errors.append(f"modify: {sid!r} init_from must be a non-empty string or null")
                else:
                    target = st_by_id.get(sid)
                    if target and target.get("agent_type", "cua") != "cua":
                        errors.append(f"modify: {sid!r} init_from requires agent_type 'cua'")
                    if ifr_val not in st_by_id and ifr_val not in set(added_ids):
                        errors.append(
                            f"modify: {sid!r} init_from {ifr_val!r} is not an existing subtask"
                        )
                    else:
                        ref = st_by_id.get(ifr_val)
                        if ref and ref.get("agent_type", "cua") != "cua":
                            errors.append(
                                f"modify: {sid!r} init_from {ifr_val!r} is not a CUA subtask"
                            )
                    if target:
                        final_deps = list(patch.get("dependencies") or target.get("dependencies", []))
                        if ifr_val not in final_deps:
                            errors.append(
                                f"modify: {sid!r} init_from {ifr_val!r} must also appear "
                                f"in dependencies"
                            )

    post_ids = surviving_ids | set(added_ids)

    for entry in cleaned_add:
        for dep in entry["dependencies"]:
            if dep not in post_ids:
                errors.append(f"add: {entry['id']!r} depends on unknown/removed id {dep!r}")
    for sid, patch in modify.items():
        if "dependencies" in patch:
            new_deps = patch.get("dependencies") or []
            if not isinstance(new_deps, list) or not all(isinstance(d, str) for d in new_deps):
                errors.append(f"modify: {sid!r} dependencies must be a list of strings")
                continue
            if sid in new_deps:
                errors.append(f"modify: {sid!r} self-depends")
                continue
            for dep in new_deps:
                if dep not in post_ids:
                    errors.append(f"modify: {sid!r} depends on unknown/removed id {dep!r}")

    for st in all_subtasks:
        sid = st["id"]
        if sid in remove_ids:
            continue
        deps_after = list((modify.get(sid) or {}).get("dependencies", st.get("dependencies", [])))
        for dep in deps_after:
            if dep in remove_ids:
                errors.append(
                    f"remove: subtask {sid!r} still depends on removed id {dep!r}; "
                    f"modify its dependencies in the same decision or keep {dep!r}"
                )

    if errors:
        return False, errors, set()

    tentative: list[dict] = []
    for st in all_subtasks:
        sid = st["id"]
        if sid in remove_ids:
            continue
        if sid in modify:
            merged = dict(st)
            for key, value in modify[sid].items():
                if key in _REPLAN_ALLOWED_MODIFY_FIELDS:
                    merged[key] = value
            tentative.append(merged)
        else:
            tentative.append(st)
    tentative.extend(cleaned_add)

    try:
        topological_sort_levels(tentative)
    except ValueError as exc:
        return False, [f"topological sort rejected decision: {exc}"], set()

    agg_nodes = [node for node in tentative if node.get("id") == "final_aggregation"]
    if len(agg_nodes) != 1:
        return False, ["post-apply graph is missing final_aggregation"], set()
    if agg_nodes[0].get("agent_type") != "manager":
        return False, ["final_aggregation agent_type must stay 'manager'"], set()

    if remove_ids:
        all_subtasks[:] = [s for s in all_subtasks if s["id"] not in remove_ids]
        for sid in remove_ids:
            st_by_id.pop(sid, None)
            all_subtask_ids.discard(sid)

    for sid, patch in modify.items():
        node = st_by_id.get(sid)
        if node is None:
            continue
        for key, value in patch.items():
            if key in _REPLAN_ALLOWED_MODIFY_FIELDS:
                node[key] = list(value) if key == "dependencies" else value

    for entry in cleaned_add:
        all_subtasks.append(entry)
        st_by_id[entry["id"]] = entry
        all_subtask_ids.add(entry["id"])

    downstream_counts.clear()
    downstream_counts.update(compute_downstream_count(all_subtasks))

    return True, [], cancel_ids


__all__ = [
    "_REPLAN_ALLOWED_MODIFY_FIELDS",
    "_annotate_graph_for_replan",
    "_append_replan_log",
    "apply_replan_decision",
    "build_merged_final_traj",
    "compute_aggregation_ancestors",
    "compute_downstream_count",
    "get_all_subtasks",
    "persist_graph_snapshot",
    "topological_sort_levels",
    "validate_subtask_ids",
]
