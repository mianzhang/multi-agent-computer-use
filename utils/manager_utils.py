from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union

from utils.graph_utils import _annotate_graph_for_replan
from utils.llm import UsageInfo, call_llm, load_prompt, parse_json_response, render_prompt
from utils.utils import extract_last_k_screenshots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP_PROMPT_PATH = PROJECT_ROOT / "prompts" / "followup.yaml"
REPLAN_PROMPT_PATH = PROJECT_ROOT / "prompts" / "replan.yaml"
AGGREGATION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "aggregation.yaml"

logger = logging.getLogger("macu")

_manager_call_counter = 0


def format_completed_results(results: dict[str, str]) -> str:
    """Format completed subtask results for the manager prompt."""
    parts = []
    for subtask_id, response in results.items():
        parts.append(f"[{subtask_id}]:\n{response}")
    return "\n\n".join(parts)


def save_manager_prompt(
    system_prompt: str,
    user_prompt: Optional[str] = None,
    run_dir: Optional[Path] = None,
    label: str = "manager",
    image_path: Optional[Union[Path, list[Path]]] = None,
    user_segments: Optional[list[dict]] = None,
) -> Path:
    """Save the full manager prompt to a YAML file in the result dir.

    Two ways to provide the user message body:

    * Legacy: ``user_prompt`` (str) + optional ``image_path`` (Path or list).
      Images are appended at the end of the file.
    * Interleaved: ``user_segments`` -- a list of ``{"type": "text", "text": ...}``
      and ``{"type": "image", "path": ...}`` dicts. Saved in order so the
      audit file shows exactly the message structure that the LLM saw.
    """
    global _manager_call_counter
    assert run_dir is not None, "run_dir is required"
    _manager_call_counter += 1
    filename = f"manager_prompt_{_manager_call_counter:03d}_{label}.yaml"
    prompt_path = run_dir / filename
    # Write as plain text sections for readability (avoid YAML quoting artifacts)
    with open(prompt_path, "w") as f:
        f.write("--- system_prompt ---\n")
        f.write(system_prompt)
        f.write("\n")
        if user_segments is not None:
            f.write("--- user_content (interleaved) ---\n")
            for seg in user_segments:
                if seg.get("type") == "text":
                    f.write(seg.get("text", ""))
                    f.write("\n")
                elif seg.get("type") == "image":
                    f.write(f"[image: {seg.get('path', '?')}]\n")
        else:
            if user_prompt:
                f.write("--- user_prompt ---\n")
                f.write(user_prompt)
                f.write("\n")
            if image_path is not None:
                paths = image_path if isinstance(image_path, list) else [image_path]
                for p in paths:
                    f.write(f"--- image_path ---\n{p}\n")
    return prompt_path


def save_manager_response(
    prompt_path: Path,
    raw_response: str,
    usage: Optional[UsageInfo] = None,
) -> Path:
    """Save the manager LLM response next to its corresponding prompt file.

    Filename mirrors :func:`save_manager_prompt` -- ``manager_response_NNN_<label>.yaml``
    -- derived from ``prompt_path`` so the counter and label match the prompt.
    """
    response_name = prompt_path.name.replace("manager_prompt_", "manager_response_", 1)
    response_path = prompt_path.with_name(response_name)
    with open(response_path, "w") as f:
        f.write("--- raw_response ---\n")
        f.write(raw_response)
        f.write("\n")
        if usage is not None:
            f.write("--- usage ---\n")
            f.write(f"model: {usage.model}\n")
            f.write(f"input_tokens: {usage.input_tokens}\n")
            f.write(f"output_tokens: {usage.output_tokens}\n")
            f.write(f"cost_usd: {usage.cost_usd:.6f}\n")
            f.write(f"inference_seconds: {usage.inference_seconds:.3f}\n")
    return response_path


def call_manager_for_subtasks(
    ready_manager_subtasks: list[dict],
    all_results: dict[str, str],
    original_task_text: str,
    args: argparse.Namespace,
    run_dir: Path,
    vm_info_by_sid: Optional[dict[str, dict]] = None,
    final_vm_choice: Optional[dict] = None,
    archive_pool: Optional[dict[str, dict]] = None,
    meta_by_sid: Optional[dict[str, dict]] = None,
) -> tuple[dict[str, str], list[UsageInfo]]:
    """Call the manager LLM to produce a final aggregation response for manager subtasks.

    CUA subtask instruction enrichment is handled by the replan ``modify``
    action (fired after each CUA completion) rather than a separate refine
    call at a coarse graph-level boundary.

    When *meta_by_sid* is provided, screenshots from completed dependency
    subtasks are included in aggregation calls (controlled by
    ``args.manager_screenshots``).

    Returns: (dict mapping subtask_id -> result/instruction, list of UsageInfo)

    When ``vm_info_by_sid`` is provided, the aggregation prompt is augmented to
    ask the manager to also pick a ``final_vm_subtask_id``. The orchestrator
    passes ``final_vm_choice`` (a mutable dict) to receive the chosen
    ``{"subtask_id": ..., "vm_path": ...}`` back.
    """
    prompt_config = load_prompt(str(AGGREGATION_PROMPT_PATH))
    usages: list[UsageInfo] = []

    manager_subtasks = [s for s in ready_manager_subtasks if s.get("agent_type") == "manager"]

    refined: dict[str, str] = {}

    # Handle manager subtasks: ask for final aggregation
    for st in manager_subtasks:
        # For aggregation, use ALL results (not just deps) if deps cover everything
        agg_dep_ids = set(st.get("dependencies", []))
        agg_results = {k: v for k, v in all_results.items() if k in agg_dep_ids}
        agg_text = format_completed_results(agg_results)

        # Build the candidate VM list. Only include subtasks that actually wrote
        # a vm_info.json (i.e., the env successfully started).
        vm_candidates: list[tuple[str, str]] = []  # (subtask_id, vm_path)
        if vm_info_by_sid:
            # Prefer dependencies of this aggregation step (most likely to hold the
            # final state); fall back to any tracked subtask if deps are empty.
            ordered_ids = [sid for sid in agg_dep_ids if sid in vm_info_by_sid]
            if not ordered_ids:
                ordered_ids = list(vm_info_by_sid.keys())
            for sid in ordered_ids:
                path = vm_info_by_sid[sid].get("vm_path")
                if path:
                    vm_candidates.append((sid, path))

        if vm_candidates:
            vm_choices_text = "\n".join(
                f"  - {sid}  (vm: {path})" for sid, path in vm_candidates
            )
            valid_ids_text = ", ".join(f'"{sid}"' for sid, _ in vm_candidates)
            action_prompt = (
                f"All subtasks are complete. Produce the final deliverable.\n\n"
                f"Aggregation instruction: {st['instruction']}\n\n"
                f"This task additionally requires designating one subtask's VM as the "
                f"FINAL AGGREGATED VM. Downstream evaluation (e.g. OSWorld reward "
                f"computation) will run against the VM you pick, so it should be the VM "
                f"whose end state (open browser tabs, files written, applications used) "
                f"most fully represents the final deliverable for the original task.\n\n"
                f"Available subtask VMs (pick exactly ONE subtask ID from this list):\n"
                f"{vm_choices_text}\n\n"
                f"If multiple VMs hold partial state (e.g. one tab each from different "
                f"subtasks), pick the subtask whose VM is closest to the final intended "
                f"state -- typically the last aggregator subtask, or the one that "
                f"explicitly opens / writes the user-visible deliverable.\n\n"
                f"Respond with JSON: {{\"final_response\": \"your synthesized response...\", "
                f"\"final_vm_subtask_id\": <one of {valid_ids_text}>}}"
            )
        else:
            action_prompt = (
                f"All subtasks are complete. Produce the final deliverable.\n\n"
                f"Aggregation instruction: {st['instruction']}\n\n"
                f"Respond with JSON: {{\"final_response\": \"your synthesized response...\"}}"
            )

        system_prompt, user_prompt = render_prompt(
            prompt_config,
            original_task=original_task_text,
            completed_results=agg_text,
            action_prompt=action_prompt,
        )

        # Collect screenshots from dependency subtasks for visual context.
        k = getattr(args, "manager_screenshots", 1)
        agg_screenshots: list[Path] = []
        if meta_by_sid and k > 0:
            for dep_sid in agg_dep_ids:
                dep_meta = meta_by_sid.get(dep_sid)
                if dep_meta is None:
                    continue
                dep_screenshots = extract_last_k_screenshots(dep_sid, run_dir, k=k)
                agg_screenshots.extend(dep_screenshots)
        agg_image_path: Optional[Union[Path, list[Path]]] = (
            agg_screenshots if len(agg_screenshots) > 1
            else agg_screenshots[0] if agg_screenshots
            else None
        )
        if agg_screenshots:
            logger.info(
                "Aggregation for [%s] including %d screenshot(s) from dependencies",
                st["id"], len(agg_screenshots),
            )

        prompt_path = save_manager_prompt(
            system_prompt, user_prompt, run_dir, f"aggregate_{st['id']}",
            image_path=agg_image_path,
        )
        logger.info("Calling manager for final aggregation: [%s] (prompt: %s)", st["id"], prompt_path)
        raw_response, usage = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=args.manager_provider,
            model=args.manager_model,
            max_tokens=args.manager_max_tokens,
            image_path=agg_image_path,
            reasoning_effort=getattr(args, "manager_reasoning_effort", None),
        )
        usages.append(usage)
        save_manager_response(prompt_path, raw_response, usage)
        logger.info(
            "Manager aggregate cost: $%.4f (%d in / %d out tokens, %s)",
            usage.cost_usd, usage.input_tokens, usage.output_tokens, usage.model,
        )

        parsed: Optional[dict] = None
        try:
            parsed = parse_json_response(raw_response)
            if isinstance(parsed, dict):
                refined[st["id"]] = parsed.get("final_response", raw_response)
            else:
                refined[st["id"]] = raw_response
        except (json.JSONDecodeError, AttributeError):
            # Use raw response if JSON parsing fails
            refined[st["id"]] = raw_response

        # Parse the manager's chosen VM (or fall back).
        if vm_candidates and final_vm_choice is not None:
            valid_ids = {sid for sid, _ in vm_candidates}
            chosen_sid: Optional[str] = None
            if isinstance(parsed, dict):
                candidate = parsed.get("final_vm_subtask_id")
                if isinstance(candidate, str) and candidate in valid_ids:
                    chosen_sid = candidate
                elif candidate is not None:
                    logger.warning(
                        "Manager returned final_vm_subtask_id=%r which is not in the "
                        "candidate list %s; falling back.", candidate, sorted(valid_ids),
                    )
            if chosen_sid is None:
                # Fallback: last candidate (deps were ordered by aggregation step's deps)
                chosen_sid = vm_candidates[-1][0]
                logger.info("Falling back to final VM subtask [%s]", chosen_sid)
            chosen_path = dict(vm_candidates)[chosen_sid]
            final_vm_choice["subtask_id"] = chosen_sid
            final_vm_choice["vm_path"] = chosen_path
            logger.info("Manager designated final VM: [%s] -> %s", chosen_sid, chosen_path)

        # Save manager response
        st_dir = Path(args.result_dir) / st.get("_task_id", "unknown") / st["id"]
        st_dir.mkdir(parents=True, exist_ok=True)
        with open(st_dir / "manager_response.txt", "w") as f:
            f.write(refined[st["id"]])

    return refined, usages

def ask_manager_for_followup(
    subtask: dict,
    cua_result: str,
    original_task_text: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[bool, str | None, str | None, UsageInfo]:
    """Ask the manager whether a completed CUA subtask needs more info."""
    prompt_config = load_prompt(str(FOLLOWUP_PROMPT_PATH))

    system_prompt, user_prompt = render_prompt(
        prompt_config,
        original_task=original_task_text,
        subtask_id=subtask["id"],
        subtask_instruction=subtask.get("instruction", ""),
        cua_result=cua_result,
    )

    k = getattr(args, "manager_screenshots", 1)
    screenshot_paths = extract_last_k_screenshots(subtask["id"], run_dir, k=k)
    if screenshot_paths:
        logger.info(
            "Manager followup for [%s] including %d screenshot(s): %s",
            subtask["id"], len(screenshot_paths), [p.name for p in screenshot_paths],
        )
    else:
        logger.info("Manager followup for [%s]: no screenshot found in trajectory", subtask["id"])
    image_path: Optional[Union[Path, list[Path]]] = (
        screenshot_paths if len(screenshot_paths) > 1
        else screenshot_paths[0] if screenshot_paths
        else None
    )

    prompt_path = save_manager_prompt(
        system_prompt, user_prompt, run_dir, f"followup_{subtask['id']}",
        image_path=image_path,
    )
    logger.info("Calling manager for followup review of [%s] (prompt: %s)", subtask["id"], prompt_path)

    try:
        raw_response, usage = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=args.manager_provider,
            model=args.manager_model,
            max_tokens=args.manager_max_tokens,
            image_path=image_path,
            reasoning_effort=getattr(args, "manager_reasoning_effort", None),
        )
    except Exception as exc:
        logger.warning("Manager followup call failed for [%s]: %s", subtask["id"], exc)
        return False, None, None, UsageInfo(model=args.manager_model)

    save_manager_response(prompt_path, raw_response, usage)
    logger.info(
        "Manager followup cost: $%.4f (%d in / %d out tokens, %s)",
        usage.cost_usd, usage.input_tokens, usage.output_tokens, usage.model,
    )
    logger.debug("Manager followup raw response for [%s]: %s", subtask["id"], raw_response)

    try:
        parsed = parse_json_response(raw_response)
        needs = parsed.get("needs_followup", False)
        if needs:
            fu_type = parsed.get("followup_type", "summary")
            fu_msg = parsed.get("followup_message", "")
            if fu_type not in ("continue", "summary"):
                logger.warning("Unknown followup_type %r, defaulting to 'summary'", fu_type)
                fu_type = "summary"
            return True, fu_type, fu_msg, usage
        return False, None, None, usage
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Failed to parse manager followup response for [%s]: %s", subtask["id"], exc)
        return False, None, None, usage


def ask_manager_for_replanning(
    focus_sid: Optional[str],
    all_subtasks: list[dict],
    results: dict[str, str],
    completed: set[str],
    running_logicals: set[str],
    original_task_text: str,
    args: argparse.Namespace,
    run_dir: Path,
    iteration: int,
    num_applied: int,
    spare_slots: int = 0,
    archive_pool: Optional[dict[str, dict]] = None,
    reason: str = "completed",
    cancelled: Optional[set[str]] = None,
    focus_step_count: Optional[int] = None,
    focus_max_steps: Optional[int] = None,
) -> tuple[Optional[dict], UsageInfo]:
    """Call the manager LLM to decide whether the pending graph should change."""
    prompt_config = load_prompt(str(REPLAN_PROMPT_PATH))

    frozen_ids = sorted(completed)
    annotated = _annotate_graph_for_replan(
        all_subtasks,
        completed,
        running_logicals,
        cancelled or set(),
    )
    current_graph_json = json.dumps(annotated, indent=2, ensure_ascii=False)

    if focus_max_steps is not None and focus_step_count is not None:
        focus_step_status = f"step {focus_step_count}/{focus_max_steps}"
    elif focus_step_count is not None:
        focus_step_status = f"step {focus_step_count}/?"
    else:
        focus_step_status = "n/a"

    running_focus_text = (
        "(still running — no final response yet; see the attached screenshot "
        "and the subtask's graph status for current state)"
    )
    if focus_sid and reason == "spare_capacity":
        just_finished_label = f"{focus_sid} (still running — consulted because spare worker slots are available)"
        just_finished_text = running_focus_text
    elif focus_sid:
        just_finished_text = _truncate_for_prompt(results.get(focus_sid, ""), limit=4000)
        just_finished_label = focus_sid
    else:
        just_finished_text = "(no subtask just finished — consulted because spare worker slots are available)"
        just_finished_label = "(none)"
    completed_text = format_completed_results(
        {sid: _truncate_for_prompt(resp, limit=1500) for sid, resp in results.items()}
    )

    running_ids_sorted = sorted(running_logicals)
    running_ids_text = json.dumps(running_ids_sorted)

    pool_entries = archive_pool or {}
    if pool_entries:
        archive_pool_text = "\n".join(
            f"- {name}  (from subtask {entry.get('source_sid', '?')}, guest path: {entry.get('guest_path', '?')})"
            for name, entry in sorted(pool_entries.items())
        )
    else:
        archive_pool_text = "(empty)"

    replan_budget_total = int(getattr(args, "max_replans", 0) or 0)
    replan_budget_remaining = max(0, replan_budget_total - num_applied)

    trigger_kind = {
        "spare_capacity": "spare-capacity consult",
    }.get(reason, "completion")

    system_prompt, user_prompt = render_prompt(
        prompt_config,
        original_task=original_task_text,
        just_finished_sid=just_finished_label,
        just_finished_response=just_finished_text,
        frozen_ids=json.dumps(frozen_ids),
        completed_results=completed_text,
        current_graph=current_graph_json,
        replan_iteration=iteration,
        replan_budget_total=replan_budget_total,
        replan_budget_remaining=replan_budget_remaining,
        spare_slots=spare_slots,
        running_ids=running_ids_text,
        archive_pool=archive_pool_text,
        focus_step_status=focus_step_status,
        trigger_kind=trigger_kind,
    )

    k = getattr(args, "manager_screenshots", 1)
    max_total_arg = getattr(args, "replan_max_screenshots", None)
    max_total = int(max_total_arg) if max_total_arg is not None else int(k)

    user_segments: list[dict] = [{"type": "text", "text": user_prompt}]
    all_shots: list[Path] = []

    screenshot_sources = []
    seen_sids: set[str] = set()

    def _collect(sid: str) -> None:
        if not sid or sid in seen_sids:
            return
        shots = extract_last_k_screenshots(sid, run_dir, k=k)
        if shots:
            screenshot_sources.append((sid, shots))
        seen_sids.add(sid)

    if focus_sid:
        _collect(focus_sid)
    for st in all_subtasks:
        sid = st.get("id")
        if not sid or sid in completed or sid in running_logicals:
            continue
        for dep_sid in st.get("dependencies", []) or []:
            if dep_sid in completed:
                _collect(dep_sid)

    if max_total > 0:
        capped: list[tuple[str, list[Path]]] = []
        count = 0
        for sid, shots in screenshot_sources:
            if count >= max_total:
                break
            take = min(len(shots), max_total - count)
            capped.append((sid, shots[:take]))
            count += take
        screenshot_sources = capped

    all_shots = [p for _, shots in screenshot_sources for p in shots]

    if screenshot_sources:
        user_segments.append({
            "type": "text",
            "text": (
                "\n\nThe following screenshot blocks are grouped by source "
                "subtask. The first block is the focus subtask (the one "
                "that triggered this replan, or the running subtask the "
                "spare-capacity hook reacted to). Subsequent blocks are "
                "completed parents of pending subtasks — use their "
                "screenshots to cross-check the text results when "
                "enriching pending dependents' instructions."
            ),
        })
        for sid, shots in screenshot_sources:
            role = "focus" if sid == focus_sid else "completed parent"
            user_segments.append({
                "type": "text",
                "text": f"\n--- Screenshots from [{sid}] ({role}, {len(shots)} image(s)) ---",
            })
            for shot in shots:
                user_segments.append({"type": "image", "path": str(shot)})

    label_tag = focus_sid or reason
    if all_shots:
        source_summary = ", ".join(
            f"{sid}×{len(shots)}" for sid, shots in screenshot_sources
        )
        logger.info(
            "Replan prompt for [%s] iter %d (%d/%d remaining) including %d "
            "screenshot(s) from %d source(s): %s",
            label_tag, iteration, replan_budget_remaining, replan_budget_total,
            len(all_shots), len(screenshot_sources), source_summary,
        )
    else:
        logger.info(
            "Replan prompt for [%s] iter %d (%d/%d remaining): no screenshot found",
            label_tag, iteration, replan_budget_remaining, replan_budget_total,
        )

    prompt_audit_path = save_manager_prompt(
        system_prompt, run_dir=run_dir,
        label=f"replan_{iteration:03d}_{label_tag}",
        user_segments=user_segments,
    )
    logger.info(
        "Calling manager for replanning (trigger: %s, iter %d, prompt: %s)",
        label_tag, iteration, prompt_audit_path,
    )

    try:
        raw_response, usage = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=args.manager_provider,
            model=args.manager_model,
            max_tokens=args.manager_max_tokens,
            user_segments=user_segments,
            reasoning_effort=getattr(args, "manager_reasoning_effort", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Replanning call failed (trigger %s, iter %d): %s", label_tag, iteration, exc)
        return None, UsageInfo(model=args.manager_model)

    save_manager_response(prompt_audit_path, raw_response, usage)
    logger.info(
        "Replan manager cost: $%.4f (%d in / %d out tokens, %s)",
        usage.cost_usd, usage.input_tokens, usage.output_tokens, usage.model,
    )
    logger.debug("Replan raw response for [%s] iter %d: %s", label_tag, iteration, raw_response)

    try:
        parsed = parse_json_response(raw_response)
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Failed to parse replan response for [%s] iter %d: %s", label_tag, iteration, exc)
        return None, usage

    if not isinstance(parsed, dict):
        logger.warning(
            "Replan response for [%s] iter %d parsed as %s, not dict; treating as no_change",
            label_tag, iteration, type(parsed).__name__,
        )
        return None, usage

    action = parsed.get("action")
    if action == "no_change":
        logger.info(
            "Replan manager chose no_change (trigger %s, reason: %s)",
            label_tag, parsed.get("reasoning", ""),
        )
        return None, usage
    if action != "update":
        logger.warning(
            "Replan manager returned unknown action %r; treating as no_change", action,
        )
        return None, usage

    remove = parsed.get("remove") or []
    add = parsed.get("add") or []
    modify = parsed.get("modify") or {}
    cancel = parsed.get("cancel") or []
    if not remove and not add and not modify and not cancel:
        logger.info("Replan manager action=update but no changes; treating as no_change")
        return None, usage

    return {
        "reasoning": parsed.get("reasoning", ""),
        "remove": list(remove),
        "add": list(add),
        "modify": dict(modify),
        "cancel": list(cancel),
    }, usage


def _truncate_for_prompt(text: str, limit: int = 1200) -> str:
    """Keep long subtask summaries bounded in manager prompts."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "\n...[truncated]"


__all__ = [
    "ask_manager_for_followup",
    "ask_manager_for_replanning",
    "call_manager_for_subtasks",
    "format_completed_results",
    "save_manager_prompt",
    "save_manager_response",
]
