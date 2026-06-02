from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from utils.llm import UsageInfo, call_llm, load_prompt, parse_json_response, render_prompt


logger = logging.getLogger("macu")


def compute_file_diff(pre_path: Path, post_path: Path) -> dict[str, list[str]]:
    """Diff two file-snapshot JSON files written by run_cua.py."""
    result = {"added": [], "modified": [], "deleted": []}
    if not pre_path.exists() or not post_path.exists():
        return result
    try:
        with open(pre_path) as f:
            pre = json.load(f)
        with open(post_path) as f:
            post = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("compute_file_diff: failed to read snapshots: %s", exc)
        return result
    if not isinstance(pre, dict) or not isinstance(post, dict):
        return result

    skip_basenames = {"files_pre.json", "files_post.json"}
    pre_keys = set(pre.keys())
    post_keys = set(post.keys())

    for path in sorted(post_keys - pre_keys):
        if os.path.basename(path) in skip_basenames:
            continue
        result["added"].append(path)
    for path in sorted(pre_keys - post_keys):
        if os.path.basename(path) in skip_basenames:
            continue
        result["deleted"].append(path)
    for path in sorted(pre_keys & post_keys):
        if os.path.basename(path) in skip_basenames:
            continue
        before = pre.get(path) or {}
        after = post.get(path) or {}
        if before.get("size") != after.get("size") or before.get("mtime") != after.get("mtime"):
            result["modified"].append(path)
    return result


def write_filemgmt_decision(subtask_dir: Path, actions: list[dict]) -> None:
    """Write the orchestrator-side file-management decision manifest."""
    path = subtask_dir / "filemgmt_decision.json"
    with open(path, "w") as f:
        json.dump({"actions": actions}, f)
    logger.info("Wrote filemgmt decision (%d actions) to %s", len(actions), path)


def read_filemgmt_downloads(subtask_dir: Path) -> dict[str, str]:
    """Read the CUA subprocess's file download manifest."""
    path = subtask_dir / "filemgmt_downloads.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _format_file_diff(diff: dict[str, list[str]]) -> str:
    """Render a ``compute_file_diff`` result for prompts/logging."""
    parts: list[str] = []
    for label in ("added", "modified", "deleted"):
        items = diff.get(label) or []
        if not items:
            continue
        parts.append(f"{label} ({len(items)}):")
        for path in items:
            parts.append(f"  - {path}")
    if not parts:
        return "(no changes)"
    return "\n".join(parts)


def _format_archive_pool(archive_pool: dict[str, dict]) -> str:
    """Render the current archive pool for the file-management prompt."""
    if not archive_pool:
        return "(none)"
    return "\n".join(
        f"- {name}  (from subtask {entry.get('source_sid', '?')}, guest path: {entry.get('guest_path', '?')})"
        for name, entry in sorted(archive_pool.items())
    )


def ask_manager_for_file_management(
    subtask: dict,
    file_diff: dict[str, list[str]],
    archive_pool: dict[str, dict],
    original_task_text: str,
    args: argparse.Namespace,
    run_dir: Path,
    *,
    prompt_yaml_path: str | Path,
    save_manager_prompt_fn: Callable[..., Path],
    save_manager_response_fn: Callable[..., Path],
    load_prompt_fn: Callable[[str], dict] = load_prompt,
    render_prompt_fn: Callable[..., tuple[str, str]] = render_prompt,
    call_llm_fn: Callable[..., tuple[str, UsageInfo]] = call_llm,
    parse_json_response_fn: Callable[[str], dict] = parse_json_response,
) -> tuple[list[dict], UsageInfo]:
    """Ask the manager which files from the diff to persist to the archive pool."""
    prompt_config = load_prompt_fn(str(prompt_yaml_path))

    diff_text = _format_file_diff(file_diff)
    existing_archives_text = _format_archive_pool(archive_pool)
    existing_names: set[str] = set(archive_pool.keys())

    system_prompt, user_prompt = render_prompt_fn(
        prompt_config,
        original_task=original_task_text,
        subtask_id=subtask["id"],
        subtask_instruction=subtask.get("instruction", ""),
        file_diff=diff_text,
        existing_archives=existing_archives_text,
    )

    prompt_path = save_manager_prompt_fn(
        system_prompt,
        user_prompt,
        run_dir,
        f"filemgmt_{subtask['id']}",
    )
    logger.info("Calling manager for file management of [%s] (prompt: %s)", subtask["id"], prompt_path)

    try:
        raw_response, usage = call_llm_fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=args.manager_provider,
            model=args.manager_model,
            max_tokens=args.manager_max_tokens,
            reasoning_effort=getattr(args, "manager_reasoning_effort", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Manager file-management call failed for [%s]: %s", subtask["id"], exc)
        return [], UsageInfo(model=args.manager_model)

    save_manager_response_fn(prompt_path, raw_response, usage)
    logger.info(
        "Manager file-mgmt cost: $%.4f (%d in / %d out tokens, %s)",
        usage.cost_usd,
        usage.input_tokens,
        usage.output_tokens,
        usage.model,
    )

    try:
        parsed = parse_json_response_fn(raw_response)
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Failed to parse manager file-mgmt response for [%s]: %s", subtask["id"], exc)
        return [], usage

    raw_actions = parsed.get("copy_actions", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_actions, list):
        logger.warning("Manager file-mgmt response for [%s]: copy_actions is not a list", subtask["id"])
        return [], usage

    valid: list[dict] = []
    seen_names: set[str] = set(existing_names)
    for action in raw_actions:
        if not isinstance(action, dict):
            continue
        src = action.get("src_path")
        if not isinstance(src, str) or not src:
            continue
        archive_name = action.get("archive_name") or os.path.basename(src)
        if not isinstance(archive_name, str) or not archive_name:
            continue
        if archive_name in seen_names:
            logger.warning(
                "Manager proposed colliding archive_name %r for [%s]; "
                "rejecting action to protect existing pool entry",
                archive_name,
                subtask["id"],
            )
            continue
        seen_names.add(archive_name)
        valid.append({"src_path": src, "archive_name": archive_name})

    logger.info(
        "Manager file-mgmt for [%s]: %d valid action(s) of %d returned",
        subtask["id"],
        len(valid),
        len(raw_actions),
    )
    return valid, usage


def apply_filemgmt_actions(
    actions: list[dict],
    downloads: dict[str, str],
    run_dir: Path,
    source_sid: str,
) -> dict:
    """Apply manager copy actions using already-downloaded files."""
    manifest: dict = {"archived": []}
    if not actions:
        return manifest

    archive_dir = run_dir / "file_artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for action in actions:
        src = action["src_path"]
        local_src = downloads.get(src)
        if not local_src or not os.path.exists(local_src):
            logger.warning(
                "apply_filemgmt_actions [%s]: source %s not in downloads manifest; "
                "CUA likely failed to fetch it (downloads has %d entries)",
                source_sid,
                src,
                len(downloads),
            )
            continue

        archive_name = action.get("archive_name") or os.path.basename(src)
        archive_path = archive_dir / archive_name
        try:
            shutil.copy2(local_src, archive_path)
        except OSError as exc:
            logger.warning("Archive copy failed (%s -> %s): %s", local_src, archive_path, exc)
            continue
        manifest["archived"].append(
            {
                "guest_path": src,
                "host_path": str(archive_path),
                "archive_name": archive_name,
                "source_sid": source_sid,
            }
        )
        logger.info("Archived %s -> %s", src, archive_path)

    return manifest


def resolve_input_files(
    input_files: list[str],
    archive_pool: dict[str, dict],
) -> list[dict]:
    """Resolve a subtask's ``input_files`` list against the archive pool."""
    resolved: list[dict] = []
    for name in input_files:
        if not isinstance(name, str) or not name:
            continue
        entry = archive_pool.get(name)
        if entry is None:
            logger.warning(
                "resolve_input_files: archive %r not in pool (available: %s); "
                "subtask will launch without it",
                name,
                sorted(archive_pool.keys()),
            )
            continue
        host_path = entry.get("host_path")
        guest_path = entry.get("guest_path")
        if not host_path or not guest_path:
            logger.warning(
                "resolve_input_files: archive %r missing host_path or guest_path: %s",
                name,
                entry,
            )
            continue
        resolved.append(
            {
                "local_path": host_path,
                "guest_path": guest_path,
                "source_sid": entry.get("source_sid", "?"),
            }
        )
    return resolved
