#!/usr/bin/env python3
"""Unit test for replan screenshot collection from focus + pending dependents' parents.

Mocks out the LLM call and screenshot extraction so the test can inspect the
user_segments structure built by ask_manager_for_replanning().
"""
import argparse
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_macu as ra
import utils.manager_utils as manager_utils
from utils import llm as _utils
from utils.llm import UsageInfo

# Bind every fake call_llm invocation against the REAL signature so that a
# missing required arg raises TypeError instead of silently passing.
_REAL_CALL_LLM_SIG = inspect.signature(_utils.call_llm)


def _bind_call_llm(**kwargs):
    """Raise TypeError if kwargs don't satisfy call_llm's real signature."""
    _REAL_CALL_LLM_SIG.bind(**kwargs)


def _mk_args() -> argparse.Namespace:
    return argparse.Namespace(
        manager_screenshots=2,
        replan_max_screenshots=12,
        manager_provider="anthropic",
        manager_model="claude-haiku-4-5-20251001",
        manager_max_tokens=1024,
        max_replans=10,
    )


def _fake_screenshots(sid, run_dir, k):
    """One fake screenshot per sid; k controls how many."""
    return [Path(f"/fake/{sid}_shot{i}.png") for i in range(k)]


def _assert_segments_shape(segments, expected_sources):
    """Ensure segments are text-then-(label,image*)-blocks-in-order.

    expected_sources: list of (sid, n_images) in the order they should appear.
    """
    # Structure:
    #   [0]         text: user_prompt
    #   [1]         text: preamble (present only when there are screenshots)
    #   [2..]       labeled blocks: text header, then N images per source
    assert segments[0]["type"] == "text" and len(segments[0]["text"]) > 0, \
        "first segment must be the main user_prompt text"

    if not expected_sources:
        # No screenshots — only the main text segment should exist.
        assert len(segments) == 1, f"expected 1 segment, got {len(segments)}"
        return

    assert segments[1]["type"] == "text" and "Screenshots from" not in segments[1]["text"], \
        "second segment should be the preamble, not a label header"

    idx = 2
    for sid, n in expected_sources:
        assert idx < len(segments), f"ran out of segments looking for {sid}"
        label = segments[idx]
        assert label["type"] == "text", f"expected text label for {sid}, got {label}"
        assert f"[{sid}]" in label["text"], f"label for {sid} missing sid, got: {label['text']}"
        idx += 1
        for i in range(n):
            assert idx < len(segments), f"ran out of segments for {sid} image {i}"
            img = segments[idx]
            assert img["type"] == "image", f"expected image for {sid}, got {img}"
            assert sid in img["path"], f"image path for {sid} has wrong sid: {img['path']}"
            idx += 1
    assert idx == len(segments), f"unexpected trailing segments: {segments[idx:]}"


def test_focus_only_when_no_pending_parents():
    """If pending subtasks have no completed parents (e.g. they're root),
    only the focus subtask's screenshots should be attached."""
    all_subtasks = [
        {"id": "a", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "b", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {
            "id": "final_aggregation", "agent_type": "manager",
            "dependencies": ["a", "b"], "instruction": "...",
        },
    ]
    completed = {"a"}  # `a` just finished; `b` pending with no completed parents yet.
    running_logicals: set[str] = set()
    captured = {}

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    def fake_save(*args, **kwargs):
        captured["save_segments"] = kwargs.get("user_segments")
        return Path("/tmp/unused.yaml")

    with patch.object(manager_utils, "extract_last_k_screenshots", side_effect=_fake_screenshots), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", side_effect=fake_save), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="a",
            all_subtasks=all_subtasks,
            results={"a": "done"},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=_mk_args(),
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )

    segs = captured["segments"]
    # Focus `a` should appear; `final_aggregation` is a pending dependent,
    # but its completed parents are only {a}, which is already the focus.
    # `b` has no completed parents.
    _assert_segments_shape(segs, [("a", 2)])
    assert segs == captured["save_segments"], "save_manager_prompt should get the same segments as call_llm"
    print("PASS  test_focus_only_when_no_pending_parents")


def test_multiple_completed_parents_attached():
    """Both completed parents of a pending subtask should contribute screenshots."""
    all_subtasks = [
        {"id": "a", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "b", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "c", "agent_type": "cua", "dependencies": ["a", "b"], "instruction": "..."},
        {
            "id": "final_aggregation", "agent_type": "manager",
            "dependencies": ["a", "b", "c"], "instruction": "...",
        },
    ]
    # `b` just finished; `a` already completed; `c` pending.
    completed = {"a", "b"}
    running_logicals: set[str] = set()
    captured = {}

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    with patch.object(manager_utils, "extract_last_k_screenshots", side_effect=_fake_screenshots), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", return_value=Path("/tmp/unused.yaml")), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="b",
            all_subtasks=all_subtasks,
            results={"a": "done-a", "b": "done-b"},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=_mk_args(),
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )

    segs = captured["segments"]
    # Focus `b` (2 imgs) then completed parent `a` (2 imgs). Order: focus first,
    # then parents of pending subtasks in graph order. `c`'s parents are (`a`, `b`);
    # `b` is already seen as focus, so only `a` is added. `final_aggregation`'s
    # parents are (`a`, `b`, `c`) — `c` is pending so skipped.
    _assert_segments_shape(segs, [("b", 2), ("a", 2)])
    # Labels should mark roles correctly:
    label_texts = [s["text"] for s in segs if s["type"] == "text" and "Screenshots from" in s["text"]]
    assert any("[b]" in t and "focus" in t for t in label_texts), \
        f"b should be labeled focus, got: {label_texts}"
    assert any("[a]" in t and "completed parent" in t for t in label_texts), \
        f"a should be labeled completed parent, got: {label_texts}"
    print("PASS  test_multiple_completed_parents_attached")


def test_no_duplicate_screenshots_for_shared_parent():
    """If two pending subtasks share a completed parent, that parent's screenshots
    should only be attached once."""
    all_subtasks = [
        {"id": "p", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "c1", "agent_type": "cua", "dependencies": ["p"], "instruction": "..."},
        {"id": "c2", "agent_type": "cua", "dependencies": ["p"], "instruction": "..."},
        {
            "id": "final_aggregation", "agent_type": "manager",
            "dependencies": ["c1", "c2"], "instruction": "...",
        },
    ]
    completed = {"p"}
    running_logicals: set[str] = set()
    captured = {}

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    with patch.object(manager_utils, "extract_last_k_screenshots", side_effect=_fake_screenshots), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", return_value=Path("/tmp/unused.yaml")), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="p",
            all_subtasks=all_subtasks,
            results={"p": "done"},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=_mk_args(),
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )

    segs = captured["segments"]
    # `p` should only appear once (as focus); neither c1 nor c2 should cause
    # a second `p` block.
    _assert_segments_shape(segs, [("p", 2)])
    print("PASS  test_no_duplicate_screenshots_for_shared_parent")


def test_cap_caps_total_screenshots():
    """replan_max_screenshots should bound the total attached images, with
    focus preserved and parents trimmed from the tail."""
    all_subtasks = [
        {"id": "p1", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "p2", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "p3", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "c", "agent_type": "cua", "dependencies": ["p1", "p2", "p3"], "instruction": "..."},
        {
            "id": "final_aggregation", "agent_type": "manager",
            "dependencies": ["c"], "instruction": "...",
        },
    ]
    completed = {"p1", "p2", "p3"}
    running_logicals: set[str] = set()
    captured = {}

    args = _mk_args()
    args.manager_screenshots = 3  # 3 per source
    args.replan_max_screenshots = 5  # cap: focus (3) + first parent (2) = 5

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    with patch.object(manager_utils, "extract_last_k_screenshots", side_effect=_fake_screenshots), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", return_value=Path("/tmp/unused.yaml")), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="p1",
            all_subtasks=all_subtasks,
            results={"p1": "", "p2": "", "p3": ""},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=args,
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )

    segs = captured["segments"]
    # Focus p1 (3 imgs) + first parent p2 (2 imgs, trimmed from 3 by cap).
    _assert_segments_shape(segs, [("p1", 3), ("p2", 2)])
    total_imgs = sum(1 for s in segs if s["type"] == "image")
    assert total_imgs == 5, f"cap should give exactly 5 images, got {total_imgs}"
    print("PASS  test_cap_caps_total_screenshots")


def test_running_subtasks_are_not_considered_parents():
    """Running (not-yet-completed) parents should NOT contribute screenshots —
    only completed parents do."""
    all_subtasks = [
        {"id": "a", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "b", "agent_type": "cua", "dependencies": [], "instruction": "..."},
        {"id": "c", "agent_type": "cua", "dependencies": ["a", "b"], "instruction": "..."},
    ]
    completed = {"a"}  # b is running, c is pending
    running_logicals = {"b"}
    captured = {}

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    with patch.object(manager_utils, "extract_last_k_screenshots", side_effect=_fake_screenshots), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", return_value=Path("/tmp/unused.yaml")), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="a",
            all_subtasks=all_subtasks,
            results={"a": "done"},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=_mk_args(),
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )

    # c's parents are {a, b}; a is focus (already seen), b is running (skip).
    # So only `a` contributes screenshots.
    _assert_segments_shape(captured["segments"], [("a", 2)])
    print("PASS  test_running_subtasks_are_not_considered_parents")


def test_no_screenshots_means_plain_text_segments():
    """With no screenshots available, user_segments should be a single text block."""
    all_subtasks = [{"id": "a", "agent_type": "cua", "dependencies": [], "instruction": "..."}]
    completed = {"a"}
    running_logicals: set[str] = set()
    captured = {}

    def fake_call_llm(**kwargs):
        _bind_call_llm(**kwargs)
        captured["segments"] = kwargs.get("user_segments")
        return '{"action":"no_change"}', UsageInfo(model="test", input_tokens=1, output_tokens=1)

    with patch.object(manager_utils, "extract_last_k_screenshots", return_value=[]), \
         patch.object(manager_utils, "call_llm", side_effect=fake_call_llm), \
         patch.object(manager_utils, "save_manager_prompt", return_value=Path("/tmp/unused.yaml")), \
         patch.object(manager_utils, "save_manager_response"):
        ra.ask_manager_for_replanning(
            focus_sid="a",
            all_subtasks=all_subtasks,
            results={"a": "done"},
            completed=completed,
            running_logicals=running_logicals,
            original_task_text="t",
            args=_mk_args(),
            run_dir=Path("/tmp"),
            iteration=1,
            num_applied=0,
        )
    _assert_segments_shape(captured["segments"], [])
    print("PASS  test_no_screenshots_means_plain_text_segments")


if __name__ == "__main__":
    test_focus_only_when_no_pending_parents()
    test_multiple_completed_parents_attached()
    test_no_duplicate_screenshots_for_shared_parent()
    test_cap_caps_total_screenshots()
    test_running_subtasks_are_not_considered_parents()
    test_no_screenshots_means_plain_text_segments()
    print("\nAll 6 tests passed.")
