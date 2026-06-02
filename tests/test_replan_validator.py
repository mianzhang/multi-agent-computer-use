"""Tests for apply_replan_decision validator — cross-batch variant_of / init_from."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_macu import apply_replan_decision


def _base_graph():
    """Return a minimal graph: subtask_1 (CUA, completed) + final_aggregation."""
    all_subtasks = [
        {
            "id": "subtask_1",
            "agent_type": "cua",
            "description": "do something",
            "instruction": "do something",
            "dependencies": [],
            "outputs": [],
            "_task_id": "test",
        },
        {
            "id": "final_aggregation",
            "agent_type": "manager",
            "description": "aggregate",
            "instruction": "aggregate results",
            "dependencies": ["subtask_1"],
            "outputs": [],
            "_task_id": "test",
        },
    ]
    st_by_id = {s["id"]: s for s in all_subtasks}
    all_ids = {s["id"] for s in all_subtasks}
    return all_subtasks, st_by_id, all_ids


# ── variant_of referencing an existing subtask (should always pass) ──

def test_variant_of_existing_subtask():
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_1_variant",
                "agent_type": "cua",
                "instruction": "try a different approach",
                "dependencies": [],
                "outputs": [],
                "variant_of": "subtask_1",
            },
        ],
        "modify": {
            "final_aggregation": {
                "dependencies": ["subtask_1", "subtask_1_variant"],
            },
        },
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert not errors


# ── variant_of referencing a subtask added in the SAME batch (the bug) ──

def test_variant_of_cross_batch_reference():
    """Manager adds subtask_1_retry AND subtask_1_variant (variant_of=subtask_1_retry).

    Before the fix, this was rejected because subtask_1_retry wasn't in
    st_by_id yet when subtask_1_variant was validated.
    """
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_1_retry",
                "agent_type": "cua",
                "instruction": "retry the task",
                "dependencies": [],
                "outputs": [],
            },
            {
                "id": "subtask_1_variant_cli",
                "agent_type": "cua",
                "instruction": "try CLI approach",
                "dependencies": [],
                "outputs": [],
                "variant_of": "subtask_1_retry",
            },
        ],
        "modify": {
            "final_aggregation": {
                "dependencies": [
                    "subtask_1",
                    "subtask_1_retry",
                    "subtask_1_variant_cli",
                ],
            },
        },
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert not errors


# ── init_from referencing a subtask added in the SAME batch ──

def test_init_from_cross_batch_reference():
    """Manager adds subtask_2 AND subtask_3 (init_from=subtask_2)."""
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_2",
                "agent_type": "cua",
                "instruction": "do step 2",
                "dependencies": [],
                "outputs": [],
            },
            {
                "id": "subtask_3",
                "agent_type": "cua",
                "instruction": "continue from step 2",
                "dependencies": ["subtask_2"],
                "outputs": [],
                "init_from": "subtask_2",
            },
        ],
        "modify": {
            "final_aggregation": {
                "dependencies": ["subtask_1", "subtask_2", "subtask_3"],
            },
        },
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert not errors


# ── variant_of referencing a non-existent subtask (should still fail) ──

def test_variant_of_nonexistent_still_rejected():
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_1_variant",
                "agent_type": "cua",
                "instruction": "try different",
                "dependencies": [],
                "outputs": [],
                "variant_of": "subtask_that_does_not_exist",
            },
        ],
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert not applied, "Should have been rejected"
    assert any("not an existing subtask" in e for e in errors)


# ── init_from referencing a non-existent subtask (should still fail) ──

def test_init_from_nonexistent_still_rejected():
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_3",
                "agent_type": "cua",
                "instruction": "continue",
                "dependencies": ["ghost"],
                "outputs": [],
                "init_from": "ghost",
            },
        ],
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert not applied, "Should have been rejected"
    assert any("not an existing subtask" in e for e in errors)


# ── Reproduce the exact pattern from the failing runs ──

def test_real_failure_pattern():
    """Reproduces the exact replan from shard 0 task 357ef137:

    add: 'subtask_2_variant_cli' variant_of 'subtask_2_explore_and_fix'
         (both being added in the same replan)
    modify: 'final_aggregation' depends on 'subtask_2_variant_cli'
    """
    all_subtasks, st_by_id, all_ids = _base_graph()
    decision = {
        "add": [
            {
                "id": "subtask_2_explore_and_fix",
                "agent_type": "cua",
                "instruction": "explore and fix the formula",
                "dependencies": ["subtask_1"],
                "outputs": [],
            },
            {
                "id": "subtask_2_variant_cli",
                "agent_type": "cua",
                "instruction": "use CLI to fix the formula",
                "dependencies": [],
                "outputs": [],
                "variant_of": "subtask_2_explore_and_fix",
            },
        ],
        "modify": {
            "final_aggregation": {
                "dependencies": [
                    "subtask_1",
                    "subtask_2_explore_and_fix",
                    "subtask_2_variant_cli",
                ],
            },
        },
    }
    applied, errors, _cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"subtask_1"},
        running_logicals=set(), task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert not errors
    # Verify both subtasks were actually added to the graph
    ids_after = {s["id"] for s in all_subtasks}
    assert "subtask_2_explore_and_fix" in ids_after
    assert "subtask_2_variant_cli" in ids_after


# ── cancel field tests ──

def _cancellable_graph():
    """Return a graph where 'b' is running and 'c_pending' depends on it."""
    all_subtasks = [
        {"id": "a", "agent_type": "cua", "description": "a", "instruction": "ia",
         "dependencies": [], "outputs": ["x"], "_task_id": "test"},
        {"id": "b", "agent_type": "cua", "description": "b", "instruction": "ib",
         "dependencies": [], "outputs": ["y"], "_task_id": "test"},
        {"id": "c_pending", "agent_type": "cua", "description": "c", "instruction": "ic",
         "dependencies": ["b"], "outputs": ["z"], "_task_id": "test"},
        {"id": "final_aggregation", "agent_type": "manager", "description": "agg",
         "instruction": "agg", "dependencies": ["a", "b", "c_pending"], "outputs": [],
         "_task_id": "test"},
    ]
    st_by_id = {s["id"]: s for s in all_subtasks}
    all_ids = {s["id"] for s in all_subtasks}
    return all_subtasks, st_by_id, all_ids


def test_cancel_with_remove_happy_path():
    """Cancel + remove the same sid — dependents must be rewired in same decision."""
    all_subtasks, st_by_id, all_ids = _cancellable_graph()
    decision = {
        "cancel": ["b"],
        "remove": ["b", "c_pending"],
        "modify": {"final_aggregation": {"dependencies": ["a"]}},
        "add": [],
    }
    applied, errors, cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"a"}, running_logicals={"b"}, task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert cancelled == {"b"}, f"Expected cancelled={{'b'}}, got {cancelled}"
    assert "b" not in {s["id"] for s in all_subtasks}


def test_cancel_only_keeps_sid_in_graph():
    """Cancel without remove — sid stays in the graph; manager prunes agg deps."""
    all_subtasks, st_by_id, all_ids = _cancellable_graph()
    decision = {
        "cancel": ["b"],
        "remove": [],
        "modify": {"final_aggregation": {"dependencies": ["a", "c_pending"]}},
        "add": [],
    }
    applied, errors, cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"a"}, running_logicals={"b"}, task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert cancelled == {"b"}
    # sid 'b' is still in the graph (not removed); orchestrator will cancel
    # its subprocess and mark state=cancelled, but node persists for audit.
    assert "b" in {s["id"] for s in all_subtasks}
    # final_aggregation deps were updated to drop 'b'.
    agg = next(s for s in all_subtasks if s["id"] == "final_aggregation")
    assert "b" not in agg["dependencies"]


def test_cancel_non_running_rejected():
    """Only running CUA subtasks can be cancelled."""
    all_subtasks, st_by_id, all_ids = _cancellable_graph()
    decision = {
        "cancel": ["c_pending"],
        "remove": [],
        "modify": {},
        "add": [],
    }
    applied, errors, cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"a"}, running_logicals={"b"}, task_id="test",
    )
    assert not applied
    assert any("not currently running" in e for e in errors), errors


def test_cancel_with_remove_dangling_dep_rejected():
    """Cancelling+removing `b` without rewiring c_pending must be rejected."""
    all_subtasks, st_by_id, all_ids = _cancellable_graph()
    decision = {"cancel": ["b"], "remove": ["b"], "modify": {}, "add": []}
    applied, errors, cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"a"}, running_logicals={"b"}, task_id="test",
    )
    assert not applied
    assert any("still depends on removed id" in e for e in errors), errors


def test_cancel_plus_aggregation_prune_in_one_decision():
    """Real-world pattern: cancel sid AND drop it from final_aggregation.deps."""
    all_subtasks, st_by_id, all_ids = _cancellable_graph()
    decision = {
        "cancel": ["b"],
        "remove": [],
        "modify": {
            "final_aggregation": {"dependencies": ["a", "c_pending"]},
            "c_pending": {"dependencies": []},
        },
        "add": [],
    }
    applied, errors, cancelled = apply_replan_decision(
        decision, all_subtasks, st_by_id, all_ids,
        downstream_counts={}, completed={"a"}, running_logicals={"b"}, task_id="test",
    )
    assert applied, f"Should have been applied, errors: {errors}"
    assert cancelled == {"b"}
    c = next(s for s in all_subtasks if s["id"] == "c_pending")
    assert c["dependencies"] == []


if __name__ == "__main__":
    tests = [
        test_variant_of_existing_subtask,
        test_variant_of_cross_batch_reference,
        test_init_from_cross_batch_reference,
        test_variant_of_nonexistent_still_rejected,
        test_init_from_nonexistent_still_rejected,
        test_real_failure_pattern,
        test_cancel_with_remove_happy_path,
        test_cancel_only_keeps_sid_in_graph,
        test_cancel_non_running_rejected,
        test_cancel_with_remove_dangling_dep_rejected,
        test_cancel_plus_aggregation_prune_in_one_decision,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            raise
    print(f"\nAll {len(tests)} tests passed.")
