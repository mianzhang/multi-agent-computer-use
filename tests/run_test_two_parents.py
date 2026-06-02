#!/usr/bin/env python3
"""Test runner for the two-parent replan enrichment test.

Runs the pre-generated graph at tests/test_two_parents.json through
``_RunSingleGraphRunner`` with replanning enabled. Verifies that the
replan log shows completed results from both parents when the child
subtask (compare_populations) becomes ready.

Usage:
    python tests/run_test_two_parents.py
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root and scripts/ to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, str(Path(project_root) / "scripts"))

import run_macu as ra


def main():
    graph_path = str(Path(__file__).resolve().parent / "test_two_parents.json")
    result_dir = "/tmp/test_macu_two_parents"

    # Build args namespace directly (bypass parse_args validation of tasks_file)
    args = argparse.Namespace(
        tasks_file=graph_path,
        result_dir=result_dir,
        cua_script=str(Path(project_root) / "scripts" / "run_cua.py"),
        osworld_root=project_root,
        python=str(Path(project_root) / ".venv" / "bin" / "python"),
        cua_provider="qwen",
        manager_provider="anthropic",
        manager_model=None,
        manager_max_tokens=8192,
        manager_screenshots=1,
        max_subtask_timeout=120,
        website="https://www.google.com",
        domain="macu",
        max_parallelism=2,
        max_replans=50,
        spare_capacity_min_steps=5,
        no_manager=False,
        evaluator_timeout=600,
        osworld_data_dir=str(Path(project_root) / "data" / "osworld" / "evaluation_examples"),
        task_id=None,
        graph_gen_max_tokens=16384,
        sleep_after_execution=None,
        max_steps=None,
        post_task_wait_seconds=None,
        reasoning_effort=None,
        reset_pool_snapshot="init_state",
        reset_pool_timeout=120,
    )

    cua_args = [
        "--headless",
        "--provider_name", "apptainer",
        "--max_steps", "5",
        "--model", "Qwen/Qwen3.6-27B",
    ]

    print(f"\n{'='*60}")
    print("TEST: Two-parent replan enrichment")
    print(f"Graph: {graph_path}")
    print(f"Result dir: {result_dir}")
    print(f"Replanning: enabled (budget={args.max_replans})")
    print(f"{'='*60}\n")

    ra._RunSingleGraphRunner(graph_path, args, cua_args).run()

    # --- Verification ---
    result_dir_path = Path(result_dir) / "test_two_parents"
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}\n")

    # Check replan logs
    replan_log = result_dir_path / "replan_log.jsonl"
    if replan_log.exists():
        print(f"Replan log: {replan_log}")
        with open(replan_log) as f:
            for line in f:
                entry = json.loads(line)
                print(f"  iter={entry.get('iteration')}, trigger={entry.get('focus_sid')}, "
                      f"applied={entry.get('applied')}, "
                      f"reasoning={entry.get('decision', {}).get('reasoning', '')[:120] if entry.get('decision') else 'N/A'}")
    else:
        print("WARNING: No replan_log.jsonl found!")

    # Check replan prompt files for both parent results
    replan_audits = sorted(result_dir_path.glob("manager_prompt_*_replan_*.yaml"))
    print(f"\nReplan prompt files: {len(replan_audits)}")
    for p in replan_audits:
        content = p.read_text()
        has_tokyo = "tokyo_pop" in content
        has_nyc = "nyc_pop" in content
        # Check for screenshot references
        has_screenshot = "image_path:" in content
        print(f"  {p.name}: "
              f"tokyo_pop_in_context={'YES' if has_tokyo else 'NO'}, "
              f"nyc_pop_in_context={'YES' if has_nyc else 'NO'}, "
              f"screenshot={'YES' if has_screenshot else 'NO'}")

    # Check for the critical replan prompt -- the one after the SECOND parent completes
    # It should have completed_results from BOTH parents
    for p in replan_audits:
        content = p.read_text()
        # The replan fired after second parent: both parents in completed_results
        if "tokyo_pop" in content and "nyc_pop" in content:
            print(f"\n  PASS: {p.name} contains results from BOTH parents")
            # Check if completed_results section has both
            if "completed_results" in content.lower() or "Prior completed" in content:
                print(f"  PASS: Both parent results appear in the completed results context")

    # Check graph snapshots for modify actions
    snapshot_dir = result_dir_path / "graph_snapshots"
    if snapshot_dir.exists():
        snapshots = sorted(snapshot_dir.glob("*.json"))
        print(f"\nGraph snapshots: {len(snapshots)}")
        for s in snapshots:
            data = json.loads(s.read_text())
            decision = data.get("decision", {})
            modify_keys = list(decision.get("modify", {}).keys())
            add_keys = [e.get("id") for e in decision.get("add", [])]
            if modify_keys or add_keys:
                print(f"  {s.name}: modified={modify_keys}, added={add_keys}")
    else:
        print("\nNo graph snapshots found (no replan changes applied)")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
