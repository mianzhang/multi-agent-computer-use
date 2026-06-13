#!/usr/bin/env python3
"""Aggregate analysis across all tasks in a MACU run directory.

Input: a run root, e.g. runs/subset36_replan5  (contains one folder per task).

Produces:
  1. A terminal report: overall success rate, per-domain breakdown, cost,
     wall-clock, replanning stats, and the per-task scoreboard. Also flags
     tasks that are stuck (claimed, no result) or not started.
  2. (optional) A CSV (--csv PATH) with one row per task for spreadsheets.

Usage:
    python scripts/analyze_run.py runs/subset36_replan5
    python scripts/analyze_run.py runs/subset36_replan5 --csv /tmp/run.csv
    python scripts/analyze_run.py runs/subset36_replan5 --tasks-file data/osworld/subset36.json
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m", "cyan": "\033[36m"}


def _c(s: str, color: str, use: bool) -> str:
    return f"{C[color]}{s}{C['reset']}" if use else s


def _load(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect(run_dir: Path) -> list[dict]:
    """One record per task folder under run_dir."""
    recs: list[dict] = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        fr = _load(d / "final_results.json")
        claimed = (d / ".claimed").exists()
        if fr is None:
            # stuck (claimed but crashed) or in-progress, no result
            recs.append({
                "task_id": d.name, "state": "stuck" if claimed else "incomplete",
                "domain": None, "score": None, "cost": None, "duration": None,
                "replans_applied": None, "replans_calls": None, "n_subtasks": None,
            })
            continue
        rp = fr.get("replanning", {}) or {}
        sub = fr.get("subtask_results", {}) or {}
        n_cua = sum(1 for k in sub if k != "final_aggregation")
        score = fr.get("osworld_evaluator_score")
        recs.append({
            "task_id": d.name,
            "state": "done",
            "domain": fr.get("domain"),
            "score": score,
            "success": isinstance(score, (int, float)) and score >= 1.0,
            "cost": fr.get("total_cost_usd"),
            "duration": fr.get("total_duration_seconds"),
            "inference": fr.get("total_inference_seconds"),
            "replans_applied": rp.get("num_applied"),
            "replans_calls": rp.get("num_calls"),
            "n_subtasks": n_cua,
            "infeasible": fr.get("osworld_infeasible_verdict"),
        })
    return recs


def _expected_from_tasks_file(tasks_file: Optional[str]) -> Optional[int]:
    if not tasks_file or not os.path.exists(tasks_file):
        return None
    try:
        d = json.load(open(tasks_file))
        if isinstance(d, dict):
            return sum(len(v) for v in d.values())
        if isinstance(d, list):
            return len(d)
    except Exception:
        pass
    return None


def _fmt_pct(num: int, den: int) -> str:
    return f"{num}/{den} ({num/den*100:.1f}%)" if den else f"{num}/0"


def report(run_dir: Path, tasks_file: Optional[str], use_color: bool) -> list[dict]:
    recs = collect(run_dir)
    done = [r for r in recs if r["state"] == "done"]
    stuck = [r for r in recs if r["state"] == "stuck"]
    incomplete = [r for r in recs if r["state"] == "incomplete"]
    scored = [r for r in done if isinstance(r["score"], (int, float))]
    succ = [r for r in scored if r["success"]]

    def h(s):
        return _c(s, "bold", use_color)

    print()
    print(h("═" * 74))
    print(h(f" MACU Run Summary: {run_dir.name}"))
    print(h("═" * 74))

    expected = _expected_from_tasks_file(tasks_file)
    total_line = f"{len(recs)} task folder(s)"
    if expected:
        total_line += f"  (expected {expected})"
    print(f"  Folders     : {total_line}")
    print(f"  Done/scored : {len(done)} done, {len(scored)} scored")
    if stuck:
        print(f"  {_c('Stuck', 'red', use_color)}       : {len(stuck)} (claimed, crashed, no result)")
    if incomplete:
        print(f"  Incomplete  : {len(incomplete)} (no result, not claimed)")

    # --- overall SR ---
    print()
    print(h("─" * 74))
    print(h(" OVERALL"))
    print(h("─" * 74))
    if scored:
        sr = len(succ) / len(scored) * 100
        partial = sum(r["score"] for r in scored) / len(scored) * 100
        sr_col = "green" if sr >= 50 else ("yellow" if sr > 0 else "red")
        print(f"  Success rate (score≥1.0) : {_c(f'{len(succ)}/{len(scored)} = {sr:.1f}%', sr_col, use_color)}")
        print(f"  Partial-credit mean      : {partial:.1f}%  (averages fractional scores)")
        if expected and len(scored) < expected:
            proj = len(succ) / expected * 100
            print(f"  {_c('NOTE', 'yellow', use_color)}: only {len(scored)}/{expected} tasks scored; "
                  f"SR over the full set is incomplete (≥{proj:.0f}% lower-bound).")
    else:
        print("  (no scored tasks)")

    total_cost = sum(r["cost"] or 0 for r in done)
    total_dur = sum(r["duration"] or 0 for r in done)
    total_inf = sum(r.get("inference") or 0 for r in done)
    tot_replan = sum(r["replans_applied"] or 0 for r in done)
    if done:
        print(f"  Total cost               : ${total_cost:.2f}  (mean ${total_cost/len(done):.3f}/task)")
        print(f"  Total wall-clock         : {total_dur/60:.0f} min  (mean {total_dur/len(done):.0f}s/task, "
              f"{total_inf/max(1,total_dur)*100:.0f}% in inference)")
        print(f"  Replans applied          : {tot_replan} across {len(done)} tasks "
              f"(mean {tot_replan/len(done):.1f}/task)")

    # --- per-domain ---
    print()
    print(h("─" * 74))
    print(h(" PER-DOMAIN"))
    print(h("─" * 74))
    bydom: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        bydom[r["domain"]].append(r)
    print(f"  {'domain':<20} {'SR':>14} {'mean$':>8} {'meanS':>7} {'replan':>7}")
    for dom in sorted(bydom):
        rs = bydom[dom]
        ok = sum(1 for r in rs if r["success"])
        mc = sum(r["cost"] or 0 for r in rs) / len(rs)
        md = sum(r["duration"] or 0 for r in rs) / len(rs)
        mr = sum(r["replans_applied"] or 0 for r in rs) / len(rs)
        col = "green" if ok == len(rs) else ("red" if ok == 0 else "yellow")
        print(f"  {dom:<20} {_c(_fmt_pct(ok,len(rs)).rjust(14), col, use_color)} "
              f"{mc:>8.3f} {md:>6.0f}s {mr:>7.1f}")

    # --- per-task scoreboard ---
    print()
    print(h("─" * 74))
    print(h(" PER-TASK"))
    print(h("─" * 74))
    print(f"  {'score':>5} {'domain':<18} {'$':>6} {'sec':>5} {'rpl':>4} {'subs':>4}  task_id")
    for r in sorted(done, key=lambda x: (x["domain"] or "", -(x["score"] or 0))):
        sc = r["score"]
        sc_col = "green" if r["success"] else ("yellow" if isinstance(sc, (int, float)) and sc > 0 else "red")
        sc_s = f"{sc:.2f}" if isinstance(sc, (int, float)) else str(sc)
        print(f"  {_c(sc_s.rjust(5), sc_col, use_color)} {(r['domain'] or '?'):<18} "
              f"{(r['cost'] or 0):>6.3f} {(r['duration'] or 0):>4.0f}s "
              f"{str(r['replans_applied'] or 0):>4} {str(r['n_subtasks'] or 0):>4}  {r['task_id'][:18]}")
    for r in stuck + incomplete:
        tag = _c(r["state"], "red" if r["state"] == "stuck" else "dim", use_color)
        print(f"  {tag:>5} {'—':<18} {'—':>6} {'—':>5} {'—':>4} {'—':>4}  {r['task_id'][:18]}")

    print()
    print(h("═" * 74))
    return recs


def write_csv(recs: list[dict], path: str) -> None:
    cols = ["task_id", "state", "domain", "score", "success", "cost",
            "duration", "inference", "replans_applied", "replans_calls",
            "n_subtasks", "infeasible"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate analysis over a MACU run dir.")
    ap.add_argument("run_dir", help="Run root, e.g. runs/subset36_replan5")
    ap.add_argument("--tasks-file", help="Original tasks JSON, to know the expected total")
    ap.add_argument("--csv", help="Write per-task rows to this CSV path")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    # guard: make sure this is a run root, not a single task dir
    if (run_dir / "final_results.json").exists():
        raise SystemExit(
            f"{run_dir} looks like a single task dir (has final_results.json). "
            "Use scripts/analyze_task.py for one task, or pass the run root here.")

    recs = report(run_dir, args.tasks_file, use_color=not args.no_color)
    if args.csv:
        write_csv(recs, args.csv)
        print(f"  CSV → {args.csv}\n")


if __name__ == "__main__":
    main()
