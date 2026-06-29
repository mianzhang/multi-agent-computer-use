#!/usr/bin/env python3
"""Aggregate analysis across all tasks in an Odysseys MACU run directory.

Odysseys-specific: scores come from the rubric eval (odysseys_eval.json, or the
corrected odysseys_eval_fulltext.json when present); tasks are grouped by
difficulty level (easy/medium/hard) rather than OSWorld domain.

Input: a run root, e.g. runs/odysseys45_qwen37plus_replan5 (one folder per task).

Produces:
  1. A terminal report: overall success rate, per-domain breakdown, cost,
     wall-clock, replanning stats, and the per-task scoreboard. Also flags
     tasks that are stuck (claimed, no result) or not started.
  2. (optional) A markdown doc (--md [PATH]) mirroring the report as tables.
     Default path: reports/<run_name>_run_summary.md

Usage:
    python scripts/analyze_run.py runs/subset36_replan5
    python scripts/analyze_run.py runs/subset36_replan5 --md
    python scripts/analyze_run.py runs/subset36_replan5 --md /tmp/run.md
    python scripts/analyze_run.py runs/subset36_replan5 --tasks-file data/osworld/subset36.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_task import node_taxonomy, replan_effects, load_eval_record  # noqa: E402


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


def collect(run_dir: Path, eval_json: Optional[str] = None) -> list[dict]:
    """One record per task folder under run_dir.

    `eval_json` selects which rubric-eval file to read scores from (passed to
    load_eval_record); None uses its default search (odysseys_eval.json)."""
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
        # Odysseys score = rubric eval's average pass-fraction (read-only;
        # produced by odysseys_eval.py). Fall back to the programmatic OSWorld
        # score only if no rubric record exists (keeps generate_report working).
        score = None
        score_kind = "rubric"
        ev = load_eval_record(d, eval_json)
        if ev is not None:
            score = ev.get("average_rubric_score")
        if score is None:
            score = fr.get("osworld_evaluator_score")
            score_kind = "osworld"
        tax = node_taxonomy(d)["counts"]
        reff = replan_effects(d)["counts"]
        recs.append({
            "task_id": d.name,
            "state": "done",
            "domain": fr.get("domain"),
            "score": score,
            "score_kind": score_kind,
            "success": isinstance(score, (int, float)) and score >= 1.0,
            "cost": fr.get("total_cost_usd"),
            "duration": fr.get("total_duration_seconds"),
            "inference": fr.get("total_inference_seconds"),
            "replans_applied": rp.get("num_applied"),
            "replans_calls": rp.get("num_calls"),
            "n_subtasks": n_cua,
            "infeasible": fr.get("osworld_infeasible_verdict"),
            "initial_cua": tax["initial_cua"],
            "added_total": tax["added_total"],
            "added_retry": tax["added_retry"],
            "added_init_from": tax["added_init_from"],
            "added_variant": tax["added_variant"],
            "final_nodes": tax["final_total"],
            "replan_calls": reff["calls"],
            "replan_no_change": reff["no_change"],
            "replan_instr_only": reff["instruction_only"],
            "replan_structural": reff["structural"],
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


_LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _lvl_key(level: Optional[str]):
    """Sort key: easy < medium < hard < anything else (alphabetical)."""
    return (_LEVEL_ORDER.get(level, 9), level or "")


def _level_map(tasks_file: Optional[str]) -> dict:
    """task_id -> difficulty level, for odysseys runs that have no domain."""
    if not tasks_file or not os.path.exists(tasks_file):
        return {}
    try:
        data = json.load(open(tasks_file))
        items = data if isinstance(data, list) else [v for vs in data.values() for v in vs]
        return {t["task_id"]: t.get("level") for t in items if isinstance(t, dict) and t.get("task_id")}
    except Exception:
        return {}


def report(run_dir: Path, tasks_file: Optional[str], use_color: bool,
           eval_json: Optional[str] = None) -> list[dict]:
    recs = collect(run_dir, eval_json)
    # Odysseys tasks have no domain; group them by difficulty level instead.
    # (The record key stays "domain" for generate_report.py compatibility, but
    # it holds the difficulty level for odysseys runs.)
    lvl = _level_map(tasks_file)
    for r in recs:
        if not r.get("domain"):
            r["domain"] = lvl.get(r["task_id"]) or "?"
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

    # --- node dynamics: how the graphs grew ---
    if done:
        print()
        print(h("─" * 74))
        print(h(" NODE DYNAMICS  (graph structure across the run)"))
        print(h("─" * 74))
        si = sum(r.get("initial_cua") or 0 for r in done)
        sa = sum(r.get("added_total") or 0 for r in done)
        sr_ = sum(r.get("added_retry") or 0 for r in done)
        sif = sum(r.get("added_init_from") or 0 for r in done)
        sv = sum(r.get("added_variant") or 0 for r in done)
        n = len(done)
        grew = sum(1 for r in done if (r.get("added_total") or 0) > 0)
        print(f"  Initial CUA subtasks     : {si} total (mean {si/n:.1f}/task)")
        print(f"  Dynamically added nodes  : {sa} total (mean {sa/n:.1f}/task) "
              f"— {grew}/{n} tasks grew their graph")
        print(f"      ├─ retry (fresh)     : {sr_}")
        print(f"      ├─ init_from (inherit state) : {sif}")
        print(f"      └─ variant (alt strategy)    : {sv}")
        # tasks that grew the most
        top = sorted(done, key=lambda r: -(r.get("added_total") or 0))[:5]
        top = [r for r in top if (r.get("added_total") or 0) > 0]
        if top:
            print(f"  Most-grown tasks:")
            for r in top:
                print(f"      +{r['added_total']:>2} (retry={r['added_retry']},"
                      f" init_from={r['added_init_from']}, variant={r['added_variant']})"
                      f"  {(r['domain'] or '?'):<16} {r['task_id'][:16]}")

    # --- replan effect: applied vs actually-structural ---
    if done:
        rc = sum(r.get("replan_calls") or 0 for r in done)
        rn = sum(r.get("replan_no_change") or 0 for r in done)
        ri = sum(r.get("replan_instr_only") or 0 for r in done)
        rs = sum(r.get("replan_structural") or 0 for r in done)
        applied = ri + rs
        print()
        print(h("─" * 74))
        print(h(" REPLAN EFFECT  (did replanning actually change graph structure?)"))
        print(h("─" * 74))
        print(f"  Replan calls             : {rc}")
        print(f"  ├─ no change (confirmed as-is) : {rn}  ({rn/rc*100:.0f}%)" if rc else "")
        print(f"  └─ applied               : {applied}  ({applied/rc*100:.0f}%)" if rc else "")
        if applied:
            print(f"       ├─ instruction-only : {ri}  ({ri/applied*100:.0f}% of applied — topology UNCHANGED)")
            print(f"       └─ structural       : {rs}  ({rs/applied*100:.0f}% of applied — nodes/edges changed)")
        tasks_structural = sum(1 for r in done if (r.get("replan_structural") or 0) > 0)
        print(f"  Tasks with ≥1 structural replan: {tasks_structural}/{len(done)}")

    # --- by difficulty level ---
    print()
    print(h("─" * 74))
    print(h(" BY LEVEL"))
    print(h("─" * 74))
    bydom: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        bydom[r["domain"]].append(r)
    print(f"  {'level':<20} {'perfect SR':>14} {'mean$':>8} {'meanS':>7} {'replan':>7}")
    for dom in sorted(bydom, key=_lvl_key):
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
    print(f"  {'score':>5} {'level':<18} {'$':>6} {'sec':>5} {'rpl':>4} {'subs':>4}  task_id")
    for r in sorted(done, key=lambda x: (_lvl_key(x["domain"]), -(x["score"] or 0))):
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


def write_md(recs: list[dict], run_dir: Path, path: str, tasks_file: Optional[str]) -> None:
    """Write the run analysis as a markdown document (tables mirroring the report)."""
    done = [r for r in recs if r["state"] == "done"]
    stuck = [r for r in recs if r["state"] == "stuck"]
    incomplete = [r for r in recs if r["state"] == "incomplete"]
    scored = [r for r in done if isinstance(r["score"], (int, float))]
    succ = [r for r in scored if r["success"]]
    n = len(done)

    L: list[str] = []
    L.append(f"# MACU Run Summary — `{run_dir.name}`\n")
    L.append(f"- **Run dir:** `{run_dir}/`")
    L.append(f"- **Generated by:** `scripts/analyze_run.py`")
    expected = _expected_from_tasks_file(tasks_file)
    folders = f"{len(recs)} task folder(s)" + (f" (expected {expected})" if expected else "")
    L.append(f"- **Folders:** {folders}")
    L.append(f"- **Done / scored:** {len(done)} done, {len(scored)} scored"
             + (f"; **{len(stuck)} stuck**" if stuck else "")
             + (f"; {len(incomplete)} incomplete" if incomplete else ""))
    L.append("")

    # --- Overall ---
    L.append("## Overall\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    if scored:
        sr = len(succ) / len(scored) * 100
        partial = sum(r["score"] for r in scored) / len(scored) * 100
        L.append(f"| Success rate (score ≥ 1.0) | {len(succ)}/{len(scored)} = {sr:.1f}% |")
        L.append(f"| Partial-credit mean | {partial:.1f}% |")
    else:
        L.append("| Success rate | (no scored tasks) |")
    if done:
        total_cost = sum(r["cost"] or 0 for r in done)
        total_dur = sum(r["duration"] or 0 for r in done)
        total_inf = sum(r.get("inference") or 0 for r in done)
        tot_replan = sum(r["replans_applied"] or 0 for r in done)
        L.append(f"| Total cost | ${total_cost:.2f} (mean ${total_cost/n:.3f}/task) |")
        L.append(f"| Total wall-clock | {total_dur/60:.0f} min (mean {total_dur/n:.0f}s/task, "
                 f"{total_inf/max(1,total_dur)*100:.0f}% inference) |")
        L.append(f"| Replans applied | {tot_replan} (mean {tot_replan/n:.1f}/task) |")
    L.append("")

    # --- Node dynamics ---
    if done:
        si = sum(r.get("initial_cua") or 0 for r in done)
        sa = sum(r.get("added_total") or 0 for r in done)
        sr_ = sum(r.get("added_retry") or 0 for r in done)
        sif = sum(r.get("added_init_from") or 0 for r in done)
        sv = sum(r.get("added_variant") or 0 for r in done)
        grew = sum(1 for r in done if (r.get("added_total") or 0) > 0)
        L.append("## Node dynamics\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Initial CUA subtasks | {si} (mean {si/n:.1f}/task) |")
        L.append(f"| Dynamically added nodes | {sa} (mean {sa/n:.1f}/task) — {grew}/{n} tasks grew |")
        L.append(f"| ├─ retry (fresh) | {sr_} |")
        L.append(f"| ├─ init_from (inherit state) | {sif} |")
        L.append(f"| └─ variant (alt strategy) | {sv} |")
        L.append("")

    # --- Replan effect ---
    if done:
        rc = sum(r.get("replan_calls") or 0 for r in done)
        rn = sum(r.get("replan_no_change") or 0 for r in done)
        ri = sum(r.get("replan_instr_only") or 0 for r in done)
        rs = sum(r.get("replan_structural") or 0 for r in done)
        applied = ri + rs
        tasks_structural = sum(1 for r in done if (r.get("replan_structural") or 0) > 0)
        L.append("## Replan effect\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Replan calls | {rc} |")
        if rc:
            L.append(f"| ├─ no change (confirmed as-is) | {rn} ({rn/rc*100:.0f}%) |")
            L.append(f"| └─ applied | {applied} ({applied/rc*100:.0f}%) |")
        if applied:
            L.append(f"| &nbsp;&nbsp;&nbsp;├─ instruction-only (topology unchanged) | {ri} ({ri/applied*100:.0f}%) |")
            L.append(f"| &nbsp;&nbsp;&nbsp;└─ structural (nodes/edges changed) | {rs} ({rs/applied*100:.0f}%) |")
        L.append(f"| Tasks with ≥1 structural replan | {tasks_structural}/{n} |")
        L.append("")

    # --- Per-domain ---
    bydom: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        bydom[r["domain"]].append(r)
    if bydom:
        L.append("## By level\n")
        L.append("| Level | Perfect SR | mean $ | mean sec | mean replan |")
        L.append("|---|---|---|---|---|")
        for dom in sorted(bydom, key=_lvl_key):
            rs = bydom[dom]
            ok = sum(1 for r in rs if r["success"])
            mc = sum(r["cost"] or 0 for r in rs) / len(rs)
            md = sum(r["duration"] or 0 for r in rs) / len(rs)
            mr = sum(r["replans_applied"] or 0 for r in rs) / len(rs)
            L.append(f"| {dom} | {_fmt_pct(ok, len(rs))} | {mc:.3f} | {md:.0f}s | {mr:.1f} |")
        L.append("")

    # --- Per-task ---
    L.append("## Per-task\n")
    L.append("| Score | Level | $ | sec | Replans | Subtasks | Init nodes | Final nodes | Task ID |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(done, key=lambda x: (_lvl_key(x["domain"]), -(x["score"] or 0))):
        sc = r["score"]
        mark = "✅" if r["success"] else ("⚠️" if isinstance(sc, (int, float)) and sc > 0 else "❌")
        sc_s = f"{sc:.2f}" if isinstance(sc, (int, float)) else str(sc)
        L.append(f"| {mark} {sc_s} | {r['domain'] or '?'} | {(r['cost'] or 0):.3f} | "
                 f"{(r['duration'] or 0):.0f}s | {r['replans_applied'] or 0} | {r['n_subtasks'] or 0} | "
                 f"{r.get('initial_cua') or 0} | {r.get('final_nodes') or 0} | `{r['task_id']}` |")
    for r in stuck + incomplete:
        L.append(f"| {r['state']} | — | — | — | — | — | — | — | `{r['task_id']}` |")
    L.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate analysis over an Odysseys MACU run dir.")
    ap.add_argument("run_dir", help="Run root, e.g. runs/odysseys45_qwen37plus_replan5")
    ap.add_argument("--tasks-file", help="Odysseys tasks JSON (levels + expected total). "
                                         "Default: data/odysseys/odysseys45.json")
    ap.add_argument("--eval-json", help="Rubric-eval JSON to read scores from. "
                                        "Default: odysseys_eval_fulltext.json if present, "
                                        "else odysseys_eval.json in the run dir.")
    ap.add_argument("--md", nargs="?", const="__DEFAULT__",
                    help="Write the analysis as a markdown doc. With no path, "
                         "defaults to reports/<run_name>_run_summary.md")
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

    # Odysseys defaults: levels file + prefer the corrected full-text eval.
    tasks_file = args.tasks_file
    if not tasks_file:
        default_tf = Path(__file__).resolve().parent.parent / "data/odysseys/odysseys45.json"
        if default_tf.exists():
            tasks_file = str(default_tf)
    eval_json = args.eval_json
    if not eval_json:
        ft = run_dir / "odysseys_eval_fulltext.json"
        if ft.exists():
            eval_json = str(ft)
            print(f"  (scores from {ft.name} — corrected full-text eval)")

    recs = report(run_dir, tasks_file, use_color=not args.no_color, eval_json=eval_json)
    if args.md is not None:
        if args.md == "__DEFAULT__":
            reports_dir = Path(__file__).resolve().parent.parent / "reports"
            md_path = str(reports_dir / f"{run_dir.name}_run_summary.md")
        else:
            md_path = args.md
        write_md(recs, run_dir, md_path, tasks_file)
        print(f"  markdown → {md_path}\n")


if __name__ == "__main__":
    main()
