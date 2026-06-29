#!/usr/bin/env python3
"""Case-by-case comparison of two Odysseys MACU runs → a markdown report.

Reads each run's rubric eval (prefers ``odysseys_eval_fulltext.json``, else
``odysseys_eval.json``), matches tasks by id, and writes a markdown report to
``reports/`` covering: overall rubric/perfect deltas, a by-level breakdown, a
per-task score table (A vs B vs Δ), the biggest movers, and — for the largest
swings — which individual rubrics flipped between the two runs.

Usage:
    python scripts/compare_runs.py runs/odysseys45_qwen37plus_nomanager \
                                   runs/odysseys45_qwen37plus_replan5
    python scripts/compare_runs.py RUN_A RUN_B --name-a single --name-b replan5 \
                                   --out reports/single_vs_replan5.md --top-swings 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_run import _level_map, _lvl_key  # noqa: E402


def _resolve_eval_json(run_dir: Path, override: Optional[str]) -> Path:
    """Prefer the corrected full-text eval, else the default eval json."""
    if override:
        return Path(override)
    ft = run_dir / "odysseys_eval_fulltext.json"
    return ft if ft.exists() else run_dir / "odysseys_eval.json"


def _as_bool(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def _load(run_dir: Path, override: Optional[str]) -> tuple[dict, str, str]:
    """Return ({task_id: rec}, eval_file_name, judge_model)."""
    path = _resolve_eval_json(run_dir, override)
    if not path.exists():
        raise SystemExit(f"No eval json found for {run_dir} (looked for {path.name})")
    data = json.load(open(path))
    tasks = {}
    for t in data.get("tasks", []):
        rubrics = {}
        for r in t.get("rubric_results", []) or []:
            rid = r.get("rubric_id") or r.get("requirement", "")[:40]
            rubrics[rid] = {
                "passed": _as_bool(r.get("success")),
                "requirement": (r.get("requirement") or "").strip(),
            }
        tasks[t["task_id"]] = {
            "score": t.get("average_rubric_score"),
            "perfect": _as_bool(t.get("perfect")) or (
                isinstance(t.get("average_rubric_score"), (int, float))
                and t["average_rubric_score"] >= 1.0
            ),
            "rubrics": rubrics,
        }
    judge = (data.get("summary") or {}).get("judge_model") or data.get("judge_model") or "?"
    return tasks, path.name, judge


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Case-by-case comparison of two Odysseys runs.")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--name-a", help="Short label for run A (default: dir name)")
    ap.add_argument("--name-b", help="Short label for run B (default: dir name)")
    ap.add_argument("--eval-json-a", help="Override eval json for run A")
    ap.add_argument("--eval-json-b", help="Override eval json for run B")
    ap.add_argument("--tasks-file", help="Odysseys tasks JSON for levels "
                                         "(default: data/odysseys/odysseys45.json)")
    ap.add_argument("--top-swings", type=int, default=8,
                    help="How many biggest-diverging tasks to show rubric-level detail for")
    ap.add_argument("--out", help="Output md path (default: reports/<A>_vs_<B>_comparison.md)")
    args = ap.parse_args()

    run_a, run_b = Path(args.run_a).resolve(), Path(args.run_b).resolve()
    name_a = args.name_a or run_a.name
    name_b = args.name_b or run_b.name

    a, eval_a, judge_a = _load(run_a, args.eval_json_a)
    b, eval_b, judge_b = _load(run_b, args.eval_json_b)

    tasks_file = args.tasks_file
    if not tasks_file:
        default_tf = Path(__file__).resolve().parent.parent / "data/odysseys/odysseys45.json"
        if default_tf.exists():
            tasks_file = str(default_tf)
    lvl = _level_map(tasks_file)

    common = sorted(set(a) & set(b), key=lambda t: (_lvl_key(lvl.get(t)), t))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    L: list[str] = []
    L.append(f"# Run comparison — `{name_a}`  vs  `{name_b}`\n")
    L.append(f"- **A:** `{run_a}/` (scores from `{eval_a}`)")
    L.append(f"- **B:** `{run_b}/` (scores from `{eval_b}`)")
    L.append(f"- **Judge:** {judge_a if judge_a == judge_b else f'{judge_a} (A) / {judge_b} (B)'}")
    L.append(f"- **Tasks compared:** {len(common)}"
             + (f"  (A-only: {len(only_a)}, B-only: {len(only_b)})" if (only_a or only_b) else ""))
    L.append(f"- **Δ = B − A** (positive ⇒ B better)\n")

    # --- Overall ---
    sa = [a[t]["score"] for t in common]
    sb = [b[t]["score"] for t in common]
    ma, mb = _mean(sa), _mean(sb)
    pa = sum(1 for t in common if a[t]["perfect"])
    pb = sum(1 for t in common if b[t]["perfect"])
    L.append("## Overall\n")
    L.append(f"| Metric | A ({name_a}) | B ({name_b}) | Δ |")
    L.append("|---|---|---|---|")
    L.append(f"| Mean rubric | {ma:.3f} | {mb:.3f} | {mb-ma:+.3f} |")
    L.append(f"| Perfect (=1.0) | {pa}/{len(common)} | {pb}/{len(common)} | {pb-pa:+d} |")
    L.append("")

    # --- By level ---
    levels = sorted({lvl.get(t) or "?" for t in common}, key=_lvl_key)
    L.append("## By level\n")
    L.append(f"| Level | n | A mean | B mean | Δ |")
    L.append("|---|---|---|---|---|")
    for lv in levels:
        ts = [t for t in common if (lvl.get(t) or "?") == lv]
        la, lb = _mean([a[t]["score"] for t in ts]), _mean([b[t]["score"] for t in ts])
        L.append(f"| {lv} | {len(ts)} | {la:.3f} | {lb:.3f} | {lb-la:+.3f} |")
    L.append("")

    # --- Per-task table ---
    def delta(t):
        return (b[t]["score"] or 0) - (a[t]["score"] or 0)

    improved = [t for t in common if delta(t) > 1e-9]
    regressed = [t for t in common if delta(t) < -1e-9]
    tied = [t for t in common if abs(delta(t)) <= 1e-9]

    L.append("## Per-task\n")
    L.append(f"Flag: 🟢 B better · 🔴 A better · ⚪ tie. "
             f"({len(improved)} improved, {len(regressed)} regressed, {len(tied)} tied.)\n")
    L.append(f"| Task | Level | A | B | Δ | |")
    L.append("|---|---|---|---|---|---|")
    for t in common:
        d = delta(t)
        flag = "🟢" if d > 1e-9 else ("🔴" if d < -1e-9 else "⚪")
        L.append(f"| `{t[:12]}` | {lvl.get(t) or '?'} | {a[t]['score']:.2f} | "
                 f"{b[t]['score']:.2f} | {d:+.2f} | {flag} |")
    L.append("")

    # --- Movers ---
    L.append("## Biggest movers\n")
    L.append(f"**B gained most** (top {min(args.top_swings, len(improved))}):  "
             + (", ".join(f"`{t[:12]}` {delta(t):+.2f}"
                          for t in sorted(improved, key=delta, reverse=True)[:args.top_swings]) or "_none_"))
    L.append("")
    L.append(f"**A gained most / B regressed** (top {min(args.top_swings, len(regressed))}):  "
             + (", ".join(f"`{t[:12]}` {delta(t):+.2f}"
                          for t in sorted(regressed, key=delta)[:args.top_swings]) or "_none_"))
    L.append("")

    # --- Rubric-level detail for the biggest swings ---
    swing = sorted(common, key=lambda t: abs(delta(t)), reverse=True)
    swing = [t for t in swing if abs(delta(t)) > 1e-9][:args.top_swings]
    if swing:
        L.append("## Rubric-level diff for the biggest swings\n")
        L.append("Only rubrics whose pass/fail **differs** between A and B are shown.\n")
        for t in swing:
            d = delta(t)
            L.append(f"### `{t[:13]}` ({lvl.get(t) or '?'}) — A {a[t]['score']:.2f} → B {b[t]['score']:.2f} ({d:+.2f})")
            ar, br = a[t]["rubrics"], b[t]["rubrics"]
            flipped = []
            for rid in sorted(set(ar) | set(br)):
                pa_ = ar.get(rid, {}).get("passed")
                pb_ = br.get(rid, {}).get("passed")
                if pa_ != pb_:
                    req = (ar.get(rid) or br.get(rid) or {}).get("requirement", "")
                    mark = lambda v: "✅" if v else ("❌" if v is False else "—")
                    flipped.append(f"- `{rid}` {mark(pa_)}→{mark(pb_)}: {req[:110]}")
            L.append("\n".join(flipped) if flipped else "_(score differs but no single rubric flipped — partial scores)_")
            L.append("")

    if only_a or only_b:
        L.append("## Tasks present in only one run\n")
        if only_a:
            L.append(f"- **A only ({name_a}):** " + ", ".join(f"`{t[:12]}`" for t in only_a))
        if only_b:
            L.append(f"- **B only ({name_b}):** " + ", ".join(f"`{t[:12]}`" for t in only_b))
        L.append("")

    out = args.out
    if not out:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        out = str(reports_dir / f"{name_a}_vs_{name_b}_comparison.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(common)} tasks compared; "
          f"B better on {len(improved)}, A better on {len(regressed)}, tied {len(tied)})")


if __name__ == "__main__":
    main()
