#!/usr/bin/env python3
"""Plot cumulative success rate (all-rubric-pass) vs agent steps for an odysseys run.

For each task we take (a) the agent-step count it used and (b) whether it was a
FULL success (all rubrics passed, i.e. average_rubric_score >= 1.0). The curve
answers: "if agents were capped at S steps, what fraction of the whole task set
would have fully succeeded?" — i.e.

    cumulative_SR(S) = #{tasks: full-success AND steps_used <= S} / N_total

It rises monotonically with S and plateaus at the run's overall perfect rate.
Each fully-successful task is credited at the step count it actually used (a
lower bound: a task is assumed to need all the steps it took).

"Agent steps" = CUA decision steps, counted from the "<n> in trajectory" markers
the runner logs per step, summed across ALL subtasks of the task (so it's the
system's total steps for single- and multi-agent alike).

Usage:
    python scripts/plot_cumulative_success.py runs/odysseys45_qwen37plus_replan5
    python scripts/plot_cumulative_success.py <run_dir> [--out PATH] [--eval-json PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def resolve_eval_json(run_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    ft = run_dir / "odysseys_eval_fulltext.json"
    return ft if ft.exists() else run_dir / "odysseys_eval.json"


def count_agent_steps(task_dir: Path) -> int:
    """CUA decision steps across all of this task's subtasks."""
    n = 0
    for log in glob.glob(str(task_dir / "**" / "subprocess.log"), recursive=True):
        try:
            n += open(log, errors="ignore").read().count(" in trajectory")
        except OSError:
            pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Cumulative success rate vs agent steps for an odysseys run.")
    ap.add_argument("run_dir")
    ap.add_argument("--eval-json", help="Rubric eval json (default: odysseys_eval_fulltext.json, else odysseys_eval.json)")
    ap.add_argument("--out", help="Output PNG (default: reports/<run_name>_cumulative_success.png)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ev_path = resolve_eval_json(run_dir, args.eval_json)
    if not ev_path.exists():
        raise SystemExit(f"No eval json found ({ev_path}). Score the run first.")
    tasks = json.load(open(ev_path))["tasks"]
    perfect = {t["task_id"]: (t.get("perfect") in (True, "True", "true")
                              or (isinstance(t.get("average_rubric_score"), (int, float))
                                  and t["average_rubric_score"] >= 1.0))
               for t in tasks}

    # (steps, is_full_success) per task that has both an eval record and a trajectory
    pts: list[tuple[int, bool]] = []
    skipped = 0
    for tid, is_perfect in perfect.items():
        td = run_dir / tid
        if not td.is_dir():
            skipped += 1
            continue
        steps = count_agent_steps(td)
        if steps == 0:
            skipped += 1
            continue
        pts.append((steps, is_perfect))
    if not pts:
        raise SystemExit("No tasks with both an eval record and a step count.")

    n_total = len(pts)
    pts.sort(key=lambda p: p[0])
    xs, ys, cum = [0], [0.0], 0
    for steps, ok in pts:
        cum += 1 if ok else 0
        xs.append(steps)
        ys.append(cum / n_total * 100.0)

    n_perfect = sum(1 for _, ok in pts if ok)
    overall = n_perfect / n_total * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(xs, ys, where="post", lw=2, color="#1f77b4")
    ax.axhline(overall, ls="--", lw=1, color="grey",
               label=f"final perfect rate = {n_perfect}/{n_total} = {overall:.1f}%")
    ax.set_xlabel("agent step budget  (CUA decision steps, summed over subtasks)")
    ax.set_ylabel("cumulative success rate  (all rubrics pass)  [%]")
    ax.set_title(f"Cumulative full-success vs agent steps — {run_dir.name}")
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(5, overall * 1.15))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = args.out or str(Path(__file__).resolve().parent.parent / "reports" / f"{run_dir.name}_cumulative_success.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    median_steps = pts[len(pts) // 2][0]
    print(f"wrote {out}")
    print(f"  tasks={n_total} (skipped {skipped} w/o traj or eval) | perfect={n_perfect} ({overall:.1f}%) | median steps/task={median_steps}")


if __name__ == "__main__":
    main()
