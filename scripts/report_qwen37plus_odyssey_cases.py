#!/usr/bin/env python3
"""Case-by-case comparison of the four qwen37plus odysseys runs.

Builds a per-task table (rubric score under nomanager / replan0 / replan1 /
replan5), flags judge-429 contamination per cell, and summarizes where planning
/ replanning helped or hurt. Output: markdown.
Usage: report_qwen37plus_odyssey_cases.py <out.md>
"""
import json, glob, os, sys, collections

CONFIGS = ["nomanager", "replan0", "replan1", "replan5"]
BASE = "runs/odysseys45_qwen37plus_{}"
TASKS_JSON = "data/odysseys/odysseys45.json"


def judge_error(reason):
    return ("Error judging rubric" in reason) or ("429" in reason) or ("Rate limit" in reason)


def load(cfg):
    d = json.load(open(os.path.join(BASE.format(cfg), "odysseys_eval.json")))
    per = {}
    for t in d["tasks"]:
        errs = sum(1 for r in t.get("rubric_results", []) if judge_error(r.get("final_reasoning", "")))
        per[t["task_id"]] = {
            "score": t["average_rubric_score"],
            "perfect": t.get("perfect", t["average_rubric_score"] >= 1.0),
            "nrub": len(t.get("rubric_results", [])),
            "judge_errs": errs,
            "task": (t.get("task") or "")[:1],  # not always present
        }
    return per, d["summary"]


def main():
    out = sys.argv[1]
    lvl = {t["task_id"]: t.get("level") for t in json.load(open(TASKS_JSON))}
    data, summ = {}, {}
    for c in CONFIGS:
        data[c], summ[c] = load(c)
    tasks = sorted(set(data["nomanager"]), key=lambda t: (lvl.get(t, "z"), t))

    L = []
    L.append("# Qwen3.7-Plus on odysseys45 — case-by-case comparison\n")
    L.append("Manager + CUA = `fireworks_ai/qwen3p7-plus` (qwen provider). Judge = gemini-3.1-flash-lite. "
             "Cell = mean rubric-pass fraction per task. `*` = that cell had ≥1 gemini-429 judge error "
             "(score under-estimated). nomanager ran at max_steps=100; replan0/1/5 at 60.\n")

    # overall
    L.append("## Overall\n")
    L.append("| Config | Mean rubric | Perfect (=1.0) | total 429 cells |")
    L.append("|---|---|---|---|")
    for c in CONFIGS:
        nperf = sum(1 for t in tasks if data[c][t]["perfect"])
        n429 = sum(1 for t in tasks if data[c][t]["judge_errs"] > 0)
        L.append(f"| {c} | {summ[c]['average_rubric_score']:.3f} | {nperf}/45 = {nperf/45*100:.1f}% | {n429} |")
    L.append("")

    # per-task table
    L.append("## Per-task scores\n")
    L.append("| Task | Level | nomanager | replan0 | replan1 | replan5 | best | planning Δ |")
    L.append("|---|---|---|---|---|---|---|---|")
    win = collections.Counter()
    helped = hurt = tied = 0
    for t in tasks:
        cells = []
        scores = {}
        for c in CONFIGS:
            d = data[c][t]
            mark = "*" if d["judge_errs"] else ""
            cells.append(f"{d['score']:.2f}{mark}")
            scores[c] = d["score"]
        nm = scores["nomanager"]
        best_replan = max(scores["replan0"], scores["replan1"], scores["replan5"])
        # which config is best
        bestc = max(CONFIGS, key=lambda c: scores[c])
        win[bestc] += 1
        delta = best_replan - nm  # >0 planning (some replan) beat nomanager
        if abs(delta) < 1e-9:
            tied += 1; tag = "tie"
        elif delta > 0:
            helped += 1; tag = f"+{delta:.2f} (planning)"
        else:
            hurt += 1; tag = f"{delta:.2f} (nomgr)"
        L.append(f"| `{t[:8]}` | {lvl.get(t,'?')} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {bestc} | {tag} |")
    L.append("")

    # summary of winners
    L.append("## Who wins per task (best config)\n")
    L.append("| Best config | # tasks |")
    L.append("|---|---|")
    for c in CONFIGS:
        L.append(f"| {c} | {win[c]} |")
    L.append("")
    L.append(f"- **Best replan ≥ nomanager (planning helped or tied):** {helped + tied} tasks "
             f"(strictly helped {helped}, tied {tied}).")
    L.append(f"- **nomanager strictly best (planning hurt):** {hurt} tasks.\n")

    # by level
    L.append("## By difficulty level (mean rubric)\n")
    L.append("| Level | nomanager | replan0 | replan1 | replan5 |")
    L.append("|---|---|---|---|---|")
    for lv in ("easy", "medium", "hard"):
        row = [lv]
        for c in CONFIGS:
            ts = [t for t in tasks if lvl.get(t) == lv]
            m = sum(data[c][t]["score"] for t in ts) / len(ts) if ts else 0
            row.append(f"{m:.3f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # biggest swings
    L.append("## Biggest swings (|best replan − nomanager|)\n")
    swings = sorted(tasks, key=lambda t: abs(max(data['replan0'][t]['score'], data['replan1'][t]['score'], data['replan5'][t]['score']) - data['nomanager'][t]['score']), reverse=True)[:8]
    L.append("| Task | Level | nomgr | best replan | Δ |")
    L.append("|---|---|---|---|---|")
    for t in swings:
        nm = data['nomanager'][t]['score']
        br = max(data['replan0'][t]['score'], data['replan1'][t]['score'], data['replan5'][t]['score'])
        L.append(f"| `{t[:8]}` | {lvl.get(t,'?')} | {nm:.2f} | {br:.2f} | {br-nm:+.2f} |")
    L.append("")
    L.append("> Caveat: cells marked `*` were depressed by gemini-429 judge errors during scoring, "
             "not task failures. Treat individual contaminated cells cautiously; the aggregate trends hold.")

    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out} ({len(tasks)} tasks; planning helped/tied {helped+tied}, hurt {hurt})")


if __name__ == "__main__":
    main()
