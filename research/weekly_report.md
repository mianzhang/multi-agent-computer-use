# Weekly Report — Multi-Agent Computer Use (MACU)

**Week of:** 2026-06-15
**Focus:** Standing up the MACU evaluation pipeline, a first benchmark run on
OSWorld, a replan-budget study, and a research direction for the dominant
failure mode.

---

## 1. Summary

- Brought up the full MACU stack locally (apptainer/QEMU VM backend) and
  validated it end-to-end on OSWorld; fixed two real bugs along the way.
- Ran a 36-task stratified OSWorld subset under two replanning budgets.
  **Headline: success rate 63.9% (replan=5) vs 55.6% (replan=1)** — a higher
  replan budget buys **+8.3 pts at near-identical cost**.
- Built analysis/visualization tooling (per-task DAG-evolution viewer with
  embedded recordings; per-run summary + CSV).
- **Key finding:** the dominant failure mode is **false success** (the agent
  reports a task done, but the evaluator scores 0) — 9 of 13 failures. This is
  structural: replanning cannot fix it because no failure signal is raised.
- Drafted a research proposal (**Proactive Speculative Execution**) that targets
  both false success and latency with a single mechanism.

## 2. Infrastructure & Setup

- Configured MACU with **manager = claude-opus-4-6**, **CUA subagent =
  gpt-5.4-mini**, both via an OpenAI-compatible proxy; VM backend = **apptainer
  (QEMU + KVM)** on a 192-core / 376 GB host.
- Validated a single OSWorld task end-to-end (manager decomposition → VM boot →
  CUA execution → OSWorld evaluator) at score 1.0.
- **Bugs fixed:**
  1. Apptainer overlay backing-chain bind — multi-level overlays
     (working → scout → base image) failed to mount the base dir; qemu errored
     with "Could not open backing file."
  2. **vmstate disk leak** — each cleaned overlay left its ~4 GB RAM snapshot
     behind, filling a 250 GB disk and crashing batch runs. Root-caused and
     fixed; verified vmstate count stays bounded across a full run.
- Parallel batch runner: N worker processes share one result dir and claim
  tasks atomically; runs are detached (tmux) so they survive SSH disconnects.

## 3. Benchmark: 36-task OSWorld subset

Stratified sample of 36 tasks across 10 OSWorld domains (proportional to the
full 369-task distribution; fixed seed for reproducibility).

### 3.1 Replan-budget study (main result)

| Metric | replan = 1 | replan = 5 |
|---|---|---|
| **Success rate (score ≥ 1.0)** | **20/36 = 55.6%** | **23/36 = 63.9%** |
| Partial-credit mean | 63.8% | 72.2% |
| Total cost | $43.41 | $45.38 |
| Total wall-clock | 475 min | 464 min |
| Replans applied | 11 | 26 |

**Takeaways:**
- Higher replan budget gives **+8.3 pts success at ~+$2 total cost** — most
  tasks succeed on the first try and never trigger replanning, so the extra
  budget is spent only on the few hard tasks. Good cost/quality trade.
- Net task-level effect of more budget: **+5 recovered, −2 regressed** (run
  variance) = +3 tasks.

### 3.2 Per-domain success rate

| Domain | replan = 1 | replan = 5 |
|---|---|---|
| libreoffice_writer | 2/2 | 2/2 |
| os | 2/2 | 2/2 |
| thunderbird | 1/1 | 1/1 |
| chrome | 3/4 | 3/4 |
| libreoffice_calc | 3/5 | 3/5 |
| libreoffice_impress | 3/5 | 4/5 |
| gimp | 1/3 | 2/3 |
| vs_code | 1/2 | 1/2 |
| vlc | 0/2 | 1/2 |
| **multi_apps** | **4/10** | **4/10** |

`multi_apps` (multi-application, long-horizon) is the clear bottleneck at 40%
under both budgets, and accounts for most of the cost and latency.

## 4. Failure Analysis (replan = 5)

Classifying all 36 tasks by outcome:

| Category | Count | % |
|---|---|---|
| ✅ Success (score ≥ 1.0) | 23 | 64% |
| 🟡 Partial (0 < score < 1) | 3 | 8% |
| ❌ **False success** (agent claims done, evaluator = 0) | **9** | **25%** |
| ❌ Infrastructure failure (no valid trajectory) | 1 | 3% |

**Main insight — false success is the dominant, structural failure mode:**
- Of 13 non-perfect tasks, **9 are false success**; only 1 is infrastructure.
- The CUA executes (23–192 steps), confidently reports completion, but the task
  is not actually done. This is a capability/self-evaluation limitation of the
  CUA model, not a framework bug.
- **Replanning cannot help here:** the subtask returns "success", so the manager
  receives no failure signal and never retries (5 of 9 had 0 replans). This is a
  structural limitation: MACU trusts the subagent's self-reported status, and
  the CUA over-reports success.

## 5. Tooling Delivered

- `scripts/analyze_task.py` — per-task analysis: terminal timeline of the
  algorithm flow (initial DAG → each replan's add/remove/modify ops → final
  state → score) and a self-contained HTML viewer that steps through the DAG's
  evolution, with full node instructions, the cause of each retry, and an
  embedded screen recording per subtask.
- `scripts/analyze_run.py` — per-run aggregate: overall/by-domain success,
  cost, latency, replan stats, per-task scoreboard, CSV export.
- Reports: `runs/subset36_replan5_analysis.md`,
  `runs/replan_budget_comparison.md`.

## 6. Research Direction (proposed)

**Proactive Speculative Execution (PSE)** — full proposal in
`research/speculative_execution_proposal.md`.

Reframing: MACU's replanning is *lazy sequential speculation* — it explores an
alternative strategy only after a failure is observed, serializing retries. PSE
instead predicts high-risk subtasks, **forks the VM snapshot**, runs **diverse
strategies in parallel**, and selects the winner by a **grounded completion
check** (read file / a11y diff / shell command) rather than self-report.

This addresses both problems with one mechanism:
- **Latency:** a serial `k`-retry chain collapses to one parallel round
  (sum → max), reducing wall-clock on hard tasks — extending MACU's "faster"
  thesis.
- **False success:** selecting on grounded environment state rather than the
  agent's claim directly defeats the dominant failure mode.

Uniquely enabled by this setting: snapshot-fork isolation, abundant idle host
compute, and multi-modal ground-truth observation — none available to web/API
agents.

## 7. Next Steps

1. **PoC (M1):** on a few `multi_apps` tasks, run the manager's diverse
   strategies in parallel with grounded selection; confirm false-success ↓ and
   latency ↓.
2. Implement a cheap **grounded completion probe** (reuse in-VM shell / a11y).
3. Implement a **risk predictor** (domain prior + trajectory "struggle"
   signals); calibrate on the existing replan1/replan5 runs.
4. Scale the benchmark toward the full 369-task OSWorld set.

---

*Data & reproducibility: `runs/subset36_replan{1,5}/` (per-task results and
HTML viewers), `runs/subset36_replan{1,5}_summary.csv`,
`runs/replan_budget_comparison.md`.*
