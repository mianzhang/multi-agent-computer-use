# Proactive Speculative Execution for Multi-Agent Computer Use

**Status:** Research proposal (draft v1)
**Context:** Built on the MACU (Multi-Agent Computer Use) framework — a manager LLM
decomposes a task into a DAG of subtasks, dispatches parallel CUA subagents, and
revises the DAG via replanning as results arrive.

---

## Abstract

Replanning in multi-agent computer use is, in effect, **lazy sequential
speculation**: the manager only explores an alternative strategy *after* it
observes a subtask fail. On long-horizon tasks this serializes a chain of
full subtask re-executions, inflating wall-clock latency; worse, when a subtask
*falsely* reports success (the dominant failure mode we observe), no failure is
ever surfaced, so the wrong result silently propagates to the final answer.

We propose **Proactive Speculative Execution (PSE)**: predict which subtasks are
high-risk, **fork the VM state**, and run *diverse* strategies **concurrently**
from the start, selecting the winner by a **grounded completion check** rather
than the agent's self-report. PSE converts the serial retry chain into a single
parallel round — reducing wall-clock latency on hard tasks — while the grounded
selection criterion simultaneously suppresses false success. The approach is
uniquely enabled by the desktop/VM setting (snapshot-fork isolation, abundant
host compute, and multi-modal ground-truth observation channels) and directly
extends MACU's central thesis that multi-agent orchestration can be *faster*.

---

## 1. Problem & Motivation

We ran MACU (manager = claude-opus-4-6, CUA = gpt-5.4-mini) on a 36-task
stratified OSWorld subset under two replan budgets:

| | Success rate | Mean wall-clock/task | Total replans |
|---|---|---|---|
| replan budget = 1 | 20/36 = 55.6% | 792 s | 11 |
| replan budget = 5 | 23/36 = 63.9% | 773 s | 26 |

Two findings motivate this proposal:

1. **The dominant failure mode is *false success*, not infrastructure or
   give-up.** Of 13 non-perfect tasks under replan=5, **9 were "false
   success"**: the CUA executed (23–192 steps), confidently reported the task
   complete, yet the OSWorld evaluator scored it 0. Only 1 was an
   infrastructure failure. **Replanning cannot fix false success**, because the
   subtask returns "success" and the manager never receives a failure signal to
   trigger a retry (5 of the 9 had 0 replans).

2. **Latency is dominated by serial retry on hard tasks.** `multi_apps` tasks
   (success rate 40%) drove both failure and latency. Example trajectories:
   - `22a4636f`: 5 serial replans, 452 s, **failed** — the manager rediscovered
     a new strategy (GUI export → CLI `libreoffice --headless` → simplified)
     only after each prior attempt failed.
   - `b5062e3e`: 5 serial replans, 1715 s, **eventually succeeded** — the
     winning strategy existed all along but was reached one failure at a time.

These two problems share a root cause: **the controller acts on a late,
unreliable completion signal** (the agent's self-report, observed only after a
full serial attempt).

## 2. Key Insight

> **MACU's replanning is a branch predictor that only fires on a miss.**

When the manager retries a failed subtask with a different strategy, it is
exploring an alternative branch — but reactively and serially, one failure at a
time. The candidate strategies are mutually independent (no data dependency);
they are serialized only by the replan loop. If we move that diversity to the
*front* and run the branches *concurrently*, a serial chain of `k` attempts
(≈ `k×` latency) collapses to a single parallel round (≈ `1×` latency).

**Empirical support (subset36, replan=5):** dynamic growth is exactly the
retry/variant machinery PSE would parallelize. Across 36 tasks the manager
added 15 nodes at runtime — 6 fresh retries, 3 strategy variants, 6 init_from
(incremental continuations) — and these concentrate on the hard tasks (e.g.
`22a4636f` added 6: five serial retries + one variant). Each of those serial
retries is an independent branch the system discovered one failure at a time;
PSE would launch them at once. (Full breakdown: §node dynamics in
`research/weekly_report.md`.)

This is the software analogue of CPU **speculative execution**: rather than
stall at an unknown branch, execute predicted paths in parallel and squash the
losers.

## 3. Method: Proactive Speculative Execution (PSE)

### 3.1 Risk prediction (the branch predictor)
A cheap predictor estimates each subtask's failure/false-success probability
*before/early in* execution, from behavioral signals available in
computer-use trajectories that are currently unused:
- domain prior (e.g. `multi_apps` ≫ single-app risk, from our data),
- "struggle" signals: repeated clicks on the same region, undo/redo cycles,
  aimless scrolling, detected error dialogs, action entropy, step count
  relative to same-domain norms.

The predictor sets the **speculation width** `k`: low risk → `k = 1` (no
speculation, no waste); high risk → `k > 1`.

### 3.2 Speculative fan-out (fork + diversify)
For a high-risk subtask the manager emits `k` **diverse** strategies up front
(distinct approaches/modalities — GUI vs CLI vs minimal — exactly what it
currently produces *across* replan rounds). Each branch runs in its own VM,
**forked from the same start snapshot** so branches are fully isolated and
non-interfering.

### 3.3 Grounded winner selection (not self-report)
The winner is the first branch to pass a **grounded completion check** — read
back the file, diff the accessibility tree, run a verifying shell command in
the VM — *not* the agent's prose claim. This is the critical coupling: PSE's
correctness depends on selecting by ground truth, which is precisely what
defeats false success.

A free by-product: if multiple diverse branches converge to the *same* grounded
end-state, we obtain a **consensus confidence signal**; divergence flags task
ambiguity/difficulty.

### 3.4 Squash & reclaim
Once a branch passes the check, the others are killed and their VMs reclaimed.
Snapshot isolation makes squashing side-effect-free.

### 3.5 Extension: DAG-level speculation
Speculatively launch a *downstream* subtask on the predicted-winning branch's
state **before** the upstream's verification completes; squash the downstream
if the upstream is later found to be a false success. This hides verification
latency across DAG edges (speculating past multiple branches).

## 4. Why Uniquely Enabled by This Setting

- **Snapshot-fork isolation.** Each speculative branch is an independent qcow2
  overlay forked from one start state — only a snapshottable VM environment
  supports clean, isolated, squashable speculation. Web/API agents cannot.
- **Abundant host compute.** Measured: 6 workers × 4 parallelism (~24 VMs) on a
  192-core host ran at load ≈ 8. Speculation trades idle compute for
  critical-path latency.
- **Multi-modal ground truth.** The VM exposes the filesystem, accessibility
  tree, process list, and shell — sub-second grounded checks that are far
  cheaper than the agent's own multi-step visual self-confirmation.
- **The manager already generates diverse strategies** (today, lazily across
  replan rounds). PSE re-times that diversity; it does not invent it.

## 5. Experimental Design

**Baselines:** (B0) no replan; (B1) sequential replanning (current MACU,
budgets 1 and 5).

**System:** PSE with grounded selection, swept over speculation width `k` and
risk threshold.

**Ablations:**
- selection by self-report vs grounded check (isolates the false-success fix),
- fixed `k` vs risk-adaptive `k` (isolates the branch predictor's value),
- strategy diversity on/off (random vs genuinely distinct approaches),
- PSE vs sequential replan at *matched total compute* (fair latency comparison).

**Metrics:**
1. Success rate (OSWorld evaluator).
2. **False-success rate** (agent self-reports done ∧ evaluator = 0).
3. Wall-clock latency per task (and on the critical path).
4. Total compute / tokens (the cost of squashed branches).
5. The **latency–compute Pareto frontier** as `k` varies.

**Benchmark:** the 36-task stratified OSWorld subset (and scale to full 369).

## 6. Hypotheses / Expected Results

- **H1 (latency):** On replan-heavy tasks, PSE reduces wall-clock by ≈ `1/k`
  relative to sequential replanning of width `k` (sum → max).
- **H2 (correctness):** Grounded winner selection lowers false-success rate vs
  self-report selection, raising success rate at fixed budget.
- **H3 (efficiency):** Risk-adaptive `k` dominates fixed `k` on the
  latency–compute Pareto frontier (speculate only where mispredict is likely).

## 7. Risks & Failure Modes

- **Correlated errors:** if all `k` branches make the same mistake (all false
  success), speculation does not help. Mitigation: enforce *real* strategy/
  modality diversity; rely on grounded (not visual) selection. Studying when
  diversity fails is itself a research question.
- **Grounded check coverage:** not every goal is cheaply machine-checkable.
  Fallback to adversarial/diverse-lens LLM verification where no programmatic
  check exists; quantify coverage.
- **Compute blow-up:** wide speculation can saturate the API token budget
  (measured key limit: 500k TPM / 120 RPM). The risk predictor must gate width;
  report the Pareto trade-off, not just best-case latency.
- **Selection latency:** the grounded check must be cheap relative to a CUA
  step or it negates the win — favor shell/a11y probes over extra LLM calls.

## 8. Related Work & Differentiation

- **Self-Refine / Reflexion / verifier models:** add a *sequential* verify-then-
  retry step. PSE differs by being *proactive and parallel*, and by selecting on
  *grounded environment state* rather than another LLM judgment.
- **Self-consistency / best-of-N:** samples `N` outputs and votes. PSE samples
  `N` *diverse executions* and selects by *grounded outcome*, with
  *risk-adaptive* `N` and *snapshot isolation* — not majority vote over text.
- **Speculative decoding:** speculation at the token level. PSE lifts the idea
  to the *task/agent* level, enabled by VM snapshot forks.
- **MACU:** introduces parallel decomposition for speed. PSE extends MACU's
  latency thesis *and* repairs its main weakness (false success) with one
  mechanism (grounded speculative selection).

## 9. Implementation Plan (grounded in this codebase)

1. **Grounded completion probe** — reuse the in-VM shell exec
   (`osworld/patches.py:_execute_shell_on_vm`) and accessibility-tree access to
   implement a cheap per-subtask "goal reached?" check.
2. **Speculative fan-out** — have the manager emit `k` diverse instructions up
   front for high-risk subtasks; fork a branch overlay per strategy via
   `_apptainer_create_overlay_from` (already used for init_from); run them
   through the existing parallel-subtask scheduler.
3. **Winner selection + squash** — first branch passing the grounded probe wins;
   terminate the rest and reclaim VMs (overlay isolation makes this clean).
4. **Risk predictor** — start with a heuristic (domain prior + step/struggle
   signals); calibrate on the existing replan1/replan5 runs.
5. **Evaluation harness** — extend `scripts/analyze_run.py` to log
   false-success rate, critical-path latency, and total compute per run.

## 10. Milestones

- **M1 — PoC:** on a handful of `multi_apps` tasks, manually run the manager's
  diverse strategies in parallel with grounded selection; confirm both
  false-success ↓ and latency ↓ (validates the core claim cheaply).
- **M2 — Integrated PSE:** the loop above wired into MACU behind a flag.
- **M3 — Full study:** baselines + ablations on the 36-task subset; produce the
  latency–compute Pareto frontier; scale to full OSWorld.

---

*Supporting data: `runs/subset36_replan{1,5}_summary.csv`,
`runs/replan_budget_comparison.md`; per-task DAG-evolution viewers at
`runs/<run>/<task_id>/analysis.html`.*
