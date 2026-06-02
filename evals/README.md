# evals

Offline evaluators for macu run directories. These scripts don't run
the harness — they read a completed (or in-progress) run dir and score
each task's saved trajectory.

## webjudge_eval.py

WebJudge (Online-Mind2Web) evaluator. An LLM judge reads the task text,
the agent's action history, and screenshots, then scores each task
`success` or `failure`. Adapted from
[OSU-NLP-Group/Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/methods/webjudge_online_mind2web.py).

### What it reads

For every `<run-dir>/<task_id>/` that contains:

- `final_results.json` — supplies `task_text` (falls back to `instruction`)
- `final_traj/traj.jsonl` — step-by-step action history
- `final_traj/*.png` — screenshots (sorted by the `step_<N>_` prefix)

Tasks missing any of the above are logged as `skipped` and do not count
toward the success rate.

### What it writes

- `<run-dir>/<task_id>/webjudge.json` — per-task verdict
  (`status`, `judge_response`, `key_points`, per-image records, token counts)
- `<run-dir>/webjudge_summary.json` — run-wide aggregate
  (per-status counts, `success_rate_over_scored`, token usage)

### Requirements

- `OPENAI_API_KEY` exported (or present in `.env` — source it before running)
- The harness venv, e.g. `.venv/bin/python` from the repository root

### Basic usage

```bash
set -a && source .env && set +a    # so OPENAI_API_KEY is exported
.venv/bin/python evals/webjudge_eval.py \
    --run-dir /path/to/macu/runs/online_m2w_run \
    --judge-model gpt-5.4-mini \
    --max-parallel 6
```

### Resuming / re-scoring

- **Default (no `--overwrite`):** any task dir that already has
  `webjudge.json` is served from the cached file. Safe to rerun on an
  actively growing run dir to score only the tasks that have completed
  since last time.
- **`--overwrite`:** re-score every task from scratch (burns tokens;
  use only if the judge prompt or model changed).

### Key flags

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--run-dir` | *(required)* | Directory containing per-task subdirs. |
| `--judge-model` | `gpt-5.4-mini` | Any OpenAI-compatible model id. |
| `--judge-base-url` | `https://api.openai.com/v1` | Override to hit another endpoint. Note: the script **does not** inherit `OPENAI_BASE_URL`, which may point at vllm in this repo. |
| `--score-threshold` | `3` | Per-image relevance cutoff (1–5). Images `>=` threshold are forwarded to the final judge prompt. |
| `--max-parallel` | `4` | Concurrent task evaluations. 6–8 is usually fine against OpenAI. |
| `--task-id ID` | *(repeatable)* | Evaluate only these task ids (comma-split by repeating). |
| `--overwrite` | off | Ignore existing `webjudge.json`. |
| `--max-tokens` | `8192` | Per-call completion budget. Reasoning models spend this on hidden reasoning too. |
| `--reasoning-effort` | `medium` | `low` / `medium` / `high`. Only used for reasoning-capable models (`gpt-5*`, `o*`); ignored otherwise. |

### Breaking down the result by level / domain

`webjudge_summary.json` has a flat list of per-task records. Join it
against your task manifest (e.g. `data/online_m2w/tasks.json`) to
slice by `level`, `website`, or any other metadata field. The summary
itself only reports `success_rate_over_scored` globally.

### Cost / tokens

Each task costs roughly one judge call per screenshot (for per-image
relevance scoring) plus one final verdict call. A 300-task online_m2w
run with `gpt-5.4-mini` is typically a few million input tokens and a
few hundred thousand output tokens. Check `webjudge_summary.json → usage`.

### Common failure modes

- **`OpenAIError: api_key client option must be set`** — `OPENAI_API_KEY`
  wasn't exported into the subprocess. Use `set -a; source .env; set +a`
  or `env OPENAI_API_KEY=... python evals/webjudge_eval.py ...`.
- **`[WARN] failed to parse image judgement`** — the judge output didn't
  match the expected `Score: N` format for one image. Non-fatal; that
  image is treated as score 0 and the task is still scored from the
  others.
- **All tasks showing `status: skipped`** — your run dir lacks
  `final_traj/` subdirs. Either the run is still warming up, or you
  pointed `--run-dir` at the wrong level.

## odysseys_eval.py

Odysseys full-trajectory, per-rubric evaluator. A multimodal judge gets
the task text, one rubric item, the full action history, and trajectory
screenshots, then marks that rubric `success` or `failure`. The task
score is the mean of its rubric scores, and `perfect` means every rubric
passed. Adapted from
[ljang0/Odysseys](https://github.com/ljang0/Odysseys/blob/main/scripts/python/run_full_trajectory_per_rubric.py).

### What it reads

`--runs-dir` can point at either one task run or a parent run directory.
The script supports:

- `steps.jsonl` or `traj.jsonl` directly in the run dir
- `<run-dir>/<task_id>/final_traj/traj.jsonl` or `steps.jsonl`
- `<run-dir>/<task_id>/final_results.json` for webjudge-style completed
  task dirs and as a fallback source for task text

Rubrics normally come from `--task-source-json`. If omitted, the script
walks up from `--runs-dir` looking for an `args.json` with
`mind2web_tasks_path`.

The task source JSON should be a list of task objects with:

- `task_id`
- `confirmed_task` or `task`
- `level` (optional; defaults to `unknown`)
- `rubrics`, a mapping from rubric id to an object containing
  `requirement` and optionally `verification`

By default only completed runs are scored. Completion is detected by a
numeric `result.txt` or, for webjudge-style dirs, a `final_results.json`.
Use `--include-incomplete` to score discovered trajectories anyway.

### What it writes

- `<runs-dir>/eval_results_full_traj_per_rubric.json` by default, or the
  path passed with `--output`

The output has:

- `summary` — total tasks/rubrics, mean per-task rubric score,
  `perfect_task_rate`, errored task count, token/cost totals, and
  optional `by_level` breakdown
- `tasks` — one record per run with `rubric_scores`, `rubric_results`,
  `average_rubric_score`, `perfect`, `num_steps`,
  `num_screenshots_sent`, `judge_model`, token counts, and `cost_usd`

The visualizer reads this file from the run-dir root and uses
`average_rubric_score` plus the per-rubric breakdown.

### Requirements

- `GEMINI_API_KEY` / `GOOGLE_API_KEY` for Gemini models, or
  `OPENAI_API_KEY` for non-Gemini models
- Dependencies from `requirements.txt` (`google-genai`, `openai`,
  `tqdm`)
- Optional `.env` loading requires `python-dotenv`; otherwise source
  `.env` before running or pass keys through the environment

### Basic usage

```bash
set -a && source .env && set +a
.venv/bin/python evals/odysseys_eval.py \
    --runs-dir /path/to/macu/runs/odysseys_run \
    --task-source-json data/odysseys/tasks.json \
    --model gemini-3.1-flash-lite-preview \
    --num-workers 8 \
    --max-concurrent-rubrics 2
```

To use an OpenAI-compatible judge instead:

```bash
.venv/bin/python evals/odysseys_eval.py \
    --runs-dir /path/to/run \
    --task-source-json /path/to/tasks.json \
    --model gpt-5.4-mini \
    --api-base https://api.openai.com/v1
```

### Resuming / re-scoring

- **Default:** if the output file already contains a successful scored
  record for a run dir, that run is served from the cached result.
- **Re-score:** write to a new `--output` path, or remove the relevant
  task records/output file before rerunning.

### Key flags

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--runs-dir` | *(required)* | Single task dir or parent run dir to discover. |
| `--task-source-json` | auto-detect from `args.json` | JSON task metadata with `rubrics`. Required in practice for rubric scoring. |
| `--output` | `<runs-dir>/eval_results_full_traj_per_rubric.json` | Aggregate result path. |
| `--model` | `gemini-3.1-flash-lite-preview` | Gemini models use Gemini APIs; all others use OpenAI chat completions. |
| `--api-base` | unset | OpenAI-compatible base URL for non-Gemini models. |
| `--api-key` | `OPENAI_API_KEY` | Explicit OpenAI-compatible API key. |
| `--gemini-api-key` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Explicit Gemini API key. |
| `--env-file` | `.env` | Load keys from a dotenv file if `python-dotenv` is installed. |
| `--max-images` | `200` | Sends the last N screenshots; `0` sends all. |
| `--max-steps` | `100` | Ignores trajectory rows with `step_num` above this; `0` is unlimited. |
| `--num-workers` | `1` | Concurrent run evaluations. |
| `--max-concurrent-rubrics` | `1` | Concurrent rubric judge calls per run. Total judge concurrency is roughly this times `--num-workers`. |
| `--include-incomplete` | off | Score runs without numeric `result.txt` / `final_results.json`. |

### Cost / tokens

Each rubric item is a separate multimodal judge call, so the cost scales
with `tasks × rubrics × screenshots sent`. Token counts are recorded per
task and summarized globally. `cost_usd` is currently only populated for
models listed in `PRICING_USD_PER_M_TOKENS`; unknown models still report
tokens but show zero cost.

### Common failure modes

- **`Gemini API key required` / `OpenAI API key required`** — export the
  correct key for the selected `--model`, or pass it explicitly.
- **`No rubric found for task_id=...`** — the task id inferred from the
  run directory is missing from `--task-source-json`, or the task source
  lacks a `rubrics` mapping.
- **`No completed runs found`** — the run dirs do not have numeric
  `result.txt` files or `final_results.json`; pass `--include-incomplete`
  if you want to score partial runs.
- **Unexpected low scores from missing end state** — increase
  `--max-steps` or `--max-images` if the decisive screenshot/action is
  being filtered out.
