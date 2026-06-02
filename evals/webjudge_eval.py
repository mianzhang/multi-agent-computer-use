"""WebJudge (Online-Mind2Web) evaluator adapted for our run directories.

Adapted from:
  https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/methods/webjudge_online_mind2web.py

Usage:
    python evals/webjudge_eval.py \\
        --run-dir runs/online_m2w_easy_nomgr_20260420_121317 \\
        --judge-model gpt-5.4-mini

Per-task inputs are discovered as:
    <run-dir>/<task_id>/final_results.json        -> task_text
    <run-dir>/<task_id>/final_traj/traj.jsonl     -> action history
    <run-dir>/<task_id>/final_traj/*.png          -> screenshots (by step)

Per-task output: <task_dir>/webjudge.json  ({status, score, key_points, record}).
Aggregate output: <run-dir>/webjudge_summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

MAX_IMAGE = 50
ACTION_LINE_RE = re.compile(r"^\s*Action:\s*(.+?)(?:\n\n|<tool_call>|$)", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(image: Image.Image) -> str:
    """PIL image -> base64-encoded JPEG string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_action_text(entry: dict) -> str:
    """Turn a traj.jsonl entry into a short human-readable action description."""
    resp = entry.get("response", "") or ""
    if isinstance(resp, str):
        m = ACTION_LINE_RE.search(resp)
        if m:
            return " ".join(m.group(1).split())
    action = entry.get("action")
    if isinstance(action, str):
        return action.strip() or "(no action)"
    if isinstance(action, dict):
        cmd = action.get("command") or action.get("action_type") or "action"
        return f"[{cmd}]"
    return "(no action)"


def _screenshot_step_idx(name: str) -> int:
    m = re.search(r"step_(\d+)_", name)
    return int(m.group(1)) if m else 10**9


def load_task_inputs(task_dir: Path) -> Optional[tuple[str, list[str], list[Path]]]:
    """Return (task_text, action_history, ordered_screenshots) or None if unusable."""
    fr = task_dir / "final_results.json"
    traj = task_dir / "final_traj" / "traj.jsonl"
    if not fr.is_file() or not traj.is_file():
        return None
    data = json.loads(fr.read_text())
    task_text = data.get("task_text") or data.get("instruction")
    if not task_text:
        return None

    actions: list[str] = []
    referenced: list[str] = []
    for line in traj.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("action") == "SUMMARY":
            continue
        actions.append(extract_action_text(entry))
        sf = entry.get("screenshot_file")
        if sf:
            referenced.append(sf)

    final_traj_dir = task_dir / "final_traj"
    images = sorted(final_traj_dir.glob("*.png"), key=lambda p: _screenshot_step_idx(p.name))
    return task_text, actions, images


# ---------------------------------------------------------------------------
# Judge model wrapper (OpenAI-compatible chat API)
# ---------------------------------------------------------------------------

@dataclass
class JudgeCall:
    input_tokens: int = 0
    output_tokens: int = 0


class JudgeModel:
    """Minimal OpenAI-compatible client with a `.generate(messages)` method.

    Intentionally does not inherit OPENAI_BASE_URL from the env (which may
    point at vllm in this repo). Pass `base_url` explicitly to override, or
    leave None to hit the official OpenAI endpoint.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        reasoning_effort: Optional[str] = "medium",
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com/v1",
        )
        self.usage = JudgeCall()

    def _is_reasoning_model(self) -> bool:
        return bool(re.match(r"^(o\d|gpt-5)", self.model))

    def generate(self, messages) -> list[str]:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
        )
        # Reasoning models (o-series, gpt-5*) don't accept temperature other than 1;
        # forward reasoning_effort instead.
        if self._is_reasoning_model():
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage.input_tokens += resp.usage.prompt_tokens or 0
            self.usage.output_tokens += resp.usage.completion_tokens or 0
        return [c.message.content for c in resp.choices]


# ---------------------------------------------------------------------------
# WebJudge — adapted verbatim (prompts unchanged)
# ---------------------------------------------------------------------------

async def identify_key_points(task: str, model: JudgeModel) -> str:
    system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": f"Task: {task}"}]},
    ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def judge_image(task: str, image_path: Path, key_points: str, model: JudgeModel) -> str:
    system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.
- Does the image show actions, progress indicators, or critical information directly related to completing the task?
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.
- **Score**: Assign a score based on the reasoning, using the following scale:
    - **1**: The image does not contain any necessary steps or relevant information.
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:
1. **Reasoning**: [Your explanation]
2. **Score**: [1-5]"""

    jpg_b64 = encode_image(Image.open(image_path))
    text = f"""**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg_b64}", "detail": "high"}},
            ],
        },
    ]
    responses = await asyncio.to_thread(model.generate, messages)
    return responses[0]


async def WebJudge_Online_Mind2Web_eval(
    task: str,
    last_actions: list[str],
    images_path: list[Path],
    model: JudgeModel,
    score_threshold: int,
):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""

    key_points = await identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except Exception:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    judge_tasks = [judge_image(task, p, key_points, model) for p in images_path]
    image_responses = await asyncio.gather(*judge_tasks)

    whole_content_img: list[dict] = []
    whole_thoughts: list[str] = []
    record: list[dict] = []
    pattern = r"[1-5]"
    for response, image_path in zip(image_responses, images_path):
        score = 0
        thought = ""
        try:
            score_text = response.split("Score")[1]
            thought = (
                response.split("**Reasoning**:")[-1].strip().lstrip("\n").split("\n\n")[0].replace("\n", " ")
            )
            score = int(re.findall(pattern, score_text)[0])
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] failed to parse image judgement: {e}", file=sys.stderr)
        record.append({"image": image_path.name, "response": response, "score": score})

        if score >= score_threshold:
            jpg_b64 = encode_image(Image.open(image_path))
            whole_content_img.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{jpg_b64}", "detail": "high"}}
            )
            if thought:
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[:MAX_IMAGE]
    whole_thoughts = whole_thoughts[:MAX_IMAGE]

    if len(whole_content_img) == 0:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""
        text = prompt.format(
            task=task,
            last_actions="\n".join(f"{i+1}. {a}" for i, a in enumerate(last_actions)),
            key_points=key_points,
        )
    else:
        text = prompt.format(
            task=task,
            last_actions="\n".join(f"{i+1}. {a}" for i, a in enumerate(last_actions)),
            key_points=key_points,
            thoughts="\n".join(f"{i+1}. {t}" for i, t in enumerate(whole_thoughts)),
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [{"type": "text", "text": text}] + whole_content_img},
    ]
    return messages, text, system_msg, record, key_points


def parse_judge_status(response: str) -> str:
    """Return 'success' / 'failure' / 'unknown'."""
    try:
        tail = response.lower().split("status:")[1]
        if "success" in tail:
            return "success"
        if "failure" in tail:
            return "failure"
    except IndexError:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

async def evaluate_task(task_dir: Path, model: JudgeModel, score_threshold: int) -> dict:
    inputs = load_task_inputs(task_dir)
    if inputs is None:
        return {"task_id": task_dir.name, "status": "skipped", "reason": "missing final_results.json or traj.jsonl"}
    task_text, last_actions, images_path = inputs
    if not images_path:
        return {"task_id": task_dir.name, "status": "skipped", "reason": "no screenshots"}

    messages, judge_prompt, _system_msg, record, key_points = await WebJudge_Online_Mind2Web_eval(
        task_text, last_actions, images_path, model, score_threshold
    )
    judge_response = (await asyncio.to_thread(model.generate, messages))[0]
    status = parse_judge_status(judge_response)
    return {
        "task_id": task_dir.name,
        "status": status,
        "judge_response": judge_response,
        "judge_prompt": judge_prompt,
        "key_points": key_points,
        "image_records": record,
        "num_images_considered": len(images_path),
        "num_images_kept": sum(1 for r in record if r["score"] >= score_threshold),
        "num_actions": len(last_actions),
    }


async def _run(run_dir: Path, task_filter: Optional[set[str]], model: JudgeModel,
               score_threshold: int, max_parallel: int, overwrite: bool) -> dict:
    task_dirs = [d for d in run_dir.iterdir() if d.is_dir() and (d / "final_results.json").exists()]
    if task_filter:
        task_dirs = [d for d in task_dirs if d.name in task_filter]
    task_dirs.sort(key=lambda p: p.name)

    print(f"[info] evaluating {len(task_dirs)} task(s) from {run_dir}", file=sys.stderr)

    sem = asyncio.Semaphore(max_parallel)

    async def _one(task_dir: Path):
        out_path = task_dir / "webjudge.json"
        if out_path.exists() and not overwrite:
            return json.loads(out_path.read_text())
        async with sem:
            try:
                result = await evaluate_task(task_dir, model, score_threshold)
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                result = {"task_id": task_dir.name, "status": "error",
                          "reason": f"{type(e).__name__}: {e}", "traceback": tb}
        if result.get("status") in ("success", "failure", "unknown"):
            persisted = {k: v for k, v in result.items() if k != "judge_prompt"}
            out_path.write_text(json.dumps(persisted, indent=2))
        print(f"[{result.get('status')}] {task_dir.name}", file=sys.stderr)
        return result

    results = await asyncio.gather(*[_one(d) for d in task_dirs])

    per_status: dict[str, int] = {}
    for r in results:
        per_status[r.get("status", "unknown")] = per_status.get(r.get("status", "unknown"), 0) + 1

    scored = [r for r in results if r.get("status") in ("success", "failure")]
    success_rate = (
        sum(1 for r in scored if r["status"] == "success") / len(scored) if scored else 0.0
    )

    summary = {
        "run_dir": str(run_dir),
        "judge_model": model.model,
        "score_threshold": score_threshold,
        "total": len(results),
        "per_status": per_status,
        "success_rate_over_scored": success_rate,
        "usage": {"input_tokens": model.usage.input_tokens, "output_tokens": model.usage.output_tokens},
        "tasks": [{k: v for k, v in r.items() if k not in ("judge_prompt", "image_records", "judge_response", "key_points")} for r in results],
    }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path, help="Directory containing per-task subdirectories")
    p.add_argument("--judge-model", default="gpt-5.4-mini", help="OpenAI-compatible model id for the judge (default: gpt-5.4-mini)")
    p.add_argument("--judge-base-url", default=None, help="Override judge API base URL (default: https://api.openai.com/v1)")
    p.add_argument("--score-threshold", type=int, default=3, help="Keep images scored >= this (1-5, default 3)")
    p.add_argument("--max-parallel", type=int, default=4, help="Max concurrent task evaluations (default: 4)")
    p.add_argument("--task-id", action="append", default=None, help="Evaluate only this task id (repeatable)")
    p.add_argument("--overwrite", action="store_true", help="Re-evaluate tasks that already have webjudge.json")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Max completion tokens per judge call (reasoning models also spend this budget on hidden reasoning tokens)")
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"],
                   help="reasoning_effort for reasoning-capable models (gpt-5*, o-series); ignored otherwise")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Run dir does not exist: {args.run_dir}")

    model = JudgeModel(
        model=args.judge_model,
        base_url=args.judge_base_url,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    task_filter = set(args.task_id) if args.task_id else None
    summary = asyncio.run(_run(args.run_dir, task_filter, model, args.score_threshold, args.max_parallel, args.overwrite))

    summary_path = args.run_dir / "webjudge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Summary -> {summary_path}")
    print(f"Total: {summary['total']}   Breakdown: {summary['per_status']}")
    if summary["per_status"].get("success", 0) + summary["per_status"].get("failure", 0) > 0:
        print(f"Success rate (over scored): {summary['success_rate_over_scored']:.3f}")
    print(f"Tokens: in={summary['usage']['input_tokens']} out={summary['usage']['output_tokens']}")


if __name__ == "__main__":
    main()
