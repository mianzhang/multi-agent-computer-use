"""Build a standalone HTML visualizer from a MACU run directory.

Given a run dir like ``runs/cafes_near_cmu`` (the ``--result-dir`` passed to
``run_macu.py``), produce a self-contained ``visualization/`` folder with an
``index.html`` you can open directly in a browser. The output mirrors the
animated player on the MACU website: a gantt chart with playback controls,
a live DAG with screenshot thumbnails on each node, a screenshot strip per
subagent, and the aggregator's final response.

Usage:
    python scripts/visualize_run.py runs/cafes_near_cmu

    # custom output dir
    python scripts/visualize_run.py runs/cafes_near_cmu --output-dir my_viewer

    # cap keyframes per subtask (default 16)
    python scripts/visualize_run.py runs/cafes_near_cmu --max-frames 24

Open ``runs/cafes_near_cmu/visualization/index.html`` in a browser. All
assets (CSS, JS, screenshots, the run JSON) are local to that folder; the
only network requests are to unpkg for React + Babel.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ImportError:
    print("error: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "_visualizer_assets"

MAX_WIDTH = 1280
JPG_QUALITY = 82
DEFAULT_MAX_FRAMES = 16

# Suffixes that indicate a "variant" of a base subtask (a sibling, not new work).
# Used to derive scheduling — variants start when their parent ends.
VARIANT_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)(_retry|_variant_.*|_continue|_fresh|_direct_url|_simple_search|_curl)$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory / screenshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def step_sort_key(p: Path) -> int:
    m = re.search(r"step_(\d+)_", p.name)
    return int(m.group(1)) if m else 0


# Action timestamps in screenshot filenames look like 20260516@165648995789
# (YYYYMMDD@HHMMSS + 6-digit microseconds).
_ACTION_TS_RE = re.compile(r"step_\d+_(\d{8}@\d{6,12})")


def parse_action_timestamp(filename: str) -> Optional[float]:
    """Return the screenshot's wall-clock timestamp as a Unix epoch float,
    or None if the filename doesn't match the expected pattern."""
    m = _ACTION_TS_RE.search(filename)
    if not m:
        return None
    raw = m.group(1)
    # strptime's %f accepts 1-6 microsecond digits; some logs append extra ms,
    # so truncate to 6 after the second-precision part.
    date_part, time_part = raw.split("@", 1)
    time_part = time_part[:12]  # HHMMSS + up to 6 micros
    try:
        dt = datetime.strptime(f"{date_part}@{time_part}", "%Y%m%d@%H%M%S%f")
    except ValueError:
        return None
    return dt.timestamp()


def first_step_timestamp(task_dir: Path, subtask: str) -> Optional[float]:
    """Earliest action timestamp across all screenshots of a subtask, or None
    if no screenshots are available."""
    sd = find_screenshot_dir(task_dir, subtask)
    if sd is None:
        return None
    earliest: Optional[float] = None
    for p in sd.glob("step_*.png"):
        ts = parse_action_timestamp(p.name)
        if ts is None:
            continue
        if earliest is None or ts < earliest:
            earliest = ts
    return earliest


def sample_evenly(items, n):
    if n >= len(items):
        return list(items)
    if n <= 1:
        return [items[0]]
    out = []
    step = (len(items) - 1) / (n - 1)
    for i in range(n):
        idx = min(int(round(i * step)), len(items) - 1)
        out.append(items[idx])
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def find_screenshot_dir(task_dir: Path, subtask: str) -> Optional[Path]:
    """Walk ``<task_dir>/<subtask>/results/pyautogui/screenshot/...`` to find the
    deepest directory containing ``step_*.png``."""
    base = task_dir / subtask / "results" / "pyautogui" / "screenshot"
    if not base.exists():
        return None
    for path in base.rglob("step_*.png"):
        return path.parent
    return None


def _clean_action_text(d: dict) -> str:
    """Pull a readable one-liner from a trajectory step. The model's response
    typically starts with prose followed by XML/tool-call dumps; we keep the
    prose. If absent, fall back to the raw pyautogui call."""
    response = (d.get("response") or "").strip()
    head = response.split("<", 1)[0].strip()
    if not head:
        head = response.split("\n", 1)[0].strip()
    for prefix in ("Action:", "ACTION:", "action:"):
        if head.startswith(prefix):
            head = head[len(prefix):].strip()
            break
    if head:
        return head
    raw_action = d.get("action") or {}
    if isinstance(raw_action, dict):
        raw = (raw_action.get("action") or "").strip()
    else:
        raw = str(raw_action).strip()
    if not raw:
        return ""
    return raw.splitlines()[0].strip()


def load_traj(screenshot_dir: Path) -> dict:
    """Parse ``traj.jsonl`` next to the screenshots into ``{step: {reasoning, action}}``."""
    traj_path = screenshot_dir / "traj.jsonl"
    if not traj_path.exists():
        return {}
    out: dict[int, dict] = {}
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = d.get("step_num")
            if step is None:
                continue
            # Keep the first response we see for each step (later batched
            # actions in the same step reuse the same reasoning).
            if int(step) in out and out[int(step)]["reasoning"]:
                continue
            out[int(step)] = {
                "reasoning": (d.get("response") or "").strip(),
                "action": _clean_action_text(d),
            }
    return out


def keyframe_budget(duration_seconds: float, actual_frames: int, max_frames: int) -> int:
    if actual_frames == 0:
        return 0
    target = math.ceil(duration_seconds / 120) + 4
    return min(actual_frames, target, max_frames)


def process_subtask_frames(
    task_dir: Path,
    subtask: str,
    duration: float,
    out_img_dir: Path,
    max_frames: int,
) -> list[dict]:
    sd = find_screenshot_dir(task_dir, subtask)
    if sd is None:
        return []
    screenshots = sorted(sd.glob("step_*.png"), key=step_sort_key)
    if not screenshots:
        return []
    n_keep = keyframe_budget(duration, len(screenshots), max_frames)
    if n_keep == 0:
        return []
    print(f"  {subtask}: {len(screenshots)} screenshots → keeping {n_keep}")
    keep = sample_evenly(screenshots, n_keep)
    traj = load_traj(sd)

    out_dir = out_img_dir / subtask
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    manifest = []
    for i, src in enumerate(keep):
        step = step_sort_key(src)
        out_name = f"{i:02d}_step{step:02d}.jpg"
        out_path = out_dir / out_name
        img = Image.open(src).convert("RGB")
        w, h = img.size
        if w > MAX_WIDTH:
            new_h = int(h * MAX_WIDTH / w)
            img = img.resize((MAX_WIDTH, new_h), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=JPG_QUALITY, optimize=True)
        info = traj.get(step, {})
        manifest.append({
            "src": f"static/images/screenshots/{subtask}/{out_name}",
            "step": step,
            "frame": i,
            "reasoning": info.get("reasoning", ""),
            "action": info.get("action", ""),
        })
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Label / schedule derivation (adapted from macu-website scripts/process_all.py)
# ─────────────────────────────────────────────────────────────────────────────

def split_variant(subtask_id: str, all_ids: set[str]) -> Optional[str]:
    m = VARIANT_SUFFIX_RE.match(subtask_id)
    if m:
        base = m.group("base")
        if base in all_ids and base != subtask_id:
            return base
    return None


_LABEL_STOPWORDS = {
    "a", "an", "and", "or", "the", "for", "with", "from", "into", "of", "to",
    "your", "open", "find", "go", "first", "then", "all", "their", "this",
    "that", "page", "site", "data", "extract", "navigate", "scroll", "click",
    "search", "subtask", "make", "use", "using", "via",
}

_SUFFIX_DISPLAY = {
    "retry": "retry", "variant": "variant", "continue": "continue",
    "fresh": "fresh", "direct": "direct", "simple": "simple", "url": "url",
    "cache": "cache", "curl": "curl", "yelp": "yelp", "google": "google",
}


def _humanize_id(subtask_id: str) -> tuple[Optional[str], Optional[str]]:
    sid = subtask_id
    if sid.startswith("subtask_"):
        rest = sid[len("subtask_"):]
        m = re.match(r"^(\d+)(_(.*))?$", rest)
        if m:
            return ("Subtask " + m.group(1), m.group(3))
        parts = rest.split("_")
        for i in range(len(parts), 0, -1):
            sfx = "_".join(parts[i - 1:])
            for k in _SUFFIX_DISPLAY:
                if sfx == k or sfx.startswith(k + "_") or sfx.endswith("_" + k):
                    base = "_".join(parts[:i - 1])
                    if base:
                        return (base.replace("_", " ").title(), sfx.replace("_", " "))
        return (rest.replace("_", " ").title(), None)
    parts = sid.split("_")
    for i in range(len(parts), 0, -1):
        sfx = "_".join(parts[i - 1:])
        for k in _SUFFIX_DISPLAY:
            if sfx == k or sfx.endswith("_" + k):
                base = "_".join(parts[:i - 1])
                if base:
                    return (base.replace("_", " ").title(), sfx.replace("_", " "))
    return (sid.replace("_", " ").title(), None)


def _first_meaningful_word(desc: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", desc or "")
    for w in words:
        if w.lower() not in _LABEL_STOPWORDS:
            return w
    return words[0] if words else ""


def derive_short_label(subtask_id: str, description: Optional[str]) -> str:
    if subtask_id == "final_aggregation":
        return "Aggregate\n& report"
    desc = (description or "").strip()
    head, suffix = _humanize_id(subtask_id)
    if head is None:
        if desc:
            w1 = _first_meaningful_word(desc)
            rest = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", desc)
            w2 = ""
            for w in rest:
                if w == w1 or w.lower() in _LABEL_STOPWORDS:
                    continue
                w2 = w
                break
            return f"{w1.capitalize()}\n{w2.capitalize()}".strip()
        return subtask_id.replace("_", " ").title()
    head_label = head
    line2 = ""
    if suffix:
        line2 = f"({suffix})"
        if head.startswith("Subtask ") and desc:
            word = _first_meaningful_word(desc)
            if word:
                head_label = f"{head} {word.capitalize()}"
    elif head.startswith("Subtask ") and desc:
        word = _first_meaningful_word(desc)
        line2 = word.capitalize() if word else ""
    elif " " in head and head.count(" ") <= 2:
        words = head.split()
        head_label = words[0]
        line2 = " ".join(words[1:])
    if line2:
        return f"{head_label}\n{line2}"
    return head_label


def derive_starts(
    subtask_order: list[str],
    timings: dict[str, dict],
    initial_subtasks: list[dict],
    snapshots_raw: list[dict],
    task_dir: Path,
) -> dict[str, float]:
    """Reconstruct simulated start times for each subtask.

    Prefers actual wall-clock timestamps embedded in screenshot filenames
    (action_timestamp = YYYYMMDD@HHMMSS......). This correctly places
    subtasks added mid-flight by spare-capacity replans, which the
    label-based heuristic would otherwise schedule after their parent
    ends. Falls back to dependency-derived timing for any subtask without
    parseable timestamps (e.g., one that never produced a screenshot).
    """
    all_ids = set(subtask_order)
    initial_ids = {s["id"] for s in initial_subtasks}
    initial_deps = {s["id"]: list(s.get("dependencies", [])) for s in initial_subtasks}
    durations = {sid: float(timings.get(sid, {}).get("duration_seconds", 0)) for sid in subtask_order}

    # Pass 1: pull a real start timestamp from each subtask's screenshots.
    real_ts: dict[str, float] = {}
    for sid in subtask_order:
        ts = first_step_timestamp(task_dir, sid)
        if ts is not None:
            real_ts[sid] = ts
    starts: dict[str, float] = {}
    if real_ts:
        zero = min(real_ts.values())
        for sid, ts in real_ts.items():
            starts[sid] = ts - zero

    # Pass 2: dependency-derived fallback for any subtask whose start is
    # still unknown (no screenshots, or unparseable filenames).
    intro_event: dict[str, str] = {}
    for snap in snapshots_raw:
        snap_label = snap.get("replan_label", "")
        for s in snap["dependency_graph"]["subtasks"]:
            if s["id"] not in intro_event:
                intro_event[s["id"]] = snap_label

    def end_of(sid: str) -> float:
        return starts.get(sid, 0.0) + durations.get(sid, 0.0)

    # Initial-snapshot subtasks: schedule by deps.
    changed = True
    while changed:
        changed = False
        for sid in subtask_order:
            if sid in starts or sid not in initial_ids:
                continue
            deps = initial_deps.get(sid, [])
            if all(d in starts for d in deps):
                starts[sid] = max((end_of(d) for d in deps), default=0.0)
                changed = True

    # Variant subtasks introduced later: align to their parent's end.
    for sid in subtask_order:
        if sid in starts:
            continue
        label = intro_event.get(sid, "")
        parent: Optional[str] = None
        if label.startswith("after_"):
            cand = label[len("after_"):]
            if cand in starts:
                parent = cand
        if parent is None:
            base = split_variant(sid, all_ids)
            if base and base in starts:
                parent = base
        starts[sid] = end_of(parent) if parent else 0.0
    return starts


def ensure_subtask_order(
    timings: dict,
    initial_subtasks: list[dict],
    snapshots_raw: list[dict],
) -> list[str]:
    initial_ids = [s["id"] for s in initial_subtasks]
    order = list(initial_ids)
    seen = set(order)
    for snap in snapshots_raw:
        for s in snap["dependency_graph"]["subtasks"]:
            sid = s["id"]
            if sid in seen:
                continue
            if sid in timings:
                order.append(sid)
                seen.add(sid)
    for sid in timings:
        if sid == "final_aggregation":
            continue
        if sid not in seen:
            order.append(sid)
            seen.add(sid)
    return order


def resolve_snapshot_times(
    snapshots_raw: list[dict],
    starts: dict[str, float],
    ends: dict[str, float],
) -> list[Optional[float]]:
    """Pin each snapshot to a moment on the simulated timeline.

    Strategy, in order of preference:
      1. ``initial`` → t=0.
      2. If the snapshot introduces new subtasks (vs the previous snapshot),
         place it at the earliest start time of those new subtasks. This
         keeps the DAG view in sync with the gantt chart even for replans
         that fire mid-flight (e.g., spare-capacity variants).
      3. Otherwise, the label-based heuristic: ``after_<sub_id>`` →
         ``ends[sub_id]``. Pure dependency-rewiring replans (no nodes
         added/removed) fall through to this branch.
    """
    times: list[Optional[float]] = [None] * len(snapshots_raw)
    prev_ids: set[str] = set()
    for i, snap in enumerate(snapshots_raw):
        lbl = snap.get("replan_label", "")
        cur_ids = {s["id"] for s in snap["dependency_graph"]["subtasks"]}
        added = cur_ids - prev_ids
        if lbl == "initial":
            times[i] = 0.0
        elif added:
            added_starts = [starts[a] for a in added if a in starts]
            if added_starts:
                times[i] = min(added_starts)
        if times[i] is None and lbl.startswith("after_"):
            sub_id = lbl[len("after_"):]
            if sub_id in ends:
                times[i] = ends[sub_id]
        prev_ids = cur_ids

    # Disambiguate snapshots that share a timestamp by spreading them out by
    # tiny offsets in iteration order, so the player walks them deterministically.
    counts: dict[float, int] = {}
    for i, t in enumerate(times):
        if t is None:
            continue
        counts[t] = counts.get(t, 0)
        times[i] = t + 0.01 * counts[t]
        counts[t] += 1
    return times


def clean_aggregator_deps(snapshots_out: list[dict], scheduled_ids: set[str]) -> None:
    last_valid: Optional[list[str]] = None
    for i, snap in enumerate(snapshots_out):
        agg = snap.get("aggregation")
        if not agg:
            continue
        raw = list(agg.get("dependencies", []))
        cleaned = [d for d in raw if d in scheduled_ids]
        if cleaned:
            agg["dependencies"] = cleaned
            last_valid = cleaned
        elif i == 0:
            agg["dependencies"] = raw
        else:
            agg["dependencies"] = last_valid or raw
    if snapshots_out and snapshots_out[-1].get("aggregation"):
        agg = snapshots_out[-1]["aggregation"]
        agg["dependencies"] = [d for d in agg["dependencies"] if d in scheduled_ids]
        if not agg["dependencies"] and last_valid:
            agg["dependencies"] = last_valid


def filter_snapshot_subtasks(snapshots_out: list[dict], scheduled_ids: set[str]) -> None:
    for i, snap in enumerate(snapshots_out):
        if i == 0:
            continue
        snap["subtasks"] = [s for s in snap["subtasks"] if s["id"] in scheduled_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def find_task_dir(run_dir: Path) -> Path:
    """Locate the inner task directory inside a ``--result-dir`` output.

    run_macu.py writes results to ``<result_dir>/<task_id>/final_results.json``.
    If ``run_dir`` itself contains ``final_results.json``, use it directly.
    Otherwise, prefer a subdirectory whose name matches ``run_dir.name``;
    fall back to the only subdirectory containing ``final_results.json``.
    """
    if (run_dir / "final_results.json").exists():
        return run_dir
    candidates = [
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "final_results.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No final_results.json found in {run_dir} or any subdirectory. "
            "Has the run completed?"
        )
    same_name = [d for d in candidates if d.name == run_dir.name]
    if same_name:
        return same_name[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Multiple task dirs found under {run_dir}: {[d.name for d in candidates]}. "
        "Pass the inner one explicitly."
    )


def process_run(task_dir: Path, out_dir: Path, max_frames: int) -> dict:
    final_results = json.loads((task_dir / "final_results.json").read_text())
    snapshots_raw = [
        json.loads(p.read_text())
        for p in sorted((task_dir / "graph_snapshots").glob("*.json"))
    ]
    if not snapshots_raw:
        raise RuntimeError(f"No graph_snapshots/*.json in {task_dir}")

    task_text = final_results.get("task_text") or final_results.get("instruction") or ""
    timings = final_results.get("task_timings", {})
    subtask_results = final_results.get("subtask_results", {})
    final_response = final_results.get("final_response", "")
    task_id = final_results.get("task_id", task_dir.name)
    cost_usd = final_results.get("total_cost_usd")

    initial_snap = snapshots_raw[0]
    initial_subtasks = initial_snap["dependency_graph"]["subtasks"]

    subtask_order = ensure_subtask_order(timings, initial_subtasks, snapshots_raw)
    # Drop subtasks that were cancelled before running (0s duration, [CANCELLED] response).
    keep_order = []
    for sid in subtask_order:
        dur = float(timings.get(sid, {}).get("duration_seconds", 0))
        resp = subtask_results.get(sid, {}).get("response", "")
        if dur <= 0 and "[CANCELLED]" in resp:
            print(f"  skip cancelled subtask: {sid}")
            continue
        keep_order.append(sid)
    subtask_order = keep_order

    out_img_dir = out_dir / "static" / "images" / "screenshots"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {task_id} from {task_dir}")
    subtask_frames = {}
    for sid in subtask_order:
        dur = float(timings.get(sid, {}).get("duration_seconds", 0))
        subtask_frames[sid] = process_subtask_frames(task_dir, sid, dur, out_img_dir, max_frames)

    starts = derive_starts(subtask_order, timings, initial_subtasks, snapshots_raw, task_dir)
    ends = {
        sid: starts[sid] + float(timings.get(sid, {}).get("duration_seconds", 0))
        for sid in subtask_order
    }
    # Add 30s of post-roll for the aggregator step.
    total_seconds = (max(ends.values()) + 30.0) if ends else 30.0

    schedule = [
        {"id": sid, "start": float(starts[sid]), "end": float(ends[sid])}
        for sid in subtask_order
    ]

    description_lookup = {}
    for snap in snapshots_raw:
        for s in snap["dependency_graph"]["subtasks"]:
            description_lookup.setdefault(s["id"], s.get("description", ""))

    short_labels = {sid: derive_short_label(sid, description_lookup.get(sid)) for sid in subtask_order}
    short_labels["final_aggregation"] = "Aggregate\n& report"

    subtasks_out = []
    for sid in subtask_order:
        res = subtask_results.get(sid, {})
        subtasks_out.append({
            "id": sid,
            "label": short_labels[sid],
            "duration_seconds": float(timings.get(sid, {}).get("duration_seconds", 0)),
            "status": res.get("status", "unknown"),
            "response": res.get("response", ""),
            "frames": subtask_frames.get(sid, []),
        })

    subtasks_out.append({
        "id": "final_aggregation",
        "label": short_labels["final_aggregation"],
        "duration_seconds": float(timings.get("final_aggregation", {}).get("duration_seconds", 0)),
        "status": "success",
        "response": final_response,
        "frames": [],
    })

    snap_times = resolve_snapshot_times(snapshots_raw, starts, ends)
    snapshots_out = []
    for raw, t in zip(snapshots_raw, snap_times):
        if t is None:
            continue
        dg = raw["dependency_graph"]
        sub_list = [
            {
                "id": st["id"],
                "dependencies": list(st.get("dependencies", [])),
                "description": st.get("description", ""),
            }
            for st in dg["subtasks"]
        ]
        agg = dg.get("aggregation", {})
        snapshots_out.append({
            "label": raw.get("replan_label", "?"),
            "t": float(t),
            "subtasks": sub_list,
            "aggregation": {
                "id": agg.get("id"),
                "dependencies": list(agg.get("dependencies", [])),
                "description": agg.get("description", ""),
            } if agg else None,
        })

    scheduled_ids = set(subtask_order)
    filter_snapshot_subtasks(snapshots_out, scheduled_ids)
    clean_aggregator_deps(snapshots_out, scheduled_ids)
    snapshots_out.sort(key=lambda s: s["t"])

    return {
        "task_id": task_id,
        "benchmark": "MACU run",
        "task_text": task_text,
        "total_seconds": float(total_seconds),
        "schedule": schedule,
        "subtasks": subtasks_out,
        "snapshots": snapshots_out,
        "final_response": final_response,
        "short_labels": short_labels,
        "subtask_order": list(subtask_order),
        "wallclock_total_seconds": final_results.get("total_duration_seconds"),
        "total_cost_usd": cost_usd,
        "source_run_dir": str(task_dir),
    }


def write_index(out_dir: Path, data: dict) -> None:
    """Render the index.html template with CSS, JS, and run data inlined.

    Everything except the screenshots is embedded in this single file so the
    viewer works equally well over http:// or directly via file:// (Chrome
    blocks XHR for local files, which is how Babel's external script-src
    loader works — inlining sidesteps that).
    """
    template = (ASSETS_DIR / "index.html").read_text()
    css = (ASSETS_DIR / "static" / "css" / "main.css").read_text()
    graph_js = (ASSETS_DIR / "static" / "js" / "graph.js").read_text()
    example_js = (ASSETS_DIR / "static" / "js" / "example.js").read_text()
    app_js = (ASSETS_DIR / "static" / "js" / "app.js").read_text()

    # Defensively escape `</` so a stray "</script>" inside the data or JS
    # source can't terminate the surrounding inline <script> tag early.
    def safe_script(s: str) -> str:
        return s.replace("</script", "<\\/script")

    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = f"{data['task_id']} — MACU run"

    html = (template
            .replace("__TITLE__", title)
            .replace("__CSS__", css)
            .replace("__DATA_JSON__", data_json)
            .replace("__GRAPH_JS__", safe_script(graph_js))
            .replace("__EXAMPLE_JS__", safe_script(example_js))
            .replace("__APP_JS__", safe_script(app_js)))
    (out_dir / "index.html").write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Path to a run directory (e.g. runs/cafes_near_cmu)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write the viewer (default: <run_dir>/visualization)")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help=f"Max keyframes per subtask (default {DEFAULT_MAX_FRAMES})")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        ap.error(f"run_dir does not exist: {run_dir}")

    task_dir = find_task_dir(run_dir)
    out_dir = (args.output_dir or (run_dir / "visualization")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = process_run(task_dir, out_dir, max_frames=args.max_frames)
    write_index(out_dir, data)

    print()
    print(f"Wrote viewer to: {out_dir}")
    print(f"Open: file://{out_dir / 'index.html'}")
    print(f"  · {len(data['subtask_order'])} subagents")
    print(f"  · {len(data['snapshots'])} DAG snapshots")
    print(f"  · {sum(len(s.get('frames', [])) for s in data['subtasks'])} screenshots")
    print(f"  · {data['total_seconds']:.0f}s total timeline")


if __name__ == "__main__":
    main()
