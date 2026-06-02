import json
from pathlib import Path

from utils.file_management_utils import (
    apply_filemgmt_actions,
    compute_file_diff,
    resolve_input_files,
)
from utils.graph_utils import build_merged_final_traj
from utils.utils import (
    extract_cua_result_from_dir,
    extract_last_k_screenshots_from_dir,
)


def _write_traj(dir_path: Path, rows: list[dict]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    traj_path = dir_path / "traj.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return traj_path


def test_build_merged_final_traj_filters_irrelevant_work_and_copies_screenshots(tmp_path):
    run_dir = tmp_path
    a_dir = run_dir / "a"
    b_dir = run_dir / "b"
    x_dir = run_dir / "x"

    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    x_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "a_step.png").write_text("a", encoding="utf-8")
    (b_dir / "b_step.png").write_text("b", encoding="utf-8")
    (x_dir / "x_step.png").write_text("x", encoding="utf-8")

    _write_traj(
        a_dir,
        [
            {
                "step_num": 1,
                "action_timestamp": "2026-01-01T00:00:01Z",
                "response": "A result",
                "action": {"command": "do_a", "input": {}},
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": "a_step.png",
            }
        ],
    )
    _write_traj(
        b_dir,
        [
            {
                "step_num": 1,
                "action_timestamp": "2026-01-01T00:00:02Z",
                "response": "B result",
                "action": {"command": "do_b", "input": {}},
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": "b_step.png",
            }
        ],
    )
    _write_traj(
        x_dir,
        [
            {
                "step_num": 1,
                "action_timestamp": "2026-01-01T00:00:03Z",
                "response": "Irrelevant branch",
                "action": {"command": "do_x", "input": {}},
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": "x_step.png",
            }
        ],
    )

    all_subtasks = [
        {"id": "a", "agent_type": "cua", "dependencies": []},
        {"id": "b", "agent_type": "cua", "dependencies": ["a"]},
        {"id": "x", "agent_type": "cua", "dependencies": []},
        {"id": "final_aggregation", "agent_type": "manager", "dependencies": ["b"]},
    ]
    graph = {"dependency_graph": {"aggregation": {"id": "final_aggregation"}}}
    results = {
        "a": "A result",
        "b": "B result",
        "x": "Irrelevant branch",
        "final_aggregation": "Manager synthesis",
    }

    out_dir = build_merged_final_traj(
        run_dir=run_dir,
        task_id="task-1",
        original_task_text="Find the answer",
        all_subtasks=all_subtasks,
        graph=graph,
        results=results,
        completion_order=[
            ("a", 1.0, None),
            ("x", 2.0, None),
            ("b", 3.0, None),
            ("final_aggregation", 4.0, None),
        ],
        final_response="Final answer",
    )

    assert out_dir == run_dir / "final_traj"
    raw_rows = [
        json.loads(line)
        for line in (out_dir / "traj_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    agent_rows = [
        json.loads(line)
        for line in (out_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    raw_text = "\n".join(row.get("response", "") for row in raw_rows)
    agent_text = "\n".join(row.get("response", "") for row in agent_rows)

    assert "Irrelevant branch" not in raw_text
    assert "[Manager: final_aggregation]" in raw_text
    assert "[Final aggregated response]" in raw_text
    assert "Final answer" in agent_text
    assert (out_dir / "a__a_step.png").exists()
    assert (out_dir / "b__b_step.png").exists()
    assert not (out_dir / "x__x_step.png").exists()


def test_extract_cua_result_from_dir_dedupes_usage_and_cleans_tool_calls(tmp_path):
    subtask_dir = tmp_path / "subtask"
    _write_traj(
        subtask_dir,
        [
            {
                "step_num": 1,
                "response": "intermediate",
                "model_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "model_time": 1.5,
                    "inference_intervals": [{"start": 10.0, "end": 11.5}],
                },
                "screenshot_file": "one.png",
            },
            {
                "step_num": 1,
                "response": "duplicate accounting entry",
                "model_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "model_time": 1.5,
                    "inference_intervals": [{"start": 10.0, "end": 11.5}],
                },
                "screenshot_file": "one.png",
            },
            {
                "step_num": 2,
                "response": "\n<tool_call>ignore me</tool_call>\n\nFinal answer\n\n\n",
                "model_usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "inference_seconds": 2.25,
                    "inference_start_time": 20.0,
                    "inference_end_time": 22.25,
                },
                "screenshot_file": "two.png",
            },
        ],
    )

    response, usage = extract_cua_result_from_dir(subtask_dir, "subtask", "gpt-test")

    assert response == "Final answer"
    assert usage.input_tokens == 15
    assert usage.output_tokens == 3
    assert usage.model == "gpt-test"
    assert usage.inference_seconds == 3.75
    assert usage.inference_intervals == [(10.0, 11.5), (20.0, 22.25)]


def test_extract_last_k_screenshots_from_dir_returns_last_existing_in_order(tmp_path):
    subtask_dir = tmp_path / "shots"
    subtask_dir.mkdir(parents=True, exist_ok=True)
    for name in ("shot1.png", "shot2.png", "shot3.png"):
        (subtask_dir / name).write_text(name, encoding="utf-8")
    _write_traj(
        subtask_dir,
        [
            {"screenshot_file": "shot1.png"},
            {"screenshot_file": "missing.png"},
            {"screenshot_file": "shot2.png"},
            {"screenshot_file": "shot3.png"},
        ],
    )

    screenshots = extract_last_k_screenshots_from_dir(subtask_dir, k=2)

    assert screenshots == [subtask_dir / "shot2.png", subtask_dir / "shot3.png"]


def test_compute_file_diff_classifies_changes_and_skips_snapshot_files(tmp_path):
    pre_path = tmp_path / "files_pre.json"
    post_path = tmp_path / "files_post.json"
    pre_path.write_text(
        json.dumps(
            {
                "/tmp/keep.txt": {"size": 1, "mtime": 1},
                "/tmp/modify.txt": {"size": 1, "mtime": 1},
                "/tmp/delete.txt": {"size": 1, "mtime": 1},
                "/tmp/files_pre.json": {"size": 1, "mtime": 1},
            }
        ),
        encoding="utf-8",
    )
    post_path.write_text(
        json.dumps(
            {
                "/tmp/keep.txt": {"size": 1, "mtime": 1},
                "/tmp/modify.txt": {"size": 2, "mtime": 1},
                "/tmp/add.txt": {"size": 1, "mtime": 1},
                "/tmp/files_post.json": {"size": 1, "mtime": 1},
            }
        ),
        encoding="utf-8",
    )

    diff = compute_file_diff(pre_path, post_path)

    assert diff == {
        "added": ["/tmp/add.txt"],
        "modified": ["/tmp/modify.txt"],
        "deleted": ["/tmp/delete.txt"],
    }


def test_apply_filemgmt_actions_archives_available_downloads(tmp_path):
    downloaded = tmp_path / "guest_dump" / "report.csv"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_text("csv-data", encoding="utf-8")

    manifest = apply_filemgmt_actions(
        actions=[
            {"src_path": "/home/user/report.csv", "archive_name": "report.csv"},
            {"src_path": "/home/user/missing.csv", "archive_name": "missing.csv"},
        ],
        downloads={"/home/user/report.csv": str(downloaded)},
        run_dir=tmp_path,
        source_sid="subtask_1",
    )

    assert manifest["archived"] == [
        {
            "guest_path": "/home/user/report.csv",
            "host_path": str(tmp_path / "file_artifacts" / "report.csv"),
            "archive_name": "report.csv",
            "source_sid": "subtask_1",
        }
    ]
    assert (tmp_path / "file_artifacts" / "report.csv").read_text(encoding="utf-8") == "csv-data"


def test_resolve_input_files_skips_missing_or_incomplete_entries():
    archive_pool = {
        "good.txt": {
            "host_path": "/tmp/good.txt",
            "guest_path": "/home/user/good.txt",
            "source_sid": "a",
        },
        "missing_host.txt": {
            "guest_path": "/home/user/missing_host.txt",
            "source_sid": "b",
        },
        "missing_guest.txt": {
            "host_path": "/tmp/missing_guest.txt",
            "source_sid": "c",
        },
    }

    resolved = resolve_input_files(
        ["good.txt", "missing_host.txt", "missing_guest.txt", "ghost.txt"],
        archive_pool,
    )

    assert resolved == [
        {
            "local_path": "/tmp/good.txt",
            "guest_path": "/home/user/good.txt",
            "source_sid": "a",
        }
    ]
