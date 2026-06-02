import json
from pathlib import Path

import pytest

import run_macu as ra
from utils.graph_utils import (
    compute_aggregation_ancestors,
    compute_downstream_count,
    get_all_subtasks,
    topological_sort_levels,
    validate_subtask_ids,
)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_args_extracts_forwarded_cua_flags(
    monkeypatch,
    tmp_path,
):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )
    result_dir = tmp_path / "runs"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_macu.py",
            str(tasks_file),
            "--result-dir",
            str(result_dir),
            "--osworld-root",
            str(tmp_path / "osworld"),
            "--",
            "--max_steps",
            "25",
            "--reasoning_effort",
            "high",
            "",
            "--provider_name",
            "apptainer",
        ],
    )

    args, cua_args = ra.parse_args()

    assert args.max_steps == 25
    assert args.reasoning_effort == "high"
    assert args.max_task_timeout == 3 * 3600
    assert args.tasks_file == str(tasks_file.resolve())
    assert args.result_dir == str(result_dir.resolve())
    assert "--max_steps" not in cua_args
    assert "--reasoning_effort" not in cua_args
    assert cua_args == ["--provider_name", "apptainer"]


def test_parse_args_accepts_google_manager_provider(monkeypatch, tmp_path):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_macu.py",
            str(tasks_file),
            "--osworld-root",
            str(tmp_path / "osworld"),
            "--manager-provider",
            "google",
            "--manager-model",
            "gemini-3.1-flash-lite-preview",
        ],
    )

    args, cua_args = ra.parse_args()

    assert args.manager_provider == "google"
    assert args.manager_model == "gemini-3.1-flash-lite-preview"
    assert cua_args == []


def test_parse_args_rejects_removed_manager_enable_flags(monkeypatch, tmp_path, capsys):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        ["run_macu.py", str(tasks_file), "--enable-replanning"],
    )

    with pytest.raises(SystemExit) as excinfo:
        ra.parse_args()
    assert excinfo.value.code == 2
    assert "manager followup and replanning are always enabled" in capsys.readouterr().err


def test_parse_args_rejects_removed_skip_evaluator_flag(monkeypatch, tmp_path, capsys):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        ["run_macu.py", str(tasks_file), "--skip-evaluator"],
    )

    with pytest.raises(SystemExit) as excinfo:
        ra.parse_args()
    assert excinfo.value.code == 2
    assert "OSWorld evaluation always runs" in capsys.readouterr().err


def test_parse_args_accepts_raw_task_string(monkeypatch, tmp_path):
    result_dir = tmp_path / "runs"
    raw_task = "Help me find 3 cafes near Ophelia in Pittsburgh"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_macu.py",
            "--task",
            raw_task,
            "--task-id",
            "cafes-near-ophelia",
            "--result-dir",
            str(result_dir),
            "--osworld-root",
            str(tmp_path / "osworld"),
        ],
    )

    args, cua_args = ra.parse_args()

    assert args.task_id == "cafes-near-ophelia"
    assert args.no_initial_setup is True
    assert args.tasks_file == str(
        (result_dir / "_input_tasks" / "cafes-near-ophelia.json").resolve()
    )
    payload = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    assert payload == [
        {
            "task_id": "cafes-near-ophelia",
            "confirmed_task": raw_task,
            "website": "",
            "no_initial_setup": True,
        }
    ]
    assert cua_args == []


def test_parse_args_defaults_worker_python_to_current_interpreter(monkeypatch, tmp_path):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(ra.sys, "executable", "/current/python")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_macu.py",
            str(tasks_file),
            "--osworld-root",
            str(tmp_path / "osworld"),
        ],
    )

    args, _cua_args = ra.parse_args()

    assert args.python == "/current/python"


def test_parse_args_rejects_python_flag(monkeypatch, tmp_path, capsys):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        ["run_macu.py", str(tasks_file), "--python", "/other/python"],
    )

    with pytest.raises(SystemExit) as excinfo:
        ra.parse_args()
    assert excinfo.value.code == 2
    assert "--python has been removed" in capsys.readouterr().err


def test_parse_args_rejects_raw_task_and_tasks_file(monkeypatch, tmp_path, capsys):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        ["run_macu.py", str(tasks_file), "--task", "Do another thing"],
    )

    with pytest.raises(SystemExit) as excinfo:
        ra.parse_args()
    assert excinfo.value.code == 2
    assert "either a tasks_file or --task" in capsys.readouterr().err


def test_infeasible_verdict_detection():
    assert ra._contains_infeasible_verdict("The task is infeasible as stated.")
    assert ra._contains_infeasible_verdict("This cannot be completed in the environment.")
    assert not ra._contains_infeasible_verdict("The task was completed successfully.")


def test_subtask_fail_action_detection(tmp_path):
    traj_path = tmp_path / "worker" / "traj.jsonl"
    traj_path.parent.mkdir(parents=True)
    traj_path.write_text(
        "\n".join(
            [
                json.dumps({"action": "WAIT"}),
                json.dumps({"action": {"action": "FAIL"}}),
                json.dumps({"action": "SUMMARY"}),
            ]
        ),
        encoding="utf-8",
    )

    assert ra._subtask_has_fail_action(tmp_path / "worker")


def test_parse_args_rejects_invalid_cua_tasks_schema(monkeypatch, tmp_path):
    tasks_file = _write_json(
        tmp_path / "bad_tasks.json",
        [{"task_id": "task-1", "instruction": "Missing confirmed_task"}],
    )

    monkeypatch.setattr("sys.argv", ["run_macu.py", str(tasks_file)])

    with pytest.raises(SystemExit, match="lacks 'confirmed_task' key"):
        ra.parse_args()


def test_parse_args_accepts_legacy_max_wave_timeout_alias(monkeypatch, tmp_path):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_macu.py",
            str(tasks_file),
            "--osworld-root",
            str(tmp_path / "osworld"),
            "--max-wave-timeout",
            "17",
        ],
    )

    args, cua_args = ra.parse_args()

    assert args.max_subtask_timeout == 17
    assert cua_args == []


def test_parse_args_requires_osworld_root(monkeypatch, tmp_path, capsys):
    tasks_file = _write_json(
        tmp_path / "tasks.json",
        [{"task_id": "task-1", "confirmed_task": "Do the thing"}],
    )

    monkeypatch.setattr("sys.argv", ["run_macu.py", str(tasks_file)])

    with pytest.raises(SystemExit) as excinfo:
        ra.parse_args()
    assert excinfo.value.code == 2
    assert "--osworld-root is required" in capsys.readouterr().err


def test_validate_subtask_ids_rejects_duplicates():
    graph = {
        "dependency_graph": {
            "subtasks": [{"id": "dup"}, {"id": "dup"}],
            "aggregation": {"id": "final_aggregation"},
        }
    }

    with pytest.raises(ValueError, match="Duplicate subtask id 'dup'"):
        validate_subtask_ids(graph)


def test_validate_subtask_ids_rejects_aggregation_collision():
    graph = {
        "dependency_graph": {
            "subtasks": [{"id": "final_aggregation"}],
            "aggregation": {"id": "final_aggregation"},
        }
    }

    with pytest.raises(ValueError, match="duplicates subtask id"):
        validate_subtask_ids(graph)


def test_get_all_subtasks_and_topological_sort_levels():
    graph = {
        "dependency_graph": {
            "subtasks": [
                {"id": "a", "dependencies": []},
                {"id": "b", "dependencies": ["a"]},
            ],
            "aggregation": {"id": "final_aggregation", "dependencies": ["b"]},
        }
    }

    with_manager = get_all_subtasks(graph)
    no_manager = get_all_subtasks(graph, no_manager=True)
    levels = topological_sort_levels(with_manager)

    assert [s["id"] for s in no_manager] == ["a", "b"]
    assert [{s["id"] for s in level} for level in levels] == [
        {"a"},
        {"b"},
        {"final_aggregation"},
    ]


def test_topological_sort_levels_rejects_cycle():
    all_subtasks = [
        {"id": "a", "dependencies": ["b"]},
        {"id": "b", "dependencies": ["a"]},
    ]

    with pytest.raises(ValueError, match="Dependency cycle or missing dependency"):
        topological_sort_levels(all_subtasks)


def test_compute_downstream_count_counts_transitive_dependents():
    all_subtasks = [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
        {"id": "c", "dependencies": ["b"]},
        {"id": "final_aggregation", "dependencies": ["b", "c"]},
    ]

    counts = compute_downstream_count(all_subtasks)

    assert counts["a"] == 3
    assert counts["b"] == 2
    assert counts["c"] == 1
    assert counts["final_aggregation"] == 0


def test_compute_aggregation_ancestors_returns_closure_and_fallback():
    all_subtasks = [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
        {"id": "c", "dependencies": []},
        {"id": "final_aggregation", "dependencies": ["b"]},
    ]

    assert compute_aggregation_ancestors(all_subtasks, "final_aggregation") == {
        "a",
        "b",
        "final_aggregation",
    }
    assert compute_aggregation_ancestors(all_subtasks, None) == {
        "a",
        "b",
        "c",
        "final_aggregation",
    }
