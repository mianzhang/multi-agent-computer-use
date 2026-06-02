import argparse
import json
from pathlib import Path

import run_macu as ra
import utils.macu_runtime as runtime
from utils.llm import UsageInfo


class _FakeLogFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, polls):
        self._polls = list(polls)
        self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.wait_calls = []

    def poll(self):
        if self._polls:
            result = self._polls.pop(0)
            if result is not None:
                self.returncode = result
            return result
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self.returncode

    def kill(self):
        self.killed += 1
        self.returncode = -9


def _write_graph(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _base_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = {
        "tasks_file": str(tmp_path / "tasks.json"),
        "result_dir": str(tmp_path / "runs"),
        "cua_script": str(tmp_path / "run_cua.py"),
        "osworld_root": str(tmp_path),
        "python": "/usr/bin/python3",
        "cua_provider": "openai",
        "manager_provider": "openai",
        "manager_model": "gpt-test",
        "manager_max_tokens": 1024,
        "manager_screenshots": 1,
        "max_subtask_timeout": 60,
        "website": "https://example.com",
        "domain": "macu",
        "max_parallelism": 1,
        "max_replans": 5,
        "spare_capacity_min_steps": 1,
        "no_manager": False,
        "evaluator_timeout": 60,
        "max_task_timeout": 3 * 3600,
        "osworld_data_dir": str(tmp_path / "osworld"),
        "task_id": None,
        "graph_gen_max_tokens": 1024,
        "sleep_after_execution": None,
        "max_steps": None,
        "post_task_wait_seconds": None,
        "reasoning_effort": None,
        "reset_pool_snapshot": "init_state",
        "reset_pool_timeout": 60,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _install_common_stubs(monkeypatch, tmp_path: Path, polls_by_sid: dict[str, list[int | None]]):
    launched: dict[str, dict] = {}
    merged_calls: list[dict] = []
    reset_calls: list[dict] = []

    def fake_launch(st, instruction, run_dir, args, cua_args, **kwargs):
        subtask_dir = Path(run_dir) / st["id"]
        subtask_dir.mkdir(parents=True, exist_ok=True)
        proc = _FakeProcess(polls_by_sid[st["id"]])
        launched[st["id"]] = {"proc": proc, "kwargs": kwargs, "instruction": instruction}
        return ra.CUAProcess(
            proc=proc,
            subtask_id=st["id"],
            log_file=_FakeLogFile(),
            subtask_dir=subtask_dir,
            result_dir=str(subtask_dir),
            instruction=instruction,
        )

    monkeypatch.setattr(runtime, "_orchestrator_alloc_vm", lambda _args: str(tmp_path / "pool.vmx"))
    monkeypatch.setattr(runtime, "_registry_path_to_absolute", lambda _args, path: Path(path))
    monkeypatch.setattr(runtime, "_take_subtask_snapshot", lambda *_args, **_kwargs: "snap")
    monkeypatch.setattr(runtime, "launch_cua_subtask", fake_launch)
    monkeypatch.setattr(runtime, "extract_latest_screenshot_from_dir", lambda _path: None)
    monkeypatch.setattr(runtime, "count_subtask_steps", lambda _path: 0)
    monkeypatch.setattr(
        ra,
        "ask_manager_for_replanning",
        lambda **_kwargs: ({}, UsageInfo(model="mgr-model", cost_usd=0.0)),
    )
    monkeypatch.setattr(
        ra,
        "build_merged_final_traj",
        lambda **kwargs: merged_calls.append(kwargs) or (Path(kwargs["run_dir"]) / "final_traj"),
    )
    monkeypatch.setattr(
        ra,
        "reset_vm_pool",
        lambda args, run_dir: reset_calls.append({"args": args, "run_dir": run_dir}),
    )
    return launched, merged_calls, reset_calls


def test_run_single_graph_no_manager_uses_only_cua_result(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-1",
            "task_text": "Do the thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Solve it", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(tmp_path, no_manager=True)
    launched, merged_calls, reset_calls = _install_common_stubs(
        monkeypatch,
        tmp_path,
        {"subtask_1": [0]},
    )
    monkeypatch.setattr(
        runtime,
        "extract_cua_result_from_dir",
        lambda _subtask_dir, sid, _model: (f"{sid} done", UsageInfo(model="cua-model", cost_usd=0.25)),
    )
    monkeypatch.setattr(
        runtime,
        "read_vm_info_from_dir",
        lambda _subtask_dir, sid: {"vm_path": f"/vm/{sid}.vmx"},
    )

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    final_results = json.loads((tmp_path / "runs" / "task-1" / "final_results.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "runs" / "task-1" / "summary.json").read_text(encoding="utf-8"))

    assert final_results["final_response"] == "subtask_1 done"
    assert final_results["final_vm_subtask_id"] == "subtask_1"
    assert final_results["final_vm_path"] == "/vm/subtask_1.vmx"
    assert final_results["subtask_results"]["subtask_1"] == {
        "status": "success",
        "response": "subtask_1 done",
    }
    assert summary["total_cost_usd"] == 0.25
    assert merged_calls[-1]["no_manager"] is True
    assert launched["subtask_1"]["kwargs"]["vm_path_override"] == str(tmp_path / "pool.vmx")
    assert reset_calls and reset_calls[-1]["run_dir"] == tmp_path / "runs" / "task-1"


def test_run_single_graph_raw_task_uses_blank_vm_setup(monkeypatch, tmp_path):
    graph_path = _write_graph(tmp_path, {"task_id": "raw-task"})
    args = _base_args(tmp_path, no_manager=True, no_initial_setup=True)
    launched, _merged_calls, _reset_calls = _install_common_stubs(
        monkeypatch,
        tmp_path,
        {"subtask_1": [0]},
    )
    monkeypatch.setattr(
        runtime,
        "extract_cua_result_from_dir",
        lambda _subtask_dir, sid, _model: (f"{sid} done", UsageInfo(model="cua-model")),
    )
    monkeypatch.setattr(
        runtime,
        "read_vm_info_from_dir",
        lambda _subtask_dir, sid: {"vm_path": f"/vm/{sid}.vmx"},
    )

    ra._RunSingleGraphRunner(
        graph_path,
        args,
        ["--model", "cua-model", "--provider_name", "vmware"],
        task_metadata={
            "task_id": "raw-task",
            "task_text": "Help me find cafes",
            "website": "",
            "no_initial_setup": True,
        },
    ).run()

    assert launched["subtask_1"]["kwargs"]["skip_canonical_setup"] is True
    assert launched["subtask_1"]["kwargs"]["website_override"] == ""


def test_run_single_graph_refreshes_stale_vm_info_at_completion(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-stale-vm-info",
            "task_text": "Do the thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Solve it", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(tmp_path, no_manager=True)
    _install_common_stubs(monkeypatch, tmp_path, {"subtask_1": [None, 0]})
    monkeypatch.setattr(ra.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        runtime,
        "extract_cua_result_from_dir",
        lambda _subtask_dir, sid, _model: (f"{sid} done", UsageInfo(model="cua-model", cost_usd=0.25)),
    )

    read_calls = []

    def fake_read_vm_info(_subtask_dir, sid):
        read_calls.append(sid)
        if len(read_calls) == 1:
            return {"vm_path": "/vm/stale.vmx"}
        return {"vm_path": "/vm/fresh.vmx"}

    monkeypatch.setattr(runtime, "read_vm_info_from_dir", fake_read_vm_info)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    final_results = json.loads(
        (tmp_path / "runs" / "task-stale-vm-info" / "final_results.json").read_text(encoding="utf-8")
    )

    assert read_calls == ["subtask_1", "subtask_1"]
    assert final_results["subtask_vm_paths"]["subtask_1"] == "/vm/fresh.vmx"
    assert final_results["final_vm_path"] == "/vm/fresh.vmx"


def test_run_single_graph_runs_manager_aggregation(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-2",
            "task_text": "Complete the workflow",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Solve it", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(tmp_path)
    _install_common_stubs(monkeypatch, tmp_path, {"worker": [0]})
    monkeypatch.setattr(
        runtime,
        "generate_validated_dependency_graph",
        lambda **_kwargs: (
            {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Solve it", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
            UsageInfo(model="mgr-model", cost_usd=0.05),
            [],
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_cua_result_from_dir",
        lambda _subtask_dir, sid, _model: (f"{sid} draft", UsageInfo(model="cua-model", cost_usd=0.15)),
    )
    monkeypatch.setattr(
        runtime,
        "read_vm_info_from_dir",
        lambda _subtask_dir, sid: {"vm_path": f"/vm/{sid}.vmx"},
    )

    def fake_call_manager_for_subtasks(
        _ready_manager_subtasks,
        _all_results,
        _original_task_text,
        _args,
        _run_dir,
        *,
        vm_info_by_sid,
        final_vm_choice,
        archive_pool,
        meta_by_sid,
    ):
        final_vm_choice["subtask_id"] = "worker"
        final_vm_choice["vm_path"] = "/vm/worker.vmx"
        return {"final_aggregation": "manager final"}, [UsageInfo(model="mgr-model", cost_usd=0.4)]

    monkeypatch.setattr(runtime, "call_manager_for_subtasks", fake_call_manager_for_subtasks)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    final_results = json.loads((tmp_path / "runs" / "task-2" / "final_results.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "runs" / "task-2" / "summary.json").read_text(encoding="utf-8"))

    assert final_results["final_response"] == "manager final"
    assert final_results["subtask_results"]["final_aggregation"] == {
        "status": "success",
        "response": "manager final",
    }
    assert final_results["final_vm_subtask_id"] == "worker"
    assert summary["total_cost_usd"] == 0.55
    assert summary["subtask_costs"]["final_aggregation"]["agent_type"] == "manager"


def test_run_single_graph_marks_timeout_as_error(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-3",
            "task_text": "Do the slow thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Wait", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(tmp_path, no_manager=True, max_subtask_timeout=0)
    launched, _, _ = _install_common_stubs(monkeypatch, tmp_path, {"subtask_1": [None, None]})
    monkeypatch.setattr(runtime, "extract_cua_result_from_dir", lambda *_args, **_kwargs: ("unused", UsageInfo()))
    monkeypatch.setattr(runtime, "read_vm_info_from_dir", lambda *_args, **_kwargs: None)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    final_results = json.loads((tmp_path / "runs" / "task-3" / "final_results.json").read_text(encoding="utf-8"))

    assert final_results["final_response"] == "[ERROR] Process timed out"
    assert final_results["subtask_results"]["subtask_1"] == {
        "status": "error",
        "response": "[ERROR] Process timed out",
    }
    assert launched["subtask_1"]["proc"].terminated == 1


def test_run_single_graph_accepts_legacy_max_wave_timeout_namespace(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-legacy-timeout",
            "task_text": "Do the slow thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Wait", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(tmp_path, no_manager=True)
    delattr(args, "max_subtask_timeout")
    args.max_wave_timeout = 0
    launched, _, _ = _install_common_stubs(monkeypatch, tmp_path, {"subtask_1": [None, None]})
    monkeypatch.setattr(runtime, "extract_cua_result_from_dir", lambda *_args, **_kwargs: ("unused", UsageInfo()))
    monkeypatch.setattr(runtime, "read_vm_info_from_dir", lambda *_args, **_kwargs: None)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    final_results = json.loads(
        (tmp_path / "runs" / "task-legacy-timeout" / "final_results.json").read_text(encoding="utf-8")
    )

    assert final_results["final_response"] == "[ERROR] Process timed out"
    assert launched["subtask_1"]["proc"].terminated == 1


def test_run_single_graph_task_timeout_cancels_running_and_writes_sentinel(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-task-timeout",
            "task_text": "Do the very slow thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Wait", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(
        tmp_path,
        no_manager=True,
        max_subtask_timeout=9999,
        max_task_timeout=2,
    )
    launched, _, _ = _install_common_stubs(monkeypatch, tmp_path, {"subtask_1": [None, None]})
    monkeypatch.setattr(runtime, "extract_cua_result_from_dir", lambda *_args, **_kwargs: ("unused", UsageInfo()))
    monkeypatch.setattr(runtime, "read_vm_info_from_dir", lambda *_args, **_kwargs: None)

    clock = {"now": 0.0}

    def fake_time():
        clock["now"] += 5.0 if launched else 0.1
        return clock["now"]

    monkeypatch.setattr(ra.time, "time", fake_time)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    run_dir = tmp_path / "runs" / "task-task-timeout"
    final_results = json.loads((run_dir / "final_results.json").read_text(encoding="utf-8"))
    final_result_alias = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))

    assert final_results["status"] == "timed_out"
    assert final_results["timed_out"] is True
    assert final_results["final_response"].startswith("[ERROR] Task timed out after")
    assert final_results["subtask_results"]["subtask_1"]["status"] == "error"
    assert final_result_alias == final_results
    assert launched["subtask_1"]["proc"].terminated == 1


def test_run_single_graph_task_timeout_still_runs_final_aggregation(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "task-task-timeout-manager",
            "task_text": "Do the very slow thing",
            "dependency_graph": {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Wait", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
        },
    )
    args = _base_args(
        tmp_path,
        max_subtask_timeout=9999,
        max_task_timeout=2,
    )
    launched, _, _ = _install_common_stubs(monkeypatch, tmp_path, {"worker": [None, None]})
    monkeypatch.setattr(
        runtime,
        "generate_validated_dependency_graph",
        lambda **_kwargs: (
            {
                "subtasks": [
                    {"id": "worker", "agent_type": "cua", "instruction": "Wait", "dependencies": []}
                ],
                "aggregation": {
                    "id": "final_aggregation",
                    "agent_type": "manager",
                    "instruction": "Aggregate",
                    "dependencies": ["worker"],
                },
            },
            UsageInfo(model="mgr-model", cost_usd=0.05),
            [],
        ),
    )
    monkeypatch.setattr(runtime, "extract_cua_result_from_dir", lambda *_args, **_kwargs: ("unused", UsageInfo()))
    monkeypatch.setattr(
        runtime,
        "read_vm_info_from_dir",
        lambda _subtask_dir, sid: {"vm_path": f"/vm/{sid}.vmx"},
    )

    manager_calls = []

    def fake_call_manager_for_subtasks(
        ready_manager_subtasks,
        all_results,
        _original_task_text,
        _args,
        _run_dir,
        *,
        vm_info_by_sid,
        final_vm_choice,
        archive_pool,
        meta_by_sid,
    ):
        manager_calls.append(
            {
                "ready_ids": [st["id"] for st in ready_manager_subtasks],
                "all_results": dict(all_results),
                "vm_info_by_sid": dict(vm_info_by_sid),
            }
        )
        final_vm_choice["subtask_id"] = "worker"
        final_vm_choice["vm_path"] = "/vm/worker.vmx"
        return {"final_aggregation": "manager saw timeout"}, [
            UsageInfo(model="mgr-model", cost_usd=0.4)
        ]

    monkeypatch.setattr(runtime, "call_manager_for_subtasks", fake_call_manager_for_subtasks)

    clock = {"now": 0.0}

    def fake_time():
        clock["now"] += 5.0 if launched else 0.1
        return clock["now"]

    monkeypatch.setattr(ra.time, "time", fake_time)

    ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"]).run()

    run_dir = tmp_path / "runs" / "task-task-timeout-manager"
    final_results = json.loads((run_dir / "final_results.json").read_text(encoding="utf-8"))
    final_result_alias = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert manager_calls
    assert manager_calls[-1]["ready_ids"] == ["final_aggregation"]
    assert manager_calls[-1]["all_results"]["worker"].startswith("[ERROR] Task timed out after")
    assert manager_calls[-1]["vm_info_by_sid"]["worker"] == {"vm_path": "/vm/worker.vmx"}
    assert final_results["status"] == "timed_out"
    assert final_results["timed_out"] is True
    assert final_results["final_response"] == "manager saw timeout"
    assert final_results["subtask_results"]["worker"]["status"] == "error"
    assert final_results["subtask_results"]["final_aggregation"] == {
        "status": "success",
        "response": "manager saw timeout",
    }
    assert final_results["final_vm_subtask_id"] == "worker"
    assert final_result_alias == final_results
    assert summary["subtask_costs"]["final_aggregation"]["agent_type"] == "manager"
    assert launched["worker"]["proc"].terminated == 1


def test_prepare_graph_osworld_generates_without_initial_screenshot(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "os-task",
            "domain": "chrome",
            "instruction": "Open the page",
        },
    )
    args = _base_args(tmp_path)
    monkeypatch.setattr(runtime, "load_canonical_osworld_config", lambda **_kwargs: {"config": []})
    monkeypatch.setattr(runtime, "_take_initial_vm_screenshot", lambda **_kwargs: (None, None))

    captured = {}

    def fake_generate_validated_dependency_graph(**kwargs):
        captured.update(kwargs)
        return (
            {
                "subtasks": [
                    {
                        "id": "worker",
                        "agent_type": "cua",
                        "instruction": "Open the page",
                        "dependencies": [],
                    }
                ]
            },
            UsageInfo(model="mgr-model", cost_usd=0.1),
            [],
        )

    monkeypatch.setattr(runtime, "generate_validated_dependency_graph", fake_generate_validated_dependency_graph)

    runner = ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"])
    runner._prepare_graph()

    assert captured["image_path"] is None
    assert runner.graph["domain"] == "chrome"
    assert runner.graph["instruction"] == "Open the page"
    assert [s["id"] for s in runner.all_subtasks] == ["worker"]


def test_prepare_graph_osworld_no_manager_skips_screenshot_and_llm(monkeypatch, tmp_path):
    graph_path = _write_graph(
        tmp_path,
        {
            "task_id": "os-task-no-manager",
            "domain": "chrome",
            "instruction": "Open the page",
        },
    )
    args = _base_args(tmp_path, no_manager=True)
    monkeypatch.setattr(runtime, "load_canonical_osworld_config", lambda **_kwargs: {"config": []})
    monkeypatch.setattr(
        runtime,
        "_take_initial_vm_screenshot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("screenshot should not run")),
    )
    monkeypatch.setattr(
        runtime,
        "generate_validated_dependency_graph",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM graph generation should not run")),
    )

    runner = ra._RunSingleGraphRunner(graph_path, args, ["--model", "cua-model", "--provider_name", "vmware"])
    runner._prepare_graph()

    assert runner.graph["domain"] == "chrome"
    assert runner.graph["dependency_graph"]["subtasks"] == [
        {
            "id": "subtask_1",
            "agent_type": "cua",
            "description": "Open the page",
            "dependencies": [],
            "instruction": "Open the page",
            "outputs": [],
            "input_files": [],
        }
    ]
