import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import utils.benchmark_utils as benchmark_utils
import utils.macu_runtime as runtime
import utils.file_management_utils as file_management_utils
import utils.manager_utils as manager_utils
from utils.llm import UsageInfo


def _manager_args(tmp_path: Path, **overrides):
    base = {
        "manager_provider": "openai",
        "manager_model": "gpt-test",
        "manager_max_tokens": 1024,
        "manager_screenshots": 2,
        "result_dir": str(tmp_path / "runs"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_save_manager_prompt_and_response_with_interleaved_segments(tmp_path):
    prompt_path = manager_utils.save_manager_prompt(
        system_prompt="SYSTEM",
        run_dir=tmp_path,
        label="aggregate_final",
        user_segments=[
            {"type": "text", "text": "Line one"},
            {"type": "image", "path": "/tmp/screenshot.png"},
            {"type": "text", "text": "Line two"},
        ],
    )

    response_path = manager_utils.save_manager_response(
        prompt_path,
        raw_response='{"final_response":"done"}',
        usage=UsageInfo(model="gpt-test", input_tokens=3, output_tokens=4, cost_usd=0.12),
    )

    prompt_text = prompt_path.read_text(encoding="utf-8")
    response_text = response_path.read_text(encoding="utf-8")

    assert prompt_path.name == "manager_prompt_001_aggregate_final.yaml"
    assert "--- user_content (interleaved) ---" in prompt_text
    assert "[image: /tmp/screenshot.png]" in prompt_text
    assert response_path.name == "manager_response_001_aggregate_final.yaml"
    assert "model: gpt-test" in response_text
    assert "cost_usd: 0.120000" in response_text


def test_ask_manager_for_file_management_filters_invalid_and_colliding_actions(
    tmp_path,
):
    def fake_call_llm(**_kwargs):
        return (
            json.dumps(
                {
                    "copy_actions": [
                        {"src_path": "/guest/already.txt", "archive_name": "already.txt"},
                        {"src_path": "/guest/new.txt", "archive_name": "new.txt"},
                        {"src_path": "/guest/fallback.txt"},
                        {"src_path": "", "archive_name": "bad.txt"},
                    ]
                }
            ),
            UsageInfo(model="gpt-test", input_tokens=1, output_tokens=2),
        )

    actions, usage = file_management_utils.ask_manager_for_file_management(
        subtask={"id": "subtask_1", "instruction": "inspect files"},
        file_diff={"added": ["/guest/new.txt"], "modified": [], "deleted": []},
        archive_pool={"already.txt": {"source_sid": "x", "guest_path": "/guest/already.txt"}},
        original_task_text="Complete the task",
        args=_manager_args(tmp_path),
        run_dir=tmp_path,
        prompt_yaml_path=tmp_path / "file_management.yaml",
        save_manager_prompt_fn=lambda *_args, **_kwargs: tmp_path / "prompt.yaml",
        save_manager_response_fn=lambda *_args, **_kwargs: tmp_path / "response.yaml",
        load_prompt_fn=lambda _path: {"name": "prompt"},
        render_prompt_fn=lambda *_args, **_kwargs: ("SYSTEM", "USER"),
        call_llm_fn=fake_call_llm,
    )

    assert actions == [
        {"src_path": "/guest/new.txt", "archive_name": "new.txt"},
        {"src_path": "/guest/fallback.txt", "archive_name": "fallback.txt"},
    ]
    assert usage.model == "gpt-test"


def test_call_manager_for_subtasks_selects_requested_final_vm(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(manager_utils, "load_prompt", lambda _path: {"name": "prompt"})
    monkeypatch.setattr(manager_utils, "render_prompt", lambda *args, **kwargs: ("SYSTEM", "USER"))
    monkeypatch.setattr(
        manager_utils,
        "extract_last_k_screenshots",
        lambda sid, run_dir, k: [Path(f"/shots/{sid}_last.png")],
    )
    monkeypatch.setattr(
        manager_utils,
        "save_manager_prompt",
        lambda _system_prompt, _user_prompt, _run_dir, _label, image_path=None, _user_segments=None: (
            captured.setdefault("image_path", image_path),
            tmp_path / "prompt.yaml",
        )[1],
    )
    monkeypatch.setattr(manager_utils, "save_manager_response", lambda *args, **kwargs: tmp_path / "response.yaml")
    monkeypatch.setattr(
        manager_utils,
        "call_llm",
        lambda **kwargs: (
            json.dumps(
                {
                    "final_response": "Merged answer",
                    "final_vm_subtask_id": "b",
                }
            ),
            UsageInfo(model="gpt-test", input_tokens=3, output_tokens=4),
        ),
    )

    final_vm_choice = {}
    refined, usages = manager_utils.call_manager_for_subtasks(
        ready_manager_subtasks=[
            {
                "id": "final_aggregation",
                "agent_type": "manager",
                "instruction": "Aggregate",
                "dependencies": ["a", "b"],
                "_task_id": "task-1",
            }
        ],
        all_results={"a": "Result A", "b": "Result B"},
        original_task_text="Do the task",
        args=_manager_args(tmp_path),
        run_dir=tmp_path,
        vm_info_by_sid={
            "a": {"vm_path": "/vm/a.vmx"},
            "b": {"vm_path": "/vm/b.vmx"},
        },
        final_vm_choice=final_vm_choice,
        meta_by_sid={"a": {}, "b": {}},
    )

    assert refined == {"final_aggregation": "Merged answer"}
    assert len(usages) == 1
    assert final_vm_choice == {"subtask_id": "b", "vm_path": "/vm/b.vmx"}
    assert set(captured["image_path"]) == {
        Path("/shots/a_last.png"),
        Path("/shots/b_last.png"),
    }


def test_ask_manager_for_followup_defaults_unknown_type_to_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(manager_utils, "load_prompt", lambda _path: {"name": "prompt"})
    monkeypatch.setattr(manager_utils, "render_prompt", lambda *args, **kwargs: ("SYSTEM", "USER"))
    monkeypatch.setattr(
        manager_utils,
        "extract_last_k_screenshots",
        lambda sid, run_dir, k: [Path("/shots/followup.png")],
    )
    monkeypatch.setattr(manager_utils, "save_manager_prompt", lambda *args, **kwargs: tmp_path / "prompt.yaml")
    monkeypatch.setattr(manager_utils, "save_manager_response", lambda *args, **kwargs: tmp_path / "response.yaml")
    monkeypatch.setattr(
        manager_utils,
        "call_llm",
        lambda **kwargs: (
            json.dumps(
                {
                    "needs_followup": True,
                    "followup_type": "invalid",
                    "followup_message": "Need one more detail",
                }
            ),
            UsageInfo(model="gpt-test", input_tokens=2, output_tokens=3),
        ),
    )

    needs, followup_type, message, usage = manager_utils.ask_manager_for_followup(
        subtask={"id": "subtask_1", "instruction": "Do work"},
        cua_result="Done",
        original_task_text="Complete task",
        args=_manager_args(tmp_path),
        run_dir=tmp_path,
    )

    assert needs is True
    assert followup_type == "summary"
    assert message == "Need one more detail"
    assert usage.model == "gpt-test"


def test_qwen_summary_extraction_failure_keeps_non_summary_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cua.py",
            "--model",
            "qwen-test",
            "--result_dir",
            str(tmp_path / "run_cua_import"),
        ],
    )
    run_cua = importlib.import_module("scripts.run_cua")

    class FakeAgent:
        model = "qwen-test"
        max_tokens = 128
        history_n = 100
        screenshots = ["encoded-image"]
        actions = ["DONE"]
        responses = ["Action: The task is completed successfully."]
        collapse_text = "[collapsed]"

        def _should_collapse_step(self, _step_num):
            return False

        def _wrap_tool_response(self, parts):
            return parts

        def _assistant_message_for_step(self, response_idx):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": self.responses[response_idx]}],
            }

        def _tool_response_messages_for_step(self, _response_idx):
            return []

        def call_llm(self, *_args, **_kwargs):
            raise RuntimeError("summary retries exhausted")

    wrote_summary = run_cua._extract_qwen_summary(
        FakeAgent(),
        "Complete the task.",
        1,
        str(tmp_path),
    )

    assert wrote_summary is False
    assert not (tmp_path / "traj.jsonl").exists()


def test_qwen_summary_extraction_defaults_to_last_five_screenshots(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cua.py",
            "--model",
            "qwen-test",
            "--result_dir",
            str(tmp_path / "run_cua_import"),
        ],
    )
    run_cua = importlib.import_module("scripts.run_cua")
    captured = {}

    class FakeAgent:
        model = "qwen-test"
        max_tokens = 128
        history_n = 100
        screenshots = [f"encoded-image-{i}" for i in range(7)]
        actions = [f"action-{i}" for i in range(7)]
        responses = [f"response-{i}" for i in range(7)]
        collapse_text = "[collapsed]"

        def _should_collapse_step(self, _step_num):
            return False

        def _wrap_tool_response(self, parts):
            return parts

        def _assistant_message_for_step(self, response_idx):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": self.responses[response_idx]}],
            }

        def _tool_response_messages_for_step(self, _response_idx):
            return []

        def call_llm(self, payload, _model):
            captured["messages"] = payload["messages"]
            return "Summary text"

    wrote_summary = run_cua._extract_qwen_summary(
        FakeAgent(),
        "Complete the task.",
        7,
        str(tmp_path),
    )

    assert wrote_summary is True
    image_urls = [
        part["image_url"]["url"]
        for message in captured["messages"]
        for part in message.get("content", [])
        if part.get("type") == "image_url"
    ]
    assert image_urls == [
        f"data:image/png;base64,encoded-image-{i}"
        for i in range(2, 7)
    ]
    text_parts = [
        part["text"]
        for message in captured["messages"]
        for part in message.get("content", [])
        if part.get("type") == "text"
    ]
    action_history_text = next(text for text in text_parts if "Full action history:" in text)
    for i in range(7):
        assert f"Step {i + 1}: action-{i}" in action_history_text
        assert f"response-{i}" in text_parts
    assert text_parts.count("[Screenshot omitted from summary context.]") == 2


def test_launch_cua_subtask_builds_expected_command(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout, stderr, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return FakeProc()

    monkeypatch.setattr(runtime, "write_subtask_osworld_example", lambda **kwargs: tmp_path / "example.json")
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime, "_child_processes", [])

    args = argparse.Namespace(
        website="https://example.com",
        python="/usr/bin/python3",
        cua_script="/repo/scripts/run_cua.py",
        osworld_root="/repo/osworld",
        cua_provider="openai",
        domain="chrome",
        sleep_after_execution=1.5,
        max_steps=10,
        post_task_wait_seconds=2.5,
        reasoning_effort="medium",
        no_manager=False,
    )

    proc = runtime.launch_cua_subtask(
        subtask={"id": "subtask_1"},
        instruction="Do work",
        run_dir=tmp_path,
        args=args,
        cua_args=["--provider_name", "apptainer"],
        staged_files=[{"local_path": "/tmp/input.txt", "guest_path": "/home/user/input.txt"}],
        vm_path_override="/vm/custom.vmx",
        temperature_override=0.7,
        skip_canonical_setup=True,
        domain="chrome",
        canonical_config={"id": "canonical"},
    )

    proc.log_file.close()

    cmd = captured["cmd"]
    assert captured["cwd"] == "/repo/osworld"
    assert "--overwrite_generated_examples" not in cmd
    assert "--temperature" in cmd and "0.7" in cmd
    assert "--path_to_vm" in cmd and str(Path("/vm/custom.vmx").resolve()) in cmd
    assert "--vm_info_path" in cmd
    assert "--enable_manager_followup" in cmd
    assert "--sleep_after_execution" in cmd and "1.5" in cmd
    assert "--max_steps" in cmd and "10" in cmd
    assert "--post_task_wait_seconds" in cmd and "2.5" in cmd
    assert "--reasoning_effort" in cmd and "medium" in cmd
    assert cmd[-2:] == ["--provider_name", "apptainer"]


def test_launch_cua_subtask_can_skip_initial_app_setup(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout, stderr, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return FakeProc()

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime, "_child_processes", [])

    args = argparse.Namespace(
        website="https://example.com",
        python="/usr/bin/python3",
        cua_script="/repo/scripts/run_cua.py",
        osworld_root="/repo/osworld",
        cua_provider="openai",
        domain="macu",
        sleep_after_execution=None,
        max_steps=None,
        post_task_wait_seconds=None,
        reasoning_effort=None,
        no_manager=False,
    )

    proc = runtime.launch_cua_subtask(
        subtask={"id": "subtask_1"},
        instruction="Start from a blank desktop",
        run_dir=tmp_path,
        args=args,
        cua_args=["--provider_name", "vmware"],
        skip_canonical_setup=True,
        website_override="",
    )

    proc.log_file.close()

    cmd = captured["cmd"]
    task_payload = json.loads((tmp_path / "subtask_1" / "task.json").read_text(encoding="utf-8"))
    example_payload = json.loads(
        (
            tmp_path
            / "subtask_1"
            / "config"
            / "examples"
            / "macu"
            / "subtask_1.json"
        ).read_text(encoding="utf-8")
    )

    assert "--overwrite_generated_examples" not in cmd
    assert "--enable_manager_followup" in cmd
    assert task_payload[0]["website"] == ""
    assert task_payload[0]["no_initial_setup"] is True
    assert example_payload["config"] == []
    assert example_payload["related_apps"] == []
    assert example_payload["metadata"]["no_initial_setup"] is True


def test_run_osworld_evaluator_subprocess_success_parses_passthrough_flags(
    monkeypatch,
    tmp_path,
):
    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path),
        evaluator_timeout=30,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_run(cmd, cwd, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        (run_dir / "result.txt").write_text("0.75\n", encoding="utf-8")
        (
            run_dir
            / "pyautogui"
            / "screenshot"
            / "test-model"
            / "chrome"
            / "task-1"
            / "result.txt"
        ).parent.mkdir(parents=True, exist_ok=True)
        (
            run_dir
            / "pyautogui"
            / "screenshot"
            / "test-model"
            / "chrome"
            / "task-1"
            / "result.txt"
        ).write_text("0.75\n", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(benchmark_utils.subprocess, "run", fake_run)

    score = benchmark_utils.run_osworld_evaluator_subprocess(
        run_dir=run_dir,
        task_id="task-1",
        domain="chrome",
        canonical_config={"id": "task-1"},
        final_vm_path="/vm/final.vmx",
        cua_model="test-model",
        cua_args=[
            "--no-headless",
            "--screen_width",
            "1280",
            "--screen_height",
            "720",
            "--client_password",
            "secret",
            "--provider_name",
            "apptainer",
        ],
        args=args,
    )

    cmd = captured["cmd"]
    assert score == 0.75
    assert captured["cwd"] == str(tmp_path)
    assert captured["timeout"] == 30
    assert "--no-headless" in cmd
    assert "--submit-fail-before-evaluate" not in cmd
    assert "--provider-name" in cmd and "apptainer" in cmd
    assert "--screen-width" in cmd and "1280" in cmd
    assert "--screen-height" in cmd and "720" in cmd
    assert "--client-password" in cmd and "secret" in cmd


def test_run_osworld_evaluator_subprocess_passes_infeasible_fail_flag(
    monkeypatch,
    tmp_path,
):
    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path),
        evaluator_timeout=30,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_results.json").write_text(
        json.dumps({"osworld_submit_fail_before_evaluate": True}),
        encoding="utf-8",
    )

    captured = {}

    def fake_run(cmd, cwd, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        (run_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(benchmark_utils.subprocess, "run", fake_run)

    score = benchmark_utils.run_osworld_evaluator_subprocess(
        run_dir=run_dir,
        task_id="task-1",
        domain="chrome",
        canonical_config={"id": "task-1"},
        final_vm_path="/vm/final.vmx",
        cua_model="test-model",
        cua_args=[],
        args=args,
    )

    assert score == 1.0
    assert "--submit-fail-before-evaluate" in captured["cmd"]


def test_run_osworld_evaluator_subprocess_accepts_legacy_infeasible_flag(
    monkeypatch,
    tmp_path,
):
    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path),
        evaluator_timeout=30,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_results.json").write_text(
        json.dumps({"osworld_infeasible_verdict": True}),
        encoding="utf-8",
    )

    captured = {}

    def fake_run(cmd, cwd, timeout):
        captured["cmd"] = cmd
        (run_dir / "result.txt").write_text("1.0\n", encoding="utf-8")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(benchmark_utils.subprocess, "run", fake_run)

    benchmark_utils.run_osworld_evaluator_subprocess(
        run_dir=run_dir,
        task_id="task-1",
        domain="chrome",
        canonical_config={"id": "task-1"},
        final_vm_path="/vm/final.vmx",
        cua_model="test-model",
        cua_args=[],
        args=args,
    )

    assert "--submit-fail-before-evaluate" in captured["cmd"]


def test_try_claim_task_does_not_reclaim_existing_claim(tmp_path):
    result_dir = tmp_path / "results"

    assert runtime._try_claim_task(result_dir, "task-1", "worker-a") is True
    assert runtime._try_claim_task(result_dir, "task-1", "worker-b") is False

    claim_path = result_dir / "task-1" / ".claimed"
    old_time = time.time() - 10
    os.utime(claim_path, (old_time, old_time))

    assert runtime._try_claim_task(result_dir, "task-1", "worker-c") is False
    claim_text = claim_path.read_text(encoding="utf-8")
    assert claim_text.startswith("worker-a\n")


def test_task_completion_marker_accepts_timeout_final_result_alias(tmp_path):
    task_dir = tmp_path / "results" / "task-1"
    task_dir.mkdir(parents=True)

    assert runtime._task_has_completion_marker(task_dir) is False
    (task_dir / "result.txt").write_text("0.0\n", encoding="utf-8")
    assert runtime._task_has_completion_marker(task_dir) is False
    assert runtime._task_has_completion_marker(task_dir, osworld_mode=True) is True

    (task_dir / "final_result.json").write_text('{"timed_out": true}\n', encoding="utf-8")

    assert runtime._task_has_completion_marker(task_dir) is True


def test_call_llm_dispatches_huggingface_with_qwen_defaults(monkeypatch):
    """call_llm(provider='huggingface') routes to call_huggingface and lets
    it apply Qwen thinking-mode defaults — i.e. does NOT forward the
    dispatcher's temperature=0.0 default (which would override 0.6)."""
    import utils.llm as llm_mod

    captured = {}

    def fake_call_huggingface(**kwargs):
        captured.update(kwargs)
        return "{}", UsageInfo(model=kwargs["model"], input_tokens=1, output_tokens=2)

    monkeypatch.setattr(llm_mod, "call_huggingface", fake_call_huggingface)

    text, usage = llm_mod.call_llm(
        system_prompt="SYSTEM",
        user_prompt="USER",
        provider="huggingface",
        model="Qwen/Qwen3.6-27B",
        max_tokens=2048,
    )

    assert text == "{}"
    assert usage.model == "Qwen/Qwen3.6-27B"
    # temperature is the dispatcher default (0.0) -> NOT forwarded
    assert "temperature" not in captured
    assert captured["model"] == "Qwen/Qwen3.6-27B"
    assert captured["max_tokens"] == 2048
    assert captured["system_prompt"] == "SYSTEM"
    assert captured["user_prompt"] == "USER"


def test_call_llm_huggingface_forwards_explicit_temperature(monkeypatch):
    """When the caller passes a non-default temperature, the dispatcher
    forwards it through to call_huggingface (overriding the Qwen default)."""
    import utils.llm as llm_mod

    captured = {}

    def fake_call_huggingface(**kwargs):
        captured.update(kwargs)
        return "ok", UsageInfo(model=kwargs["model"])

    monkeypatch.setattr(llm_mod, "call_huggingface", fake_call_huggingface)

    llm_mod.call_llm(
        system_prompt="S",
        user_prompt="U",
        provider="huggingface",
        model=None,           # exercise default-model branch
        temperature=0.3,
    )

    assert captured["temperature"] == 0.3
    assert captured["model"] == "Qwen/Qwen3.6-27B"


def test_call_llm_dispatches_google(monkeypatch):
    import utils.llm as llm_mod

    captured = {}

    def fake_call_google(**kwargs):
        captured.update(kwargs)
        return "{}", UsageInfo(model=kwargs["model"], input_tokens=3, output_tokens=4)

    monkeypatch.setattr(llm_mod, "call_google", fake_call_google)

    text, usage = llm_mod.call_llm(
        system_prompt="SYSTEM",
        user_prompt="USER",
        provider="google",
        model="gemini-3.1-flash-lite-preview",
        max_tokens=2048,
        reasoning_effort="high",
    )

    assert text == "{}"
    assert usage.model == "gemini-3.1-flash-lite-preview"
    assert captured["model"] == "gemini-3.1-flash-lite-preview"
    assert captured["max_tokens"] == 2048
    assert captured["system_prompt"] == "SYSTEM"
    assert captured["user_prompt"] == "USER"
    assert "reasoning_effort" not in captured


def test_call_llm_google_uses_gemini_flash_lite_default(monkeypatch):
    import utils.llm as llm_mod

    captured = {}

    def fake_call_google(**kwargs):
        captured.update(kwargs)
        return "{}", UsageInfo(model=kwargs["model"])

    monkeypatch.setattr(llm_mod, "call_google", fake_call_google)

    _text, usage = llm_mod.call_llm(
        system_prompt="SYSTEM",
        user_prompt="USER",
        provider="google",
    )

    assert captured["model"] == "gemini-3.1-flash-lite-preview"
    assert usage.model == "gemini-3.1-flash-lite-preview"


def test_compute_cost_includes_gemini_rates():
    import utils.llm as llm_mod

    assert llm_mod.compute_cost(
        "gemini-3.1-pro-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    ) == 14.0
    assert llm_mod.compute_cost(
        "gemini-3.1-flash-lite-preview",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    ) == 1.75


def test_call_llm_unknown_provider_lists_all_valid_options(monkeypatch):
    """Error message for unknown providers should mention all valid options."""
    import utils.llm as llm_mod
    import pytest

    with pytest.raises(ValueError) as excinfo:
        llm_mod.call_llm(
            system_prompt="S",
            user_prompt="U",
            provider="not_real",
        )
    msg = str(excinfo.value)
    assert all(provider in msg for provider in ("openai", "anthropic", "huggingface", "google"))
