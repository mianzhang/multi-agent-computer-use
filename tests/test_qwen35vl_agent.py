import base64
import json
import re

import pytest

import osworld.mm_agents.qwen35vl_agent as qwen_agent_mod
from osworld.mm_agents.qwen35vl_agent import Qwen35VLAgent


def test_qwen35vl_defaults_match_qwen35_recommended_sampling():
    agent = Qwen35VLAgent()

    assert agent.temperature == 0.6
    assert agent.top_p == 0.95
    assert agent.top_k == 20
    assert agent.min_p == 0.0
    assert agent.presence_penalty == 0.0
    assert agent.repetition_penalty == 1.0
    assert agent.enable_thinking is True
    assert agent.keep_reasoning is False
    assert agent.preserve_reasoning_content is False
    assert agent.max_tool_calls_per_turn == 10


def test_parse_response_maps_terminate_failure_to_fail_action():
    agent = Qwen35VLAgent()
    response = """
Action: Report that the task is infeasible.
<tool_call>
<function=computer_use>
<parameter=action>
terminate
</parameter>
<parameter=status>
failure
</parameter>
</function>
</tool_call>
"""

    low_level_instruction, actions = agent.parse_response(response)

    assert low_level_instruction == "Report that the task is infeasible."
    assert actions == ["FAIL"]


def test_parse_response_surfaces_plain_infeasible_verdict_as_fail_action():
    agent = Qwen35VLAgent()

    low_level_instruction, actions = agent.parse_response(
        "Action: Stop because the task is infeasible without the required account."
    )

    assert low_level_instruction == "Stop because the task is infeasible without the required account."
    assert actions == ["FAIL"]


def test_assistant_history_does_not_replay_reasoning_by_default():
    agent = Qwen35VLAgent()
    agent.responses = ["Action: Click the button.\n<tool_call>...</tool_call>"]
    agent.reasonings = ["The button is visible in the lower-right corner."]

    message = agent._assistant_message_for_step(0)

    assert message["content"] == [
        {"type": "text", "text": "Action: Click the button.\n<tool_call>...</tool_call>"}
    ]
    assert "reasoning" not in message
    assert "reasoning_content" not in message


def test_assistant_history_can_preserve_reasoning_side_channel():
    agent = Qwen35VLAgent(preserve_reasoning_content=True)
    agent.responses = ["Action: Click the button.\n<tool_call>...</tool_call>"]
    agent.reasonings = ["The button is visible in the lower-right corner."]

    message = agent._assistant_message_for_step(0)

    assert message["content"] == [
        {"type": "text", "text": "Action: Click the button.\n<tool_call>...</tool_call>"}
    ]
    assert message["reasoning"] == "The button is visible in the lower-right corner."
    assert message["reasoning_content"] == "The button is visible in the lower-right corner."


def test_assistant_history_strips_embedded_think_when_reasoning_is_logged():
    agent = Qwen35VLAgent(keep_reasoning=True)
    agent.responses = [
        "<think>\nNeed the search box.\n</think>\n\n"
        "Action: Click search.\n<tool_call>...</tool_call>"
    ]
    agent.reasonings = ["Need the search box."]

    message = agent._assistant_message_for_step(0)

    assert message["content"] == [
        {"type": "text", "text": "Action: Click search.\n<tool_call>...</tool_call>"}
    ]
    assert "reasoning" not in message
    assert "reasoning_content" not in message


def test_parse_response_normalizes_split_python_style_key_list():
    agent = Qwen35VLAgent()
    response = """
Action: Press Ctrl+H.
<tool_call>
<function=computer_use>
<parameter=action>
key
</parameter>
<parameter=keys>
["['ctrl", "h']"]
</parameter>
</function>
</tool_call>
"""

    low_level_instruction, actions = agent.parse_response(response)

    assert low_level_instruction == "Press Ctrl+H."
    assert actions == ['pyautogui.hotkey("ctrl", "h")']


def test_parse_response_accepts_python_style_key_list():
    agent = Qwen35VLAgent()
    response = """
Action: Press Ctrl+L.
<tool_call>
<function=computer_use>
<parameter=action>
key
</parameter>
<parameter=keys>
['ctrl', 'l']
</parameter>
</function>
</tool_call>
"""

    _, actions = agent.parse_response(response)

    assert actions == ['pyautogui.hotkey("ctrl", "l")']


def test_decode_type_text_converts_single_escaped_controls():
    assert Qwen35VLAgent._decode_type_text(r"a\nb\tc\r") == "a\nb\tc\r"


def test_decode_type_text_preserves_double_escaped_controls():
    assert Qwen35VLAgent._decode_type_text(r"echo '1\\n2'") == r"echo '1\\n2'"


def test_parse_response_type_uses_clipboard_with_decoded_text():
    agent = Qwen35VLAgent()
    response = r"""
Action: Insert two spreadsheet rows.
<tool_call>
<function=computer_use>
<parameter=action>
type
</parameter>
<parameter=text>
Year\tApplied\r\n2023\t578
</parameter>
</function>
</tool_call>
"""

    _, actions = agent.parse_response(response)

    assert len(actions) == 1
    assert "pyperclip.copy(_text)" in actions[0]
    encoded = re.search(r"base64\.b64decode\('([^']+)'\)", actions[0]).group(1)
    assert base64.b64decode(encoded).decode("utf-8") == "Year\tApplied\r\n2023\t578"


def test_extract_reasoning_text_accepts_vllm_field_names():
    class Message:
        content = "Action: Wait.\n<tool_call>...</tool_call>"
        reasoning = None
        reasoning_content = "Need the page to finish loading."

    assert (
        Qwen35VLAgent._extract_reasoning_text(Message())
        == "Need the page to finish loading."
    )


def test_snapshot_restore_round_trips_reasoning_state():
    agent = Qwen35VLAgent()
    agent.responses = ["Action: Wait.\n<tool_call>...</tool_call>"]
    agent.reasonings = ["Need the page to load."]
    agent.screenshots = ["base64"]
    agent.actions = ["Waiting"]
    agent.folded_prefix_k = 3
    state = agent.snapshot_state()

    agent.responses.append("polluting failed retry")
    agent.reasonings.append("polluting reasoning")
    agent.screenshots.append("extra screenshot")
    agent.actions.append("polluting action")
    agent.folded_prefix_k = 99

    agent.restore_state(state)

    assert agent.responses == ["Action: Wait.\n<tool_call>...</tool_call>"]
    assert agent.reasonings == ["Need the page to load."]
    assert agent.screenshots == ["base64"]
    assert agent.actions == ["Waiting"]
    assert agent.folded_prefix_k == 3


def test_parse_response_maps_terminate_fail_alias_to_fail_action():
    agent = Qwen35VLAgent()
    response = """
Action: Report failure.
<tool_call>
<function=computer_use>
<parameter=action>terminate</parameter>
<parameter=status>fail</parameter>
</function>
</tool_call>
"""

    low_level_instruction, actions = agent.parse_response(response)

    assert low_level_instruction == "Report failure."
    assert actions == ["FAIL"]


def test_parse_response_maps_impossible_status_to_fail_action():
    agent = Qwen35VLAgent()
    response = """
Action: Stop because it cannot be completed.
<tool_call>
<function=computer_use>
<parameter=action>terminate</parameter>
<parameter=status>impossible</parameter>
</function>
</tool_call>
"""

    _, actions = agent.parse_response(response)

    assert actions == ["FAIL"]


def test_parse_response_success_status_with_infeasible_text_still_fails():
    agent = Qwen35VLAgent()
    response = """
Action: Stop because this is impossible without the required account.
<tool_call>
<function=computer_use>
<parameter=action>terminate</parameter>
<parameter=status>success</parameter>
</function>
</tool_call>
"""

    _, actions = agent.parse_response(response)

    assert actions == ["FAIL"]


def test_parse_response_accepts_qwen_agent_json_tool_call_format():
    agent = Qwen35VLAgent()
    response = """
Action: Click the visible button.
<tool_call>
{"name":"computer_use","arguments":{"action":"left_click","coordinate":[100,200]}}
</tool_call>
"""

    low_level_instruction, actions = agent.parse_response(response)

    assert low_level_instruction == "Click the visible button."
    assert actions == ["pyautogui.click(100, 200)"]


def test_parse_response_respects_single_tool_call_cap():
    agent = Qwen35VLAgent(max_tool_calls_per_turn=1)
    response = """
Action: Move to the list, then scroll.
<tool_call>
<function=computer_use>
<parameter=action>mouse_move</parameter>
<parameter=coordinate>[100, 200]</parameter>
</function>
</tool_call>
<tool_call>
<function=computer_use>
<parameter=action>scroll</parameter>
<parameter=pixels>-5</parameter>
</function>
</tool_call>
"""

    _, actions = agent.parse_response(response)

    assert actions == ["pyautogui.moveTo(100, 200)"]


def test_request_payload_includes_native_tool_calling_by_default():
    agent = Qwen35VLAgent()
    _, tools_def = agent.build_tool_prompt(processed_width=1000, processed_height=1000)

    payload = agent._request_payload([], tools_def)

    assert payload["tools"] == [tools_def]
    assert payload["tool_choice"] == "auto"


def test_normalize_native_tool_calls_caps_to_one_before_serializing():
    agent = Qwen35VLAgent(max_tool_calls_per_turn=1)
    tool_calls = [
        {
            "function": {
                "name": "computer_use",
                "arguments": json.dumps({"action": "mouse_move", "coordinate": [100, 200]}),
            }
        },
        {
            "function": {
                "name": "computer_use",
                "arguments": json.dumps({"action": "scroll", "pixels": -5}),
            }
        },
    ]

    normalized = agent._normalize_native_tool_calls(tool_calls)
    text = agent._serialize_native_tool_calls(normalized)

    assert len(normalized) == 1
    assert text.count("<tool_call>") == 1
    assert "<parameter=action>\nmouse_move\n</parameter>" in text
    assert "<parameter=coordinate>\n[100, 200]\n</parameter>" in text


def test_call_llm_passes_and_serializes_native_tool_calls(monkeypatch):
    captured_kwargs = {}

    class FakeMessage:
        content = "Action: Click the target."
        reasoning_content = "The target is visible."
        tool_calls = [
            {
                "function": {
                    "name": "computer_use",
                    "arguments": json.dumps(
                        {"action": "left_click", "coordinate": [100, 200]}
                    ),
                }
            }
        ]

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("osworld.mm_agents.qwen35vl_agent.openai.OpenAI", FakeClient)
    agent = Qwen35VLAgent()
    tools_def = {"type": "function", "function": {"name": "computer_use"}}

    response = agent.call_llm(
        {
            "messages": [],
            "tools": [tools_def],
            "tool_choice": "auto",
        },
        "qwen-test",
    )

    assert captured_kwargs["tools"] == [tools_def]
    assert captured_kwargs["tool_choice"] == "auto"
    assert "Action: Click the target." in response
    assert "<function=computer_use>" in response
    assert "<parameter=action>\nleft_click\n</parameter>" in response
    assert agent._last_reasoning == "The target is visible."


def test_call_llm_retry_backoff_is_short_and_skips_final_sleep(monkeypatch):
    calls = []
    sleeps = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise qwen_agent_mod.SSLError("request timed out")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr(qwen_agent_mod.openai, "OpenAI", FakeClient)
    monkeypatch.setattr(qwen_agent_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    agent = Qwen35VLAgent()

    with pytest.raises(qwen_agent_mod.SSLError):
        agent.call_llm(
            {
                "messages": [],
                "max_retries": 3,
                "retry_backoff_seconds": 0.25,
                "retry_backoff_max_seconds": 0.5,
            },
            "qwen-test",
        )

    assert len(calls) == 3
    assert sleeps == [0.25, 0.5]
