import ast
import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai
from PIL import Image
from requests.exceptions import SSLError

from .utils.qwen_vl_utils import smart_resize


logger = None

MAX_RETRY_TIMES = int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5"))
DEFAULT_RETRY_BACKOFF_SECONDS = float(os.getenv("OSWORLD_OPENAI_RETRY_BACKOFF_SECONDS", "1.0"))
DEFAULT_RETRY_BACKOFF_MAX_SECONDS = float(os.getenv("OSWORLD_OPENAI_RETRY_BACKOFF_MAX_SECONDS", "5.0"))
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
FAILURE_TERMINATE_STATUSES = {
    "failure",
    "fail",
    "failed",
    "error",
    "impossible",
    "infeasible",
    "unfeasible",
    "not_feasible",
    "not possible",
    "not_possible",
    "cannot",
    "false",
}
SUCCESS_TERMINATE_STATUSES = {
    "success",
    "succeeded",
    "done",
    "completed",
    "complete",
    "ok",
    "true",
}
INFEASIBLE_VERDICT_PATTERNS = (
    re.compile(r"\binfeasible\b", re.IGNORECASE),
    re.compile(r"\bunfeasible\b", re.IGNORECASE),
    re.compile(r"\bimpossible\b", re.IGNORECASE),
    re.compile(r"\bnot\s+feasible\b", re.IGNORECASE),
    re.compile(r"\bnot\s+possible\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+be\s+(?:done|completed|performed|fulfilled|satisfied)\b", re.IGNORECASE),
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in FALSE_ENV_VALUES


def process_image(
    image_bytes: bytes,
    *,
    min_pixels: int = 16 * 16 * 4 * 16,
    max_pixels: int = 16 * 16 * 4 * 12800,
) -> str:
    """Resize + re-encode screenshot and return base64 PNG.

    Defaults match the Qwen3-VL computer_use cookbook (local-model path):
    min_pixels = patch*patch*merge^2 * 16,  max_pixels = patch*patch*merge^2 * 6400
    with patch_size=16, merge_size=2.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image = image.resize((resized_width, resized_height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _contains_infeasible_verdict(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in INFEASIBLE_VERDICT_PATTERNS)


class Qwen35VLAgent:
    """
    Lightweight Qwen3.5/3.6 agent.

    Characteristics:
    - OpenAI-compatible API only.
    - Native OpenAI tool calls by default, with XML fallback parsing.
    - Preserve thinking traces in history by default.
    - History truncation by `history_n`.
    - Old screenshot folding by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen35-vl",
        max_tokens: int = 32768,
        top_p: float = 0.95,
        temperature: float = 0.6,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 100,
        add_thought_prefix: bool = False,
        coordinate_type: str = "relative",
        api_backend: str = "openai",
        image_max: int = 20,
        fold_size: int = 10,
        collapse_text: Optional[str] = None,
        keep_reasoning: bool = False,
        preserve_reasoning_content: bool = True,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        enable_thinking: bool = True,
        min_pixels: int = 16 * 16 * 4 * 16,
        max_pixels: int = 16 * 16 * 4 * 6400,
        use_native_tool_calling: Optional[bool] = None,
        max_tool_calls_per_turn: int = 10,
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = history_n
        self.add_thought_prefix = add_thought_prefix
        self.coordinate_type = coordinate_type
        self.api_backend = api_backend
        self.image_max = int(image_max)
        self.fold_size = int(fold_size)
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        self.keep_reasoning = keep_reasoning
        self.preserve_reasoning_content = preserve_reasoning_content
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.enable_thinking = enable_thinking
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if use_native_tool_calling is None:
            use_native_tool_calling = _env_flag("QWEN35VL_USE_NATIVE_TOOLS", True)
        self.use_native_tool_calling = bool(use_native_tool_calling)
        self.max_tool_calls_per_turn = int(max_tool_calls_per_turn)

        if action_space != "pyautogui":
            raise ValueError("qwen35vl_agent only supports pyautogui action space")
        if observation_type != "screenshot":
            raise ValueError("qwen35vl_agent only supports screenshot observations")
        if api_backend != "openai":
            raise ValueError("qwen35vl_agent only supports OpenAI-compatible APIs")
        if self.image_max < 1:
            raise ValueError("image_max must be >= 1")
        if self.fold_size < 1:
            raise ValueError("fold_size must be >= 1")
        if self.max_tool_calls_per_turn < 1:
            raise ValueError("max_tool_calls_per_turn must be >= 1")

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.reasonings: List[str] = []
        self.assistant_contents: List[str] = []
        self.native_tool_calls: List[List[Dict[str, Any]]] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0
        self._last_reasoning = ""
        self._last_assistant_content = ""
        self._last_native_tool_calls: List[Dict[str, Any]] = []
        self.inference_time_total = 0.0
        self.inference_intervals: List[Dict[str, float]] = []
        self._last_inference_time = 0.0
        self._last_inference_intervals: List[Dict[str, float]] = []

    @staticmethod
    def _py_string(text: str) -> str:
        return json.dumps("" if text is None else str(text), ensure_ascii=False)

    @staticmethod
    def _decode_type_text(text: Any) -> str:
        """Decode model-emitted one-backslash control escapes for type actions.

        Qwen often writes XML text parameters like ``foo\nbar`` instead of
        literal newlines. Decode those before pasting, but leave double-escaped
        sequences such as ``\\n`` intact when the task really needs backslash-n.
        """
        text = "" if text is None else str(text)
        replacements = {"n": "\n", "r": "\r", "t": "\t"}

        def replace_match(match: re.Match) -> str:
            return replacements[match.group(1)]

        return re.sub(r"(?<!\\)\\([nrt])", replace_match, text)

    @staticmethod
    def _parse_jsonish(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return stripped
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value

    @classmethod
    def _coerce_tool_arguments(cls, arguments: Any) -> Dict:
        arguments = cls._parse_jsonish(arguments)
        return arguments if isinstance(arguments, dict) else {}

    @staticmethod
    def _get_field(value: Any, field_name: str) -> Any:
        if isinstance(value, dict):
            return value.get(field_name)
        return getattr(value, field_name, None)

    @classmethod
    def _extract_tool_calls(cls, msg) -> List[Any]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            return list(tool_calls)

        for attr in ("model_extra", "__dict__"):
            extra = getattr(msg, attr, None)
            if isinstance(extra, dict) and extra.get("tool_calls"):
                return list(extra["tool_calls"])

        if hasattr(msg, "model_dump"):
            try:
                dumped = msg.model_dump()
            except Exception:
                dumped = {}
            if isinstance(dumped, dict) and dumped.get("tool_calls"):
                return list(dumped["tool_calls"])
        return []

    def _normalize_native_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        step_index = len(self.responses) + 1

        for index, tool_call in enumerate(tool_calls):
            function = self._get_field(tool_call, "function")
            name = self._get_field(function, "name") or self._get_field(tool_call, "name")
            if name != "computer_use":
                continue

            arguments = (
                self._get_field(function, "arguments")
                if function is not None
                else self._get_field(tool_call, "arguments")
            )
            params = self._coerce_tool_arguments(arguments)
            if not params:
                continue

            call_id = self._get_field(tool_call, "id") or f"call_{step_index}_{index + 1}"
            normalized.append(
                {
                    "id": str(call_id),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(params, ensure_ascii=False),
                    },
                }
            )
            if len(normalized) >= self.max_tool_calls_per_turn:
                break

        return normalized

    @classmethod
    def _serialize_native_tool_calls(cls, tool_calls: List[Dict[str, Any]]) -> str:
        serialized: List[str] = []
        for tool_call in tool_calls:
            function = cls._get_field(tool_call, "function")
            name = cls._get_field(function, "name") or cls._get_field(tool_call, "name")
            if name != "computer_use":
                continue

            arguments = (
                cls._get_field(function, "arguments")
                if function is not None
                else cls._get_field(tool_call, "arguments")
            )
            params = cls._coerce_tool_arguments(arguments)
            if not params:
                continue

            lines = ["<tool_call>", "<function=computer_use>"]
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (list, dict)):
                    value_text = json.dumps(value, ensure_ascii=False)
                else:
                    value_text = str(value)
                lines.extend([f"<parameter={key}>", value_text, "</parameter>"])
            lines.extend(["</function>", "</tool_call>"])
            serialized.append("\n".join(lines))
        return "\n".join(serialized)

    @staticmethod
    def _build_clipboard_paste_command(text: str) -> str:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return (
            "import base64, time, pyautogui\n"
            f"_text = base64.b64decode('{encoded}').decode('utf-8')\n"
            "try:\n"
            "    import pyperclip\n"
            "    pyperclip.copy(_text)\n"
            "    time.sleep(0.1)\n"
            "    pyautogui.hotkey('ctrl', 'v')\n"
            "    time.sleep(0.1)\n"
            "except Exception:\n"
            "    _text = _text.replace('\\r\\n', '\\n').replace('\\r', '\\n')\n"
            "    for _line_index, _line in enumerate(_text.split('\\n')):\n"
            "        for _part_index, _part in enumerate(_line.split('\\t')):\n"
            "            if _part:\n"
            "                pyautogui.typewrite(_part)\n"
            "            if _part_index < len(_line.split('\\t')) - 1:\n"
            "                pyautogui.press('tab')\n"
            "        if _line_index < len(_text.split('\\n')) - 1:\n"
            "            pyautogui.press('enter')"
        )

    def _update_folding_state(self, total_screenshots: int) -> None:
        while (total_screenshots - self.folded_prefix_k) > self.image_max:
            self.folded_prefix_k += self.fold_size
        if self.folded_prefix_k > total_screenshots:
            self.folded_prefix_k = total_screenshots

    def _should_collapse_step(self, step_num_1based: int) -> bool:
        return step_num_1based <= self.folded_prefix_k

    def _wrap_tool_response(self, parts: List[Dict]) -> List[Dict]:
        return (
            [{"type": "text", "text": "<tool_response>\n"}]
            + parts
            + [{"type": "text", "text": "\n</tool_response>"}]
        )

    def build_tool_prompt(
        self,
        *,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[str, Dict]:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if self.coordinate_type == "absolute"
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status. Use `status=failure` when the task is infeasible or cannot be completed.
* `answer`: Answer a question."""

        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "scroll",
                                "hscroll",
                                "wait",
                                "terminate",
                                "answer",
                            ],
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`."},
                        "text": {
                            "type": "string",
                            "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                        },
                        "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                        "pixels": {"type": "number", "description": "Scroll amount."},
                        "time": {"type": "number", "description": "Seconds to wait."},
                        "status": {
                            "type": "string",
                            "description": "Task status for terminate.",
                            "enum": ["success", "failure"],
                        },
                    },
                },
            },
        }

        system_prompt = self._build_system_prompt(tools_def)
        return system_prompt, tools_def

    def _build_system_prompt(self, tools_def: Dict, use_native: Optional[bool] = None) -> str:
        if use_native is None:
            use_native = self.use_native_tool_calling
        if use_native:
            if self.max_tool_calls_per_turn <= 1:
                tool_call_budget = (
                    "- Call at most one computer_use tool per turn; wait for the next screenshot before another GUI action.\n"
                    "- Do not attempt multiple GUI actions in one turn."
                )
            else:
                tool_call_budget = (
                    f"- You may call up to {self.max_tool_calls_per_turn} computer_use tools in one turn.\n"
                    "- Use multiple tool calls only for short, safe action sequences that do not require inspecting an intermediate screenshot.\n"
                    "- If the next action depends on what happened, stop after one tool call and wait for the next screenshot."
                )
            return (
                "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
                "Use the computer_use tool to interact with the desktop GUI.\n\n"
                "<IMPORTANT>\n"
                "Reminder:\n"
                f"{tool_call_budget}\n"
                "- Required parameters MUST be specified.\n"
                "- You may provide optional reasoning for your tool call in natural language before the tool call.\n"
                "- If finishing successfully, call computer_use with action=terminate and status=success.\n"
                "- If the task is infeasible or impossible, call computer_use with action=terminate and status=failure.\n"
                f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
                f"- Collapsed screenshots appear as text: {self.collapse_text}\n"
                "</IMPORTANT>\n\n"
                "# Response format\n\n"
                "Response format for every step:\n"
                "1) Action: a short imperative describing what to do in the UI.\n"
                "2) One or more computer_use tool calls when safe.\n\n"
                "Rules:\n"
                "- Be brief: one sentence for Action."
            )

        return (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def)
            + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- Call at most one function per turn; wait for the next screenshot before another GUI action.\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
            f"- Collapsed screenshots appear as text: {self.collapse_text}\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing successfully, use action=terminate with status=success in the tool call.\n"
            "- If the task is infeasible or impossible, use action=terminate with status=failure in the tool call."
        )

    def _request_payload(self, messages: List[Dict], tools_def: Dict) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        if self.use_native_tool_calling:
            payload["tools"] = [tools_def]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _sanitize_messages_for_dump(messages: List[Dict]) -> List[Dict]:
        sanitized: List[Dict] = []
        for message in messages:
            cloned: Dict[str, Any] = {"role": message.get("role")}
            for reasoning_key in ("reasoning", "reasoning_content"):
                reasoning_content = message.get(reasoning_key)
                if reasoning_content:
                    reasoning_text = str(reasoning_content)
                    cloned[reasoning_key] = (
                        reasoning_text[:240] + "...<omitted>"
                        if len(reasoning_text) > 240
                        else reasoning_text
                    )
            if message.get("tool_calls"):
                cloned["tool_calls"] = message.get("tool_calls")
            if message.get("tool_call_id"):
                cloned["tool_call_id"] = message.get("tool_call_id")

            content = message.get("content")
            if isinstance(content, list):
                cloned["content"] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = ((part.get("image_url") or {}).get("url")) or ""
                        if url.startswith("data:image/"):
                            cloned["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url[:40] + "...<omitted>"},
                                }
                            )
                        else:
                            cloned["content"].append(part)
                    else:
                        cloned["content"].append(part)
            else:
                cloned["content"] = content
            sanitized.append(cloned)
        return sanitized

    @staticmethod
    def _split_embedded_reasoning(response: str) -> Tuple[str, str]:
        if not response or "</think>" not in response:
            return response or "", ""
        before, after = response.split("</think>", 1)
        if "<think>" not in before:
            return response, ""
        reasoning = before.split("<think>", 1)[-1].strip()
        return after.lstrip("\n"), reasoning

    def _assistant_message_for_step(self, response_idx: int) -> Dict[str, Any]:
        response = self.responses[response_idx] if response_idx < len(self.responses) else ""
        content_text = (
            self.assistant_contents[response_idx]
            if response_idx < len(self.assistant_contents)
            else ""
        )
        if content_text:
            embedded_reasoning = ""
        else:
            content_text, embedded_reasoning = self._split_embedded_reasoning(response)
        reasoning = (
            self.reasonings[response_idx]
            if response_idx < len(self.reasonings)
            else embedded_reasoning
        )
        tool_calls = (
            self.native_tool_calls[response_idx]
            if response_idx < len(self.native_tool_calls)
            else []
        )

        if tool_calls:
            message: Dict[str, Any] = {"role": "assistant", "content": content_text or ""}
            message["tool_calls"] = tool_calls
        else:
            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
            }
        if self.preserve_reasoning_content and reasoning:
            # vLLM accepts historical thinking as "reasoning" and then maps it
            # to "reasoning_content" before Qwen3.5's chat template is applied.
            message["reasoning"] = reasoning
            # Keep the template-native key for direct tokenizer rendering and
            # vLLM versions that pass it through.
            message["reasoning_content"] = reasoning
        return message

    def _tool_response_messages_for_step(self, response_idx: int) -> List[Dict[str, Any]]:
        tool_calls = (
            self.native_tool_calls[response_idx]
            if response_idx < len(self.native_tool_calls)
            else []
        )
        messages: List[Dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": (
                        f"computer_use call {index + 1}/{len(tool_calls)} was executed. "
                        "The updated screenshot is provided in the following user message."
                    ),
                }
            )
        return messages

    def record_response(self, response: str) -> None:
        self.responses.append(response or "")
        self.reasonings.append(self._last_reasoning or "")
        self.assistant_contents.append(self._last_assistant_content or "")
        self.native_tool_calls.append(list(self._last_native_tool_calls or []))
        self._last_reasoning = ""
        self._last_assistant_content = ""
        self._last_native_tool_calls = []

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "thoughts": list(self.thoughts),
            "actions": list(self.actions),
            "observations": list(self.observations),
            "responses": list(self.responses),
            "reasonings": list(self.reasonings),
            "assistant_contents": list(self.assistant_contents),
            "native_tool_calls": list(self.native_tool_calls),
            "screenshots": list(self.screenshots),
            "folded_prefix_k": self.folded_prefix_k,
            "_last_reasoning": self._last_reasoning,
            "_last_assistant_content": self._last_assistant_content,
            "_last_native_tool_calls": list(self._last_native_tool_calls),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.thoughts = list(state.get("thoughts", []))
        self.actions = list(state.get("actions", []))
        self.observations = list(state.get("observations", []))
        self.responses = list(state.get("responses", []))
        self.reasonings = list(state.get("reasonings", []))
        self.assistant_contents = list(state.get("assistant_contents", []))
        self.native_tool_calls = list(state.get("native_tool_calls", []))
        self.screenshots = list(state.get("screenshots", []))
        self.folded_prefix_k = int(state.get("folded_prefix_k", 0))
        self._last_reasoning = str(state.get("_last_reasoning", ""))
        self._last_assistant_content = str(state.get("_last_assistant_content", ""))
        self._last_native_tool_calls = list(state.get("_last_native_tool_calls", []))

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]

        original_img = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = original_img.size

        processed_b64 = process_image(
            screenshot_bytes,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
        processed_width, processed_height = processed_img.size

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self._update_folding_state(total_steps)

        start_step = max(1, total_steps - self.history_n)

        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}"
            for i in range(0, min(start_step - 1, len(self.actions)))
        ]
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        system_prompt, tools_def = self.build_tool_prompt(
            processed_width=processed_width,
            processed_height=processed_height,
        )

        instruction_prompt = (
            f"\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n"
            f"{previous_actions_str}"
        )

        messages: List[Dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        for step_num in range(start_step, total_steps + 1):
            is_first_turn = step_num == start_step
            is_collapsed = self._should_collapse_step(step_num)

            if is_collapsed:
                parts = [{"type": "text", "text": self.collapse_text}]
                if is_first_turn:
                    # Align with the formal agent: when the first historical image is
                    # collapsed, keep only the instruction/user text instead of adding
                    # an extra placeholder block.
                    user_content = [{"type": "text", "text": instruction_prompt}]
                else:
                    user_content = self._wrap_tool_response(parts)
                messages.append({"role": "user", "content": user_content})
            else:
                img_url = f"data:image/png;base64,{self.screenshots[step_num - 1]}"
                if is_first_turn:
                    user_content = [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": instruction_prompt},
                    ]
                else:
                    user_content = self._wrap_tool_response(
                        [{"type": "image_url", "image_url": {"url": img_url}}]
                    )
                messages.append({"role": "user", "content": user_content})

            if step_num <= total_steps - 1 and (step_num - 1) < len(self.responses):
                response_idx = step_num - 1
                messages.append(self._assistant_message_for_step(response_idx))
                messages.extend(self._tool_response_messages_for_step(response_idx))

        try:
            draft_dir = "./draft/message_cache"
            os.makedirs(draft_dir, exist_ok=True)
            step_idx = total_steps - 1
            message_file_path = os.path.join(draft_dir, f"qwen35vl_messages_step_{step_idx}.json")
            with open(message_file_path, "w", encoding="utf-8") as file_obj:
                json.dump(self._sanitize_messages_for_dump(messages), file_obj, ensure_ascii=False, indent=2)
        except Exception as exc:
            if logger:
                logger.warning("[Qwen35VLAgent] failed to dump debug messages: %s", exc)

        response = self.call_llm(self._request_payload(messages, tools_def), self.model)

        if logger:
            logger.info("Qwen35VL Output: %s", response)
        self.record_response(response or "")

        low_level_instruction, pyautogui_code = self.parse_response(
            response or "",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

        if logger:
            logger.info("Low level instruction: %s", low_level_instruction)
            logger.info("Pyautogui code: %s", pyautogui_code)

        self.actions.append(low_level_instruction)
        return response or "", pyautogui_code

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        if not response or not response.strip():
            return low_level_instruction, pyautogui_code

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            if not (original_width and original_height):
                return int(x), int(y)
            if self.coordinate_type == "absolute":
                if processed_width and processed_height:
                    x_scale = original_width / processed_width
                    y_scale = original_height / processed_height
                    return int(x * x_scale), int(y * y_scale)
                return int(x), int(y)
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
            params: Dict = {}
            func_match = re.search(r"<function=([^>]+)>", xml_content)
            if not func_match or func_match.group(1) != "computer_use":
                return None

            for match in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL):
                name = match.group(1)
                value = match.group(2).strip()
                if value.startswith("[") or value.startswith("{"):
                    parsed_value = self._parse_jsonish(value)
                    if parsed_value is not value:
                        params[name] = parsed_value
                        continue
                params[name] = value
            return params

        def parse_json_tool_call(tool_call_content: str) -> Optional[Dict]:
            payload = self._parse_jsonish(tool_call_content.strip())
            if not isinstance(payload, dict):
                return None

            name = (
                payload.get("name")
                or payload.get("function")
                or ((payload.get("function_call") or {}).get("name") if isinstance(payload.get("function_call"), dict) else None)
            )
            arguments = (
                payload.get("arguments")
                if "arguments" in payload
                else payload.get("parameters")
            )
            if arguments is None and isinstance(payload.get("function_call"), dict):
                arguments = payload["function_call"].get("arguments")
            if name != "computer_use":
                return None
            params = self._coerce_tool_arguments(arguments)
            return params or None

        def parse_tool_call(tool_call_content: str) -> Optional[Dict]:
            return parse_xml_tool_call(tool_call_content) or parse_json_tool_call(tool_call_content)

        def parse_keys(raw_keys):
            if isinstance(raw_keys, str):
                raw_keys = raw_keys.strip()
                try:
                    raw_keys = json.loads(raw_keys)
                except Exception:
                    try:
                        raw_keys = ast.literal_eval(raw_keys)
                    except Exception:
                        raw_keys = raw_keys.split("+") if "+" in raw_keys else [raw_keys]

            def normalize_key(key) -> List[str]:
                if isinstance(key, list):
                    keys = []
                    for item in key:
                        keys.extend(normalize_key(item))
                    return keys
                key_text = str(key).strip()
                if not key_text:
                    return []
                if "," in key_text and (
                    key_text.startswith("[")
                    or key_text.endswith("]")
                    or key_text.startswith("(")
                    or key_text.endswith(")")
                ):
                    parts = key_text.split(",")
                else:
                    parts = [key_text]
                cleaned = []
                for part in parts:
                    part = part.strip().strip("[]()").strip().strip("\"'").strip()
                    if part:
                        cleaned.append(part.lower())
                return cleaned

            if isinstance(raw_keys, list):
                keys = []
                for key in raw_keys:
                    keys.extend(normalize_key(key))
                return keys
            return normalize_key(raw_keys)

        def parse_coordinate(raw_coord):
            if isinstance(raw_coord, str):
                try:
                    raw_coord = json.loads(raw_coord)
                except Exception:
                    return None
            if isinstance(raw_coord, list) and len(raw_coord) >= 2:
                return raw_coord[0], raw_coord[1]
            return None

        def process_tool_call_params(params: Dict) -> None:
            action = str(params.get("action", "")).strip().lower()
            if not action:
                return

            coordinate = parse_coordinate(params.get("coordinate"))
            text = params.get("text")

            def press_modifier_keys() -> None:
                if text:
                    for key in str(text).split("+"):
                        key = key.strip().lower()
                        if key:
                            pyautogui_code.append(f"pyautogui.keyDown({self._py_string(key)})")

            def release_modifier_keys() -> None:
                if text:
                    keys = [key.strip().lower() for key in str(text).split("+") if key.strip()]
                    for key in reversed(keys):
                        pyautogui_code.append(f"pyautogui.keyUp({self._py_string(key)})")

            if action == "left_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.click({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.click()")
                release_modifier_keys()
            elif action == "right_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.rightClick()")
                release_modifier_keys()
            elif action == "middle_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.middleClick()")
                release_modifier_keys()
            elif action == "double_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()
            elif action == "triple_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()
            elif action == "type":
                text = self._decode_type_text(params.get("text", ""))
                pyautogui_code.append(self._build_clipboard_paste_command(text))
            elif action == "key":
                keys = parse_keys(params.get("keys", []))
                keys_str = ", ".join(self._py_string(key) for key in keys)
                if len(keys) > 1:
                    pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                else:
                    pyautogui_code.append(f"pyautogui.press({keys_str})")
            elif action in {"scroll", "hscroll"}:
                press_modifier_keys()
                pixels = params.get("pixels", 0)
                try:
                    pixels = int(float(pixels))
                except Exception:
                    pixels = 0
                pyautogui_code.append(f"pyautogui.scroll({pixels})")
                release_modifier_keys()
            elif action == "wait":
                pyautogui_code.append("WAIT")
            elif action == "terminate":
                status = str(params.get("status", "")).strip().lower()
                if status in FAILURE_TERMINATE_STATUSES or _contains_infeasible_verdict(response):
                    pyautogui_code.append("FAIL")
                elif not status or status in SUCCESS_TERMINATE_STATUSES:
                    pyautogui_code.append("DONE")
                else:
                    pyautogui_code.append("DONE")
            elif action == "answer":
                pyautogui_code.append("DONE")
            elif action == "mouse_move":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.moveTo(0, 0)")
            elif action == "left_click_drag":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    duration = 0.5
                    if "duration" in params:
                        try:
                            duration = float(params["duration"])
                        except Exception:
                            duration = 0.5
                    pyautogui_code.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
                else:
                    pyautogui_code.append("pyautogui.dragTo(0, 0)")

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                low_level_instruction = stripped.split(":", 1)[-1].strip()
                break

        processed_tool_calls = 0
        for tool_call_match in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
            params = parse_tool_call(tool_call_match.group(1))
            if params:
                process_tool_call_params(params)
                processed_tool_calls += 1
                if processed_tool_calls >= self.max_tool_calls_per_turn:
                    break

        if not low_level_instruction and pyautogui_code:
            first_code = pyautogui_code[0]
            if first_code == "DONE":
                low_level_instruction = "Task completed"
            elif first_code == "FAIL":
                low_level_instruction = "Task infeasible"
            elif first_code == "WAIT":
                low_level_instruction = "Waiting"
            elif "." in first_code:
                low_level_instruction = f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
            else:
                low_level_instruction = "Performing action"
        elif not pyautogui_code and _contains_infeasible_verdict(response):
            low_level_instruction = low_level_instruction or "Task infeasible"
            pyautogui_code.append("FAIL")

        return low_level_instruction, pyautogui_code

    @staticmethod
    def _extract_content_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(part.get("text", ""))
                else:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            return "".join(parts)
        return str(content)

    @classmethod
    def _extract_reasoning_text(cls, msg) -> str:
        reasoning = cls._extract_content_text(getattr(msg, "reasoning", None))
        if reasoning:
            return reasoning
        reasoning = cls._extract_content_text(getattr(msg, "reasoning_content", None))
        if reasoning:
            return reasoning

        for attr in ("model_extra", "__dict__"):
            extra = getattr(msg, attr, None)
            if isinstance(extra, dict):
                reasoning = cls._extract_content_text(
                    extra.get("reasoning") or extra.get("reasoning_content")
                )
                if reasoning:
                    return reasoning

        if hasattr(msg, "model_dump"):
            try:
                dumped = msg.model_dump()
            except Exception:
                dumped = {}
            if isinstance(dumped, dict):
                return cls._extract_content_text(
                    dumped.get("reasoning") or dumped.get("reasoning_content")
                )
        return ""

    def _record_inference_call(self, started_at: float, ended_at: float) -> None:
        duration = max(0.0, ended_at - started_at)
        interval = {"start": started_at, "end": ended_at}
        self.inference_time_total += duration
        self.inference_intervals.append(interval)
        self._last_inference_time += duration
        self._last_inference_intervals.append(interval)

    def call_llm(self, payload: Dict, model: str) -> str:
        self._last_inference_time = 0.0
        self._last_inference_intervals = []
        base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        default_timeout = str(
            float(os.environ.get("OSWORLD_HTTP_CONNECT_TIMEOUT", "10"))
            + float(os.environ.get("OSWORLD_HTTP_READ_TIMEOUT", "120"))
        )
        timeout_s = float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", default_timeout))

        try:
            client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        except TypeError:
            client = openai.OpenAI(base_url=base_url, api_key=api_key)

        retryable_types = tuple(
            exc
            for exc in [
                SSLError,
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "RateLimitError", None),
                getattr(openai, "InternalServerError", None),
            ]
            if isinstance(exc, type)
        )

        extra_body: Dict = {}
        if self.top_k is not None and self.top_k > 0:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None and self.min_p > 0:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None and self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty
        # Qwen3.5's chat template defaults thinking ON. Make this explicit so
        # vllm honors the per-call choice regardless of how it was launched.
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}

        create_kwargs: Dict = dict(
            model=model,
            messages=payload["messages"],
            max_tokens=payload.get("max_tokens", self.max_tokens),
            temperature=payload.get("temperature", self.temperature),
            top_p=payload.get("top_p", self.top_p),
        )
        if payload.get("tools"):
            create_kwargs["tools"] = payload["tools"]
            create_kwargs["tool_choice"] = payload.get("tool_choice", "auto")
        if self.presence_penalty is not None and self.presence_penalty != 0.0:
            create_kwargs["presence_penalty"] = self.presence_penalty
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        max_retries = int(payload.get("max_retries", MAX_RETRY_TIMES))
        retry_backoff_seconds = float(
            payload.get("retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS)
        )
        retry_backoff_max_seconds = float(
            payload.get("retry_backoff_max_seconds", DEFAULT_RETRY_BACKOFF_MAX_SECONDS)
        )
        last_err: Optional[Exception] = None
        bad_request_type = getattr(openai, "BadRequestError", None)
        downgraded_tool_calling = False
        for attempt in range(1, max_retries + 1):
            try:
                inference_started_at = time.time()
                try:
                    response = client.chat.completions.create(**create_kwargs)
                finally:
                    self._record_inference_call(inference_started_at, time.time())
                msg = response.choices[0].message
                content = msg.content
                text = self._extract_content_text(content)
                reasoning = self._extract_reasoning_text(msg)
                tool_calls = self._normalize_native_tool_calls(self._extract_tool_calls(msg))
                tool_text = self._serialize_native_tool_calls(tool_calls)
                self._last_native_tool_calls = tool_calls
                self._last_assistant_content = text or ""
                if tool_text:
                    text = f"{text.rstrip()}\n\n{tool_text}".strip()
                # Fallback: some models (e.g. Qwen3.5-2B) put all output
                # into reasoning/thinking tokens when served with
                # --reasoning-parser, leaving content empty.
                if not text and not tool_calls and reasoning:
                    text = reasoning
                    reasoning = ""
                    self._last_assistant_content = text
                self._last_reasoning = reasoning
                # Optionally prepend reasoning in <think> tags so it is
                # preserved in conversation history for subsequent turns.
                if self.keep_reasoning and reasoning and text != reasoning:
                    text = f"<think>\n{reasoning}\n</think>\n\n{text}"
                    if not self.preserve_reasoning_content:
                        self._last_assistant_content = text
                return text
            except Exception as exc:
                if (
                    bad_request_type
                    and isinstance(exc, bad_request_type)
                    and "tools" in create_kwargs
                    and not downgraded_tool_calling
                ):
                    downgraded_tool_calling = True
                    fallback_tools_def = create_kwargs["tools"][0]
                    fallback_system_prompt = self._build_system_prompt(
                        fallback_tools_def,
                        use_native=False,
                    )
                    fallback_messages = [dict(message) for message in create_kwargs["messages"]]
                    if fallback_messages:
                        system_message = dict(fallback_messages[0])
                        content = system_message.get("content")
                        if isinstance(content, list):
                            content = list(content)
                            replaced_text = False
                            for index, part in enumerate(content):
                                if isinstance(part, dict) and part.get("type") == "text":
                                    updated_part = dict(part)
                                    updated_part["text"] = fallback_system_prompt
                                    content[index] = updated_part
                                    replaced_text = True
                                    break
                            if not replaced_text:
                                content.insert(0, {"type": "text", "text": fallback_system_prompt})
                            system_message["content"] = content
                        else:
                            system_message["content"] = [{"type": "text", "text": fallback_system_prompt}]
                        fallback_messages[0] = system_message
                        create_kwargs["messages"] = fallback_messages
                    create_kwargs.pop("tools", None)
                    create_kwargs.pop("tool_choice", None)
                    if logger:
                        logger.warning(
                            "[Qwen35VLAgent] native tool calling rejected by server; retrying with XML fallback prompt: %s",
                            exc,
                        )
                    continue
                if not isinstance(exc, retryable_types):
                    raise
                last_err = exc
                if logger:
                    logger.warning(
                        "[Qwen35VLAgent] call_llm failed attempt %d/%d: %s",
                        attempt,
                        max_retries,
                        exc,
                    )
                if attempt < max_retries and retry_backoff_seconds > 0:
                    time.sleep(
                        min(retry_backoff_seconds * attempt, retry_backoff_max_seconds)
                    )

        if last_err is not None:
            raise last_err
        return ""

    def reset(self, _logger=None, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen35vl_agent")
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.reasonings = []
        self.assistant_contents = []
        self.native_tool_calls = []
        self.screenshots = []
        self.folded_prefix_k = 0
        self._last_reasoning = ""
        self._last_assistant_content = ""
        self._last_native_tool_calls = []
        self.inference_time_total = 0.0
        self.inference_intervals = []
        self._last_inference_time = 0.0
        self._last_inference_intervals = []
