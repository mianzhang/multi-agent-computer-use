"""Utilities for calling LLM APIs (OpenAI, Anthropic/Claude, Google Gemini)."""

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

# Triple-backtick fenced code block with an optional language tag
# (e.g. ```json, ```javascript). The language tag is stripped; only the
# content between the fences is captured.
_FENCE_PATTERN = re.compile(
    r"```(?:\w+)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Single-backtick inline code span. Intentionally excludes backticks so it
# doesn't swallow neighbouring fences, but allows newlines so multi-line
# inline spans work.
_INLINE_PATTERN = re.compile(r"`([^`]+)`", re.DOTALL)

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

TOKENS_PER_MILLION = 1_000_000

# Rates per 1M tokens (USD). Qwen / vllm-served models are intentionally
# absent — local inference, no API charge — so compute_cost returns $0.00.
MODEL_RATES = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-20250514": {"input": 5.0, "output": 25.0},
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.5},
}

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a given model and token counts. Returns 0.0 for unknown/free models."""
    rates = MODEL_RATES.get(model)
    if not rates:
        return 0.0
    input_cost = input_tokens / TOKENS_PER_MILLION * rates["input"]
    output_cost = output_tokens / TOKENS_PER_MILLION * rates["output"]
    return input_cost + output_cost


@dataclass
class UsageInfo:
    """Token usage info from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    inference_seconds: float = 0.0
    inference_intervals: list[tuple[float, float]] = field(default_factory=list)


def _read_image(image_path: Union[str, Path], *, as_base64: bool) -> tuple[Union[str, bytes], str]:
    """Read an image file and return (data, media_type).

    ``data`` is base64 text when ``as_base64`` is true, otherwise raw bytes.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    with open(path, "rb") as f:
        raw_data = f.read()
    if as_base64:
        return base64.standard_b64encode(raw_data).decode("ascii"), media_type
    return raw_data, media_type


def _normalize_image_paths(
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]],
) -> list[Union[str, Path]]:
    """Return image inputs as a flat list, preserving caller order.

    Accepts a single str/Path, a sequence of str/Path, or None. None entries
    inside a sequence are filtered out so callers can pass a list assembled
    from possibly-missing screenshots without sprinkling None checks.
    """
    if image_path is None:
        return []
    if isinstance(image_path, (str, Path)):
        return [image_path]
    return [p for p in image_path if p is not None]


def _segments_to_provider_content(segments: list[dict], provider: str) -> list[dict]:
    """Convert a list of ``{type: text|image, ...}`` segments to a provider message body.

    Each segment is one of:
    * ``{"type": "text", "text": "..."}``
    * ``{"type": "image", "path": "/abs/path.png"}``

    The output is the provider-specific ``content`` list for the user
    message, preserving segment order so callers can interleave text and
    images at exactly the positions they want (e.g. one screenshot
    immediately after each branch's text summary).
    """
    out: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError(f"segment must be a dict, got {type(seg).__name__}")
        seg_type = seg.get("type")
        if seg_type == "text":
            text = seg.get("text", "")
            if text:
                out.append({"type": "text", "text": text})
        elif seg_type == "image":
            path = seg.get("path")
            if not path:
                continue
            data, media_type = _read_image(path, as_base64=True)
            if provider == "openai":
                out.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                )
            else:  # anthropic
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
        else:
            raise ValueError(f"unknown segment type: {seg_type!r}")
    return out


def _segments_to_google_contents(segments: list[dict]) -> list[object]:
    """Convert interleaved text/image segments to Google Gen AI content parts."""
    from google.genai import types

    out: list[object] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError(f"segment must be a dict, got {type(seg).__name__}")
        seg_type = seg.get("type")
        if seg_type == "text":
            text = seg.get("text", "")
            if text:
                out.append(types.Part.from_text(text=text))
        elif seg_type == "image":
            path = seg.get("path")
            if not path:
                continue
            data, media_type = _read_image(path, as_base64=False)
            assert isinstance(data, bytes)
            out.append(types.Part.from_bytes(data=data, mime_type=media_type))
        else:
            raise ValueError(f"unknown segment type: {seg_type!r}")
    return out


_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _openai_accepts_reasoning_effort(model: str) -> bool:
    """True if the OpenAI model accepts the ``reasoning_effort`` parameter.

    Reasoning effort is only honored by reasoning-capable families (o-series,
    gpt-5). For legacy chat models (gpt-4o, gpt-4-turbo, ...) the OpenAI SDK
    rejects the parameter, so callers must skip passing it through.
    """
    if not model:
        return False
    m = model.lower()
    return m.startswith(_OPENAI_REASONING_PREFIXES)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    user_segments: Optional[list[dict]] = None,
    reasoning_effort: Optional[str] = None,
) -> tuple[str, UsageInfo]:
    """Call the OpenAI API with the given prompts.

    Three ways to populate the user message, in priority order:

    1. ``user_segments`` -- a list of ``{"type": "text", ...}`` and
       ``{"type": "image", "path": ...}`` dicts. Used as-is, in order, so
       callers can interleave text and images at exact positions. Wins over
       both ``user_prompt`` and ``image_path``.
    2. ``user_prompt`` + ``image_path`` -- legacy form. All images are
       prepended to the user message as ``image_url`` content blocks
       followed by the text.
    3. Just ``user_prompt`` -- a plain text user message.

    Returns:
        Tuple of (response_text, UsageInfo).
    """
    from openai import OpenAI

    # Manager-side base URL override. When call_openai is invoked inside a
    # harness that has OPENAI_BASE_URL pointing at a vllm server (for the CUA
    # client), manager calls for real OpenAI models need a separate endpoint.
    # Set MACU_MANAGER_BASE_URL (e.g. https://api.openai.com/v1) to route
    # call_openai to OpenAI even when OPENAI_BASE_URL is taken by vllm.
    mgr_base_url = os.environ.get("MACU_MANAGER_BASE_URL")
    if mgr_base_url:
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=mgr_base_url)
    else:
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    if user_segments is not None:
        user_content = _segments_to_provider_content(user_segments, provider="openai")
    else:
        image_paths = _normalize_image_paths(image_path)
        if image_paths:
            user_content = []
            for item in image_paths:
                data, media_type = _read_image(item, as_base64=True)
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                )
            user_content.append({"type": "text", "text": user_prompt})
        else:
            user_content = user_prompt

    create_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=max_tokens,
    )
    if _openai_accepts_reasoning_effort(model):
        # Reasoning models (o-series, gpt-5) only accept the default
        # temperature=1, so omit the parameter entirely rather than sending
        # whatever the caller passed.
        if reasoning_effort:
            create_kwargs["reasoning_effort"] = reasoning_effort
    else:
        create_kwargs["temperature"] = temperature

    inference_started_at = time.time()
    response = client.chat.completions.create(**create_kwargs)
    inference_ended_at = time.time()
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    cost = compute_cost(model, input_tokens, output_tokens)
    inference_seconds = inference_ended_at - inference_started_at
    return response.choices[0].message.content, UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost_usd=cost,
        inference_seconds=inference_seconds,
        inference_intervals=[(inference_started_at, inference_ended_at)],
    )


def call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    user_segments: Optional[list[dict]] = None,
) -> tuple[str, UsageInfo]:
    """Call the Anthropic (Claude) API with the given prompts.

    Three ways to populate the user message, in priority order:

    1. ``user_segments`` -- a list of ``{"type": "text", ...}`` and
       ``{"type": "image", "path": ...}`` dicts. Used as-is, in order, so
       callers can interleave text and images at exact positions. Wins over
       both ``user_prompt`` and ``image_path``.
    2. ``user_prompt`` + ``image_path`` -- legacy form. All images are
       prepended to the user message as ``image`` content blocks followed
       by the text.
    3. Just ``user_prompt`` -- a plain text user message.

    Returns:
        Tuple of (response_text, UsageInfo).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    if user_segments is not None:
        user_content = _segments_to_provider_content(user_segments, provider="anthropic")
    else:
        image_paths = _normalize_image_paths(image_path)
        if image_paths:
            user_content = []
            for item in image_paths:
                data, media_type = _read_image(item, as_base64=True)
                user_content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
            user_content.append({"type": "text", "text": user_prompt})
        else:
            user_content = user_prompt

    inference_started_at = time.time()
    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    inference_ended_at = time.time()
    usage = response.usage
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    cost = compute_cost(model, input_tokens, output_tokens)
    inference_seconds = inference_ended_at - inference_started_at
    return response.content[0].text, UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost_usd=cost,
        inference_seconds=inference_seconds,
        inference_intervals=[(inference_started_at, inference_ended_at)],
    )


# Cheap insurance for vllm deployments without --reasoning-parser qwen3:
# the raw model emits "<think>...</think>final answer" and we want the answer.
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def call_huggingface(
    system_prompt: str,
    user_prompt: str,
    model: str = "Qwen/Qwen3.6-27B",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    user_segments: Optional[list[dict]] = None,
) -> tuple[str, UsageInfo]:
    """Call a vllm-served HuggingFace model via the OpenAI-compatible API.

    Resolves base_url from env in priority: ``MACU_MANAGER_BASE_URL`` →
    ``OPENAI_BASE_URL`` → ``http://127.0.0.1:8000/v1``. ``api_key`` defaults
    to ``OPENAI_API_KEY`` env var or ``"dummy"`` (vllm doesn't authenticate).

    Sampling defaults target Qwen3 thinking-mode:
    ``temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=0.0, repetition_penalty=1.0``. ``top_k``, ``min_p``,
    and ``repetition_penalty`` are vllm extensions and are passed through
    ``extra_body``. Explicit ``temperature`` / ``top_p`` arguments always
    override the defaults.

    The user message is built the same way as ``call_openai`` (vllm
    speaks the OpenAI ``image_url`` format for multimodal models).

    Cost is reported as $0.00 for models not listed in ``MODEL_RATES``.
    """
    from openai import OpenAI

    base_url = (
        os.environ.get("MACU_MANAGER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    )
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
    client = OpenAI(base_url=base_url, api_key=resolved_key)

    if user_segments is not None:
        user_content = _segments_to_provider_content(user_segments, provider="openai")
    else:
        image_paths = _normalize_image_paths(image_path)
        if image_paths:
            user_content = []
            for item in image_paths:
                data, media_type = _read_image(item, as_base64=True)
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                )
            user_content.append({"type": "text", "text": user_prompt})
        else:
            user_content = user_prompt

    eff_temperature = temperature if temperature is not None else 0.6
    eff_top_p = top_p if top_p is not None else 0.95

    create_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=eff_temperature,
        top_p=eff_top_p,
        presence_penalty=0.0,
        max_completion_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        },
    )

    # Retry transient connection errors from local or remote vllm deployments.
    # 8 attempts with jittered exponential backoff (1, 2, 4, 8, 16, 32,
    # 64 s + jitter; max ~127 s total). Jitter prevents the herd of
    # shards from synchronizing their retries on a chain-wide outage.
    import random as _random
    from openai import APIConnectionError as _APIConnectionError
    try:
        from httpx import RemoteProtocolError as _RemoteProtocolError, ReadTimeout as _ReadTimeout
    except Exception:  # httpx is an openai dep, but stay defensive
        _RemoteProtocolError = ()  # type: ignore[assignment]
        _ReadTimeout = ()  # type: ignore[assignment]

    _retry_excs: tuple = (_APIConnectionError,)
    if _RemoteProtocolError:
        _retry_excs = _retry_excs + (_RemoteProtocolError,)
    if _ReadTimeout:
        _retry_excs = _retry_excs + (_ReadTimeout,)

    inference_started_at = time.time()
    last_exc = None
    _MAX_ATTEMPTS = 8
    for _attempt in range(_MAX_ATTEMPTS):  # 1 initial + 7 retries
        try:
            response = client.chat.completions.create(**create_kwargs)
            break
        except _retry_excs as exc:
            last_exc = exc
            if _attempt == _MAX_ATTEMPTS - 1:
                raise
            base = min(2 ** _attempt, 64)        # 1, 2, 4, 8, 16, 32, 64
            time.sleep(base * (0.5 + _random.random()))  # 0.5x..1.5x jitter
    else:  # pragma: no cover — for never break
        raise last_exc  # type: ignore[misc]
    inference_ended_at = time.time()
    text = response.choices[0].message.content or ""
    text = _THINK_PREFIX_RE.sub("", text)

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    cost = compute_cost(model, input_tokens, output_tokens)
    inference_seconds = inference_ended_at - inference_started_at
    return text, UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost_usd=cost,
        inference_seconds=inference_seconds,
        inference_intervals=[(inference_started_at, inference_ended_at)],
    )


def call_google(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3.1-flash-lite-preview",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    user_segments: Optional[list[dict]] = None,
) -> tuple[str, UsageInfo]:
    """Call Gemini through the Google Gen AI SDK.

    Uses ``GEMINI_API_KEY`` from the environment. ``GOOGLE_API_KEY`` is
    accepted as a compatibility fallback. The content construction mirrors
    the other providers: ``user_segments`` preserves interleaving, while
    ``image_path`` prepends images before the user text.
    """
    from google import genai
    from google.genai import types

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError("GEMINI_API_KEY is required for provider='google'")

    client = genai.Client(api_key=resolved_key)

    if user_segments is not None:
        contents = _segments_to_google_contents(user_segments)
    else:
        contents: list[object] = []
        for item in _normalize_image_paths(image_path):
            data, media_type = _read_image(item, as_base64=False)
            assert isinstance(data, bytes)
            contents.append(types.Part.from_bytes(data=data, mime_type=media_type))
        if user_prompt:
            contents.append(types.Part.from_text(text=user_prompt))
        if not contents:
            contents = [types.Part.from_text(text="")]

    inference_started_at = time.time()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    inference_ended_at = time.time()
    text = response.text or ""

    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    if total_tokens >= input_tokens and total_tokens:
        output_tokens = total_tokens - input_tokens
    else:
        output_tokens = candidate_tokens + thoughts_tokens
    cost = compute_cost(model, input_tokens, output_tokens)
    inference_seconds = inference_ended_at - inference_started_at
    return text, UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost_usd=cost,
        inference_seconds=inference_seconds,
        inference_intervals=[(inference_started_at, inference_ended_at)],
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    image_path: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    user_segments: Optional[list[dict]] = None,
    reasoning_effort: Optional[str] = None,
) -> tuple[str, UsageInfo]:
    """Unified interface for calling LLM APIs.

    Pass ``user_segments`` (a list of ``{type: text|image, ...}`` dicts) to
    interleave text and images at exact positions in the user message.
    Otherwise pass ``user_prompt`` + optional ``image_path`` for the legacy
    text-then-images form.

    ``reasoning_effort`` is forwarded to OpenAI reasoning-capable models
    (o-series / gpt-5) and ignored for other providers and non-reasoning
    OpenAI models.

    Returns:
        Tuple of (response_text, UsageInfo).
    """
    if provider == "openai":
        return call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or "gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            image_path=image_path,
            user_segments=user_segments,
            reasoning_effort=reasoning_effort,
        )
    elif provider == "anthropic":
        return call_anthropic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or "claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            image_path=image_path,
            user_segments=user_segments,
        )
    elif provider == "huggingface":
        # Dispatcher default temperature=0.0 is greedy decoding, which is
        # wrong for Qwen thinking. Forward temperature only when the caller
        # explicitly overrode the default; otherwise let call_huggingface
        # pick the Qwen default.
        kw: dict = dict(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or "Qwen/Qwen3.6-27B",
            max_tokens=max_tokens,
            api_key=api_key,
            image_path=image_path,
            user_segments=user_segments,
        )
        if temperature != 0.0:
            kw["temperature"] = temperature
        return call_huggingface(**kw)
    elif provider == "google":
        return call_google(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or "gemini-3.1-flash-lite-preview",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            image_path=image_path,
            user_segments=user_segments,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'openai', 'anthropic', 'huggingface', "
            "or 'google'."
        )


def load_prompt(yaml_path: str) -> dict:
    """Load a prompt YAML file and return its contents as a dict.

    Args:
        yaml_path: Path to the YAML prompt file.

    Returns:
        Dict with keys like 'system_prompt', 'user_prompt_template', etc.
    """
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def render_prompt(prompt_config: dict, **kwargs) -> tuple[str, str]:
    """Render a prompt config into (system_prompt, user_prompt).

    Args:
        prompt_config: Dict loaded from a YAML prompt file.
        **kwargs: Variables to substitute into the user_prompt_template.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = prompt_config["system_prompt"]
    user_prompt = prompt_config["user_prompt_template"].format(**kwargs)
    return system_prompt, user_prompt


def parse_json_response(response: str) -> dict:
    """Parse a JSON object from an LLM response.

    Tolerates the wrappers that LLMs add around JSON -- markdown fences,
    reasoning prose, trailing commentary -- even when the prompt explicitly
    asks for "JSON only". Strategies are tried in order and the first one
    that yields valid JSON wins:

    1. ``json.loads`` on the full (stripped) response -- fast path for models
       that obey the instruction.
    2. Scan for triple-backtick fenced code blocks anywhere in the text and
       parse each one's contents. An optional language tag (``` ```json``` ```,
       ``` ```javascript``` ```, etc.) is stripped.
    3. Scan for single-backtick inline code spans and parse each one's
       contents.
    4. Fall back to extracting the outermost JSON bracket pair: find the
       first ``{`` or ``[`` and the last matching ``}`` or ``]`` and parse
       that slice. Handles the common "manager wrote reasoning paragraphs,
       then bare JSON, no fence" shape that claude-opus produces under some
       prompts. Tried last so that fenced content is always preferred over
       a potentially-sloppy bracket match.

    Raises ``json.JSONDecodeError`` if none of the strategies succeeds.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON as a dict.
    """
    text = response.strip()

    # Strategy 1: the entire response is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: triple-backtick fenced block. Try every fence in the
    # response so a degenerate empty fence doesn't block a later real one.
    for m in _FENCE_PATTERN.finditer(text):
        content = m.group(1).strip()
        if not content:
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            continue

    # Strategy 3: single-backtick inline code span.
    for m in _INLINE_PATTERN.finditer(text):
        content = m.group(1).strip()
        if not content:
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            continue

    # Strategy 4: scan for top-level balanced bracket groups (respecting
    # string literals so braces inside quoted JSON strings don't throw off
    # the depth counter) and try each as standalone JSON. Prefer the LAST
    # group that parses — when an LLM emits a draft object followed by a
    # revised one separated by prose ("Wait, let me fix that. {...}"), the
    # second is the intended answer.
    last_parsed = None
    for candidate in _iter_balanced_groups(text):
        try:
            last_parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
    if last_parsed is not None:
        return last_parsed

    # Strategy 5: greedy outermost bracket fallback for malformed text where
    # the balanced scan finds nothing parseable (e.g., unbalanced or truncated
    # output). Pick whichever opener appears first and trim to the last
    # matching closer.
    first_obj = text.find("{")
    first_arr = text.find("[")
    candidates = [i for i in (first_obj, first_arr) if i != -1]
    if candidates:
        start = min(candidates)
        end_char = "}" if text[start] == "{" else "]"
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("no JSON content found in response", text, 0)


_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = set(_OPENERS.values())


def _iter_balanced_groups(text: str):
    """Yield substrings for every top-level balanced ``{...}`` or ``[...]``
    group in ``text``, skipping over double-quoted string literals so brackets
    inside JSON string values don't corrupt the depth counter. Tracks both
    bracket kinds on a single stack so a ``[`` nested inside ``{`` is not
    treated as top-level.
    """
    stack: list[str] = []
    start = -1
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            # skip past the string literal, honoring backslash escapes
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch in _OPENERS:
            if not stack:
                start = i
            stack.append(_OPENERS[ch])
        elif ch in _CLOSERS and stack:
            if ch == stack[-1]:
                stack.pop()
                if not stack and start != -1:
                    yield text[start : i + 1]
                    start = -1
            else:
                # mismatched closer -- abandon this group and resync
                stack.clear()
                start = -1
        i += 1
