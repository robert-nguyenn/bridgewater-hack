"""Shared Anthropic client wrapper.

Centralizes model selection, prompt caching, tool use for structured
output, and logging. All agents call into this so we can tweak caching
and retries in one place.

Caching strategy: the system prompt and any injected context (channel
catalog, event catalog) are cached with ephemeral cache_control. The
user message is the only per call unique payload.

Logging: every call writes a line to data/runs/<run_id>/llm_log.jsonl
with timestamp, model, tool name, input tokens, output tokens, cache
read/write tokens.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR_ROOT = PROJECT_ROOT / "data" / "runs"

DEFAULT_MODEL = "claude-opus-4-7"
FAST_MODEL = "claude-sonnet-4-6"  # for volume calls where cost matters


_client: Optional[Anthropic] = None
_async_client: Optional[AsyncAnthropic] = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = Anthropic(api_key=api_key)
    return _client


def get_async_client() -> AsyncAnthropic:
    global _async_client
    if _async_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _async_client = AsyncAnthropic(api_key=api_key)
    return _async_client


def new_run_id() -> str:
    """Create a run id and its log directory."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    (RUN_DIR_ROOT / run_id).mkdir(parents=True, exist_ok=True)
    return run_id


def _log_call(run_id: Optional[str], payload: dict) -> None:
    if run_id is None:
        return
    log_file = RUN_DIR_ROOT / run_id / "llm_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def call_tool(
    *,
    system: str,
    cacheable_context: Optional[str],
    user: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    run_id: Optional[str] = None,
    caller: str = "",
) -> dict:
    """Call Claude with a single forced tool. Return the tool input as a dict.

    Prompt caching applies to `system` and `cacheable_context` if provided.
    """
    client = get_client()

    # Build the system content with cache_control on the stable parts
    sys_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    if cacheable_context:
        sys_blocks.append({
            "type": "text",
            "text": cacheable_context,
            "cache_control": {"type": "ephemeral"},
        })

    tools = [{
        "name": tool_name,
        "description": tool_description,
        "input_schema": tool_schema,
    }]

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=sys_blocks,
        tools=tools,
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0

    # Extract the tool call input
    tool_input: Optional[dict] = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input
            break

    usage = resp.usage
    _log_call(run_id, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": caller,
        "model": resp.model,
        "tool": tool_name,
        "stop_reason": resp.stop_reason,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    })

    if tool_input is None:
        raise RuntimeError(
            f"Model did not invoke tool '{tool_name}'. stop_reason={resp.stop_reason}"
        )
    return tool_input


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=8))
def call_text(
    *,
    system: str,
    cacheable_context: Optional[str] = None,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    run_id: Optional[str] = None,
    caller: str = "",
) -> str:
    """Plain text call with prompt caching."""
    client = get_client()
    sys_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    if cacheable_context:
        sys_blocks.append({
            "type": "text",
            "text": cacheable_context,
            "cache_control": {"type": "ephemeral"},
        })

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=sys_blocks,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0

    usage = resp.usage
    _log_call(run_id, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": caller,
        "model": resp.model,
        "tool": "(text)",
        "stop_reason": resp.stop_reason,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
    })

    text_parts = [b.text for b in resp.content if b.type == "text"]
    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Async variants for parallel execution
# ---------------------------------------------------------------------------
async def call_tool_async(
    *,
    system: str,
    cacheable_context: Optional[str],
    user: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    run_id: Optional[str] = None,
    caller: str = "",
    enable_web_search: bool = False,
    web_search_max_uses: int = 5,
) -> dict:
    """Async version of call_tool for parallel invocations. Same caching and logging.

    Set enable_web_search=True to give Claude access to Anthropic's server-side
    web_search tool. Claude uses it transparently; the final response still
    comes back via tool_name. web_search usage is logged as
    web_search_requests in the run log.
    """
    client = get_async_client()

    sys_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    if cacheable_context:
        sys_blocks.append({
            "type": "text",
            "text": cacheable_context,
            "cache_control": {"type": "ephemeral"},
        })

    tools: list[dict[str, Any]] = [{
        "name": tool_name,
        "description": tool_description,
        "input_schema": tool_schema,
    }]
    if enable_web_search:
        tools.append({
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": web_search_max_uses,
        })

    # Force custom tool only when NOT using web search (otherwise Claude
    # needs freedom to call web_search first, then the custom tool).
    tool_choice: dict[str, Any]
    if enable_web_search:
        tool_choice = {"type": "auto"}
    else:
        tool_choice = {"type": "tool", "name": tool_name}

    t0 = time.time()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=sys_blocks,
        tools=tools,
        tool_choice=tool_choice,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0

    tool_input: Optional[dict] = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input
            break

    usage = resp.usage
    server_use = getattr(usage, "server_tool_use", None)
    web_search_requests = getattr(server_use, "web_search_requests", 0) if server_use else 0
    _log_call(run_id, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": caller,
        "model": resp.model,
        "tool": tool_name,
        "stop_reason": resp.stop_reason,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "web_search_requests": web_search_requests,
    })

    if tool_input is None:
        raise RuntimeError(
            f"Model did not invoke tool '{tool_name}'. stop_reason={resp.stop_reason}"
        )
    return tool_input


async def call_with_images_async(
    *,
    system: str,
    text: str,
    images: list[tuple[str, bytes]],  # list of (media_type, raw_bytes)
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 3000,
    run_id: Optional[str] = None,
    caller: str = "",
) -> dict:
    """Vision call with one or more images. Used by the review agent to verify
    that plots match textual claims."""
    import base64

    client = get_async_client()
    sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    content: list[dict[str, Any]] = []
    for media_type, data in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        })
    content.append({"type": "text", "text": text})

    tools = [{
        "name": tool_name,
        "description": tool_description,
        "input_schema": tool_schema,
    }]

    t0 = time.time()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=sys_blocks,
        tools=tools,
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": content}],
    )
    elapsed = time.time() - t0

    tool_input: Optional[dict] = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input
            break

    usage = resp.usage
    _log_call(run_id, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": caller,
        "model": resp.model,
        "tool": tool_name,
        "stop_reason": resp.stop_reason,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "n_images": len(images),
    })

    if tool_input is None:
        raise RuntimeError(f"Model did not invoke tool '{tool_name}'.")
    return tool_input
