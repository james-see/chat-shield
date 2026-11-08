"""Chat Shield middleware callbacks.

``on_llm_request`` runs before the provider call and redacts sensitive data.
``on_llm_execution`` wraps the provider call and restores placeholders in the
response (streaming and non-streaming).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from . import redactor
from . import detectors as det_mod
from . import stats

logger = logging.getLogger(__name__)

# Module-level config + engine holder, set during registration
_config: dict[str, Any] = {}
_rampart_engine = None  # RampartEngine or None


def set_config(config: dict) -> None:
    global _config
    _config = config


def set_rampart(engine) -> None:
    global _rampart_engine
    _rampart_engine = engine


def on_llm_request(**kwargs) -> dict | None:
    """llm_request middleware — redact sensitive data before provider call."""
    if _config.get("mode") == "off":
        return None

    request = dict(kwargs.get("request") or {})
    request_id = kwargs.get("api_request_id", "") or f"fallback-{time.monotonic_ns()}"

    active_detectors = det_mod.get_active_detectors(_config)
    rampart = _rampart_engine if _config.get("rampart_enabled", True) else None

    # Find messages — OpenAI uses "messages", Responses API uses "input"
    messages = request.get("messages")
    if messages is None:
        messages = request.get("input")
    if not isinstance(messages, list):
        return None

    # Work on copies so we don't mutate the original
    messages = [_copy_msg(m) for m in messages]

    if _config.get("mode") == "warn":
        # Warn mode: detect but don't redact
        _warn_only(messages, active_detectors, rampart)
        return None

    # Redact mode — only redact the LAST user message, not the entire history.
    # The LLM already saw historical content in prior turns; redacting it now is
    # counterproductive and breaks the conversation context with 100+ placeholders.
    last_user_msg = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_msg = msg  # keep ref to the last one

    if last_user_msg is None:
        return None  # no user message in this request

    # Redact only the last user message
    single = [last_user_msg]
    full_mapping = redactor.redact_messages(single, active_detectors, rampart)

    if not full_mapping:
        return None

    # Copy the redacted content back into the full message list
    last_user_msg["content"] = single[0].get("content", last_user_msg.get("content"))

    # Force non-streaming so we can reliably restore placeholders in the response.
    # Streaming restoration is fragile across providers — we'll handle that post-hackathon.
    request["stream"] = False
    if "stream_options" in request:
        del request["stream_options"]

    # Store mapping for restoration in llm_execution
    redactor.store_mapping(request_id, full_mapping)

    # Update request with redacted messages
    if "messages" in request:
        request["messages"] = messages
    elif "input" in request:
        request["input"] = messages

    # Track stats
    for placeholder in full_mapping:
        parts = placeholder.split(":")
        if len(parts) >= 2:
            stats.record_redaction(parts[1])

    logger.info(
        "Chat Shield redacted %d items for request %s",
        len(full_mapping),
        request_id[:16],
    )

    return {
        "request": request,
        "source": "chat-shield",
        "reason": f"redacted {len(full_mapping)} sensitive items",
    }


def on_llm_execution(**kwargs) -> Any:
    """llm_execution middleware — restore placeholders in provider response."""
    request_id = kwargs.get("api_request_id", "") or f"fallback-{time.monotonic_ns()}"

    mapping = redactor.get_mapping(request_id)
    if not mapping:
        # Nothing was redacted — pass through
        return kwargs["next_call"](kwargs["request"])

    response = kwargs["next_call"](kwargs["request"])

    # Determine if response is streaming or non-streaming
    if _is_streaming(response):
        return _wrap_stream(response, mapping, request_id)
    else:
        _restore_non_stream(response, mapping)
        redactor.cleanup_mapping(request_id)
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_msg(msg) -> dict:
    """Deep-ish copy of a message dict (shallow for nested objects)."""
    if not isinstance(msg, dict):
        return msg
    out = {}
    for k, v in msg.items():
        if isinstance(v, list):
            out[k] = [
                _copy_msg(item) if isinstance(item, dict) else item for item in v
            ]
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def _warn_only(messages, active_detectors, rampart) -> None:
    """Detect sensitive data and log warnings without redacting."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        spans = det_mod.scan_text(content, active_detectors)
        if rampart is not None:
            spans.extend(rampart.detect(content))
        if spans:
            logger.warning(
                "Chat Shield detected %d sensitive items (warn mode, not redacted): %s",
                len(spans),
                [s[1] for s in spans],
            )


def _is_streaming(response) -> bool:
    """Check if the response is a streaming iterator."""
    if response is None:
        return False
    # Streaming responses have __iter__ and __next__ but aren't basic types
    if isinstance(response, (str, bytes, dict, list, int, float, bool, type(None))):
        return False
    return hasattr(response, "__iter__") and hasattr(response, "__next__")


def _wrap_stream(original_stream, mapping: dict, request_id: str):
    """Wrap a streaming response, restoring placeholders in each chunk."""

    def _restored():
        buffer = ""
        try:
            for chunk in original_stream:
                text = _extract_delta(chunk)
                if text:
                    released, buffer = redactor.restore_streaming_chunk(
                        text, mapping, buffer
                    )
                    if released:
                        _patch_chunk_text(chunk, released)
                yield chunk
            # Flush remaining buffer
            if buffer:
                restored, _ = redactor.restore_text(buffer, mapping)
                if restored != buffer:
                    # Emit a final text chunk with the restored content
                    yield _make_text_chunk(restored)
        finally:
            redactor.cleanup_mapping(request_id)

    return _restored()


def _extract_delta(chunk) -> str:
    """Extract the text delta from a streaming chunk (provider-agnostic)."""
    # OpenAI ChatCompletionChunk
    try:
        if hasattr(chunk, "choices"):
            for choice in chunk.choices:
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    return choice.delta.content or ""
        # Dict-style (some providers)
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    return content
    except (AttributeError, TypeError, KeyError):
        pass
    return ""


def _patch_chunk_text(chunk, text: str) -> None:
    """Replace the text delta in a streaming chunk."""
    try:
        if hasattr(chunk, "choices"):
            for choice in chunk.choices:
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    choice.delta.content = text
                    return
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                if "content" in delta:
                    delta["content"] = text
                    return
    except (AttributeError, TypeError, KeyError):
        pass


def _make_text_chunk(text: str):
    """Create a simple text chunk for flushing remaining buffer."""
    # Return a dict-style chunk (works with most providers)
    return {
        "choices": [{"delta": {"content": text}, "index": 0}],
    }


def _restore_non_stream(response, mapping: dict) -> None:
    """Restore placeholders in a non-streaming ChatCompletion response."""
    try:
        # OpenAI ChatCompletion object
        if hasattr(response, "choices"):
            for choice in response.choices:
                msg = getattr(choice, "message", None)
                if msg is None:
                    continue
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    restored, count = redactor.restore_text(content, mapping)
                    if count > 0:
                        msg.content = restored
        # Dict-style response
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                content = msg.get("content")
                if isinstance(content, str) and content:
                    restored, count = redactor.restore_text(content, mapping)
                    if count > 0:
                        msg["content"] = restored
    except (AttributeError, TypeError, KeyError) as exc:
        logger.warning("Chat Shield restore error: %s", exc)
