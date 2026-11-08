"""Unified redaction + restoration engine.

Combines Rampart ML detection with regex detectors into a single pass that
produces a placeholder mapping.  Restoration reverses the mapping.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

from . import detectors as det_mod
from .rampart_engine import RampartEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-request state management
# ---------------------------------------------------------------------------

_request_maps: dict[str, dict[str, str]] = {}
_global_lock = threading.Lock()
_TTL_SECONDS = 300


def store_mapping(request_id: str, mapping: dict[str, str]) -> None:
    with _global_lock:
        _request_maps[request_id] = mapping


def get_mapping(request_id: str) -> Optional[dict[str, str]]:
    with _global_lock:
        return _request_maps.get(request_id)


def cleanup_mapping(request_id: str) -> None:
    with _global_lock:
        _request_maps.pop(request_id, None)


def cleanup_expired() -> None:
    """Remove mappings older than TTL (called periodically)."""
    # Simple approach: if we have >100 entries, clear the oldest
    with _global_lock:
        if len(_request_maps) > 100:
            # Keep only the most recent 50
            items = list(_request_maps.items())
            _request_maps.clear()
            _request_maps.update(items[-50:])


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact_text(
    text: str,
    active_detectors: list[det_mod.Detector],
    rampart: Optional[RampartEngine] = None,
    counters: Optional[dict[str, int]] = None,
) -> tuple[str, dict[str, str]]:
    """Scan text for sensitive patterns and replace with placeholders.

    Returns (redacted_text, mapping) where mapping is
    {placeholder: original_value}.
    """
    if counters is None:
        counters = {}

    mapping: dict[str, str] = {}

    # Collect all spans from both layers
    spans: list[tuple[str, str, int, int]] = []  # (matched, type, start, end)

    # Layer 1: Rampart ML detection
    if rampart is not None:
        ml_spans = rampart.detect(text)
        spans.extend(ml_spans)

    # Layer 2: Regex detection
    regex_spans = det_mod.scan_text(text, active_detectors)
    spans.extend(regex_spans)

    if not spans:
        return text, mapping

    # Sort spans by start position; on overlap, keep the longer span
    spans.sort(key=lambda s: (s[2], -(s[3] - s[2])))

    # Filter overlapping spans — keep first (longest) at each position
    filtered: list[tuple[str, str, int, int]] = []
    last_end = -1
    for matched, type_tag, start, end in spans:
        if start >= last_end:
            filtered.append((matched, type_tag, start, end))
            last_end = end

    # Build redacted text by replacing from end to start (preserve indices)
    redacted = text
    for matched, type_tag, start, end in reversed(filtered):
        n = counters.get(type_tag, 0) + 1
        counters[type_tag] = n
        placeholder = f"[REDACTED:{type_tag}:{n}]"
        mapping[placeholder] = matched
        redacted = redacted[:start] + placeholder + redacted[end:]

    return redacted, mapping


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[A-Z_]+:\d+\]")


def restore_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Replace placeholders with original values.

    Returns (restored_text, count_of_replacements).
    """
    if not mapping:
        return text, 0

    restored = text
    count = 0
    for placeholder, original in mapping.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, original)
            count += 1

    return restored, count


def restore_streaming_chunk(text: str, mapping: dict[str, str], buffer: str) -> tuple[str, str]:
    """Restore placeholders in a streaming chunk with a carry buffer.

    Placeholders might span chunk boundaries, so we buffer text and only
    release text when we're sure no partial placeholder is at the end.

    Returns (text_to_release, new_buffer).
    """
    combined = buffer + text

    # Find the last potential partial placeholder
    # Placeholders look like [REDACTED:TYPE:N] — min length ~15 chars
    # Check if the end of combined could be the start of a placeholder
    release_until = len(combined)

    # Search backwards for '[' that might start an incomplete placeholder
    for i in range(len(combined) - 1, max(len(combined) - 30, -1), -1):
        if combined[i] == "[":
            # Check if this could be a partial [REDACTED:...]
            tail = combined[i:]
            if _PLACEHOLDER_RE.match(tail) and not _PLACEHOLDER_RE.fullmatch(tail):
                # Partial placeholder at the end — hold it back
                release_until = i
                break

    to_release = combined[:release_until]
    new_buffer = combined[release_until:]

    # Restore complete placeholders in the release text
    restored, count = restore_text(to_release, mapping)
    return restored, new_buffer


# ---------------------------------------------------------------------------
# Redact full message list (used by middleware)
# ---------------------------------------------------------------------------


def redact_messages(
    messages: list,
    active_detectors: list[det_mod.Detector],
    rampart: Optional[RampartEngine] = None,
) -> dict[str, str]:
    """Redact all message contents in a provider request.

    Mutates messages in place.  Returns full mapping {placeholder: original}.
    """
    full_mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")
        if isinstance(content, str) and content:
            redacted, mapping = redact_text(content, active_detectors, rampart, counters)
            if mapping:
                msg["content"] = redacted
                full_mapping.update(mapping)

        # Also handle content as list of parts (vision/multimodal)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text_val = part.get("text")
                    if isinstance(text_val, str) and text_val:
                        redacted, mapping = redact_text(
                            text_val, active_detectors, rampart, counters
                        )
                        if mapping:
                            part["text"] = redacted
                            full_mapping.update(mapping)

        # Check tool_calls / function arguments
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                args_str = func.get("arguments", "")
                if isinstance(args_str, str) and args_str:
                    redacted_args, arg_mapping = redact_text(
                        args_str, active_detectors, rampart, counters
                    )
                    if arg_mapping:
                        func["arguments"] = redacted_args
                        full_mapping.update(arg_mapping)

    return full_mapping
