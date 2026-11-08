"""Redaction statistics tracking (thread-safe)."""

from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_total = 0
_by_type: dict[str, int] = defaultdict(int)
_mode = "redact"


def record_redaction(type_id: str, count: int = 1) -> None:
    global _total
    with _lock:
        _total += count
        _by_type[type_id] += count


def set_mode(mode: str) -> None:
    global _mode
    with _lock:
        _mode = mode


def get_summary() -> dict:
    with _lock:
        return {
            "mode": _mode,
            "total": _total,
            "by_type": dict(_by_type),
        }


def reset() -> None:
    global _total
    with _lock:
        _total = 0
        _by_type.clear()
