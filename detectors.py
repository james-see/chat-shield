"""Specialized regex detectors for PII / sensitive identifiers.

Each detector is a named regex with optional context heuristics to reduce
false positives on ambiguous patterns (CAGE codes, DUNS, AWS secrets, DoD IDs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Detector:
    id: str
    type: str  # placeholder type tag
    pattern: re.Pattern
    description: str
    context_required: bool = False
    context_keywords: tuple[str, ...] = ()
    luhn_validate: bool = False  # for credit cards


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------

BUILTIN_DETECTORS: list[Detector] = [
    # --- Coordinates (default ON) ---
    Detector(
        id="coord_mgrs",
        type="MGRS",
        pattern=re.compile(r"\d{1,2}[A-Z][A-Z]{2}\d{4,10}"),
        description="MGRS military grid coordinates",
    ),
    Detector(
        id="coord_latlon",
        type="COORD",
        pattern=re.compile(r"-?\d{1,3}\.\d{2,7},\s*-?\d{1,3}\.\d{2,7}"),
        description="Decimal latitude/longitude coordinates",
    ),
    Detector(
        id="coord_dms",
        type="DMS",
        pattern=re.compile(
            r"\d{1,3}°\d{1,2}'\d{1,2}\"[NS],?\s*\d{1,3}°\d{1,2}'\d{1,2}\"[EW]",
            re.IGNORECASE,
        ),
        description="DMS coordinates (degrees/minutes/seconds)",
    ),
    # --- Government / defense (default OFF — opt in if needed) ---
    Detector(
        id="mil_email",
        type="MIL",
        pattern=re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.mil\b"),
        description=".mil email addresses",
    ),
    Detector(
        id="cage",
        type="CAGE",
        pattern=re.compile(r"\b[A-Z0-9]{5}\b"),
        description="CAGE codes (5-char alphanumeric)",
        context_required=True,
        context_keywords=("cage", "cage code", "cage:"),
    ),
    Detector(
        id="duns",
        type="DUNS",
        pattern=re.compile(r"\b\d{9}\b"),
        description="DUNS numbers (9-digit)",
        context_required=True,
        context_keywords=("duns", "duns:", "uei"),
    ),
    Detector(
        id="clearance",
        type="CLR",
        pattern=re.compile(
            r"\b(TS/SCI|Top Secret|SECRET|CONFIDENTIAL)\b", re.IGNORECASE
        ),
        description="Security clearance levels",
    ),
    Detector(
        id="cui",
        type="CUI",
        pattern=re.compile(r"\bCUI(?://|//SP-|//FGI|//OC|//PRVCY|//DS)", re.IGNORECASE),
        description="CUI (Controlled Unclassified Information) markers",
    ),
    Detector(
        id="dod_id",
        type="DODID",
        pattern=re.compile(r"\b\d{10}\b"),
        description="DoD ID numbers (10-digit EDIPI)",
        context_required=True,
        context_keywords=("dod id", "edipi", "id number", "dod-id"),
    ),
    # --- General PII / financial (default ON) ---
    Detector(
        id="email",
        type="EMAIL",
        pattern=re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        description="Email addresses",
    ),
    Detector(
        id="ssn",
        type="SSN",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        description="Social Security Numbers",
    ),
    Detector(
        id="phone",
        type="PHONE",
        pattern=re.compile(r"\b(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        description="Phone numbers",
    ),
    Detector(
        id="ip_addr",
        type="IP",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        description="IPv4 addresses",
    ),
    # --- Cloud / API credentials (default ON) ---
    Detector(
        id="aws_key",
        type="AWS",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        description="AWS access key IDs",
    ),
    Detector(
        id="aws_secret",
        type="AWSSEC",
        pattern=re.compile(r"\b[A-Za-z0-9/+=]{40}\b"),
        description="AWS secret access keys (40-char base64)",
        context_required=True,
        context_keywords=("aws_secret", "secret key", "secret_access_key", "aws secret"),
    ),
    Detector(
        id="github_pat",
        type="GH",
        pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
        description="GitHub personal access tokens",
    ),
    Detector(
        id="stripe_key",
        type="STRIPE",
        pattern=re.compile(r"\b(?:sk|pk)_(?:live|test)_[0-9a-zA-Z]{24,}\b"),
        description="Stripe API keys",
    ),
    # --- Financial (default ON) ---
    Detector(
        id="credit_card",
        type="CC",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        description="Credit card numbers (Luhn-validated)",
        luhn_validate=True,
    ),
    Detector(
        id="iban",
        type="IBAN",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        description="International Bank Account Numbers (IBAN)",
    ),
]


# Default on/off state — keys not listed default to True
# email is on by default (works in regex-only mode too)
DEFAULT_OFF = {"mil_email", "cage", "duns", "clearance", "cui", "dod_id"}


def _luhn_valid(number_str: str) -> bool:
    """Luhn checksum validation for credit card numbers."""
    digits = [int(ch) for ch in number_str if ch.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def get_active_detectors(config: dict) -> list[Detector]:
    """Return detectors active under the current config."""
    overrides: dict = config.get("detectors", {}) or {}
    custom_patterns: list = config.get("custom_patterns", []) or []
    strict: bool = config.get("strict", False)

    detectors: list[Detector] = []
    for d in BUILTIN_DETECTORS:
        # Per-detector override takes priority, then DEFAULT_OFF
        if d.id in overrides:
            if not overrides[d.id]:
                continue
        elif d.id in DEFAULT_OFF:
            continue

        # In strict mode, context_required is relaxed
        det = Detector(
            id=d.id,
            type=d.type,
            pattern=d.pattern,
            description=d.description,
            context_required=d.context_required and not strict,
            context_keywords=d.context_keywords,
            luhn_validate=d.luhn_validate,
        )
        detectors.append(det)

    # Custom user patterns
    for cp in custom_patterns:
        name = cp.get("name", "custom")
        pat = cp.get("pattern", "")
        if not pat:
            continue
        detectors.append(
            Detector(
                id=f"custom_{name}",
                type=f"CUSTOM_{name.upper()}",
                pattern=re.compile(pat),
                description=cp.get("description", f"Custom pattern: {name}"),
            )
        )

    return detectors


def scan_text(text: str, detectors: list[Detector]) -> list[tuple[str, str, int, int]]:
    """Scan text and return list of (matched_value, type_tag, start, end).

    Context-gated detectors only match when their keywords appear in the
    80-char window before the match.  Luhn-validated detectors reject
    non-valid numbers.
    """
    results: list[tuple[str, str, int, int]] = []

    for det in detectors:
        for m in det.pattern.finditer(text):
            original = m.group(0)

            # Context check
            if det.context_required:
                window_start = max(0, m.start() - 80)
                context = text[window_start : m.start()].lower()
                if not any(kw in context for kw in det.context_keywords):
                    continue

            # Luhn check
            if det.luhn_validate and not _luhn_valid(original):
                continue

            results.append((original, det.type, m.start(), m.end()))

    return results
