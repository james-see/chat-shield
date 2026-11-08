"""Chat Shield — PII redaction middleware plugin.

Registers ``llm_request`` and ``llm_execution`` middleware that redact
personally identifiable information before it reaches external LLM providers
and restore original values in the response.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import middleware as mw
from . import detectors as det_mod
from . import redactor
from . import stats
from .rampart_engine import RampartEngine, DEFAULT_KEEP

logger = logging.getLogger(__name__)

# Singleton engine — loaded once, reused across requests
_rampart: RampartEngine | None = None


def _handle_shield_command(raw_args: str) -> str:
    """Handle /chat-shield slash command."""
    parts = raw_args.strip().split()
    if not parts or parts[0] in ("status", "stats"):
        s = stats.get_summary()
        lines = [f"🛡️ Chat Shield", f"Mode: {s['mode']}", f"Total redactions: {s['total']}"]
        if s["by_type"]:
            lines.append("By type:")
            for k, v in sorted(s["by_type"].items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    elif parts[0] == "test":
        test_text = (
            " ".join(parts[1:])
            or "Contact Sarah Mitchell at sarah@example.com or (555) 123-4567. "
            "SSN: 123-45-6789. Card: 4111 1111 1111 1111. "
            "Lives at 221B Baker Street, Arlington, VA 22201."
        )
        from . import detectors as d

        config = mw._config if mw._config else {}
        active = d.get_active_detectors(config)
        rampart = _rampart if config.get("rampart_enabled", True) else None
        redacted, mapping = redactor.redact_text(test_text, active, rampart)
        lines = [
            f"Input:  {test_text}",
            f"Redacted: {redacted}",
            f"Items detected: {len(mapping)}",
        ]
        for ph, orig in mapping.items():
            lines.append(f"  {ph} ← {orig}")
        return "\n".join(lines)

    elif parts[0] == "reset":
        stats.reset()
        return "🛡️ Chat Shield stats reset."

    elif parts[0] == "mode":
        current = stats.get_summary()["mode"]
        return (
            f"Current mode: {current}\n"
            "Change with: hermes config set "
            "plugins.entries.chat-shield.settings.mode <redact|warn|off>"
        )

    return (
        "Usage: /chat-shield [status|test <text>|reset|mode]\n"
        "  status — show redaction stats\n"
        "  test <text> — test detection on text\n"
        "  reset — clear stats\n"
        "  mode — show current mode"
    )


def register(ctx):
    """Register Chat Shield middleware and slash command."""
    # Read config
    config = {
        "mode": ctx.get_config("mode", default="redact"),
        "strict": ctx.get_config("strict", default=False),
        "rampart_enabled": ctx.get_config("rampart_enabled", default=True),
        "rampart_threshold": ctx.get_config("rampart_threshold", default=0.5),
        "rampart_keep_labels": ctx.get_config("rampart_keep_labels", default=list(DEFAULT_KEEP)),
        "detectors": ctx.get_config("detectors", default={}),
        "custom_patterns": ctx.get_config("custom_patterns", default=[]),
        "show_indicator": ctx.get_config("show_indicator", default=True),
    }

    if config["mode"] == "off":
        logger.info("Chat Shield is disabled (mode=off)")
        return

    # Initialize Rampart engine (lazy — model loads on first use)
    global _rampart
    if config["rampart_enabled"]:
        try:
            # Use the documented plugin_data_dir() API for a stable data path
            from pathlib import Path as _P

            try:
                # Hermes >=0.22 exposes plugin_data_dir() on ctx
                data_dir = Path(ctx.plugin_data_dir())
            except AttributeError:
                # Fallback for older versions
                data_dir = _P(str(_P.home() / ".hermes" / "plugin-data" / "chat-shield"))

            keep_set = set(config.get("rampart_keep_labels", []))
            _rampart = RampartEngine(
                cache_dir=data_dir,
                threshold=config["rampart_threshold"],
                keep_labels=keep_set,
            )
            logger.info("Rampart engine initialized (model will load on first request)")
        except Exception as exc:
            logger.warning("Rampart init failed: %s — regex-only mode", exc)
            _rampart = None

    # Push config + engine to middleware module
    mw.set_config(config)
    mw.set_rampart(_rampart)
    stats.set_mode(config["mode"])

    # Register middleware
    ctx.register_middleware("llm_request", mw.on_llm_request)
    ctx.register_middleware("llm_execution", mw.on_llm_execution)

    # Register slash command
    ctx.register_command(
        "chat-shield",
        _handle_shield_command,
        description="Show redaction stats or test detection",
    )

    # Register bundled skill
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, str(skill_md))
            logger.info("Chat Shield skill registered: %s", child.name)

    logger.info(
        "Chat Shield active (mode=%s, rampart=%s, detectors=%d)",
        config["mode"],
        _rampart is not None,
        len(det_mod.get_active_detectors(config)),
    )
