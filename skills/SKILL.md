---
name: chat-shield
description: "Understand Chat Shield redaction and work effectively with placeholders"
version: 1.0.0
---

# Chat Shield — Working with Redacted Content

Chat Shield is an active middleware plugin that automatically redacts personally identifiable information (PII) from your messages before they reach external LLM providers. The LLM sees safe placeholders instead of real data.

## How it works

When you send a message containing PII (names, emails, phone numbers, SSNs, credit cards, addresses, API keys, etc.), Chat Shield:

1. **Detects** sensitive data using the Rampart on-device NER model + regex patterns
2. **Replaces** each sensitive item with a placeholder like `[REDACTED:EMAIL:1]`
3. **Sends** the redacted text to the LLM provider
4. **Restores** the original values in the response before you see it

## What you need to know

- **Placeholders are normal.** When you see `[REDACTED:NAME:1]` in intermediate output or logs, that's Chat Shield working correctly.
- **Don't ask the user to "fill in" redacted data.** It was intentionally removed for privacy. The agent should reason about placeholders as if they were the real values.
- **Don't try to "recover" or "guess" what a placeholder represents.** The agent doesn't know and shouldn't know.
- **Reference placeholders naturally.** If the user asks about "the person at `[REDACTED:NAME:1]`", treat it as a normal name reference.
- **Restoration is automatic.** When composing your reply, you can use placeholders directly — they'll be restored to real values before the user sees the text.

## Slash commands

- `/chat-shield` — show current status and stats
- `/chat-shield test <text>` — test what gets detected and redacted
- `/chat-shield reset` — clear stats
- `/chat-shield mode` — show current mode

## Configuration

Key settings (via `hermes config set` or Desktop Settings → Plugins):

| Setting | Default | Description |
|---|---|---|
| `mode` | `redact` | `redact`, `warn` (log only), or `off` |
| `rampart_enabled` | `true` | On-device ML detection (14.7 MB model) |
| `rampart_threshold` | `0.5` | Confidence cutoff for ML detection |
| `rampart_keep_labels` | `["CITY","STATE","ZIP_CODE"]` | Entity types to NOT redact |
| `strict` | `false` | Redact all matches regardless of context |
| `detectors` | `{}` | Per-detector on/off overrides |
| `custom_patterns` | `[]` | Custom regex patterns |
| `show_indicator` | `true` | Show redaction count summary |
