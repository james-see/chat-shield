# Chat Shield

> Automatically redacts PII before it reaches external LLM providers. On-device ML + regex. Nothing leaves your machine.

## What it does

Chat Shield is a [Hermes Agent](https://hermes-agent.nousresearch.com) middleware plugin that sits between your conversation and the LLM provider. Before any text is sent to OpenAI, Anthropic, Google, or any other external provider, Chat Shield scans it for personally identifiable information and replaces sensitive items with safe placeholders like `[REDACTED:EMAIL:1]`. When the LLM responds, Chat Shield restores the original values so you see everything normally.

**The LLM never sees your sensitive data.**

## Why you need this

If you use Hermes with external LLM providers, every message you send — including conversation history — goes to that provider's servers. Chat Shield ensures that PII like names, addresses, phone numbers, SSNs, credit card numbers, and API keys never make that trip.

Use cases:
- **Healthcare** — HIPAA-compliant agent interactions
- **Legal** — protect client identities and case details
- **Finance** — redact account numbers, cards, routing numbers
- **HR** — protect employee PII in agent workflows
- **Business** — keep customer data private
- **Journalism** — protect source identities
- **General privacy** — don't send your personal data to cloud LLMs

## Two-layer detection

### Layer 1: Rampart NER Model (ML)

Uses the [Rampart](https://huggingface.co/nationaldesignstudio/rampart) on-device NER model — a 14.7 MB quantized ONNX token-classification model that runs locally on your CPU. No data leaves your machine.

Detects 17 entity types:
- Names (given name, surname)
- Contact info (email, phone, URL)
- Government IDs (SSN/tax ID, passport, driver's license, government ID)
- Financial (bank account, routing number)
- Address components (building number, street, secondary, city, state, ZIP)

### Layer 2: Regex Detectors

Specialized patterns for structured identifiers:
- Credit cards (Luhn-validated)
- IBAN
- AWS access keys and secrets
- GitHub PATs
- Stripe API keys
- IP addresses
- Coordinates (MGRS, decimal, DMS)
- CAGE codes, DUNS numbers (context-gated)
- Custom user-defined patterns

## Installation

```bash
hermes plugins install james-see/chat-shield
hermes plugins enable chat-shield
```

The Rampart model downloads automatically on first use (~15 MB, cached locally).

## Configuration

```yaml
plugins:
  entries:
    chat-shield:
      settings:
        mode: redact              # redact | warn | off
        rampart_enabled: true     # on-device ML detection
        rampart_threshold: 0.5    # confidence cutoff
        rampart_keep_labels:      # entities to NOT redact
          - CITY
          - STATE
          - ZIP_CODE
        strict: false             # redact all matches (higher FP rate)
        detectors:                # per-detector overrides
          coord_mgrs: true
          credit_card: true
          mil_email: false
        custom_patterns:          # your own regex
          - name: project_codename
            pattern: 'PROJECT-\w+-\d+'
            description: "Internal project codenames"
        show_indicator: true      # show redaction count
```

## Slash commands

| Command | Description |
|---|---|
| `/chat-shield` | Show status and redaction stats |
| `/chat-shield test <text>` | Test detection on provided text |
| `/chat-shield reset` | Clear stats |
| `/chat-shield mode` | Show current mode |

## How it works

```
You: "Email Sarah at sarah@example.com, card 4111 1111 1111 1111"
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Chat Shield llm_request          │
    │  Detects: NAME, EMAIL, CC          │
    │  Replaces with placeholders        │
    └──────────────┬───────────────────┘
                   │ "Email [REDACTED:NAME:1] at
                   │  [REDACTED:EMAIL:1], card
                   │  [REDACTED:CC:1]"
                   ▼
    ┌──────────────────────────────────┐
    │  External LLM Provider            │  ← sees only placeholders
    │  (OpenAI / Anthropic / etc.)      │
    └──────────────┬───────────────────┘
                   │ "I'll contact [REDACTED:NAME:1]
                   │  at [REDACTED:EMAIL:1]..."
                   ▼
    ┌──────────────────────────────────┐
    │  Chat Shield llm_execution        │
    │  Restores placeholders → originals│
    └──────────────┬───────────────────┘
                   │ "I'll contact Sarah at
                   │  sarah@example.com..."
                   ▼
              You see real values
```

## Privacy

- **No cloud calls.** The Rampart model runs entirely on your device via ONNX Runtime.
- **No data collection.** Chat Shield does not phone home, does not log sensitive data, does not report usage.
- **Fail-safe.** If the model can't load, Chat Shield falls back to regex-only mode. If regex also fails, text passes through unredacted (fail-open, per Hermes middleware contract).

## License

MIT (plugin code) / CC BY 4.0 (Rampart model)
