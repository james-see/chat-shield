# Chat Shield — Demo Video Script (<90 seconds)

## Setup before recording

- Hermes Desktop open, chat visible
- Chat Shield enabled (Rampart + regex detectors on)
- Have this message ready to paste:

```
Please draft an email to Sarah Mitchell at sarah.mitchell@gmail.com reminding her about the payment of $4,200 due on her card 4111 1111 1111 1111. Her SSN for the account is 123-45-6789 and her phone is (555) 123-4567. She lives at 221B Baker Street, Arlington, VA 22201.
```

---

## Recording

### [0:00–0:10] Intro

> "Chat Shield is a Hermes Agent middleware plugin that redacts personal data before it reaches external LLM providers — using on-device machine learning and regex pattern detection."

### [0:10–0:20] Show config

Show Desktop → Settings → Plugins → Chat Shield. Briefly scroll through the settings form: mode=redact, Rampart enabled, detectors list.

> "It runs entirely locally — no data leaves your machine."

### [0:20–0:45] The test

Type `/chat-shield test` in the chat, paste the prepared message. Show the output.

> "The plugin detected multiple items: Sarah's name and address from the Rampart ML model, her email, phone, SSN, credit card, and ZIP code from regex patterns. Each is replaced with a placeholder like [REDACTED:EMAIL:1]."

### [0:45–0:70] Live redaction

Send the actual message to the agent. Show the agent's reply — all real values are present (restored).

> "The LLM only saw placeholders. It never saw Sarah's real email, card number, or address. But the response I see has all the original values restored automatically."

### [0:70–0:85] Close

> "Chat Shield — because a prompt can't redact itself."

---

## Tips

- Screen record with QuickTime (Cmd+Shift+5) or OBS
- Keep cuts tight — the `/chat-shield test` output is the visual proof
- The live send → restored response is the "wow" moment
- Post on X with `@NousResearch` tagged and link the PR: https://github.com/NousResearch/hermes-agent/pull/151047
