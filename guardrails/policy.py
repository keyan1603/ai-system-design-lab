"""Guardrail policy: what to do with each check's verdict, and what to do when a check itself fails.

Fail-closed vs fail-open is an explicit decision per check, not an accident of
exception handling:

* security checks (injection, output leak) fail CLOSED: if the check cannot
  run, the ticket goes to a human rather than through unguarded;
* PII redaction fails CLOSED too, because sending raw PII to a hosted model is
  the harm it exists to prevent;
* nothing in this module fails open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from guardrails.checks import check_reply, redact_pii, screen_injection


@dataclass
class InputDecision:
    action: str                     # allow | redacted | block
    text: str                       # what downstream may see
    reasons: list = field(default_factory=list)
    pii: dict = field(default_factory=dict)


@dataclass
class OutputDecision:
    action: str                     # release | hold
    problems: list = field(default_factory=list)


class GuardrailPolicy:
    def __init__(self, allowed_email_domains: frozenset = frozenset()) -> None:
        self.allowed_email_domains = frozenset(d.lower() for d in allowed_email_domains)

    def check_input(self, text: str) -> InputDecision:
        try:
            verdict = screen_injection(text)
            if verdict.blocked:
                return InputDecision("block", "", [f"injection:{m}" for m in verdict.matched])
            pii = redact_pii(text)
            if pii.found:
                return InputDecision("redacted", pii.text, ["pii_redacted"], pii.findings)
            return InputDecision("allow", text)
        except Exception as exc:  # noqa: BLE001 - a broken guard must not become an open door
            return InputDecision("block", "", [f"guard_error:{type(exc).__name__}"])

    def check_output(self, reply: str, allowed_text: str) -> OutputDecision:
        try:
            v = check_reply(reply, allowed_text, self.allowed_email_domains)
            return OutputDecision("release" if v.ok else "hold", v.problems)
        except Exception as exc:  # noqa: BLE001
            return OutputDecision("hold", [f"guard_error:{type(exc).__name__}"])
