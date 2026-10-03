"""Deterministic guardrail checks: PII redaction, injection screening, output validation.

All three are plain functions with no model call, so they are fast, free,
testable offline, and cannot themselves be prompt-injected. They are one layer
of a defense, not the defense: pattern matching is bypassable, which is why
the tools are read-only and the output check re-verifies what the model says
against what the request was allowed to contain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

# ---- PII redaction ---------------------------------------------------------------------
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class PIIResult:
    text: str
    findings: dict = field(default_factory=dict)   # kind -> count

    @property
    def found(self) -> bool:
        return bool(self.findings)


def redact_pii(text: str) -> PIIResult:
    findings: dict[str, int] = {}

    def swap(pattern, label, validator=None):
        nonlocal text

        def repl(m):
            raw = m.group(0)
            if validator and not validator(re.sub(r"\D", "", raw)):
                return raw
            findings[label] = findings.get(label, 0) + 1
            return f"[{label}]"

        text = pattern.sub(repl, text)

    # Order matters: cards before phones, or a card number is mislabelled a phone.
    swap(_EMAIL, "EMAIL")
    swap(_SSN, "SSN")
    swap(_CARD, "CARD", _luhn_ok)
    swap(_PHONE, "PHONE")
    return PIIResult(text, findings)


# ---- injection screening ---------------------------------------------------------------
_INJECTION_PATTERNS = {
    "override_instructions": re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all|your)\b.{0,30}\b(instructions?|rules?|prompt)", re.I),
    "reveal_prompt": re.compile(r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\b.{0,15}\b(prompt|instructions?)", re.I),
    "role_override": re.compile(r"\b(you are now|developer mode|system override|act as (?:an? )?(?:admin|root))\b", re.I),
    "bulk_exfiltration": re.compile(r"\b(list|dump|paste|export)\b.{0,40}\b(every|all)\b.{0,30}\b(customer|record|user)s?\b", re.I),
}


@dataclass
class InjectionVerdict:
    blocked: bool
    matched: list = field(default_factory=list)


def screen_injection(text: str) -> InjectionVerdict:
    matched = [name for name, pat in _INJECTION_PATTERNS.items() if pat.search(text)]
    return InjectionVerdict(blocked=bool(matched), matched=matched)


# ---- output validation -----------------------------------------------------------------
class TriageResult(BaseModel):
    category: Literal["billing", "technical", "account"]
    severity: Literal["low", "medium", "high"]
    customer_id: Optional[str] = None
    order_id: Optional[str] = None
    summary: str


_ID = re.compile(r"\b[CO]-\d{4}\b")


def parse_triage(text: str) -> TriageResult:
    """Parse the triage agent's JSON, tolerating markdown fences. Raises ValueError if invalid."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return TriageResult.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"triage output is not valid TriageResult JSON: {exc}") from exc


@dataclass
class OutputVerdict:
    ok: bool
    problems: list = field(default_factory=list)


def check_reply(reply: str, allowed_text: str, allowed_email_domains: frozenset = frozenset()) -> OutputVerdict:
    """Verify a customer-facing reply before it leaves the system.

    * No PII patterns (the model must not echo or invent contact details).
      Email addresses on ``allowed_email_domains`` are the company's own
      (for example the no-reply address a knowledge-base article quotes) and
      are not customer PII; without that allowlist the check would hold
      perfectly good replies.
    * Every customer/order id in the reply must already appear in the request
      that was allowed in. An id that did not come from the ticket means the
      model pulled in another customer's data, which is the leak to stop.
    """
    problems = []
    scrubbed = _EMAIL.sub(lambda m: "" if m.group(0).rsplit("@", 1)[1].lower() in allowed_email_domains else m.group(0), reply)
    pii = redact_pii(scrubbed)
    if pii.found:
        problems.append(f"reply contains PII patterns: {sorted(pii.findings)}")
    allowed_ids = set(_ID.findall(allowed_text))
    stray = sorted(set(_ID.findall(reply)) - allowed_ids)
    if stray:
        problems.append(f"reply references ids not present in the ticket: {stray}")
    return OutputVerdict(ok=not problems, problems=problems)
