"""The enterprise knowledge assistant.

    question -> input guard -> ACL-filtered retrieval -> answer agent -> citation check -> output guard -> answered | not_found | held | escalated

Design rules, each enforced in code rather than in the prompt:

* Authorization happens at retrieval. The model only ever sees chunks the user
  may read, so it cannot leak what it was never given.
* Every answer must cite sources, and every cited id must be one of the chunks
  actually retrieved for this user. An answer that cites nothing, or cites a
  document it was not shown, is held.
* The gateway's cache is scoped by the user's *access fingerprint*, not just
  the organization. Scoping by organization alone would let a cached answer
  written for an engineer be served to a contractor who asks the same question.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from requisite import Agent, RequestContext
from requisite.core.exceptions import CostLimitException
from requisite.telemetry.otel import get_tracer

from gateway.provider import GatewayExhausted, GatewayProvider
from gateway.routing import degraded_routes_used
from guardrails.policy import GuardrailPolicy
from knowledge.corpus import User
from knowledge.index import Hit, SecureIndex

_tracer = get_tracer("ai_system_design.knowledge")

NOT_FOUND = "I can't find that in the documents you have access to."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided in the user message. "
    "Cite every claim with the source id in square brackets, for example [HR-002]. "
    "Quote exact numbers and names from the sources. "
    f"If the sources do not contain the answer, reply with exactly this sentence and nothing else: {NOT_FOUND}"
)

_CITATION = re.compile(r"\[([A-Z]+-\d{3})\]")


@dataclass
class Answer:
    action: str                       # answered | not_found | held | escalated
    text: str = ""
    citations: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)       # doc ids the model was shown
    reasons: list = field(default_factory=list)
    latency_s: float = 0.0
    degraded: bool = False            # served (in part) by the degraded tier


def access_fingerprint(user: User) -> str:
    return hashlib.sha256("|".join(sorted(user.groups)).encode()).hexdigest()[:8]


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[{h.doc_id}] {h.title}\n{h.text}" for h in hits) or "(no sources available)"
    return f"Sources:\n{sources}\n\nQuestion: {question}"


class KnowledgeAssistant:
    """`on_degraded` decides what a degraded-tier answer becomes: "hold" withholds it, "label" releases it with a visible banner."""

    def __init__(self, gateway: GatewayProvider, index: SecureIndex, policy: Optional[GuardrailPolicy] = None,
                 enforce_acl: bool = True, scope_cache_by_acl: bool = True, top_k: int = 4, on_degraded: str = "hold") -> None:
        if on_degraded not in ("hold", "label"):
            raise ValueError("on_degraded must be 'hold' or 'label'")
        self.gateway, self.index = gateway, index
        self.policy = policy or GuardrailPolicy(allowed_email_domains=frozenset({"example.com"}))
        self.enforce_acl, self.scope_cache_by_acl, self.top_k, self.on_degraded = enforce_acl, scope_cache_by_acl, top_k, on_degraded
        # One agent for everyone: who is asking travels in the request context, not in the agent.
        self._agent = Agent(name="answerer", provider=gateway, system_prompt=SYSTEM_PROMPT, max_iterations=1)

    def ask(self, user: User, question: str, correlation_id: str = "") -> Answer:
        t0 = time.perf_counter()
        corr = correlation_id or uuid.uuid4().hex[:12]
        # The context's tenant slot is the gateway's cache and budget scope: the organization plus the
        # user's access fingerprint, so a cached answer is never served across access levels.
        scope = f"{user.org}:{access_fingerprint(user)}" if self.scope_cache_by_acl else user.org
        ctx = RequestContext(user=user.name, tenant=scope, correlation_id=corr)

        def done(span, action, **kw) -> Answer:
            span.set_attribute("knowledge.action", action)
            return Answer(action, latency_s=time.perf_counter() - t0, **kw)

        with _tracer.start_as_current_span("knowledge.ask") as span:
            span.set_attribute("knowledge.user", user.name)
            span.set_attribute("knowledge.groups", ",".join(user.groups))
            decision = self.policy.check_input(question)
            if decision.action == "block":
                return done(span, "escalated", reasons=decision.reasons)

            try:
                hits = self.index.search(user, decision.text, top_k=self.top_k, enforce_acl=self.enforce_acl)
            except Exception as exc:  # noqa: BLE001 - e.g. the embedding API is down: fail closed, never answer ungrounded
                return done(span, "held", reasons=[f"retrieval_unavailable: {type(exc).__name__}"])
            retrieved = sorted({h.doc_id for h in hits})
            span.set_attribute("knowledge.retrieved", ",".join(retrieved))
            prompt = build_prompt(decision.text, hits)
            try:
                text = self._agent.run(prompt, context=ctx).content.strip()
            except CostLimitException:
                return done(span, "held", retrieved=retrieved, reasons=["budget_exhausted"])
            except GatewayExhausted:
                return done(span, "held", retrieved=retrieved, reasons=["model_unavailable"])
            except Exception as exc:  # noqa: BLE001
                return done(span, "held", retrieved=retrieved, reasons=[f"pipeline_error: {type(exc).__name__}"])

            degraded = degraded_routes_used([r for r in self.gateway.audit.records if r.correlation_id == corr], self.gateway.routes)
            if text == NOT_FOUND:
                return done(span, "not_found", text=text, retrieved=retrieved, degraded=bool(degraded))

            citations = sorted(set(_CITATION.findall(text)))
            problems = []
            if not citations:
                problems.append("uncited_answer")
            stray = [c for c in citations if c not in retrieved]
            if stray:
                problems.append(f"cites_unretrieved:{stray}")
            problems += self.policy.check_output(text, prompt).problems
            if degraded and self.on_degraded == "hold":
                problems.append(f"served_by_degraded_tier: {','.join(degraded)}")
            elif degraded:
                text = f"[Answered by a fallback model; verify before relying on it] {text}"
            return done(span, "held" if problems else "answered", text=text, citations=citations, retrieved=retrieved,
                        reasons=problems, degraded=bool(degraded))
