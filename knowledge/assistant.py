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
from dataclasses import dataclass, field
from typing import Optional

from requisite import Agent
from requisite.telemetry.otel import get_tracer

from gateway.provider import GatewayProvider
from gateway.tenant import TenantProvider
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


def access_fingerprint(user: User) -> str:
    return hashlib.sha256("|".join(sorted(user.groups)).encode()).hexdigest()[:8]


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[{h.doc_id}] {h.title}\n{h.text}" for h in hits) or "(no sources available)"
    return f"Sources:\n{sources}\n\nQuestion: {question}"


class KnowledgeAssistant:
    def __init__(self, gateway: GatewayProvider, index: SecureIndex, policy: Optional[GuardrailPolicy] = None,
                 enforce_acl: bool = True, scope_cache_by_acl: bool = True, top_k: int = 4) -> None:
        self.gateway, self.index = gateway, index
        self.policy = policy or GuardrailPolicy(allowed_email_domains=frozenset({"example.com"}))
        self.enforce_acl, self.scope_cache_by_acl, self.top_k = enforce_acl, scope_cache_by_acl, top_k
        self._agents: dict[str, Agent] = {}

    def _agent_for(self, user: User) -> Agent:
        scope = f"{user.org}:{access_fingerprint(user)}" if self.scope_cache_by_acl else user.org
        if scope not in self._agents:
            self._agents[scope] = Agent(name="answerer", provider=TenantProvider(self.gateway, scope),
                                        system_prompt=SYSTEM_PROMPT, max_iterations=1)
        return self._agents[scope]

    def ask(self, user: User, question: str) -> Answer:
        import time
        t0 = time.perf_counter()
        with _tracer.start_as_current_span("knowledge.ask") as span:
            span.set_attribute("knowledge.user", user.name)
            span.set_attribute("knowledge.groups", ",".join(user.groups))
            decision = self.policy.check_input(question)
            if decision.action == "block":
                span.set_attribute("knowledge.action", "escalated")
                return Answer("escalated", reasons=decision.reasons, latency_s=time.perf_counter() - t0)

            hits = self.index.search(user, decision.text, top_k=self.top_k, enforce_acl=self.enforce_acl)
            retrieved = sorted({h.doc_id for h in hits})
            span.set_attribute("knowledge.retrieved", ",".join(retrieved))
            prompt = build_prompt(decision.text, hits)
            text = self._agent_for(user).run(prompt).content.strip()

            if text == NOT_FOUND:
                span.set_attribute("knowledge.action", "not_found")
                return Answer("not_found", text, [], retrieved, latency_s=time.perf_counter() - t0)

            citations = sorted(set(_CITATION.findall(text)))
            problems = []
            if not citations:
                problems.append("uncited_answer")
            stray = [c for c in citations if c not in retrieved]
            if stray:
                problems.append(f"cites_unretrieved:{stray}")
            out = self.policy.check_output(text, prompt)
            problems += out.problems
            action = "held" if problems else "answered"
            span.set_attribute("knowledge.action", action)
            return Answer(action, text, citations, retrieved, problems, time.perf_counter() - t0)
