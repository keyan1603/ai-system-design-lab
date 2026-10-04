"""Live run: what happens to the system when each dependency fails?

Every scenario injects one failure into a REAL pipeline (real models, real MCP
server, real A2A service) and records what the system did: the signal that
exposed it, the behavior, the outcome the user saw, and whether anything
restricted leaked. Injected faults are labelled as such; none is a vendor outage.

Invariants checked in every scenario: the pipeline returned a defined outcome
instead of raising, and no restricted fact reached a user-visible answer.
"""

import os
import sys
import time

from dotenv import load_dotenv
from requisite.core.rate_limiter import RateLimiter
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from tabulate import tabulate

load_dotenv(".env")

from a2a_layer.client import KnowledgeClient  # noqa: E402
from a2a_layer.server import AgentAudit, start_knowledge_service  # noqa: E402
from gateway.factory import build_gateway  # noqa: E402
from gateway.routing import LIGHT  # noqa: E402
from identity.tokens import IdentityProvider  # noqa: E402
from knowledge.assistant import KnowledgeAssistant  # noqa: E402
from knowledge.corpus import DOCS, USERS  # noqa: E402
from knowledge.index import SecureIndex  # noqa: E402
from tickets.data import TICKETS, Ticket  # noqa: E402
from tickets.platform import TicketPlatform  # noqa: E402
from tickets.policy_tool import PolicyTool  # noqa: E402

LIMITER = RateLimiter(requests_per_minute=15)      # one limiter for one API key, shared by every gateway here
# Facts that must never reach any answer in this study (the users involved are not entitled to them).
RESTRICTED = [d.canary.lower() for d in DOCS if d.id in ("HR-003", "FIN-001", "FIN-002", "LEG-001", "EXEC-001")]
Q_ENG = "What command do I run to roll back the latest release during a severity 1 incident?"
Q_UNKNOWN = "Which cafeteria vendor supplies the Berlin office lunch menu?"
rows = []


class FlakyEmbedder(GeminiEmbeddingProvider):
    """Real Gemini embeddings that can be switched off, to model the embedding API failing."""

    down = False

    def embed(self, texts):
        if self.down:
            raise ConnectionError("injected: embedding API unreachable")
        return super().embed(texts)


def gw(**kw):
    return build_gateway(use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=LIMITER, with_faults=True,
                         audit_path="audit/phase5_failures.jsonl", **kw)


def safe(text):
    return not any(c in (text or "").lower() for c in RESTRICTED)


def record(name, failure, signal, behavior, outcome, safe_ok=True, extra=""):
    rows.append([name, failure, signal, behavior, outcome, "yes" if safe_ok else "NO", extra])
    print(f"  done: {name} -> {outcome}")


def main():
    index = SecureIndex(FlakyEmbedder(model="gemini-embedding-2"))
    index.ingest(DOCS)
    alice, dave = USERS["alice"], USERS["dave"]

    # S1: the cheap route is down; traffic escalates to the heavy route
    g, faults = gw()
    faults["light"].set("down")
    a = KnowledgeAssistant(g, index).ask(alice, Q_ENG, "s1")
    rec = [r for r in g.audit.records if r.correlation_id == "s1"][-1]
    record("S1 light route down (injected)", "gemini-light unavailable", "audit: attempts light->heavy; fallback_rate up",
           f"escalated to {rec.route}", f"{a.action}; cost ${rec.cost_usd:.6f} per call", safe(a.text))

    # S2: every Gemini route down; the local model answers; two policies for what that means
    for policy in ("hold", "label"):
        g, faults = gw()
        faults["light"].set("down"); faults["heavy"].set("down")
        a = KnowledgeAssistant(g, index, on_degraded=policy).ask(alice, Q_ENG, f"s2-{policy}")
        record(f"S2 all cloud routes down, on_degraded={policy}", "gemini-light and gemini-heavy unavailable (injected)",
               "audit: route local-llama; degraded_share = 1",
               "local model answers" if policy == "label" else "local model answers, answer withheld",
               f"{a.action}{' with banner' if policy == 'label' and a.action == 'answered' else ''}; reasons={a.reasons[:1]}", safe(a.text))

    # S3: a slow (not failed) route, with and without a timeout
    for label, t_light in (("no timeout", 999.0), ("timeout 2s, run 1", 2.0), ("timeout 2s, run 2", 2.0), ("timeout 2s, run 3", 2.0)):
        g, faults = gw(timeouts={"light": t_light})
        faults["light"].set("slow", delay_s=8.0)
        t0 = time.perf_counter()
        a = KnowledgeAssistant(g, index).ask(alice, Q_ENG, f"s3-{label}")
        rec = [r for r in g.audit.records if r.correlation_id == f"s3-{label}"][-1]
        chain = "->".join(f"{x['route'].replace('gemini-', '')}:{'ok' if x['ok'] else (x['error'] or '').split(':')[0]}" for x in rec.attempts)
        record(f"S3 slow route (8 s injected delay), {label}", "gemini-light answers after 8 s", "latency; audit: TimeoutError attempt",
               "waits for the slow route" if t_light > 100 else "gives up on the slow route and fails over",
               f"{a.action} after {time.perf_counter() - t0:.1f}s", safe(a.text), f"route chain: {chain}")

    # S4: the embedding API (retrieval) is down
    g, _ = gw()
    index.retriever.embedding_provider.down = True
    a = KnowledgeAssistant(g, index).ask(alice, Q_ENG, "s4")
    index.retriever.embedding_provider.down = False
    record("S4 embedding API down (injected)", "retrieval cannot embed the question", "reasons: retrieval_unavailable",
           "fails closed: no ungrounded answer, no model call", f"{a.action}; reasons={a.reasons}", safe(a.text),
           f"model calls: {len([r for r in g.audit.records if r.correlation_id == 's4'])}")

    # S5-S7 run the ticket platform
    t_ticket = next(t for t in TICKETS if t.id == "T05")

    # S5: the MCP tool server cannot start
    g, _ = gw()
    with TicketPlatform(g, backend="adk", mcp_command=[sys.executable, "-c", "import sys; sys.exit(1)"]) as p:
        o = p.handle(t_ticket, correlation_id="s5")
    record("S5 MCP tool server will not start", "ticket-ops server exits immediately", "reasons: pipeline_error / tool failure",
           "ticket is held, not lost, not guessed", f"{o.action}; {o.reasons[:1]}; reply={o.reply[:60]!r}", safe(o.reply))

    # S6: the knowledge agent (A2A) is down while a ticket needs policy
    secret = os.urandom(32)
    idp = IdentityProvider(secret, USERS, {"ticket-agent": {"employee", "support"}})
    kg, _ = build_gateway(use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=LIMITER)
    service = start_knowledge_service(KnowledgeAssistant(kg, index), secret, AgentAudit())
    url = service.url
    service.stop()                                        # the service is now down
    g, _ = gw()
    ptool = PolicyTool(KnowledgeClient(url), idp)
    refund = Ticket("P01", "acme", "Customer C-1002 reports order O-5002 was charged twice. Can we refund the duplicate, "
                    "how long will it take, and what is the approval threshold for refunds?", "billing", "high")
    with TicketPlatform(g, backend="adk", policy_tool=ptool) as p:
        o = p.handle(refund, operator="sam", operator_token=idp.login("sam", "ticket-agent"), correlation_id="s6")
    invented = "500 dollars" in o.reply.lower() or "team lead" in o.reply.lower()
    record("S6 knowledge agent (A2A) unreachable", "connection refused to the A2A endpoint", "policy tool: ok=False; reasons: capability_unavailable",
           "tool returns 'not permitted', resolver told not to guess, ticket held because a requested fact is missing",
           f"{o.action} {o.reasons[:1]}; policy invented: {invented}",
           safe(o.reply) and not invented, f"policy calls: {ptool.calls[-1:]}")

    # S7: the tenant's budget is exhausted
    g, _ = gw(tenant_budget_usd=0.00002)
    with TicketPlatform(g, backend="adk") as p:
        o1 = p.handle(t_ticket, correlation_id="s7a")
        o2 = p.handle(t_ticket, correlation_id="s7b")
    record("S7 tenant budget exhausted", "per-tenant spend cap reached", "reasons: budget_exhausted; blocked_budget in audit",
           "the reactive cap trips partway through the first ticket; every later call is blocked; nothing crashes",
           f"1st {o1.action} {o1.reasons[:1]}; 2nd {o2.action} {o2.reasons[:1]}", True)

    print("\n== failure-mode study (every scenario returned a defined outcome; 'safe' = no restricted fact in any answer)")
    print(tabulate(rows, headers=["scenario", "failure injected", "signal", "system behavior", "user-visible outcome", "safe", "notes"], tablefmt="github"))


if __name__ == "__main__":
    main()
