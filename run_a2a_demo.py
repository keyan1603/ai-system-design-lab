"""Phase 4 live run: two agents, two services, one verified identity, one trace.

    operator --login--> ticket platform (ADK + Requisite) --A2A over HTTP--> knowledge agent (ADK to_a2a + Requisite)

Scenarios
  1. a support operator resolves a refund ticket; the resolver asks the knowledge agent for policy over A2A
  2. an engineer opens the same ticket; the knowledge agent must refuse the support-only policy
  3. attacks on the A2A endpoint: no token, forged, expired, wrong audience
  4. confused deputy: an executive acting through the ticket agent vs calling the knowledge agent directly
  5. one trace across both services, joined to the audit trail by correlation id
"""

import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv
from requisite.core.rate_limiter import RateLimiter
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from tabulate import tabulate

load_dotenv(".env")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

exporter = InMemorySpanExporter()
_tp = TracerProvider()
_tp.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(_tp)

from a2a_layer.client import KnowledgeClient, RemoteDenied  # noqa: E402
from a2a_layer.server import AgentAudit, start_knowledge_service  # noqa: E402
from gateway.factory import build_gateway  # noqa: E402
from gateway.routing import LIGHT  # noqa: E402
from identity.tokens import IdentityProvider, issue  # noqa: E402
from knowledge.assistant import KnowledgeAssistant  # noqa: E402
from knowledge.corpus import DOCS, USERS  # noqa: E402
from knowledge.index import SecureIndex  # noqa: E402
from tickets.data import Ticket  # noqa: E402
from tickets.platform import TicketPlatform  # noqa: E402
from tickets.policy_tool import PolicyTool  # noqa: E402

REFUND_TICKET = Ticket(
    "P01", "acme",
    "Customer C-1002 reports order O-5002 was charged twice. Can we refund the duplicate, how long will it take, "
    "and what is the approval threshold for refunds?",
    "billing", "high", must_mention=["500"])


def tree(trace_id):
    spans = [s for s in exporter.get_finished_spans() if s.context.trace_id == trace_id]
    by_parent = {}
    for s in spans:
        by_parent.setdefault(s.parent.span_id if s.parent else None, []).append(s)
    out = []

    def walk(pid, depth):
        for s in sorted(by_parent.get(pid, []), key=lambda x: x.start_time):
            a = s.attributes or {}
            note = ""
            if s.name == "gateway.chat":
                note = f"  route={a.get('gateway.route')} in={a.get('gen_ai.usage.input_tokens')} out={a.get('gen_ai.usage.output_tokens')}"
            if s.name in ("knowledge.ask", "ticket.handle"):
                note = f"  {a.get('knowledge.user', a.get('ticket.id', ''))} action={a.get('knowledge.action', a.get('ticket.action', ''))}"
            noisy = "event_queue" in s.name or "task_manager" in s.name or "_deliver" in s.name or "_dispatch_loop" in s.name
            keep = (not noisy) and (s.name.startswith(("ticket.", "knowledge.", "gateway.", "requisite.agent.tool_call", "invoke_agent"))
                                    or s.name.endswith(("send_message_streaming", "handle_requests")))
            if keep:
                out.append(f"{'  ' * depth}{s.name} ({(s.end_time - s.start_time) / 1e6:.0f} ms){note}")
            walk(s.context.span_id, depth + (1 if keep else 0))

    walk(None, 0)
    return "\n".join(out)


def main():
    limiter = RateLimiter(requests_per_minute=15)     # one limiter for one API key, shared by both services
    secret = os.urandom(32)
    idp = IdentityProvider(secret, USERS, {"ticket-agent": {"employee", "support"}}, ttl_s=300)

    # ---- the knowledge service (its own gateway, index and audit) ----
    kgw, _ = build_gateway(use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=limiter, audit_path="audit/phase4_knowledge_gateway.jsonl")
    index = SecureIndex(GeminiEmbeddingProvider(model="gemini-embedding-2"))
    index.ingest(DOCS)
    audit = AgentAudit(path="audit/phase4_agent_audit.jsonl")
    service = start_knowledge_service(KnowledgeAssistant(kgw, index), secret, audit)
    print("knowledge agent serving at", service.url)

    # ---- the ticket platform (its own gateway), linked to the knowledge agent over A2A ----
    tgw, _ = build_gateway(use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=limiter, audit_path="audit/phase4_ticket_gateway.jsonl")
    ptool = PolicyTool(KnowledgeClient(service.url), idp)
    results = {}
    with TicketPlatform(tgw, backend="adk", policy_tool=ptool) as platform:
        for label, operator in (("1. support operator (sam)", "sam"), ("2. engineer (alice)", "alice")):
            corr = f"corr-{operator}"
            op_token = idp.login(operator, "ticket-agent")
            out = platform.handle(REFUND_TICKET, operator_token=op_token, correlation_id=corr)
            results[operator] = (out, corr)
            print(f"\n== {label}: action={out.action} tools={sorted(set(out.tools_used))} policy_calls={ptool.calls[-1:] }")
            print("   reply:", out.reply[:420].replace("\n", " "))
            reply = out.reply.lower()
            # "500" alone would match the order id O-5002, so look for the policy's own wording.
            print("   reply states the support-only refund threshold:", "500 dollars" in reply or "team lead" in reply)

    # ---- attacks on the A2A endpoint (no model is called for any of these) ----
    print("\n== 3. attacks on the A2A endpoint")
    client = KnowledgeClient(service.url)
    q = "What is the Q3 revenue forecast?"
    attacks = [
        ("no token", None),
        ("forged signature (attacker key, claims exec)", issue(b"attacker-key", "mallory", ["exec", "finance"], "knowledge-agent")),
        ("expired token", issue(secret, "sam", ["employee", "support"], "knowledge-agent", ttl_s=-5)),
        ("token for another audience (ticket-agent)", issue(secret, "sam", ["employee", "support"], "ticket-agent")),
    ]
    rows = []
    calls_before = len(kgw.audit.records)
    for name, token in attacks:
        try:
            asyncio.run(client.ask(q, token, "corr-attack"))
            result = "ACCEPTED"
        except RemoteDenied as e:
            result = "rejected (401)" if "401" in str(e) else f"rejected: {str(e)[:50]}"
        rows.append([name, result, audit.records[-1].outcome])
    print(tabulate(rows, headers=["attack", "client saw", "service audit outcome"], tablefmt="github"))
    print("   model calls made by the knowledge service during the attacks:", len(kgw.audit.records) - calls_before)

    # ---- confused deputy ----
    print("\n== 4. confused deputy: executive via the ticket agent vs direct")
    falcon = "How much is the Project Falcon acquisition and how long is exclusivity?"
    erin_via = idp.exchange(idp.login("erin", "ticket-agent"), "ticket-agent", "knowledge-agent")
    erin_direct = issue(secret, "erin", USERS["erin"].groups, "knowledge-agent", 300)
    rows = []
    for name, token in (("erin via ticket agent (groups capped by its ceiling)", erin_via), ("erin directly", erin_direct)):
        text = asyncio.run(client.ask(falcon, token, f"corr-erin-{len(rows)}"))
        rec = audit.records[-1]
        rows.append([name, ",".join(rec.effective_groups), "LEG-001" in rec.retrieved, rec.outcome, text[:70].replace("\n", " ")])
    print(tabulate(rows, headers=["caller", "effective groups", "LEG-001 retrieved", "outcome", "answer"], tablefmt="github"))

    # ---- one trace, one correlation id ----
    out_sam, corr_sam = results["sam"]
    print("\n== 5. audit trail (knowledge service) for the two ticket runs")
    print(tabulate([[r.correlation_id, r.service, r.outcome, r.subject, r.acting_service, ",".join(r.effective_groups), ",".join(r.cited) or "-"]
                    for r in audit.records if r.correlation_id in ("corr-sam", "corr-alice")],
                   headers=["correlation id", "service", "outcome", "subject", "acting service", "groups", "cited"], tablefmt="github"))
    root = next(s for s in exporter.get_finished_spans() if s.name == "ticket.handle" and s.attributes.get("ticket.correlation_id") == corr_sam)
    print("\ntrace across both services (same trace id):")
    print(tree(root.context.trace_id))

    card = httpx.get(f"{service.url}/.well-known/agent-card.json").json()
    print("\nagent card (public):", json.dumps({k: card.get(k) for k in ("name", "description", "url", "version", "capabilities", "securitySchemes", "skills")}, default=str)[:700])
    service.stop()


if __name__ == "__main__":
    main()
