"""Offline integration test of the A2A layer: real HTTP on localhost, real ADK A2A
server and client, scripted model and bag-of-words embeddings (no network, no API key)."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from a2a_layer.client import KnowledgeClient, RemoteDenied
from a2a_layer.server import AgentAudit, start_knowledge_service
from identity.tokens import IdentityProvider, issue
from knowledge.corpus import DOCS, USERS
from knowledge.index import SecureIndex
from test_knowledge_mock import BowEmbedder, assistant as make_assistant

SECRET = b"a2a-test-secret"
IDP = IdentityProvider(SECRET, USERS, {"ticket-agent": {"employee", "support"}}, ttl_s=120)
EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module")
def service():
    tp = trace.get_tracer_provider()
    if not hasattr(tp, "add_span_processor"):
        tp = TracerProvider()
        trace.set_tracer_provider(tp)
    tp.add_span_processor(SimpleSpanProcessor(EXPORTER))
    idx = SecureIndex(BowEmbedder())
    idx.ingest(DOCS)
    bot, provider = make_assistant(idx)
    audit = AgentAudit()
    server = start_knowledge_service(bot, SECRET, audit)
    yield server, audit, provider
    server.stop()


def ask(server, question, token, corr="corr-1"):
    return asyncio.run(KnowledgeClient(server.url).ask(question, token, corr))


def knowledge_token(user, via_ticket_agent=True):
    if via_ticket_agent:
        return IDP.exchange(IDP.login(user, "ticket-agent"), "ticket-agent", "knowledge-agent")
    return issue(SECRET, user, USERS[user].groups, "knowledge-agent", 120)


def test_agent_card_is_public_and_describes_the_agent(service):
    server, _, _ = service
    card = httpx.get(f"{server.url}/.well-known/agent-card.json").json()
    assert card["name"] == "knowledge_agent"
    assert "bearer" in str(card.get("securitySchemes"))           # the card tells clients a token is required


def test_valid_delegated_token_gets_a_cited_answer_and_an_audit_record(service):
    server, audit, _ = service
    text = ask(server, "How are duplicate charges refunded?", knowledge_token("sam"), "corr-sam")
    assert "[" in text and "]" in text                                    # scripted model cites a retrieved source
    rec = [r for r in audit.records if r.correlation_id == "corr-sam"][-1]
    assert rec.outcome == "answered" and rec.subject == "sam" and rec.acting_service == "ticket-agent"
    assert "SUP-001" in rec.retrieved and set(rec.effective_groups) == {"employee", "support"}


@pytest.mark.parametrize("token_factory,reason", [
    (lambda: None, "missing_token"),
    (lambda: "garbage.token", "malformed"),
    (lambda: issue(SECRET, "sam", ["support"], "knowledge-agent", ttl_s=-5), "expired"),
    (lambda: issue(SECRET, "sam", ["support"], "ticket-agent", ttl_s=60), "wrong_audience"),
    (lambda: issue(b"attacker-key", "sam", ["exec"], "knowledge-agent", ttl_s=60), "bad_signature"),
])
def test_bad_credentials_are_rejected_before_any_agent_runs(service, token_factory, reason):
    server, audit, provider = service
    before = len(provider.prompts)
    with pytest.raises(RemoteDenied):
        ask(server, "What is the Q3 forecast?", token_factory(), f"corr-{reason}")
    assert len(provider.prompts) == before                                # the model was never called
    assert audit.records[-1].outcome == f"denied_auth:{reason}"


def test_delegation_removes_exec_access_that_a_direct_call_would_have(service):
    server, audit, provider = service
    q = "How much is the Project Falcon acquisition?"
    ask(server, q, knowledge_token("erin", via_ticket_agent=True), "corr-erin-via")
    via = audit.records[-1]
    ask(server, q, knowledge_token("erin", via_ticket_agent=False), "corr-erin-direct")
    direct = audit.records[-1]
    assert "LEG-001" not in via.retrieved and set(via.effective_groups) == {"employee"}
    assert "LEG-001" in direct.retrieved and "exec" in direct.effective_groups


def test_one_trace_spans_client_and_server(service):
    server, _, _ = service
    tracer = trace.get_tracer("test")
    EXPORTER.clear()

    async def go():
        with tracer.start_as_current_span("caller.root"):
            return await KnowledgeClient(server.url).ask("How are duplicate charges refunded?", knowledge_token("sam"), "corr-trace")

    asyncio.run(go())
    spans = EXPORTER.get_finished_spans()
    root = next(s for s in spans if s.name == "caller.root")
    server_side = [s for s in spans if s.name == "knowledge.ask"]
    assert server_side and server_side[-1].context.trace_id == root.context.trace_id
