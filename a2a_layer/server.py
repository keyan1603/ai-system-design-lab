"""The knowledge assistant as an A2A service, built with Google ADK's `to_a2a`.

    A2A client --HTTP--> auth middleware --> ADK A2A executor --> KnowledgeA2AAgent --> KnowledgeAssistant (Requisite)

* ADK supplies the A2A server, agent card and task handling (`to_a2a`).
* Requisite supplies the agent logic: the existing `KnowledgeAssistant`.
  `KnowledgeA2AAgent` is a thin ADK `BaseAgent` that hands the question to it.
* The middleware is where identity is enforced. It verifies the bearer token
  (signature, expiry, audience), and the agent then derives the caller's
  groups from the VERIFIED token, never from anything the caller merely says
  in the message. The agent card stays public, as the A2A discovery flow needs.
* W3C `traceparent` and `X-Correlation-Id` are honored so one request is one
  trace and one correlation id across both services.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import uvicorn
from a2a.types import HTTPAuthSecurityScheme, SecurityRequirement, SecurityScheme, StringList
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types as genai_types
from opentelemetry import context as otel_context
from opentelemetry import propagate
from pydantic import PrivateAttr
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from identity.tokens import AuthError, Claims, verify
from knowledge.assistant import KnowledgeAssistant
from knowledge.corpus import User

CLAIMS: contextvars.ContextVar[Optional[Claims]] = contextvars.ContextVar("claims", default=None)
CORRELATION: contextvars.ContextVar[str] = contextvars.ContextVar("correlation", default="")

AUDIENCE = "knowledge-agent"
WITHHELD = "The answer was withheld by policy checks."


@dataclass
class AgentAuditRecord:
    correlation_id: str
    service: str
    outcome: str                       # answered | not_found | held | escalated | denied_auth:<reason>
    subject: str = ""
    acting_service: str = ""
    effective_groups: list = field(default_factory=list)
    question_sha256: str = ""
    retrieved: list = field(default_factory=list)
    cited: list = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class AgentAudit:
    """Cross-service audit trail. Questions are hashed, never stored."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.records: list[AgentAuditRecord] = []
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, rec: AgentAuditRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret: bytes, audience: str, audit: AgentAudit, service: str) -> None:
        super().__init__(app)
        self.secret, self.audience, self.audit, self.service = secret, audience, audit, service

    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/.well-known/"):      # agent card discovery is public
            return await call_next(request)
        corr = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:12]
        header = request.headers.get("authorization", "")
        try:
            if not header.lower().startswith("bearer "):
                raise AuthError("missing_token")
            claims = verify(self.secret, header[7:].strip(), self.audience)
        except AuthError as exc:
            self.audit.write(AgentAuditRecord(corr, self.service, f"denied_auth:{exc.reason}"))
            return JSONResponse({"error": "unauthorized", "reason": exc.reason}, status_code=401)
        # Continue the caller's trace, then expose identity to the agent via context.
        ctx_token = otel_context.attach(propagate.extract(dict(request.headers)))
        c1, c2 = CLAIMS.set(claims), CORRELATION.set(corr)
        try:
            return await call_next(request)
        finally:
            CLAIMS.reset(c1)
            CORRELATION.reset(c2)
            otel_context.detach(ctx_token)


class KnowledgeA2AAgent(BaseAgent):
    """ADK agent that answers with Requisite's KnowledgeAssistant, as the verified caller."""

    _assistant: Any = PrivateAttr()
    _audit: Any = PrivateAttr()

    def __init__(self, *, assistant: KnowledgeAssistant, audit: AgentAudit, **data: Any) -> None:
        super().__init__(**data)
        self._assistant, self._audit = assistant, audit

    async def _run_async_impl(self, ctx: Any):
        parts = (ctx.user_content.parts if ctx.user_content else None) or []
        question = " ".join(p.text for p in parts if getattr(p, "text", None)).strip()
        claims, corr = CLAIMS.get(), CORRELATION.get()
        if claims is None:                                     # defence in depth: middleware always sets this
            text, outcome, groups, retrieved, cited, sub, act = "Unauthorized.", "denied_auth:no_claims", [], [], [], "", ""
        else:
            user = User(claims.sub, "acme", tuple(claims.groups))
            # to_thread keeps the blocking Requisite call off the event loop and copies the trace context.
            ans = await asyncio.to_thread(self._assistant.ask, user, question, corr)
            text = ans.text if ans.action in ("answered", "not_found") else WITHHELD
            outcome, groups, retrieved, cited = ans.action, list(claims.groups), ans.retrieved, ans.citations
            sub, act = claims.sub, claims.act or ""
        self._audit.write(AgentAuditRecord(corr, self.name, outcome, sub, act, groups,
                                           hashlib.sha256(question.encode()).hexdigest(), retrieved, cited))
        yield Event(invocation_id=ctx.invocation_id, author=self.name,
                    content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    """Runs a Starlette app on localhost in a background thread (real HTTP, one process)."""

    def __init__(self, app, port: int) -> None:
        self.port = port
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self._thread = threading.Thread(target=lambda: asyncio.run(self._server.serve()), daemon=True)

    def start(self, timeout: float = 15.0) -> "ServerThread":
        self._thread.start()
        deadline = time.time() + timeout
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("A2A server did not start")
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def start_knowledge_service(assistant: KnowledgeAssistant, secret: bytes, audit: AgentAudit) -> ServerThread:
    port = free_port()
    agent = KnowledgeA2AAgent(
        name="knowledge_agent",
        description="Answers internal policy and runbook questions with citations, limited to what the verified caller may read.",
        assistant=assistant, audit=audit)
    # ADK's automatic card declares no security scheme, so a client could not learn that a
    # bearer token is required. Build the card ourselves with the scheme the middleware enforces.
    scheme = SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(
        scheme="bearer", bearer_format="signed-token", description="Short-lived, audience-bound token from the identity provider."))
    card = asyncio.run(AgentCardBuilder(agent=agent, rpc_url=f"http://127.0.0.1:{port}",
                                        security_schemes={"bearer": scheme}).build())
    card.security_requirements.append(SecurityRequirement(schemes={"bearer": StringList(list=[])}))
    app = to_a2a(agent, host="127.0.0.1", port=port, agent_card=card)
    app.add_middleware(AuthMiddleware, secret=secret, audience=AUDIENCE, audit=audit, service="knowledge-agent")
    return ServerThread(app, port).start()
