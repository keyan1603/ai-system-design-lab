"""A2A client for the knowledge service, using ADK's `RemoteA2aAgent`.

The ticket platform calls the knowledge assistant the way any A2A client would:
fetch the public agent card, then send the question with a bearer token. The
token, a correlation id and the W3C trace context travel as HTTP headers via an
httpx auth hook, so the receiving service can verify who is asking and join
the caller's trace.
"""

from __future__ import annotations

import uuid
from typing import Optional

import httpx
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from opentelemetry import propagate


class RemoteDenied(Exception):
    """The remote service refused the call (authentication or policy)."""


class _Auth(httpx.Auth):
    def __init__(self, token: Optional[str], correlation_id: str) -> None:
        self.token, self.correlation_id = token, correlation_id

    def auth_flow(self, request):
        if self.token:
            request.headers["Authorization"] = f"Bearer {self.token}"
        request.headers["X-Correlation-Id"] = self.correlation_id
        carrier: dict = {}
        propagate.inject(carrier)                      # traceparent from the caller's current span
        request.headers.update(carrier)
        yield request


class KnowledgeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def ask(self, question: str, token: Optional[str], correlation_id: Optional[str] = None, subject: str = "caller") -> str:
        corr = correlation_id or uuid.uuid4().hex[:12]
        async with httpx.AsyncClient(auth=_Auth(token, corr), timeout=90.0) as http:
            remote = RemoteA2aAgent(name="knowledge_agent",
                                    agent_card=f"{self.base_url}/.well-known/agent-card.json", httpx_client=http)
            runner = InMemoryRunner(agent=remote, app_name="ticket-agent")
            session = await runner.session_service.create_session(app_name="ticket-agent", user_id=subject)
            message = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
            texts: list[str] = []
            try:
                async for event in runner.run_async(user_id=subject, session_id=session.id, new_message=message):
                    if getattr(event, "error_message", None):
                        raise RemoteDenied(event.error_message)
                    if event.content and event.content.parts:
                        texts += [p.text for p in event.content.parts if getattr(p, "text", None)]
            except RemoteDenied:
                raise
            except Exception as exc:  # noqa: BLE001 - surface transport/auth failures as one error type
                raise RemoteDenied(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
            return "\n".join(texts).strip()
