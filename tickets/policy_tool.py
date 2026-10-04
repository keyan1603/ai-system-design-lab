"""The resolver's link to the knowledge agent: an async Requisite tool that calls it over A2A.

The call is made AS THE HUMAN OPERATOR handling the ticket, not as the service:
the operator's session token is exchanged for a token addressed to the knowledge
agent whose groups are capped by the ticket agent's own ceiling (see
identity/tokens.py). The knowledge agent therefore enforces the operator's
access, further limited to what this service is allowed to lend.

How the tool knows whose ticket it is (Requisite request context): it declares a
`RequestContext` parameter, which Requisite injects and hides from the model.
The operator's *credential* is deliberately not in that context (a context is
for identity and tracing, and can end up in logs); it sits in a service-side
`SessionTokens` store keyed by the request's correlation id. Both are isolated
per request, so tickets for different operators can run concurrently.
"""

from __future__ import annotations

import threading
from typing import Optional

from requisite import RequestContext
from requisite.tools import tool

from a2a_layer.client import KnowledgeClient, RemoteDenied
from identity.tokens import AuthError, IdentityProvider

DENIED_TEXT = "Policy lookup was not permitted for this operator."


class SessionTokens:
    """Operator session tokens held by the ticket service, keyed by correlation id."""

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, correlation_id: str, token: str) -> None:
        with self._lock:
            self._tokens[correlation_id] = token

    def get(self, correlation_id: str) -> Optional[str]:
        with self._lock:
            return self._tokens.get(correlation_id)

    def pop(self, correlation_id: str) -> None:
        with self._lock:
            self._tokens.pop(correlation_id, None)


class PolicyTool:
    def __init__(self, client: KnowledgeClient, idp: IdentityProvider, service: str = "ticket-agent",
                 target_audience: str = "knowledge-agent") -> None:
        self.client, self.idp, self.service, self.target = client, idp, service, target_audience
        self.sessions = SessionTokens()
        self.calls: list[dict] = []

    def failures(self, correlation_id: str) -> list:
        """Calls for this request that did not get an answer (service down, token refused, access denied)."""
        return [c for c in self.calls if c["correlation_id"] == correlation_id and not c["ok"]]

    def as_tool(self):
        holder = self

        @tool
        async def ask_policy(ctx: RequestContext, question: str) -> str:
            """Ask the company knowledge agent a policy question (refund rules, approval limits, escalation paths). Returns its cited answer."""
            try:
                operator_token = holder.sessions.get(ctx.correlation_id) or ""
                exchanged = holder.idp.exchange(operator_token, holder.service, holder.target)
                text = await holder.client.ask(question, exchanged, ctx.correlation_id, subject="ticket-agent")
                holder.calls.append({"correlation_id": ctx.correlation_id, "operator": ctx.user, "ok": True})
                return text or DENIED_TEXT
            except (AuthError, RemoteDenied) as exc:
                holder.calls.append({"correlation_id": ctx.correlation_id, "operator": ctx.user, "ok": False, "error": str(exc)[:120]})
                return DENIED_TEXT

        return ask_policy
