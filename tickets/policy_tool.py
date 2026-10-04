"""The resolver's link to the knowledge agent: an async Requisite tool that calls it over A2A.

The call is made AS THE HUMAN OPERATOR handling the ticket, not as the service:
the operator's session token is exchanged for a token addressed to the knowledge
agent whose groups are capped by the ticket agent's own ceiling (see
identity/tokens.py). The knowledge agent therefore enforces the operator's
access, further limited to what this service is allowed to lend.

Known limit: one operator context is held on the tool at a time, so tickets
are handled one at a time per platform instance. A concurrent design would
bind the context per request (for example per-ticket agents).
"""

from __future__ import annotations

from typing import Optional

from requisite.tools import tool

from a2a_layer.client import KnowledgeClient, RemoteDenied
from identity.tokens import AuthError, IdentityProvider

DENIED_TEXT = "Policy lookup was not permitted for this operator."


class PolicyTool:
    def __init__(self, client: KnowledgeClient, idp: IdentityProvider, service: str = "ticket-agent",
                 target_audience: str = "knowledge-agent") -> None:
        self.client, self.idp, self.service, self.target = client, idp, service, target_audience
        self._operator_token: Optional[str] = None
        self._correlation_id: str = ""
        self.calls: list[dict] = []

    def set_operator(self, operator_token: Optional[str], correlation_id: str) -> None:
        self._operator_token, self._correlation_id = operator_token, correlation_id

    def as_tool(self):
        holder = self

        @tool
        async def ask_policy(question: str) -> str:
            """Ask the company knowledge agent a policy question (refund rules, approval limits, escalation paths). Returns its cited answer."""
            try:
                exchanged = holder.idp.exchange(holder._operator_token or "", holder.service, holder.target)
                text = await holder.client.ask(question, exchanged, holder._correlation_id, subject="ticket-agent")
                holder.calls.append({"question": question, "ok": True})
                return text or DENIED_TEXT
            except (AuthError, RemoteDenied) as exc:
                holder.calls.append({"question": question, "ok": False, "error": str(exc)[:120]})
                return DENIED_TEXT

        return ask_policy
