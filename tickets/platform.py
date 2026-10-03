"""The support-ticket platform: guardrails, three Requisite agents, one workflow, one gateway.

    ticket -> input guard -> [triage -> resolver (MCP tools) -> responder] -> output guard -> released | held | escalated

* The three agents are plain Requisite `Agent`s. Their provider is a
  `TenantProvider` over the gateway, so every model call is routed, cached,
  budgeted and audited per tenant without the agents knowing.
* The same `Workflow` runs on Requisite's native engine or on Google ADK
  (`use_adk()`); only the coordinator changes. Which backend is active is a
  constructor argument, which is what makes the backend comparison fair.
* The resolver's tools come from an MCP server over stdio, via Requisite's
  `MCPClient`.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from requisite import Agent, Workflow
from requisite.mcp import MCPClient
from requisite.telemetry.otel import get_tracer

from gateway.provider import GatewayProvider
from gateway.routing import degraded_routes_used
from gateway.tenant import TenantProvider
from guardrails.checks import TriageResult, parse_triage
from guardrails.policy import GuardrailPolicy
from tickets.data import Ticket

_tracer = get_tracer("ai_system_design.tickets")

TRIAGE_PROMPT = (
    "You triage customer support tickets. Reply with ONLY a JSON object, no prose, with keys: "
    '"category" (billing, technical or account), "severity" (low, medium or high), '
    '"customer_id" (like C-1001, or null), "order_id" (like O-5001, or null), '
    'and "summary" (one sentence that restates the problem and keeps any customer or order ids).'
)
RESOLVER_PROMPT = (
    "You are a support resolver. You receive a triage JSON. Use the tools to look up the facts you need "
    "(get_order, get_customer, search_kb with one of: password_reset, duplicate_charge, suspended_account, mobile_crash). "
    "Never guess. Reply with a short resolution note that restates the ticket in one sentence, then lists the facts "
    "you looked up using their exact values (for example a billing status or a number of minutes)."
)
RESPONDER_PROMPT = (
    "You write the customer-facing reply for a support ticket from a resolution note. Be brief and polite (3 sentences). "
    "Use only facts that appear in the note, quote exact values, and never mention any customer or order id that is not in the note."
)


@dataclass
class Outcome:
    ticket_id: str
    tenant: str
    action: str                  # released | held | escalated
    reasons: list = field(default_factory=list)
    triage: Optional[TriageResult] = None
    reply: str = ""
    tools_used: list = field(default_factory=list)
    latency_s: float = 0.0
    backend: str = ""
    model_calls: int = 0


class TicketPlatform:
    def __init__(self, gateway: GatewayProvider, backend: str = "adk", policy: Optional[GuardrailPolicy] = None):
        if backend not in ("adk", "native"):
            raise ValueError("backend must be 'adk' or 'native'")
        self.gateway, self.backend = gateway, backend
        # example.com is the synthetic company domain used by the knowledge base.
        self.policy = policy or GuardrailPolicy(allowed_email_domains=frozenset({"example.com"}))
        # MCP server as a subprocess over stdio; tools are discovered once and reused.
        self._mcp = MCPClient.stdio(name="ticket-ops", command=sys.executable, args=["-m", "tickets.mcp_server"])
        self._tools = self._mcp.discover_tools()
        self._workflows: dict[str, Workflow] = {}

    def _workflow_for(self, tenant: str) -> Workflow:
        if tenant not in self._workflows:
            provider = TenantProvider(self.gateway, tenant)
            wf = Workflow()
            wf.add(Agent(name="triage", provider=provider, system_prompt=TRIAGE_PROMPT, max_iterations=1))
            wf.add(Agent(name="resolver", provider=provider, system_prompt=RESOLVER_PROMPT, tools=self._tools, max_iterations=6))
            wf.add(Agent(name="responder", provider=provider, system_prompt=RESPONDER_PROMPT, max_iterations=1))
            wf.use_adk() if self.backend == "adk" else wf.use_native()
            self._workflows[tenant] = wf
        return self._workflows[tenant]

    def handle(self, ticket: Ticket) -> Outcome:
        t0 = time.perf_counter()
        with _tracer.start_as_current_span("ticket.handle") as span:
            span.set_attribute("ticket.id", ticket.id)
            span.set_attribute("ticket.tenant", ticket.tenant)
            span.set_attribute("ticket.backend", self.backend)

            decision = self.policy.check_input(ticket.text)
            span.set_attribute("guardrail.input_action", decision.action)
            if decision.action == "block":
                span.set_attribute("ticket.action", "escalated")
                return Outcome(ticket.id, ticket.tenant, "escalated", decision.reasons,
                               latency_s=time.perf_counter() - t0, backend=self.backend)

            before = len(self.gateway.audit.records)
            result = self._workflow_for(ticket.tenant).run(decision.text)
            calls = len(self.gateway.audit.records) - before
            tools = [t for step in result.steps for t in getattr(step, "tool_calls_executed", [])]

            try:
                triage = parse_triage(result.steps[0].content)
            except ValueError as exc:
                span.set_attribute("ticket.action", "held")
                return Outcome(ticket.id, ticket.tenant, "held", [f"triage_invalid: {exc}"[:200]], reply=result.content,
                               tools_used=tools, latency_s=time.perf_counter() - t0, backend=self.backend, model_calls=calls)

            out = self.policy.check_output(result.content, decision.text)
            span.set_attribute("guardrail.output_action", out.action)
            action = "released" if out.action == "release" else "held"
            # The output guard checks for leaks, not for correctness. An answer
            # produced by the degraded-tier model passes the leak check but is
            # not trustworthy, so it is held for review instead of released.
            degraded = degraded_routes_used(self.gateway.audit.records[before:], self.gateway.routes)
            if degraded and action == "released":
                action = "held"
                out.problems.append(f"served_by_degraded_tier: {','.join(degraded)}")
            span.set_attribute("ticket.degraded_routes", ",".join(degraded))
            span.set_attribute("ticket.action", action)
            return Outcome(ticket.id, ticket.tenant, action, out.problems + decision.reasons, triage, result.content,
                           tools, time.perf_counter() - t0, self.backend, calls)
