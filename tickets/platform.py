"""The support-ticket platform: guardrails, three Requisite agents, one workflow, one gateway.

    ticket -> input guard -> [triage -> resolver (MCP + A2A tools) -> responder] -> output guard -> released | held | escalated

* The three agents are plain Requisite `Agent`s sharing one `Workflow`. Who a
  run is for (operator, tenant, correlation id) travels in Requisite's
  `RequestContext`, so the gateway reads the tenant itself, tools see
  the operator, and concurrent tickets for different operators stay isolated.
* The same `Workflow` runs on Requisite's native engine or on Google ADK,
  OpenAI Agents, Strands or Microsoft Agent Framework; only the coordinator
  changes. Which backend is active is a constructor argument.
* The resolver's tools come from an MCP server over stdio (a session the agent
  owns) and, optionally, from the knowledge agent over A2A.
* Nothing in `handle` raises: every failure becomes a *held* ticket with a
  reason, because an exception in a support pipeline is a lost ticket.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from requisite import Agent, RequestContext, Workflow
from requisite.core.exceptions import CostLimitException
from requisite.mcp import MCPClient
from requisite.telemetry.otel import get_tracer

from gateway.provider import GatewayExhausted, GatewayProvider
from gateway.routing import degraded_routes_used
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
POLICY_ADDENDUM = (
    " For company policy questions (refund rules, approval limits, escalation paths) call ask_policy with one specific "
    "question and include its answer, with its source ids in square brackets, in your note. If ask_policy says the "
    "lookup was not permitted, say so plainly and do not guess the policy."
)
RESPONDER_PROMPT = (
    "You write the customer-facing reply for a support ticket from a resolution note. Be brief and polite (3 sentences). "
    "Use only facts that appear in the note, quote exact values, and never mention any customer or order id that is not in the note."
)

_BACKENDS = {"adk": "use_adk", "native": "use_native", "openai_agents": "use_openai_agents",
             "strands": "use_strands", "agent_framework": "use_agent_framework"}


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
    correlation_id: str = ""


class TicketPlatform:
    def __init__(self, gateway: GatewayProvider, backend: str = "adk", policy: Optional[GuardrailPolicy] = None,
                 policy_tool=None, mcp_command: Optional[list] = None):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        self.gateway, self.backend = gateway, backend
        # example.com is the synthetic company domain used by the knowledge base.
        self.policy = policy or GuardrailPolicy(allowed_email_domains=frozenset({"example.com"}))
        self.policy_tool = policy_tool
        self._mcp_command = mcp_command or [sys.executable, "-m", "tickets.mcp_server"]
        self._workflow: Optional[Workflow] = None
        self._resolver: Optional[Agent] = None

    def close(self) -> None:
        """Close the resolver agent, which closes the MCP session it owns."""
        if self._resolver is not None:
            self._resolver.close()
        self._resolver = self._workflow = None

    def __enter__(self) -> "TicketPlatform":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _build(self) -> Workflow:
        if self._workflow is None:
            wf = Workflow()
            wf.add(Agent(name="triage", provider=self.gateway, system_prompt=TRIAGE_PROMPT, max_iterations=1))
            # The resolver owns its MCP session (Requisite persistent MCP sessions): one server process, reused for every call.
            mcp = MCPClient.stdio(name="ticket-ops", command=self._mcp_command[0], args=self._mcp_command[1:])
            prompt = RESOLVER_PROMPT + (POLICY_ADDENDUM if self.policy_tool else "")
            extra = [self.policy_tool.as_tool()] if self.policy_tool else []
            self._resolver = Agent(name="resolver", provider=self.gateway, system_prompt=prompt, tools=extra,
                                   mcp_clients=[mcp], max_iterations=6)
            wf.add(self._resolver)
            wf.add(Agent(name="responder", provider=self.gateway, system_prompt=RESPONDER_PROMPT, max_iterations=1))
            getattr(wf, _BACKENDS[self.backend])()
            self._workflow = wf
        return self._workflow

    def handle(self, ticket: Ticket, operator: str = "system", operator_token: Optional[str] = None,
               correlation_id: str = "") -> Outcome:
        t0 = time.perf_counter()
        corr = correlation_id or uuid.uuid4().hex[:12]

        def finish(span, action, reasons, **kw) -> Outcome:
            span.set_attribute("ticket.action", action)
            return Outcome(ticket.id, ticket.tenant, action, reasons, latency_s=time.perf_counter() - t0,
                           backend=self.backend, correlation_id=corr, **kw)

        with _tracer.start_as_current_span("ticket.handle") as span:
            span.set_attribute("ticket.id", ticket.id)
            span.set_attribute("ticket.tenant", ticket.tenant)
            span.set_attribute("ticket.backend", self.backend)
            span.set_attribute("ticket.correlation_id", corr)

            decision = self.policy.check_input(ticket.text)
            span.set_attribute("guardrail.input_action", decision.action)
            if decision.action == "block":
                return finish(span, "escalated", decision.reasons)

            ctx = RequestContext(user=operator, tenant=ticket.tenant, correlation_id=corr, attributes={"ticket": ticket.id})
            if self.policy_tool and operator_token:
                self.policy_tool.sessions.put(corr, operator_token)     # credential stays service-side, not in the context
            try:
                result = self._build().run(decision.text, context=ctx)
            except CostLimitException as exc:
                return finish(span, "held", [f"budget_exhausted: {str(exc)[:80]}"])
            except GatewayExhausted:
                return finish(span, "held", ["model_unavailable: every route failed"])
            except Exception as exc:  # noqa: BLE001 - a support pipeline must never lose a ticket to an exception
                return finish(span, "held", [f"pipeline_error: {type(exc).__name__}: {str(exc)[:120]}"])
            finally:
                if self.policy_tool:
                    self.policy_tool.sessions.pop(corr)

            mine = [r for r in self.gateway.audit.records if r.correlation_id == corr]
            tools = [t for step in result.steps for t in getattr(step, "tool_calls_executed", [])]
            try:
                triage = parse_triage(result.steps[0].content)
            except ValueError as exc:
                return finish(span, "held", [f"triage_invalid: {exc}"[:200]], reply=result.content, tools_used=tools,
                              model_calls=len(mine))

            out = self.policy.check_output(result.content, decision.text)
            span.set_attribute("guardrail.output_action", out.action)
            action = "released" if out.action == "release" else "held"
            reasons = list(out.problems) + list(decision.reasons)
            # The output guard checks for leaks, not for correctness. An answer produced by the
            # degraded-tier model passes the leak check but is not trustworthy, so it is held for review.
            degraded = degraded_routes_used(mine, self.gateway.routes)
            if degraded and action == "released":
                action = "held"
                reasons.append(f"served_by_degraded_tier: {','.join(degraded)}")
            # A tool that could not answer is a degraded capability too. The model was told not to guess,
            # but a reply that quietly omits what the customer asked for must not be released as complete.
            if self.policy_tool and self.policy_tool.failures(corr) and action == "released":
                action = "held"
                reasons.append("capability_unavailable: ask_policy")
            span.set_attribute("ticket.degraded_routes", ",".join(degraded))
            return finish(span, action, reasons, triage=triage, reply=result.content, tools_used=tools, model_calls=len(mine))
