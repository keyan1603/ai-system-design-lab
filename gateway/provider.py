"""GatewayProvider: an in-process AI gateway that *is* a Requisite provider.

Because it subclasses `requisite.providers.base.BaseProvider`, anything in the
framework that accepts a provider accepts the gateway: `AI(provider=gw)`,
`Agent(provider=gw)`, and therefore every orchestrator backend (native, ADK,
OpenAI Agents, Strands, ...). Routing, caching, failover, budgets and audit
apply to every model call without any caller knowing the gateway exists.

What it deliberately is NOT: a network service. A shared multi-tenant gateway
(virtual keys, per-team quotas, a dashboard) belongs at the deployment tier,
and the Phase 5/6 write-up compares that buy-vs-build choice. This class is
the in-process policy engine such a service would embed.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Optional

from pydantic import BaseModel
from requisite import current_context, submit_with_context
from requisite.core.cost_limiter import CostLimiter
from requisite.core.exceptions import CostLimitException, ProviderException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk
from requisite.providers.base import BaseProvider
from requisite.telemetry.otel import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    genai_request_attributes,
    get_tracer,
)

from gateway.audit import AuditLog, AuditRecord
from gateway.cache import SemanticCache, make_scope, prompt_text
from gateway.routing import Classifier, Route, default_classifier, fallback_order

_tracer = get_tracer("ai_system_design.gateway")
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gateway-call")

# kwargs consumed by the gateway itself, never forwarded to a provider
_GATEWAY_KWARGS = ("tenant", "tier", "no_cache")


class GatewayExhausted(ProviderException):
    """Every eligible route failed or was skipped by its circuit breaker."""


class GatewayProvider(BaseProvider):
    def __init__(
        self,
        routes: Sequence[Route],
        *,
        classifier: Classifier = default_classifier,
        cache: Optional[SemanticCache] = None,
        cost_limiter: Optional[CostLimiter] = None,
        tenant_budgets: Optional[Callable[[str], CostLimiter]] = None,
        audit: Optional[AuditLog] = None,
        default_tenant: str = "default",
    ) -> None:
        if not routes:
            raise ValueError("GatewayProvider needs at least one route")
        super().__init__(api_key="gateway", model=routes[0].provider.model, timeout=0)
        self.routes = list(routes)
        self.classifier = classifier
        self.cache = cache
        self.cost_limiter = cost_limiter
        # One CostLimiter per tenant, created on first use, so one tenant's
        # spend can never block another's. The global limiter (if any) is the
        # fleet-wide ceiling above them.
        self._tenant_budget_factory = tenant_budgets
        self._tenant_budgets: dict[str, CostLimiter] = {}
        # Which route served the tool-calling turns of each conversation, so the
        # rest of that tool loop can be pinned to it (see _plan).
        self._loop_routes: "dict[str, Route]" = {}
        self.audit = audit or AuditLog()
        self.default_tenant = default_tenant

    # -- BaseProvider surface -------------------------------------------------
    @property
    def name(self) -> str:
        return "gateway"

    @property
    def model(self) -> str:
        return "gateway-routed"

    def validate_config(self) -> None:
        for r in self.routes:
            r.provider.validate_config()

    def tenant_budget(self, tenant: str) -> Optional[CostLimiter]:
        if self._tenant_budget_factory is None:
            return None
        if tenant not in self._tenant_budgets:
            self._tenant_budgets[tenant] = self._tenant_budget_factory(tenant)
        return self._tenant_budgets[tenant]

    def _check_budgets(self, tenant: str) -> None:
        for limiter in (self.cost_limiter, self.tenant_budget(tenant)):
            if limiter is not None:
                limiter.check()

    def _record_spend(self, tenant: str, response: ChatResponse) -> None:
        for limiter in (self.cost_limiter, self.tenant_budget(tenant)):
            if limiter is not None:
                limiter.record(response.usage, response.model)

    def _call_sync(self, route, messages, temperature, tools, response_model, kwargs):
        if route.limiter is not None:
            route.limiter.acquire()
        if route.timeout_s is None:
            return route.provider.chat(messages, temperature=temperature, tools=tools, response_model=response_model, **kwargs)
        # The provider call runs in a worker thread (carrying the request context) so the
        # gateway can stop waiting. The abandoned call still finishes in the background;
        # a timeout bounds latency, it does not cancel work or its cost.
        future = submit_with_context(_TIMEOUT_POOL, route.provider.chat, messages, temperature=temperature, tools=tools,
                                     response_model=response_model, **kwargs)
        try:
            return future.result(timeout=route.timeout_s)
        except FutureTimeout as exc:
            raise TimeoutError(f"route {route.name} exceeded {route.timeout_s}s") from exc

    async def _call_async(self, route, messages, temperature, tools, response_model, kwargs):
        if route.limiter is not None:
            await route.limiter.aacquire()
        call = route.provider.achat(messages, temperature=temperature, tools=tools, response_model=response_model, **kwargs)
        if route.timeout_s is None:
            return await call
        return await asyncio.wait_for(call, timeout=route.timeout_s)

    def preflight(self, probe: bool = False) -> dict:
        """Check every route *before* an outage forces the gateway to use it.

        A fallback route that has never been exercised is a hope, not a
        control: a missing SDK extra or a stopped local server only shows up
        at the worst moment. ``probe=True`` makes one tiny real call per route.
        """
        report = {}
        for r in self.routes:
            try:
                r.provider.validate_config()
                if probe:
                    r.provider.chat([Message(role=Role.USER, content="ping")], temperature=0.0)
                report[r.name] = "ok"
            except Exception as exc:  # noqa: BLE001
                report[r.name] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return report

    # -- shared planning / bookkeeping ---------------------------------------
    def _plan(self, messages, tools, response_model, kwargs):
        opts = {k: kwargs.pop(k) for k in _GATEWAY_KWARGS if k in kwargs}
        # Tenant and correlation id come from Requisite's request-scoped context
        # (0.42.0+) unless a caller passes `tenant=` explicitly; no wrapper has to carry them.
        ctx = current_context()
        tenant = opts.get("tenant") or (ctx.tenant if ctx and ctx.tenant else self.default_tenant)
        tier = opts.get("tier") or self.classifier(messages, bool(tools), response_model is not None)
        # Tool-calling responses are never cached: they describe an action to
        # take, and replaying a stale action is worse than paying for a call.
        cacheable = self.cache is not None and not tools and not opts.get("no_cache", False)
        text = prompt_text(messages)
        # Conversation identity for sticky routing: everything up to and
        # including the first user message, which is the same on every turn.
        head = []
        for m in messages:
            head.append(f"{getattr(m.role, 'value', m.role)}:{m.content}")
            if getattr(m.role, "value", m.role) == "user":
                break
        loop_key = hashlib.sha256("|".join(head).encode()).hexdigest()
        candidates = fallback_order(self.routes, tier)
        # Sticky tool loops. Once a conversation contains tool results, its
        # history holds provider-specific state (for Gemini 3, thought
        # signatures on function-call parts) that another model rejects or
        # mishandles. Failing over in the middle of a tool loop therefore
        # breaks it; the loop stays on the route that started it, and if that
        # route fails the error goes to the caller, which restarts the whole
        # task on another route.
        in_tool_loop = any(getattr(m.role, "value", m.role) == "tool" for m in messages)
        if in_tool_loop and loop_key in self._loop_routes:
            candidates = [self._loop_routes[loop_key]]
        return {
            "has_tools": bool(tools),
            "loop_key": loop_key,
            "request_id": uuid.uuid4().hex[:12],
            "correlation_id": (ctx.correlation_id if ctx and ctx.correlation_id else ""),
            "tenant": tenant,
            "tier": tier,
            "cacheable": cacheable,
            "text": text,
            "scope": make_scope(tenant, messages),
            "candidates": candidates,
        }

    def _record(self, plan, t0, outcome, route=None, response=None, attempts=None, cost=0.0, sim=None):
        rec = AuditRecord(
            request_id=plan["request_id"],
            correlation_id=plan["correlation_id"],
            tenant=plan["tenant"],
            tier=plan["tier"],
            outcome=outcome,
            route=route,
            model=getattr(response, "model", None),
            attempts=attempts or [],
            prompt_sha256=hashlib.sha256(plan["text"].encode()).hexdigest(),
            prompt_chars=len(plan["text"]),
            # A cache hit consumed no tokens; the tokens the cached answer
            # originally cost are reported separately as savings.
            input_tokens=response.usage.prompt_tokens if response and not outcome.startswith("cache") else 0,
            output_tokens=response.usage.completion_tokens if response and not outcome.startswith("cache") else 0,
            saved_tokens=(response.usage.prompt_tokens + response.usage.completion_tokens) if response and outcome.startswith("cache") else 0,
            cost_usd=cost,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            cache_similarity=sim,
            prompt=plan["text"],
        )
        self.audit.write(rec)
        return rec

    def _span_attrs(self, plan, route: Optional[Route], response: Optional[ChatResponse], rec: AuditRecord) -> dict:
        # OpenTelemetry GenAI semantic conventions (still marked
        # "Development" upstream, names can change): gen_ai.* for the model
        # call, gateway.* for what only a gateway knows.
        # gen_ai.* names and the provider-name mapping come from Requisite
        # (0.39.0+), so the lab and the framework share one vocabulary.
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: "chat",
            "gateway.request_id": plan["request_id"],
            "gateway.tenant": plan["tenant"],
            "gateway.tier": plan["tier"],
            "gateway.outcome": rec.outcome,
            "gateway.attempts": len(rec.attempts),
            "gateway.cost_usd": rec.cost_usd,
        }
        if route is not None and response is not None:
            attrs.update(genai_request_attributes(response.provider, route.provider.model))
            attrs.update({
                GEN_AI_RESPONSE_MODEL: response.model,
                GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
                GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
                "gateway.route": route.name,
            })
        return attrs

    def _annotate(self, span, plan, route, response, rec) -> None:
        for k, v in self._span_attrs(plan, route, response, rec).items():
            span.set_attribute(k, v)

    def _cache_response(self, hit) -> ChatResponse:
        return hit.response.model_copy(update={"provider": "gateway-cache"})

    def _finish_ok(self, plan, t0, route, response, attempts, store_cache=True):
        cost = route.cost_fn(response.usage, response.model)
        self._record_spend(plan["tenant"], response)
        route.breaker.record_success()
        if plan["has_tools"]:
            self._loop_routes[plan["loop_key"]] = route
            while len(self._loop_routes) > 1024:
                self._loop_routes.pop(next(iter(self._loop_routes)))
        # Structured (parsed) responses are not cached: the parsed object is not
        # guaranteed to survive a copy, and a cached text-only hit would drop it.
        if store_cache and plan["cacheable"] and response.parsed is None:
            self.cache.store(plan["scope"], plan["text"], response)
        return self._record(plan, t0, "ok", route.name, response, attempts, cost)

    # -- sync chat -----------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Any]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        t0 = time.perf_counter()
        plan = self._plan(messages, tools, response_model, kwargs)
        with _tracer.start_as_current_span("gateway.chat") as span:
            try:
                self._check_budgets(plan["tenant"])
            except CostLimitException:
                rec = self._record(plan, t0, "blocked_budget")
                for k, v in self._span_attrs(plan, None, None, rec).items():
                    span.set_attribute(k, v)
                raise

            if plan["cacheable"]:
                hit = self.cache.lookup(plan["scope"], plan["text"])
                if hit:
                    rec = self._record(plan, t0, f"cache_{hit.kind}", "cache", hit.response, [], 0.0, hit.similarity)
                    for k, v in self._span_attrs(plan, None, None, rec).items():
                        span.set_attribute(k, v)
                    return self._cache_response(hit)

            attempts: list[dict] = []
            for route in plan["candidates"]:
                if not route.breaker.allow():
                    attempts.append({"route": route.name, "ok": False, "error": "circuit_open"})
                    continue
                try:
                    response = self._call_sync(route, messages, temperature, tools, response_model, kwargs)
                except Exception as exc:  # any provider/transport failure triggers failover
                    route.breaker.record_failure()
                    attempts.append({"route": route.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                attempts.append({"route": route.name, "ok": True, "error": None})
                rec = self._finish_ok(plan, t0, route, response, attempts)
                for k, v in self._span_attrs(plan, route, response, rec).items():
                    span.set_attribute(k, v)
                return response

            rec = self._record(plan, t0, "exhausted", None, None, attempts)
            for k, v in self._span_attrs(plan, None, None, rec).items():
                span.set_attribute(k, v)
            raise GatewayExhausted(
                f"All routes failed or were skipped: {attempts}", provider="gateway", details={"attempts": attempts}
            )

    # -- async chat ----------------------------------------------------------
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Any]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        t0 = time.perf_counter()
        plan = self._plan(messages, tools, response_model, kwargs)
        # Same span as the sync path: orchestrators such as ADK drive agents
        # through the async API, and the gateway must be visible in those traces.
        with _tracer.start_as_current_span("gateway.chat") as span:
            try:
                self._check_budgets(plan["tenant"])
            except CostLimitException:
                rec = self._record(plan, t0, "blocked_budget")
                self._annotate(span, plan, None, None, rec)
                raise
            if plan["cacheable"]:
                # The cache's embedding call is synchronous; keep it off the event loop.
                hit = await asyncio.to_thread(self.cache.lookup, plan["scope"], plan["text"])
                if hit:
                    rec = self._record(plan, t0, f"cache_{hit.kind}", "cache", hit.response, [], 0.0, hit.similarity)
                    self._annotate(span, plan, None, None, rec)
                    return self._cache_response(hit)
            attempts: list[dict] = []
            for route in plan["candidates"]:
                if not route.breaker.allow():
                    attempts.append({"route": route.name, "ok": False, "error": "circuit_open"})
                    continue
                try:
                    response = await self._call_async(route, messages, temperature, tools, response_model, kwargs)
                except Exception as exc:
                    route.breaker.record_failure()
                    attempts.append({"route": route.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                attempts.append({"route": route.name, "ok": True, "error": None})
                if plan["cacheable"] and response.parsed is None:
                    await asyncio.to_thread(self.cache.store, plan["scope"], plan["text"], response)
                rec = self._finish_ok(plan, t0, route, response, attempts, store_cache=False)
                self._annotate(span, plan, route, response, rec)
                return response
            rec = self._record(plan, t0, "exhausted", None, None, attempts)
            self._annotate(span, plan, None, None, rec)
            raise GatewayExhausted(
                f"All routes failed or were skipped: {attempts}", provider="gateway", details={"attempts": attempts}
            )

    # -- streaming: failover only before the first chunk ---------------------
    # Once a chunk has reached the caller the response is partly delivered; a
    # silent restart on another model would splice two different answers
    # together, so a mid-stream failure is surfaced, not retried.
    def stream(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> Iterator[StreamChunk]:
        t0 = time.perf_counter()
        plan = self._plan(messages, tools, None, kwargs)
        attempts: list[dict] = []
        for route in plan["candidates"]:
            if not route.breaker.allow():
                attempts.append({"route": route.name, "ok": False, "error": "circuit_open"})
                continue
            it = route.provider.stream(messages, temperature=temperature, tools=tools, **kwargs)
            try:
                first = next(it)
            except StopIteration:
                route.breaker.record_success()
                attempts.append({"route": route.name, "ok": True, "error": None})
                self._record(plan, t0, "ok", route.name, None, attempts)
                return
            except Exception as exc:
                route.breaker.record_failure()
                attempts.append({"route": route.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            attempts.append({"route": route.name, "ok": True, "error": None})
            try:
                yield first
                yield from it
            except Exception:
                route.breaker.record_failure()
                raise
            route.breaker.record_success()
            self._record(plan, t0, "ok", route.name, None, attempts)
            return
        self._record(plan, t0, "exhausted", None, None, attempts)
        raise GatewayExhausted(
            f"All routes failed or were skipped: {attempts}", provider="gateway", details={"attempts": attempts}
        )

    async def astream(self, messages, *, model=None, temperature=None, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        plan = self._plan(messages, tools, None, kwargs)
        attempts: list[dict] = []
        for route in plan["candidates"]:
            if not route.breaker.allow():
                attempts.append({"route": route.name, "ok": False, "error": "circuit_open"})
                continue
            it = route.provider.astream(messages, temperature=temperature, tools=tools, **kwargs).__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                route.breaker.record_success()
                return
            except Exception as exc:
                route.breaker.record_failure()
                attempts.append({"route": route.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            yield first
            try:
                async for chunk in it:
                    yield chunk
            except Exception:
                route.breaker.record_failure()
                raise
            route.breaker.record_success()
            return
        raise GatewayExhausted(
            f"All routes failed or were skipped: {attempts}", provider="gateway", details={"attempts": attempts}
        )
