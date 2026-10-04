"""Offline tests for the gateway: no network, no API key, no sleeping.

Fake providers script exactly the failures we want (outage, flapping, slow
recovery) so every resilience rule is checked deterministically before any
real model is involved.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from requisite.core.cost_limiter import CostLimiter, cost_per_token
from requisite.core.exceptions import CostLimitException, ProviderException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk, Usage
from requisite.providers.base import BaseProvider

from gateway.audit import AuditLog
from gateway.cache import SemanticCache
from gateway.provider import GatewayExhausted, GatewayProvider
from gateway.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from gateway.routing import DEGRADED, HEAVY, LIGHT, Route, fallback_order

COST = cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=2.0)  # illustrative rates


class FakeProvider(BaseProvider):
    """Scripted provider: `fail_next` failures, then success."""

    def __init__(self, name="fake", model="fake-model", fail_next=0, text="ok"):
        super().__init__(api_key="x", model=model)
        self._name, self.fail_next, self.text, self.calls = name, fail_next, text, 0

    name = property(lambda self: self._name)

    def chat(self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kw):
        self.calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ProviderException("boom", provider=self._name)
        return ChatResponse(
            content=self.text, model=self._model, provider=self._name,
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

    async def achat(self, messages, **kw):
        return self.chat(messages, **kw)

    def stream(self, messages, **kw):
        self.calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ProviderException("boom", provider=self._name)
        yield StreamChunk(delta="a", is_final=False)
        yield StreamChunk(delta="b", is_final=True)

    def astream(self, messages, **kw):  # pragma: no cover - not exercised
        raise NotImplementedError


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def user(text):
    return [Message(role=Role.USER, content=text)]


def route(name, tier, provider=None, clock=None, **bk):
    return Route(name, provider or FakeProvider(name=name), tier, COST,
                 CircuitBreaker(clock=clock or Clock(), **bk))


def toy_embed(text):
    """Deterministic bag-of-letters embedding: similar strings land close together."""
    v = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1
    return v


# ---- routing -------------------------------------------------------------------------
def test_fallback_order_escalates_then_degrades():
    rs = [route("l", LIGHT), route("h", HEAVY), route("d", DEGRADED)]
    assert [r.name for r in fallback_order(rs, LIGHT)] == ["l", "h", "d"]
    assert [r.name for r in fallback_order(rs, HEAVY)] == ["h", "l", "d"]
    assert [r.name for r in fallback_order(rs, DEGRADED)] == ["d", "l", "h"]


def test_default_classifier_sends_tools_and_long_prompts_heavy():
    light, heavy = route("l", LIGHT), route("h", HEAVY)
    gw = GatewayProvider([light, heavy])
    gw.chat(user("short"))
    assert light.provider.calls == 1 and heavy.provider.calls == 0
    gw.chat(user("x" * 2000))
    assert heavy.provider.calls == 1
    gw.chat(user("short with tools"), tools=[object()])
    assert heavy.provider.calls == 2


def test_explicit_tier_overrides_classifier():
    light, heavy = route("l", LIGHT), route("h", HEAVY)
    GatewayProvider([light, heavy]).chat(user("short"), tier=HEAVY)
    assert heavy.provider.calls == 1 and light.provider.calls == 0


# ---- failover + breaker -------------------------------------------------------------
def test_failover_to_next_route_and_audit_records_attempts():
    bad = route("l", LIGHT, FakeProvider("l", fail_next=99))
    good = route("h", HEAVY)
    gw = GatewayProvider([bad, good])
    resp = gw.chat(user("hi"))
    assert resp.provider == "h"
    rec = gw.audit.records[-1]
    assert rec.outcome == "ok" and rec.route == "h"
    assert [a["ok"] for a in rec.attempts] == [False, True]


def test_all_routes_failing_raises_and_is_audited():
    gw = GatewayProvider([route("l", LIGHT, FakeProvider("l", fail_next=99)),
                          route("d", DEGRADED, FakeProvider("d", fail_next=99))])
    with pytest.raises(GatewayExhausted):
        gw.chat(user("hi"))
    assert gw.audit.records[-1].outcome == "exhausted"


def test_breaker_opens_after_threshold_and_skips_route():
    clock = Clock()
    flaky = FakeProvider("l", fail_next=99)
    gw = GatewayProvider([route("l", LIGHT, flaky, clock, failure_threshold=2), route("h", HEAVY)])
    gw.chat(user("a")); gw.chat(user("b"))          # 2 failures trip the breaker
    assert gw.routes[0].breaker.state == OPEN
    calls_before = flaky.calls
    gw.chat(user("c"))                               # skipped without calling the provider
    assert flaky.calls == calls_before
    assert gw.audit.records[-1].attempts[0]["error"] == "circuit_open"


def test_breaker_half_open_probe_then_recovers():
    clock = Clock()
    flaky = FakeProvider("l", fail_next=2)
    gw = GatewayProvider([route("l", LIGHT, flaky, clock, failure_threshold=2, reset_timeout_s=30), route("h", HEAVY)])
    gw.chat(user("a")); gw.chat(user("b"))
    assert gw.routes[0].breaker.state == OPEN
    clock.t = 31
    assert gw.routes[0].breaker.state == HALF_OPEN
    resp = gw.chat(user("c"))                        # the probe succeeds (fail_next exhausted)
    assert resp.provider == "l" and gw.routes[0].breaker.state == CLOSED


def test_failed_probe_reopens_breaker():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, reset_timeout_s=10, clock=clock)
    b.record_failure()
    clock.t = 11
    assert b.allow() and not b.allow()               # one probe only
    b.record_failure()
    assert b.state == OPEN


# ---- cache --------------------------------------------------------------------------
def test_exact_cache_hit_skips_provider_and_costs_nothing():
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed))
    gw.chat(user("reset my password"))
    r2 = gw.chat(user("reset my password"))
    assert r2.provider == "gateway-cache" and light.provider.calls == 1
    rec = gw.audit.records[-1]
    assert rec.outcome == "cache_exact" and rec.cost_usd == 0.0


def test_semantic_hit_above_threshold_and_miss_below():
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed, threshold=0.99))
    gw.chat(user("reset my password please"))
    gw.chat(user("reset my password please!"))       # near-identical letters -> semantic hit
    assert gw.audit.records[-1].outcome == "cache_semantic"
    gw.chat(user("zzzz qqqq xxxx jjjj"))              # unrelated -> miss
    assert gw.audit.records[-1].outcome == "ok" and light.provider.calls == 2


def test_cache_never_crosses_tenants():
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed))
    gw.chat(user("salary bands for engineering"), tenant="hr")
    gw.chat(user("salary bands for engineering"), tenant="support")
    assert light.provider.calls == 2                 # second tenant must NOT see hr's cached answer


def test_cache_scope_includes_system_prompt():
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed))
    q = Message(role=Role.USER, content="same question")
    gw.chat([Message(role=Role.SYSTEM, content="be terse"), q])
    gw.chat([Message(role=Role.SYSTEM, content="be verbose"), q])
    assert light.provider.calls == 2


def test_tool_calls_and_no_cache_flag_bypass_cache():
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed))
    gw.chat(user("hello"), tools=[object()]); gw.chat(user("hello"), tools=[object()])
    gw.chat(user("hello2"), no_cache=True); gw.chat(user("hello2"), no_cache=True)
    assert light.provider.calls == 4


def test_cache_ttl_and_eviction():
    clock = Clock()
    cache = SemanticCache(toy_embed, ttl_s=10, max_entries=2, clock=clock)
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=cache)
    gw.chat(user("one")); clock.t = 11
    gw.chat(user("one"))                              # expired -> real call
    assert light.provider.calls == 2
    gw.chat(user("two")); gw.chat(user("three"))
    assert cache.evictions >= 1


# ---- budget -------------------------------------------------------------------------
def test_budget_blocks_after_spend_and_is_audited():
    # one call = 100 prompt tokens * $1/1k + 50 completion * $2/1k = $0.20
    limiter = CostLimiter(budget_usd=0.25, cost_fn=COST)
    gw = GatewayProvider([route("l", LIGHT)], cost_limiter=limiter)
    gw.chat(user("a")); gw.chat(user("b"))            # reactive limiter: second call still allowed ($0.20 < $0.25)
    with pytest.raises(CostLimitException):
        gw.chat(user("c"))
    assert gw.audit.records[-1].outcome == "blocked_budget"
    assert gw.audit.summary()["total_cost_usd"] == pytest.approx(0.40)


# ---- audit privacy ------------------------------------------------------------------
def test_audit_does_not_store_prompt_by_default(tmp_path):
    log = AuditLog(path=str(tmp_path / "a.jsonl"))
    GatewayProvider([route("l", LIGHT)], audit=log).chat(user("my SSN is 123-45-6789"))
    assert "123-45-6789" not in (tmp_path / "a.jsonl").read_text()
    assert log.records[0].prompt_sha256 and log.records[0].prompt is None


# ---- streaming + async --------------------------------------------------------------
def test_stream_fails_over_before_first_chunk():
    gw = GatewayProvider([route("l", LIGHT, FakeProvider("l", fail_next=99)), route("h", HEAVY)])
    assert "".join(c.delta for c in gw.stream(user("hi"))) == "ab"
    assert gw.audit.records[-1].route == "h"


def test_async_chat_failover_and_cache():
    light = route("l", LIGHT)
    gw = GatewayProvider([route("bad", LIGHT, FakeProvider("bad", fail_next=99)), light],
                         cache=SemanticCache(toy_embed))

    async def go():
        a = await gw.achat(user("async q"))
        b = await gw.achat(user("async q"))
        return a, b

    a, b = asyncio.run(go())
    assert a.provider == "l" and b.provider == "gateway-cache"


# ---- telemetry ----------------------------------------------------------------------
def test_span_carries_genai_and_gateway_attributes():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tp = trace.get_tracer_provider()
    if not hasattr(tp, "add_span_processor"):       # the global provider can be installed only once per process
        tp = TracerProvider()
        trace.set_tracer_provider(tp)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    GatewayProvider([route("l", LIGHT)]).chat(user("hi"), tenant="t1")
    spans = [s for s in exporter.get_finished_spans() if s.name == "gateway.chat"]
    assert spans, "gateway.chat span was not exported"
    attrs = spans[-1].attributes
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.usage.input_tokens"] == 100 and attrs["gen_ai.usage.output_tokens"] == 50
    assert attrs["gateway.tenant"] == "t1" and attrs["gateway.route"] == "l"


# ---- preflight, cache accounting, rate-limit decorator -------------------------------
def test_preflight_reports_bad_route_before_an_outage_needs_it():
    class Broken(FakeProvider):
        def validate_config(self):
            raise ProviderException("ollama package not installed")
    gw = GatewayProvider([route("l", LIGHT), route("d", DEGRADED, Broken("d"))])
    rep = gw.preflight()
    assert rep["l"] == "ok" and "not installed" in rep["d"]


def test_preflight_probe_makes_a_real_call_per_route():
    gw = GatewayProvider([route("l", LIGHT), route("h", HEAVY)])
    gw.preflight(probe=True)
    assert all(r.provider.calls == 1 for r in gw.routes)


def test_cache_hit_records_savings_not_spend():
    gw = GatewayProvider([route("l", LIGHT)], cache=SemanticCache(toy_embed))
    gw.chat(user("q")); gw.chat(user("q"))
    hit = gw.audit.records[-1]
    assert hit.input_tokens == 0 and hit.output_tokens == 0 and hit.saved_tokens == 150
    assert gw.audit.summary()["tokens_saved_by_cache"] == 150


def test_rate_limited_provider_acquires_before_each_call():
    from gateway.ratelimit import RateLimitedProvider

    class Counting:
        n = 0
        def acquire(self): Counting.n += 1
        async def aacquire(self): Counting.n += 1

    inner = FakeProvider("l")
    p = RateLimitedProvider(inner, Counting())
    p.chat(user("x")); p.chat(user("y"))
    assert Counting.n == 2 and inner.calls == 2


# ---- per-tenant budgets, tenant pinning ----------------------------------------------
def test_tenant_budget_blocks_only_that_tenant():
    gw = GatewayProvider([route("l", LIGHT)],
                         tenant_budgets=lambda t: CostLimiter(budget_usd=0.15, cost_fn=COST))
    gw.chat(user("a"), tenant="acme")                 # $0.20 spent, over acme's $0.15 cap
    with pytest.raises(CostLimitException):
        gw.chat(user("b"), tenant="acme")
    assert gw.chat(user("c"), tenant="globex").content == "ok"   # globex unaffected
    assert gw.audit.records[-2].outcome == "blocked_budget"


def test_global_budget_still_applies_above_tenant_budgets():
    gw = GatewayProvider([route("l", LIGHT)], cost_limiter=CostLimiter(budget_usd=0.15, cost_fn=COST),
                         tenant_budgets=lambda t: CostLimiter(budget_usd=100, cost_fn=COST))
    gw.chat(user("a"), tenant="acme")
    with pytest.raises(CostLimitException):
        gw.chat(user("b"), tenant="globex")           # fleet ceiling reached by someone else's spend


def test_tenant_provider_pins_tenant_for_audit_and_cache_scope():
    from gateway.tenant import TenantProvider
    light = route("l", LIGHT)
    gw = GatewayProvider([light], cache=SemanticCache(toy_embed))
    a, b = TenantProvider(gw, "acme"), TenantProvider(gw, "globex")
    a.chat(user("same text")); b.chat(user("same text"))
    assert light.provider.calls == 2
    assert [r.tenant for r in gw.audit.records] == ["acme", "globex"]


def test_async_path_emits_gateway_span_too():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tp = trace.get_tracer_provider()
    if not hasattr(tp, "add_span_processor"):       # a real provider was not installed by an earlier test
        tp = TracerProvider(); trace.set_tracer_provider(tp)
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    gw = GatewayProvider([route("l", LIGHT)])
    asyncio.run(gw.achat(user("hi"), tenant="t9"))
    spans = [s for s in exporter.get_finished_spans() if s.name == "gateway.chat"]
    assert spans and spans[-1].attributes["gateway.tenant"] == "t9" and spans[-1].attributes["gateway.route"] == "l"


# ---- sticky tool loops, degraded-tier detection --------------------------------------
def tool_msg():
    return Message(role=Role.TOOL, content="result", name="t", tool_call_id="1")


def test_tool_loop_stays_on_the_route_that_started_it_and_never_fails_over():
    flaky = FakeProvider("l", fail_next=0)
    gw = GatewayProvider([route("l", LIGHT, flaky), route("h", HEAVY)])
    first = [Message(role=Role.USER, content="do the task")]
    gw.chat(first, tools=[object()], tier=LIGHT)                     # turn 1 served by l, loop pinned to l
    flaky.fail_next = 5
    loop = first + [Message(role=Role.ASSISTANT, content=""), tool_msg()]
    with pytest.raises(GatewayExhausted):                             # l fails mid-loop: must NOT hop to h
        gw.chat(loop, tools=[object()], tier=LIGHT)
    assert gw.routes[1].provider.calls == 0
    assert [a["route"] for a in gw.audit.records[-1].attempts] == ["l"]


def test_first_turn_of_a_tool_conversation_can_still_fail_over():
    gw = GatewayProvider([route("l", LIGHT, FakeProvider("l", fail_next=9)), route("h", HEAVY)])
    resp = gw.chat(user("start the task"), tools=[object()])
    assert resp.provider == "h"


def test_degraded_routes_used_reports_only_degraded_tier():
    from gateway.routing import degraded_routes_used
    gw = GatewayProvider([route("l", LIGHT, FakeProvider("l", fail_next=9)), route("d", DEGRADED)])
    gw.chat(user("a"))
    assert degraded_routes_used(gw.audit.records, gw.routes) == ["d"]
    ok = GatewayProvider([route("l", LIGHT), route("d", DEGRADED)])
    ok.chat(user("a"))
    assert degraded_routes_used(ok.audit.records, ok.routes) == []
