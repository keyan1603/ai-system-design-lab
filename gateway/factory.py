"""Builds the lab's standard gateway from environment configuration.

Every provider here is a real Requisite provider (GeminiProvider,
OllamaProvider, GeminiEmbeddingProvider); the gateway adds policy around them.
"""

from __future__ import annotations

import os
from typing import Optional

from requisite.core.cost_limiter import CostFn, CostLimiter, cost_per_token
from requisite.core.rate_limiter import RateLimiter
from requisite.providers.gemini_provider import GeminiProvider
from requisite.providers.ollama_provider import OllamaProvider
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider

from gateway.audit import AuditLog
from gateway.cache import SemanticCache
from gateway.faults import FaultInjector
from gateway.provider import GatewayProvider
from gateway.resilience import CircuitBreaker
from gateway.routing import DEGRADED, HEAVY, LIGHT, Route

# Published Gemini API paid-tier standard prices, USD per 1M tokens, read from
# https://ai.google.dev/gemini-api/docs/pricing on 2026-10-03. Prices change;
# these are configuration, not a promise. (input, output)
PRICES_PER_1M = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
}


def gemini_cost_fn(model: str) -> CostFn:
    """Cost from token usage.

    Output tokens are billed *including* thinking tokens, and a provider's
    `completion_tokens` may exclude them, so output is taken as
    `max(completion_tokens, total_tokens - prompt_tokens)`.
    """
    in_rate, out_rate = PRICES_PER_1M[model]

    def _cost(usage, _model: str) -> float:
        out_tokens = max(usage.completion_tokens, usage.total_tokens - usage.prompt_tokens)
        return usage.prompt_tokens * in_rate / 1e6 + out_tokens * out_rate / 1e6

    return _cost


def model_cost_fn(usage, model: str) -> float:
    """Price a response by the model that actually answered (limiters need this:
    a budget shared across tiers must not price a heavy call at light rates).
    Models with no published price here, such as the local one, cost 0."""
    name = model.removeprefix("models/")
    return gemini_cost_fn(name)(usage, name) if name in PRICES_PER_1M else 0.0


def build_gateway(
    *,
    with_faults: bool = False,
    cache_threshold: float = 0.90,
    budget_usd: Optional[float] = None,
    tenant_budget_usd: Optional[float] = None,
    use_cache: bool = True,
    classifier=None,
    limiter: Optional[RateLimiter] = None,
    timeouts: Optional[dict] = None,
    audit_path: Optional[str] = None,
    breaker_reset_s: float = 30.0,
):
    timeouts = timeouts or {}
    key = os.environ["GEMINI_API_KEY"]
    light_model = os.environ.get("LIGHT_MODEL", "gemini-3.5-flash-lite")
    heavy_model = os.environ.get("HEAVY_MODEL", "gemini-3.5-flash")
    rpm = int(os.environ.get("REQUESTS_PER_MINUTE", "15"))

    # Both Gemini routes draw on one API key, so they share ONE limiter instance.
    # Pass one `limiter` to several gateways that share an API key: two
    # limiter instances do not know about each other and together can exceed the quota.
    limiter = limiter or RateLimiter(requests_per_minute=rpm)
    light = GeminiProvider(api_key=key, model=light_model)
    heavy = GeminiProvider(api_key=key, model=heavy_model)
    local = OllamaProvider(model="llama3.2:1b", timeout=180.0)
    faults: dict[str, FaultInjector] = {}
    if with_faults:
        light = faults.setdefault("light", FaultInjector(light, "gemini-light"))
        heavy = faults.setdefault("heavy", FaultInjector(heavy, "gemini-heavy"))

    free = cost_per_token(prompt_rate_per_1k=0.0, completion_rate_per_1k=0.0)
    routes = [
        Route("gemini-light", light, LIGHT, gemini_cost_fn(light_model), CircuitBreaker(reset_timeout_s=breaker_reset_s),
              timeout_s=timeouts.get("light", 30.0), limiter=limiter),
        Route("gemini-heavy", heavy, HEAVY, gemini_cost_fn(heavy_model), CircuitBreaker(reset_timeout_s=breaker_reset_s),
              timeout_s=timeouts.get("heavy", 45.0), limiter=limiter),
        Route("local-llama", local, DEGRADED, free, CircuitBreaker(reset_timeout_s=breaker_reset_s),
              timeout_s=timeouts.get("local", 120.0)),
    ]
    embedder = GeminiEmbeddingProvider(api_key=key, model=os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2"))
    cache = SemanticCache(embedder.embed_one, threshold=cache_threshold)
    spend_cap = CostLimiter(budget_usd=budget_usd, cost_fn=model_cost_fn) if budget_usd else None
    gw = GatewayProvider(
        routes,
        cache=cache if use_cache else None,
        **({"classifier": classifier} if classifier else {}),
        cost_limiter=spend_cap,
        tenant_budgets=(lambda _t: CostLimiter(budget_usd=tenant_budget_usd, cost_fn=model_cost_fn)) if tenant_budget_usd else None,
        audit=AuditLog(path=audit_path),
    )
    return gw, faults
