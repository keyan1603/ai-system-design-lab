# ai-system-design-lab

Companion repo for the AI System Design post in the AI/agents blog series. It is built **phase by phase**; this is the state after **Phase 1: the gateway layer**.

The goal is one reference system serving two scenarios (a support-ticket platform and an enterprise knowledge assistant) on top of [Requisite](https://github.com/keyan1603/requisite-ai), Google ADK and the provider SDKs, with the architecture decisions made explicit and measured.

## Phase 1: `gateway/`

An in-process AI gateway that **is** a Requisite provider (`GatewayProvider(BaseProvider)`), so it plugs into `AI`, `Agent` and every orchestrator backend unchanged.

| Module | What it does |
|---|---|
| `provider.py` | `GatewayProvider`: tier routing, ordered failover, budget check, cache, audit, OpenTelemetry span per request, sync/async/stream, `preflight()` |
| `routing.py` | `Route`, tier classifier, fallback order (same tier, then escalate, then degrade, local model last) |
| `resilience.py` | `CircuitBreaker` (closed / open / half-open, injectable clock) |
| `cache.py` | `SemanticCache`: exact hash lookup, then embedding similarity; entries are scoped by tenant and system prompt |
| `audit.py` | One `AuditRecord` per request; prompts are hashed, not stored, unless you opt in |
| `ratelimit.py` | `RateLimitedProvider`: a Requisite `RateLimiter` placed next to the one real API it protects |
| `faults.py` | `FaultInjector`: switch an outage on or off in front of a real provider |
| `factory.py` | Standard build: Gemini light + heavy (Requisite `GeminiProvider`), local Ollama (Requisite `OllamaProvider`), Requisite `GeminiEmbeddingProvider` for the cache, Requisite `CostLimiter` for the budget |

Reused from Requisite as-is: providers, embeddings, `RateLimiter`, `CostLimiter`, `cost_per_token`, `get_tracer`, `Message`/`ChatResponse`/`Usage`.

## Run it

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY
ollama pull llama3.2:1b         # the degraded-tier route
venv/Scripts/python -m pytest tests_offline -q     # 23 offline tests, no network
venv/Scripts/python -u run_gateway_demo.py         # real run, 6 scenarios
```

## What the real run showed

- **Routing:** short tickets served by `gemini-3.5-flash-lite`; a 744-token prompt routed to `gemini-3.5-flash` at about 58x the cost of a light call.
- **Cache:** an exact repeat cost 0 ms and 0 tokens. A paraphrase hit at similarity 0.987. A *different* request ("cancel both charges **and close the account**") also hit at 0.946 and was answered with the refund answer. A similarity threshold is a correctness setting, not just a cost setting.
- **Tenant isolation:** the same ticket from two tenants produced two model calls; only a repeat from the same tenant hit the cache.
- **Failover (injected outage):** light down escalated to heavy at roughly 40x the per-request cost; the breaker opened after 3 failures and later requests skipped the dead route with no timeout; with both Gemini routes down the local model answered; after the cool-down a probe closed the breaker.
- **Budget:** the reactive limiter lets one call cross the cap, then blocks every later call.
- **Preflight:** a fallback route that has never run is untested; `preflight(probe=True)` exercises every route before an outage needs it.

Outage numbers come from a `FaultInjector` in front of real providers (an injected fault, not a real vendor outage).

## Not in Phase 1

Per-tenant budgets, guardrails, MCP tools, RAG, A2A and the ADK orchestration arrive in later phases.

Prices in `factory.py` are the published Gemini API paid-tier prices read on 2026-10-03; they are configuration and will change.
