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

---

## Phase 2: the support-ticket platform (`tickets/`, `guardrails/`)

```
ticket -> input guard -> triage -> resolver (MCP tools) -> responder -> output guard -> released | held | escalated
```

| Piece | Built with |
|---|---|
| Three agents (triage, resolver, responder) | Requisite `Agent`, provider = `TenantProvider(gateway, tenant)` |
| Orchestration, switchable | Requisite `Workflow` on **Google ADK** (`use_adk()`) or the native engine (`use_native()`) |
| Tools | Requisite `MCPServer` (3 read-only tools, stdio) consumed by Requisite `MCPClient` |
| Guardrails | `guardrails/`: PII redaction (Luhn-checked cards), injection screening, output validation; fail-closed policy |
| Tracing | OpenTelemetry: ADK spans, Requisite `agent`/`ai` spans (GenAI attributes from Requisite 0.39.0) and the gateway span nest in one trace |
| Per-tenant budgets, sticky tool loops, degraded-tier hold | `gateway/provider.py`, `tickets/platform.py` |

```bash
venv/Scripts/python -m pytest tests_offline -q                        # 46 offline tests
venv/Scripts/python -u run_ticket_eval.py --pin-light --show          # both backends, same model, 9 labelled tickets
```

### What the real runs showed (Gemini free tier, `gemini-3.5-flash-lite`, 9 synthetic tickets)

- **Both orchestrators reach the same results.** ADK and native: category 7/7, severity 4/7 (the same three severity calls differ from my labels on both), expected facts present 93% (ADK) and 100% (native) on 7 tickets. With n=7 that difference is noise, not a finding. Median latency was dominated by per-call MCP process start-up, not by the orchestrator.
- **Guardrails:** both injection tickets were escalated with **zero model or tool calls**; the PII ticket reached the model with the email, phone and card replaced by `[EMAIL]`, `[PHONE]`, `[CARD]` and was still resolved correctly.
- **Output guard needs an allowlist.** A reply quoting the company's own `no-reply@` address was held as "PII" until the guard learned the company domain.
- **Silent failover hides a quota problem.** In the tiered run the heavy model (`gemini-3.5-flash`) returned `429` (free-tier limit: 20 requests per day) and the gateway quietly served those calls from the light model. Availability held; the only trace was the audit log.
- **Failover keeps the system up, not the answers good.** A second backend run built its own rate limiter, together exceeded the real quota, tripped both Gemini breakers, and 22 of 28 calls were answered by the local 1B model: category accuracy fell to 4/7 and expected facts to 36%, yet **the guardrails released all seven replies** (they check for leaks, not correctness). The platform now holds any ticket served by the degraded tier. Share one `RateLimiter` per API key.
- **Mid-tool-loop failover breaks tool calling.** One request failed with `400: Function call is missing a thought_signature` after a route change inside a tool loop; the gateway now pins a tool loop to the route that started it (verified offline with scripted fakes; the live failure was seen once).
- **MCP over stdio reconnects on every tool call**, and `initialize` dominated each call (about 5.9 s of 6.4 s in one trace, 2.0 s of 2.3 s in another).
- **ADK's per-agent token counters read empty** (`in=None out=None`) because Requisite's ADK shim does not pass usage back to ADK.

Cost figures are computed from published paid-tier prices; the runs themselves used the free tier.


---

## Phase 3: the enterprise knowledge assistant (`knowledge/`)

```
question -> input guard -> ACL-filtered retrieval -> answer agent -> citation check -> output guard -> answered | not_found | held | escalated
```

| Piece | Built with |
|---|---|
| Embeddings, vector store, chunking | Requisite `GeminiEmbeddingProvider` (`gemini-embedding-2`), `InMemoryVectorStore`, `Retriever.add_texts` |
| Access control | `knowledge/index.py`: one boolean flag per allowed group on every chunk, one **pre-filtered** store search per group the user belongs to, merged. No groups means no results |
| Answer agent | Requisite `Agent` on the shared gateway, provider pinned to the user's *access fingerprint* so the cache never crosses access levels |
| Checks | Every answer must cite sources, every citation must be a chunk retrieved for that user, plus the Phase 2 input and output guards |

Authorization is enforced in code at retrieval, never by asking the model to behave. A leak is measured, not judged: each restricted document carries a canary fact that appears nowhere else.

```bash
venv/Scripts/python -m pytest tests_offline -q          # 68 offline tests
venv/Scripts/python -u run_knowledge_eval.py            # real run: ACL on, ACL off (control), cache-scope experiment
```

### What the real run showed (12 synthetic documents, 5 users, 13 questions, `gemini-3.5-flash-lite`)

- **Access control ON: 13/13 correct, 0/5 restricted documents leaked.** All six authorized questions were answered with the right citation and exact facts; the five questions a user was not allowed to see returned "I can't find that in the documents you have access to"; the unanswerable question did too; the injection attempt was escalated with no model call.
- **Access control OFF (control): 8/13 correct, 5/5 restricted documents leaked.** Same model, same questions, retrieval ignoring groups: the model answered every one of them, with a valid citation. The **citation check passed all five leaks**, because it verifies that an answer is grounded in what was retrieved, not that retrieval was authorized. Grounding checks and authorization are different controls.
- **The model never receives unauthorized text** (asserted offline by capturing the exact prompt for every question).
- **Cache scope:** with an access-fingerprint scope and with an organization-only scope, the second user was not served the first user's answer at similarity threshold 0.90, because the cache keys on the whole prompt, sources included, and the two users' sources differed. That is an accident of protection, not a guarantee. Offline, a permissive threshold (0.5) under an organization-only scope *does* serve one user's answer to another; there the citation check held the answer because it cited a document the second user was never shown. Keep the access fingerprint in the scope.
- **Limits of this run:** 12 documents of one chunk each, so it exercises authorization and citation mechanics, not retrieval quality at scale.

### Requisite 0.40.0 adopted (agent-owned persistent MCP sessions, ADK token usage)

On the same 9 tickets and model as Phase 2, the resolver's three MCP tool calls dropped from **about 6.4 s each (initialize dominated) to 3 to 4 ms each**, with one 974 ms `initialize` per agent, and ADK's per-agent token counts are now populated (`in=1174 out=182` instead of `None`). End-to-end ticket latency still shows 37 to 46 s outliers: each is a single model call that includes waiting on the shared 15-requests-per-minute rate limiter, which this run did not instrument separately, so no orchestrator or MCP conclusion is drawn from them.
