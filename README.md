# ai-system-design-lab

A runnable reference system for designing enterprise AI: one AI gateway, two real scenarios (a support-ticket platform and an enterprise knowledge assistant), agent-to-agent calls with verified identity, drift monitoring, a failure-mode study and a cost study. Every claim in the companion blog post, *How to Design an AI System That Doesn't Leak, Lie or Fall Over*, comes from code and measurements in this repo.

It is built on [Requisite](https://github.com/requisite-ai/requisite-ai) (providers, agents, multi-agent workflows, RAG, MCP, rate and cost limits, tracing), Google's Agent Development Kit (ADK, including its A2A support), and the provider SDKs. Requisite supplies the building blocks; this repo adds the policy layer on top: gateway, guardrails, access control, identity and monitoring.

## Architecture

```
operator --login--> ticket platform ------A2A over HTTP, delegated identity------> knowledge assistant
                    (triage, resolver, responder)                                 (access-filtered retrieval, cited answers)
                           |                                                                |
                           +-------------------- every model call --------------------------+
                                                         |
                                                    AI gateway
                          tier routing, failover, circuit breakers, timeouts, rate limits,
                          semantic cache, per-tenant budgets, audit record per request
                                                         |
                          Gemini light | Gemini heavy | local Ollama model (degraded tier)
```

| Layer | What it does | Where |
|---|---|---|
| AI gateway | A Requisite provider that every agent and every orchestrator uses unchanged: tier routing, ordered failover, circuit breaker, per-route timeout and rate limiter, scoped semantic cache, per-tenant budgets, audit, an OpenTelemetry span per request, fault injection | `gateway/` |
| Guardrails | PII redaction (Luhn-checked cards), prompt-injection screening, output checks (including a company-domain allowlist); fail closed | `guardrails/` |
| Support-ticket platform | Guard, triage, resolver (MCP tools), responder, guard; one `Workflow` that runs on five interchangeable coordinators; outcomes are released, held or escalated, and it never raises | `tickets/` |
| Knowledge assistant | Access-filtered retrieval, cited answers, a check that every citation was retrieved for that user, hold-or-label policy on the degraded tier | `knowledge/` |
| Identity | Short-lived, audience-bound, signed tokens, and token exchange where delegated access can only shrink (lab-grade; production uses OIDC / OAuth 2.1) | `identity/` |
| Agent-to-agent | The ticket platform's resolver calls the knowledge assistant over ADK's A2A support, with bearer auth, a declared agent card security scheme, correlation id and W3C `traceparent` | `a2a_layer/` |
| Monitoring | Quality, security and operations metrics checked against a deliberately saved baseline | `monitoring/` |
| Decisions | Ten decision records, each tied to something measured here | `docs/decisions.md` |

### How Requisite and ADK are used

| Need | Used from |
|---|---|
| Model providers, embeddings, cost per token | Requisite `GeminiProvider`, `OllamaProvider`, `GeminiEmbeddingProvider` |
| Agents and multi-agent workflow | Requisite `Agent` and `Workflow`; coordinator switchable between Requisite native, Google ADK, OpenAI Agents SDK, Strands and Microsoft Agent Framework |
| Tools | Requisite `MCPServer` and `MCPClient` (persistent agent-owned session) |
| Retrieval | Requisite `HybridRetriever` with metadata filters (`{"groups": {"$in": [...]}}`), `InMemoryVectorStore` |
| Per-request identity | Requisite `RequestContext` (tenant, operator and correlation id reach the gateway and tools without wrappers) |
| Limits | Requisite `RateLimiter` and `CostLimiter` |
| Tracing | Requisite and ADK GenAI span attributes, one OpenTelemetry trace across services |
| Agent-to-agent | Google ADK `to_a2a`, `RemoteA2aAgent`, `AgentCardBuilder` |

## Quick start

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY
ollama pull llama3.2:1b         # the degraded-tier model
venv/Scripts/python -m pytest tests_offline -q      # 102 offline tests, no network, no API key
```

Prices in `gateway/factory.py` are the published Gemini API paid-tier prices read on 2026-10-03; they are configuration and will change.

## Run it

| Script | What it does |
|---|---|
| `run_gateway_demo.py` | Six gateway scenarios on real models: routing, cache, tenant isolation, injected outage and failover, budget, preflight |
| `run_ticket_eval.py` | The ticket platform on nine labelled tickets. `--backend adk,native,openai_agents,strands,agent_framework` compares coordinators; `--pin-light` fixes the model |
| `run_knowledge_eval.py` | 13 questions, 5 users: access control on, access control off (control), cache-scope experiment |
| `run_a2a_demo.py` | Two services over A2A: delegation, four attacks on the endpoint, confused deputy, one trace |
| `run_drift_demo.py` | Baseline, repeat, injected outage, dropped access filter, each checked against the baseline |
| `run_failure_study.py` | Seven injected failures against the real pipelines |
| `run_finops.py` | Measured cost per path, a live cache experiment, and projections labelled as assumptions |

Everything runs on the Gemini free tier plus a local 1B model, on synthetic data (customers, orders, a 14-document knowledge base, 9 tickets, 13 questions). Outages are switched on by a `FaultInjector` in front of real providers; they are injected faults, not vendor outages. Dollar figures are the same tokens priced at the published paid rates.

## What the runs showed

### Gateway

- **Routing.** Short tickets were served by `gemini-3.5-flash-lite`; a 744-token prompt routed to `gemini-3.5-flash` at about 58x the cost of a light call.
- **Cache.** An exact repeat cost 0 ms and 0 tokens. A paraphrase hit at similarity 0.987. A different request ("cancel both charges **and close the account**") also hit at 0.946 and got the refund answer, so the similarity threshold is a correctness setting, not just a cost setting. The same ticket from two tenants produced two model calls.
- **Failover.** With the light route down, requests escalated to the heavy model at roughly 40x the per-request cost; the breaker opened after 3 failures and later requests skipped the dead route with no timeout; with both Gemini routes down the local model answered; after the cool-down a probe closed the breaker.
- **Budget.** The reactive limiter lets one call cross the cap, then blocks every later call. Budgets are per tenant.
- **Degraded tier.** Failover keeps the system up, not the answers good. With the local 1B model answering, category accuracy fell from 7/7 to 4/7 and expected facts from 93 percent to 36 percent, yet the guardrails released every reply, because they check for leaks, not correctness. The platform therefore holds any ticket served by the degraded tier, and the knowledge assistant holds or labels such answers by policy.
- **Tool loops are sticky.** A tool loop stays on the route that started it, because switching provider mid-conversation breaks tool calling.
- **Preflight.** A fallback route that has never run is untested; `preflight(probe=True)` exercises every route before an outage needs it.

### Support-ticket platform

- **Guardrails.** Both injection tickets were escalated with zero model or tool calls; the PII ticket reached the model with the email, phone and card replaced by `[EMAIL]`, `[PHONE]`, `[CARD]` and was still resolved correctly.
- **Five coordinators, one result.** The same nine tickets on Google ADK, Requisite native, OpenAI Agents SDK, Strands and Microsoft Agent Framework: category 7/7 and severity 4/7 on every one, all 28 model calls through the gateway to the same model. The coordinators used 7,829 to 10,362 input tokens for the same work (a 32 percent spread, 14 percent in dollars). Latency is not comparable across them because the shared 15-requests-per-minute limiter dominates.
- **Persistent MCP session.** With the resolver owning its MCP session, its three tool calls took 3 to 4 ms each after one 974 ms initialize, instead of about 6.4 s each when every call started a new server process.

### Knowledge assistant

- **Access control on: 13/13 correct, 0 of 5 restricted documents leaked.** The five questions a user was not allowed to see returned "I can't find that in the documents you have access to"; the unanswerable question did too; the injection attempt was escalated with no model call.
- **Access control off (control): 8/13 correct, 5 of 5 leaked.** Same model, same questions, retrieval ignoring groups. The citation check passed all five leaks, because it verifies that an answer is grounded in what was retrieved, not that retrieval was authorized. Grounding checks and authorization are different controls.
- **The model never receives unauthorized text**, asserted offline by capturing the exact prompt for every question. A leak is measured with canary facts that appear in one restricted document and nowhere else.
- **Cache scope.** Keep the access fingerprint in the cache scope. With a permissive threshold and an organization-only scope, one user's answer is served to another; in that case the citation check held it, but that is a second line of defence, not the design.

### Identity and agent-to-agent

- **The callee enforces identity.** The knowledge agent derives groups from a verified token, never from message text. Four attacks (no token, forged signature claiming `exec`, expired token, token for another audience) were rejected with `401`, each audited, and the knowledge service made zero model calls.
- **Delegation shrinks access.** An executive calling the knowledge agent directly got the confidential acquisition answer; the same executive acting through the ticket agent got "I can't find that", because the exchanged token's groups are the user's groups intersected with the ticket agent's own ceiling.
- **The audit trail is the evidence.** With the support operator the resolver's A2A call cited the refund policy (audit: groups `employee,support`); with an engineer it got "not found" (audit: groups `employee`). Reply wording varies because the model summarises; the audit does not.
- **One trace across the HTTP hop.** The ticket workflow, the ADK agents, the resolver's tool call, the A2A client and server spans, the knowledge call and the gateway span share one trace id.
- **The agent card declares its auth.** ADK's automatic card has no security scheme, so a client cannot discover that a bearer token is required; the lab builds the card with `AgentCardBuilder(security_schemes=...)`.

### Drift, failures and cost

- **Drift: three failure shapes, one monitor.** Against a saved baseline (13/13 correct, 0 leaks, no fallbacks) an unchanged repeat read **stable**. An injected outage read **critical** on `correct_rate` (1.00 to 0.62) and `degraded_share` (0 to 1): the small model answered the authorized questions but ignored the refusal rule, filling denied and unknown questions with invented text carrying valid citations. Dropping the access filter read **critical** on `leaks` (0 to 5) while latency and cost barely moved. During the outage cost per request fell to zero: a cost dashboard looks healthier exactly when the system is degraded.
- **Seven injected failures, seven defined outcomes, no restricted fact in any answer.** Light route down: escalated to the heavy model. All cloud routes down: the local model answered and the answer was held (or labelled, by policy). Embedding API down: failed closed with no model call. MCP server will not start: ticket held. Knowledge agent unreachable: ticket held because a requested fact was missing, no policy invented. Budget exhausted: held, nothing crashed.
- **A shorter timeout can make the common case slower.** With an 8 s injected delay, no timeout answered in 9.7 s; with a 2 s timeout the request moved to the heavy model and answered in 17.0 s and 15.6 s in two runs, and fell through to the local model (held) in a third.
- **Measured unit costs** at published paid-tier prices: a ticket (4 model calls) $0.001104 on flash-lite and $0.004598 if every call ran on the heavy model; a knowledge answer $0.000167 and $0.000767.
- **Cache experiment** on a deliberately repetitive 22-question workload: 22 model calls and $0.004065 with the cache off, 6 calls and $0.001116 with it on (threshold 0.90, scoped by access), mean latency 3.17 s to 1.12 s, zero wrong answers in both. The 73 percent hit rate belongs to that workload, and the cache's own embedding calls are not counted.
- **Projections** (monthly cost at assumed volumes, break-even against an assumed hosting cost) are arithmetic on those measurements and are labelled as assumptions in the output.

## Limits

- Tokens are HMAC-signed with a shared secret and live in one process; production uses an OIDC / OAuth 2.1 identity provider and the security schemes the agent card declares.
- The operator's credential sits in a service-side store keyed by correlation id, so concurrent tickets for different operators are isolated.
- Fourteen short documents and 9 to 13 questions demonstrate mechanisms, not retrieval quality or throughput at scale. With 7 to 9 cases, small differences between runs are noise.
- The gateway is an in-process policy engine, not a network service. A shared multi-tenant gateway (virtual keys, team quotas, dashboards) belongs at the deployment tier; `docs/decisions.md` covers when to buy one.
- Synthetic data only.

## Layout

```
gateway/        routing, circuit breaker, semantic cache, audit, fault injection, factory
guardrails/     PII redaction, injection screen, output checks, fail-closed policy
tickets/        support-ticket platform, MCP tool server, policy tool
knowledge/      corpus, access-filtered index, assistant
identity/       tokens, login, token exchange
a2a_layer/      A2A server (auth middleware, agent card) and client
monitoring/     drift rules, baseline, checks
docs/           decision records
tests_offline/  102 tests, scripted models, no network
audit/          audit logs and per-case outcomes from the real runs
baselines/      the saved drift baseline
```
