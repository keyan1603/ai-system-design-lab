# Architecture decisions for the reference system

Short decision records, each tied to something measured in this lab. They are the design half of the AI System Design post; the code and the run logs are the evidence. Format: context, decision, evidence, consequences.

## D1. Authorization is enforced in code at retrieval, never by prompt
- **Context.** Two scenarios share one model and one corpus, but users may read different documents.
- **Decision.** Every chunk carries the groups allowed to read it. Every retrieval for a user passes a filter built from that user's verified groups, applied inside the store and to every retrieval path, before ranking. No groups means no results.
- **Evidence.** Same model, same 13 questions: filter on, 13/13 correct and 0 of 5 restricted documents leaked; filter off, 8/13 and 5/5 leaked, and the citation check passed every leak.
- **Consequences.** A grounding check is not an authorization control. Any retrieval product adopted later must be tested for a filter that runs before ranking and covers the keyword path (Requisite's `HybridRetriever` once did not, fixed in 0.41.0).

## D2. A gateway sits between every agent and every model
- **Context.** Agents call models in many places; routing, quota, budget and audit cannot be left to each caller.
- **Decision.** One gateway implements the provider interface, so every agent and every orchestrator backend uses it unchanged. It owns tier routing, ordered failover, circuit breakers, per-route timeouts and rate limits, a scoped cache, per-tenant budgets, and an audit record per request.
- **Evidence.** Most of the real incidents in this lab surfaced at the gateway: a quota 429 silently failing over, a mid-tool-loop route change breaking tool calls, a limiter that burned quota on failover, a slow route stalling requests.
- **Consequences.** For several teams, buy a gateway service (shared keys, quotas, dashboards). Keep these policies explicit either way and verify a product supports them: sticky tool loops, hold on degraded tier, tenant-scoped cache.

## D3. Failover keeps the system up, not the answers good
- **Context.** When cloud routes fail, a local model can still answer.
- **Decision.** Answers served by the degraded tier are held (tickets) or labelled (knowledge), by explicit per-use-case policy, and the audit marks them.
- **Evidence.** With the local 1B model answering, the guardrails released every reply in one run (they check for leaks, not correctness); it ignored the refusal rule and filled denied and unknown questions with irrelevant text carrying valid citations. A cost chart showed cost per request falling to zero.
- **Consequences.** Monitor `fallback_rate` and `degraded_share` as first-class metrics. Never treat a falling cost line as health.

## D4. A tool loop never changes model mid-flight
- **Context.** Conversation history holds provider-specific state.
- **Decision.** Once a conversation contains tool results, its remaining turns stay on the route that started it; if that route fails, the error goes to the caller, who restarts the task.
- **Evidence.** One live request failed with a Gemini 400 about a missing thought signature after a route change inside a tool loop.
- **Consequences.** Cross-provider failover is a conversation-boundary decision.

## D5. Timeouts are tuned against the fallback, not in isolation
- **Context.** A slow route stalls every request without a timeout.
- **Decision.** Each route has a timeout; the rate-limiter wait is outside it.
- **Evidence.** An 8 s injected delay: no timeout answered in 9.7 s. With a 2 s timeout the request moved to the heavy model and answered in 17.0 s and 15.6 s in two runs, and reached the local model (held) in the third. The timeout bounded the slow route but made the common case slower, because the next route was slower than waiting.
- **Consequences.** Set a timeout from the slow route's tail latency and compare it with the fallback's latency. A shorter timeout is not automatically safer.

## D6. Identity is verified by the callee, and delegation can only shrink access
- **Context.** Two services call each other on behalf of a person.
- **Decision.** The callee derives access from a verified, short-lived, audience-bound token, never from message text. A calling service exchanges the user's token for one whose groups are the user's groups intersected with the service's own ceiling.
- **Evidence.** Four attacks (no token, forged signature, expired, wrong audience) rejected with 401 and zero model calls. An executive acting through the ticket agent lost exec access that a direct call kept.
- **Consequences.** Production uses an OIDC / OAuth 2.1 provider and the security schemes the A2A agent card declares (ADK's automatic card declares none, so the card is built explicitly).

## D7. Every dependency failure ends in a defined outcome
- **Context.** A support pipeline that raises loses tickets.
- **Decision.** Each failure becomes a held or withheld outcome with a reason: no ungrounded answer when retrieval is down, no guessed policy when the policy agent is down, no crash when a budget is exhausted or a tool server will not start.
- **Evidence.** Seven injected failures against real pipelines: seven defined outcomes, no restricted fact in any answer (see `phase5_failure_study.log`).
- **Consequences.** A tool that cannot answer is a degraded capability: the ticket is held when a requested fact is missing, not released as complete.

## D8. One orchestration coordinator is not a one-way door
- **Context.** Several agent frameworks exist and will keep changing.
- **Decision.** Agents and tools are written once against Requisite; the coordinator (native, Google ADK, OpenAI Agents, Strands, Microsoft Agent Framework) is a one-line switch.
- **Evidence.** See the backend comparison in the README: the same nine tickets on every backend.
- **Consequences.** Choose a coordinator for the ecosystem around it (ADK for A2A, deployment and evaluation), not for sequential-pipeline capability. Keep model calls, tools, guards and identity out of the coordinator.

## D9. Self-hosting is a quality and operations decision before it is a cost decision
- **Evidence.** Hosted flash-lite cost about $0.0011 per ticket and $0.00017 per knowledge answer at published prices; the break-even against an assumed $500 per month of hosting is about 100,000 questions per day, and that excludes the quality the 1B local model gave up.
- **Decision.** Hosted by default; a local model only as the degraded tier, or where data residency or very high volume justify its operating cost.

## D10. Observability is OpenTelemetry end to end, plus system-level drift
- **Decision.** One trace across both services (traceparent over A2A) and a correlation id in every audit record; a drift monitor over quality, security and operations metrics with a deliberately saved baseline, where any leak is critical regardless of baseline.
- **Evidence.** An unchanged repeat read stable; an injected outage and a dropped access filter each read critical on different metrics.
