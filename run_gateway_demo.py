"""Phase 1 live demo: the gateway in front of real Gemini and local Ollama models.

Six scenarios, each printing real audit records. Outage scenarios use the
FaultInjector (an *injected* failure in front of a real provider), and are
labelled as such in the output.
"""

import sys
import time

from dotenv import load_dotenv
from requisite.core.interfaces import Message, Role
from tabulate import tabulate

load_dotenv(".env")

from gateway.factory import build_gateway  # noqa: E402

SYSTEM = ("You triage customer support tickets. Reply in one line: "
          "'<category: billing|technical|account> | <one-sentence summary>'.")


def ask(gw, text, **kw):
    msgs = [Message(role=Role.SYSTEM, content=SYSTEM), Message(role=Role.USER, content=text)]
    try:
        r = gw.chat(msgs, temperature=0.0, **kw)
        return r
    except Exception as e:  # noqa: BLE001 - demo prints every outcome
        print(f"   ! {type(e).__name__}: {str(e)[:140]}")
        return None


def show(gw, since, title):
    rows = []
    for r in gw.audit.records[since:]:
        rows.append([r.tenant, r.tier, r.outcome, r.route, "->".join(("ok" if a["ok"] else a["error"].split(":")[0]) for a in r.attempts) or "-",
                     f"{r.latency_ms:.0f}", r.input_tokens, r.output_tokens, f"${r.cost_usd:.6f}",
                     f"{r.cache_similarity:.3f}" if r.cache_similarity is not None else "-"])
    print(f"\n== {title}")
    print(tabulate(rows, headers=["tenant", "tier", "outcome", "route", "attempts", "ms", "in", "out", "cost", "sim"], tablefmt="github"))


def main():
    gw, faults = build_gateway(with_faults=True, breaker_reset_s=3.0, audit_path="audit/phase1_demo.jsonl")

    print("Preflight (real 1-token probe per route):", gw.preflight(probe=True))

    t1 = "I was charged twice for my October subscription and need one of the charges refunded."
    t2 = "The mobile app crashes every time I open the invoices screen on Android 15."
    t3 = "I can't log in, the password reset email never arrives."

    # 1. normal routing: short tickets go to the light tier
    n = len(gw.audit.records)
    for t in (t1, t2, t3):
        r = ask(gw, t)
        if r:
            print("  ", r.content[:110])
    show(gw, n, "1. Normal routing (3 distinct tickets, expect gemini-light)")

    # 2. cache: exact repeat, paraphrase, and a near-miss that is a DIFFERENT question
    n = len(gw.audit.records)
    ask(gw, t1)
    ask(gw, "My October subscription was billed two times, please refund one of the charges.")
    ask(gw, "I was charged twice for my October subscription and need both charges cancelled and the account closed.")
    show(gw, n, "2. Cache: exact repeat / paraphrase / similar-but-different question")
    print("   cache:", gw.cache.stats())

    # 3. tenant isolation: same text, different tenant must be a miss
    n = len(gw.audit.records)
    ask(gw, t2, tenant="acme")
    ask(gw, t2, tenant="globex")
    ask(gw, t2, tenant="acme")
    show(gw, n, "3. Tenant isolation (acme, globex, acme again)")

    # 4. long prompt goes heavy
    n = len(gw.audit.records)
    long_ticket = "Ticket history follows. " + " ".join(f"[{i}] customer reports intermittent sync failure after update." for i in range(60))
    ask(gw, long_ticket)
    show(gw, n, "4. Long prompt (expect gemini-heavy)")

    # 5. injected outages: light down -> escalate; both down -> local degraded; recover
    n = len(gw.audit.records)
    faults["light"].set("down")
    for i in range(4):
        ask(gw, f"Outage probe {i}: dashboard shows wrong totals after migration number {i}.", no_cache=True)
    faults["heavy"].set("down")
    for i in range(2):
        ask(gw, f"Both cloud routes down probe {i}: export button does nothing on report {i}.", no_cache=True)
    faults["light"].set("up"); faults["heavy"].set("up")
    time.sleep(3.2)  # let breakers reach half-open
    for i in range(2):
        ask(gw, f"Recovery probe {i}: email notifications delayed for ticket {i}.", no_cache=True)
    show(gw, n, "5. INJECTED outages: light down -> heavy; both down -> local; recovery")
    print("   breaker states:", {r.name: r.breaker.state for r in gw.routes})

    # 6. budget
    gw2, _ = build_gateway(budget_usd=0.00005, audit_path="audit/phase1_budget.jsonl")
    for i in range(4):
        ask(gw2, f"Budget probe {i}: invoice {i} total looks wrong.", no_cache=True)
    show(gw2, 0, "6. Budget cap ($0.00005): reactive limiter lets spend cross once, then blocks")

    print("\nOverall summary (scenarios 1-5):")
    print(gw.audit.summary())


if __name__ == "__main__":
    sys.exit(main())
