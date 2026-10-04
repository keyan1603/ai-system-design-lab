"""Live run: system drift monitoring on the knowledge assistant.

    1. baseline   a reviewed run on the normal configuration (saved deliberately)
    2. repeat     the same configuration again: must read as stable (tolerance for model noise)
    3. outage     Gemini routes taken down by fault injection, so the local fallback answers
    4. no-filter  the access filter dropped, a configuration mistake that leaves quality, cost and latency untouched

Runs 3 and 4 use real models; run 3's outage is an injected fault, not a vendor outage.
"""

from dotenv import load_dotenv
from requisite.core.rate_limiter import RateLimiter
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from tabulate import tabulate

load_dotenv(".env")

from gateway.factory import build_gateway  # noqa: E402
from gateway.routing import LIGHT  # noqa: E402
from knowledge.assistant import KnowledgeAssistant  # noqa: E402
from knowledge.corpus import CASES, DOCS, USERS  # noqa: E402
from knowledge.index import SecureIndex  # noqa: E402
from monitoring.drift import check_drift, save_baseline, summarize  # noqa: E402
from run_knowledge_eval import judge, leaked  # noqa: E402

BASELINE = "baselines/knowledge_baseline.json"


def run_once(label, index, limiter, outage=False, enforce_acl=True):
    gw, faults = build_gateway(with_faults=True, use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=limiter,
                               audit_path=f"audit/phase4_drift_{label}.jsonl", breaker_reset_s=600)
    if outage:
        faults["light"].set("down")
        faults["heavy"].set("down")
    bot = KnowledgeAssistant(gw, index, enforce_acl=enforce_acl)
    results = []
    for c in CASES:
        ans = bot.ask(USERS[c.user], c.question)
        results.append((c, ans, judge(c, ans), leaked(c, ans)))
    # Keep every answer so a flagged case can be inspected afterwards instead of guessed at.
    import json
    with open(f"audit/phase4_drift_{label}_cases.json", "w", encoding="utf-8") as f:
        json.dump([{"id": c.id, "user": c.user, "expect": c.expect, "action": a.action, "retrieved": a.retrieved,
                    "citations": a.citations, "reasons": a.reasons, "correct": ok, "leaked": lk, "text": a.text}
                   for c, a, ok, lk in results], f, indent=1)
    metrics = summarize(results, gw.audit.summary(), degraded_routes={"local-llama"})
    print(f"  {label}: {metrics}")
    return metrics


def main():
    limiter = RateLimiter(requests_per_minute=15)
    index = SecureIndex(GeminiEmbeddingProvider(model="gemini-embedding-2"))
    index.ingest(DOCS)

    base = run_once("baseline", index, limiter)
    save_baseline(base, BASELINE)                    # deliberate: this run was reviewed by a human before saving
    runs = {"repeat (same config)": run_once("repeat", index, limiter),
            "outage (injected)": run_once("outage", index, limiter, outage=True),
            "access filter dropped": run_once("nofilter", index, limiter, enforce_acl=False)}

    metrics = list(base)
    rows = [[m, base[m]] + [r[m] for r in runs.values()] for m in metrics]
    print("\n== metrics")
    print(tabulate(rows, headers=["metric", "baseline"] + list(runs), tablefmt="github"))
    print("\n== drift verdicts against the saved baseline")
    for name, m in runs.items():
        rep = check_drift(m, BASELINE)
        print(f"\n{name}: {rep['status'].upper()}")
        for f in rep["findings"]:
            print(f"   [{f['severity']}] {f['metric']}: baseline {f['baseline']} -> {f['current']}  (rule: {f['rule']})")


if __name__ == "__main__":
    main()
