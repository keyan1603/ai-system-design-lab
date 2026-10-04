"""Phase 5 FinOps study: what does each path cost, what does a cache save, where does hosting break even?

Measured (real runs): tokens, dollars at published paid-tier prices, and latency per ticket and per question,
read from the gateway audit logs of the earlier runs; plus a live cache experiment on a repeated workload.
Projected (arithmetic on the measured numbers, assumptions stated): monthly cost and self-hosting break-even.
The runs themselves used the free tier; dollars are what the same tokens cost at the published paid prices.
"""

import json
import statistics
from pathlib import Path

from dotenv import load_dotenv
from requisite.core.rate_limiter import RateLimiter
from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from tabulate import tabulate

load_dotenv(".env")

from gateway.factory import PRICES_PER_1M, build_gateway  # noqa: E402
from gateway.routing import LIGHT  # noqa: E402
from knowledge.assistant import KnowledgeAssistant  # noqa: E402
from knowledge.corpus import CASES, DOCS, USERS  # noqa: E402
from knowledge.index import SecureIndex  # noqa: E402
from run_knowledge_eval import judge  # noqa: E402

AUDIT = Path("audit")


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reprice(recs, model):
    in_rate, out_rate = PRICES_PER_1M[model]
    return sum(r["input_tokens"] * in_rate / 1e6 + r["output_tokens"] * out_rate / 1e6 for r in recs)


def measured_paths():
    rows = []
    t = [r for r in load(AUDIT / "phase2_adk_pinned_persistent.jsonl") if r["outcome"] == "ok"]
    tickets = len(t) // 4                                  # 4 model calls per ticket in this pipeline
    k = [r for r in load(AUDIT / "phase4_drift_baseline.jsonl") if r["outcome"] == "ok"]
    for name, recs, units, unit in (("ticket pipeline (triage, resolver with tools, responder)", t, tickets, "ticket"),
                                    ("knowledge answer (1 model call)", k, len(k), "question")):
        tin, tout = sum(r["input_tokens"] for r in recs) / units, sum(r["output_tokens"] for r in recs) / units
        for model in ("gemini-3.5-flash-lite", "gemini-3.5-flash"):
            rows.append([name, model, round(tin), round(tout), f"${reprice(recs, model) / units:.6f}", f"{statistics.median(r['latency_ms'] for r in recs):.0f} ms/call"])
    print("\n== measured: tokens and cost per unit of work (published paid-tier prices)")
    print(tabulate(rows, headers=["path", "if every call ran on", "input tok / unit", "output tok / unit", "cost / unit", "median latency"], tablefmt="github"))
    return {"ticket": reprice(t, "gemini-3.5-flash-lite") / tickets, "ticket_heavy": reprice(t, "gemini-3.5-flash") / tickets,
            "question": reprice(k, "gemini-3.5-flash-lite") / len(k)}


# A repeated workload: six distinct questions asked three times each, plus paraphrases of two of them.
PARAPHRASES = [("alice", "Which command rolls back the newest release in a sev 1 incident?", "K01"),
               ("bob", "What's our revenue forecast for the third quarter?", "K03"),
               ("alice", "What is the command to roll back the latest release for a severity 1 incident?", "K01"),
               ("bob", "Tell me the Q3 revenue forecast.", "K03")]


def cache_experiment(index, limiter):
    by_id = {c.id: c for c in CASES}
    base = [by_id[i] for i in ("K01", "K03", "K05", "K07", "K08", "K11")]
    workload = [(c.user, c.question, c.id) for _ in range(3) for c in base] + PARAPHRASES
    out = {}
    for label, use_cache in (("cache off", False), ("cache on (threshold 0.90, scoped by access)", True)):
        gw, _ = build_gateway(use_cache=use_cache, cache_threshold=0.90, classifier=lambda m, t, s: LIGHT, limiter=limiter,
                              audit_path=f"audit/phase5_cache_{'on' if use_cache else 'off'}.jsonl")
        bot = KnowledgeAssistant(gw, index)
        wrong, lat = 0, []
        for user, q, cid in workload:
            ans = bot.ask(USERS[user], q)
            lat.append(ans.latency_s)
            case = by_id[cid]
            wrong += 0 if judge(case, ans) else 1
        s = gw.audit.summary()
        calls = s["by_outcome"].get("ok", 0)
        out[label] = {"asks": len(workload), "model_calls": calls, "cache_hits": len(workload) - calls,
                      "gateway_cost": s["total_cost_usd"], "mean_latency_s": statistics.mean(lat), "wrong_answers": wrong}
        print("  ", label, out[label])
    rows = [[k, v["asks"], v["model_calls"], v["cache_hits"], f"${v['gateway_cost']:.6f}", f"{v['mean_latency_s']:.2f}", v["wrong_answers"]] for k, v in out.items()]
    print("\n== live: cache economics on a repeated workload (22 asks)")
    print(tabulate(rows, headers=["condition", "asks", "model calls", "cache hits", "model cost", "mean latency (s)", "wrong answers"], tablefmt="github"))
    return out


def projections(unit_costs, cache):
    print("\n== projected (arithmetic on measured numbers; volumes and hosting costs are ASSUMPTIONS, not measurements)")
    rows = []
    for tickets_per_day, questions_per_day in ((1_000, 5_000), (10_000, 50_000), (100_000, 500_000)):
        light = 30 * (tickets_per_day * unit_costs["ticket"] + questions_per_day * unit_costs["question"])
        heavy = 30 * (tickets_per_day * unit_costs["ticket_heavy"] + questions_per_day * unit_costs["question"] * 5)
        hit = cache["cache on (threshold 0.90, scoped by access)"]["cache_hits"] / cache["cache off"]["asks"]
        cached = 30 * (tickets_per_day * unit_costs["ticket"] + questions_per_day * unit_costs["question"] * (1 - hit))
        rows.append([f"{tickets_per_day:,} tickets + {questions_per_day:,} questions per day", f"${light:,.0f}", f"${cached:,.0f}", f"${heavy:,.0f}"])
    print(tabulate(rows, headers=["daily volume", "all calls on flash-lite", f"flash-lite + {hit:.0%} cache hit on questions", "all calls on flash (heavy)"], tablefmt="github"))
    print("   (the heavy column uses the measured token mix at flash prices; question cost x5 approximates flash vs flash-lite)")

    print("\nSelf-hosting break-even (hosted flash-lite cost per month vs an assumed fixed monthly cost to host a model)")
    rows = []
    per_q = unit_costs["question"]
    for host in (100, 500, 2_000, 10_000):
        be = host / (30 * per_q)
        rows.append([f"${host:,}/month", f"{be:,.0f} questions/day", "quality and operations NOT included: the local 1B model failed the refusal rule in Phase 4"])
    print(tabulate(rows, headers=["assumed hosting cost", "break-even volume vs hosted flash-lite", "caveat"], tablefmt="github"))


def main():
    unit = measured_paths()
    limiter = RateLimiter(requests_per_minute=15)
    index = SecureIndex(GeminiEmbeddingProvider(model="gemini-embedding-2"))
    index.ingest(DOCS)
    cache = cache_experiment(index, limiter)
    projections(unit, cache)


if __name__ == "__main__":
    main()
