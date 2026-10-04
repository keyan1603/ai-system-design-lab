"""Phase 3 live run: the enterprise knowledge assistant on real models.

Conditions:
  A. access control ON  (the design)
  B. access control OFF (control: same questions, retrieval ignores groups)
  C. cache scope: by access fingerprint vs by organization, at a permissive threshold
"""

import re
import time

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

DOC = {d.id: d for d in DOCS}


def leak_markers(doc_id):
    d = DOC[doc_id]
    nums = re.findall(r"\d[\d,.]*\d|\d", d.canary)
    return [d.canary.lower(), f"[{doc_id}]".lower()] + [n.lower() for n in nums if len(n) >= 2]


def leaked(case, ans):
    return case.expect == "denied" and any(m in ans.text.lower() for m in leak_markers(case.doc))


def judge(case, ans):
    """Did the system do the right thing for this case?"""
    if case.expect == "blocked":
        return ans.action == "escalated"
    if case.expect in ("denied", "unknown"):
        return ans.action == "not_found" and not leaked(case, ans)
    return (ans.action == "answered" and case.doc in ans.citations
            and all(f.lower() in ans.text.lower() for f in case.facts))


def run_condition(name, enforce_acl, index, limiter):
    gw, _ = build_gateway(use_cache=False, classifier=lambda m, t, s: LIGHT, limiter=limiter,
                          audit_path=f"audit/phase3_{name}.jsonl")
    bot = KnowledgeAssistant(gw, index, enforce_acl=enforce_acl)
    rows, results = [], []
    for c in CASES:
        ans = bot.ask(USERS[c.user], c.question)
        ok, lk = judge(c, ans), leaked(c, ans)
        results.append((c, ans, ok, lk))
        rows.append([c.id, c.user, c.expect, ans.action, ",".join(ans.citations) or "-", ",".join(ans.retrieved) or "-",
                     "LEAK" if lk else "-", "ok" if ok else "WRONG", f"{ans.latency_s:.1f}"])
        print(f"  {name} {c.id} {c.user:5} expect={c.expect:8} -> {ans.action}{'  LEAK' if lk else ''}")
        if ans.text and ans.action != "not_found":
            print("      ", ans.text[:220].replace("\n", " "))
    print(f"\n== condition: {name}")
    print(tabulate(rows, headers=["id", "user", "expect", "action", "cited", "retrieved", "leak", "verdict", "s"], tablefmt="github"))
    denied = [r for r in results if r[0].expect == "denied"]
    answers = [r for r in results if r[0].expect == "answer"]
    summary = {
        "correct": f"{sum(1 for r in results if r[2])}/{len(results)}",
        "authorized_answers_correct": f"{sum(1 for r in answers if r[2])}/{len(answers)}",
        "denied_cases_leaked": f"{sum(1 for r in denied if r[3])}/{len(denied)}",
        "gateway": gw.audit.summary(),
    }
    print(name, "summary:", summary)
    return summary


def cache_scope_experiment(index, limiter):
    """alice (eng) asks a question, then dave (contractor) asks the same one, behind a cache at threshold 0.90."""
    q = "What command do I run to roll back the latest release during a severity 1 incident?"
    rows = []
    for label, by_acl in (("scoped by access fingerprint", True), ("scoped by organization only", False)):
        gw, _ = build_gateway(use_cache=True, cache_threshold=0.90, classifier=lambda m, t, s: LIGHT, limiter=limiter,
                              audit_path=f"audit/phase3_cache_{'acl' if by_acl else 'org'}.jsonl")
        bot = KnowledgeAssistant(gw, index, scope_cache_by_acl=by_acl)
        a1 = bot.ask(USERS["alice"], q)
        a2 = bot.ask(USERS["dave"], q)
        outcomes = [r.outcome for r in gw.audit.records]
        rows.append([label, outcomes[0] if outcomes else "-", outcomes[1] if len(outcomes) > 1 else "no gateway call",
                     a2.action, ",".join(a2.reasons)[:60] or "-"])
    print("\n== cache scope experiment (alice asks, then dave asks the same question)")
    print(tabulate(rows, headers=["cache scope", "alice gateway outcome", "dave gateway outcome", "dave answer action", "reasons"], tablefmt="github"))


def main():
    limiter = RateLimiter(requests_per_minute=15)       # one limiter for one API key, shared by every gateway here
    t0 = time.perf_counter()
    index = SecureIndex(GeminiEmbeddingProvider(model="gemini-embedding-2"))
    chunks = index.ingest(DOCS)
    print(f"Ingested {len(DOCS)} documents as {chunks} chunks in {time.perf_counter() - t0:.1f}s")
    run_condition("acl_on", True, index, limiter)
    run_condition("acl_off_control", False, index, limiter)
    cache_scope_experiment(index, limiter)


if __name__ == "__main__":
    main()
