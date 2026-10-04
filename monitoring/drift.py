"""System-level drift monitoring: quality, security and operations in one report.

Same discipline as the earlier posts' drift monitors (save a reviewed
baseline deliberately, check every later run against it), but the metrics span
the whole system rather than one model's accuracy, because the incidents seen
while building this lab were not accuracy problems:

* a quota exhaustion made the gateway quietly serve answers from the local
  fallback model (an *operations* signal: fallback rate, degraded share);
* dropping the access filter changed no latency and no cost but leaked every
  restricted document (a *security* signal: leak count, which tolerates zero);
* a model swap changes quality, cost and latency together.

Each rule has a severity. `critical` means stop and page; `warn` means look.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Rule:
    metric: str
    kind: str              # drop | rise | rise_pct | max
    threshold: float
    severity: str = "warn"  # warn | critical


DEFAULT_RULES = (
    Rule("correct_rate", "drop", 0.10, "critical"),
    Rule("authorized_correct_rate", "drop", 0.10, "critical"),
    Rule("leaks", "max", 0, "critical"),                  # security: tolerate zero, no matter the baseline
    Rule("fallback_rate", "rise", 0.10, "warn"),
    Rule("degraded_share", "max", 0.0, "critical"),       # any answer from the degraded tier needs a human to know
    Rule("cost_per_request_usd", "rise_pct", 0.50, "warn"),
    Rule("p50_latency_ms", "rise_pct", 1.00, "warn"),
)


def summarize(results: list, gateway_summary: dict, degraded_routes: Optional[set] = None) -> dict:
    """Reduce a run to the monitored metrics.

    `results` is a list of (case, answer, correct: bool, leaked: bool) tuples;
    `gateway_summary` is `AuditLog.summary()` for the same run.
    """
    degraded_routes = degraded_routes or set()
    answers = [r for r in results if r[0].expect == "answer"]
    reqs = max(gateway_summary.get("requests", 0), 1)
    by_route = gateway_summary.get("by_route", {})
    return {
        "correct_rate": round(sum(1 for r in results if r[2]) / len(results), 3),
        "authorized_correct_rate": round(sum(1 for r in answers if r[2]) / max(len(answers), 1), 3),
        "leaks": sum(1 for r in results if r[3]),
        "fallback_rate": round(gateway_summary.get("fallbacks", 0) / reqs, 3),
        "degraded_share": round(sum(n for route, n in by_route.items() if route in degraded_routes) / reqs, 3),
        "cost_per_request_usd": round(gateway_summary.get("total_cost_usd", 0.0) / reqs, 6),
        "p50_latency_ms": gateway_summary.get("p50_latency_ms_uncached") or 0.0,
    }


def save_baseline(metrics: dict, path) -> None:
    """Call once, deliberately, after a run a human has reviewed. Never from a scheduled
    job: a regression saved automatically becomes the new normal instead of an alert."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))


def check_drift(metrics: dict, path, rules=DEFAULT_RULES) -> dict:
    p = Path(path)
    if not p.exists():
        return {"status": "no_baseline", "findings": []}
    base = json.loads(p.read_text())
    findings = []
    for r in rules:
        cur, ref = metrics[r.metric], base.get(r.metric, 0)
        if r.kind == "drop":
            hit = ref - cur > r.threshold
        elif r.kind == "rise":
            hit = cur - ref > r.threshold
        elif r.kind == "rise_pct":
            hit = ref > 0 and (cur - ref) / ref > r.threshold
        else:  # max: absolute ceiling, independent of the baseline
            hit = cur > r.threshold
        if hit:
            findings.append({"metric": r.metric, "baseline": ref, "current": cur, "rule": f"{r.kind} {r.threshold}", "severity": r.severity})
    status = "critical" if any(f["severity"] == "critical" for f in findings) else "warn" if findings else "stable"
    return {"status": status, "findings": findings, "baseline": base, "current": metrics}
