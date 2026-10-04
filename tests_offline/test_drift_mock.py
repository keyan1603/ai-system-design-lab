import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.drift import check_drift, save_baseline, summarize

GOOD = {"correct_rate": 1.0, "authorized_correct_rate": 1.0, "leaks": 0, "fallback_rate": 0.0,
        "degraded_share": 0.0, "cost_per_request_usd": 0.0002, "p50_latency_ms": 1000.0}


def case(expect):
    return NS(expect=expect)


def test_summarize_computes_every_metric():
    results = [(case("answer"), None, True, False), (case("answer"), None, False, False),
               (case("denied"), None, False, True), (case("denied"), None, True, False)]
    gw = {"requests": 10, "fallbacks": 3, "by_route": {"gemini-light": 6, "local-llama": 4},
          "total_cost_usd": 0.002, "p50_latency_ms_uncached": 1200.0}
    m = summarize(results, gw, {"local-llama"})
    assert m == {"correct_rate": 0.5, "authorized_correct_rate": 0.5, "leaks": 1, "fallback_rate": 0.3,
                 "degraded_share": 0.4, "cost_per_request_usd": 0.0002, "p50_latency_ms": 1200.0}


def test_no_baseline_is_reported_not_assumed_stable(tmp_path):
    assert check_drift(GOOD, tmp_path / "none.json")["status"] == "no_baseline"


def test_unchanged_run_is_stable(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(GOOD, p)
    assert check_drift(dict(GOOD, p50_latency_ms=1300.0), p)["status"] == "stable"       # +30% is within tolerance


def test_quality_drop_is_critical(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(GOOD, p)
    r = check_drift(dict(GOOD, correct_rate=0.6), p)
    assert r["status"] == "critical" and r["findings"][0]["metric"] == "correct_rate"


def test_any_leak_is_critical_even_if_baseline_had_one(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(dict(GOOD, leaks=1), p)
    r = check_drift(dict(GOOD, leaks=1), p)
    assert r["status"] == "critical" and [f["metric"] for f in r["findings"]] == ["leaks"]


def test_silent_fallback_to_degraded_tier_is_flagged_without_any_accuracy_change(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(GOOD, p)
    r = check_drift(dict(GOOD, fallback_rate=0.5, degraded_share=0.5), p)
    assert {f["metric"] for f in r["findings"]} == {"fallback_rate", "degraded_share"} and r["status"] == "critical"


def test_cost_and_latency_regressions_warn(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(GOOD, p)
    r = check_drift(dict(GOOD, cost_per_request_usd=0.0005, p50_latency_ms=2500.0), p)
    assert r["status"] == "warn" and len(r["findings"]) == 2
