"""Phase 2 live run: the ticket platform on real models, scored against labelled tickets.

    python -u run_ticket_eval.py                 # both backends, all tickets
    python -u run_ticket_eval.py --backend adk --only T01
"""

import argparse
import statistics

from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv(".env")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from gateway.factory import build_gateway  # noqa: E402
from tickets.data import TICKETS  # noqa: E402
from tickets.platform import TicketPlatform  # noqa: E402

SHOW = False
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)


def mention_rate(t, o):
    if not t.must_mention:
        return None
    hits = sum(1 for m in t.must_mention if m.lower() in o.reply.lower())
    return hits / len(t.must_mention)


def trace_tree(trace_id):
    spans = [s for s in exporter.get_finished_spans() if s.context.trace_id == trace_id]
    by_parent = {}
    for s in spans:
        by_parent.setdefault(s.parent.span_id if s.parent else None, []).append(s)
    lines = []

    def walk(parent_id, depth):
        for s in sorted(by_parent.get(parent_id, []), key=lambda x: x.start_time):
            a = s.attributes or {}
            extra = ""
            if "gen_ai.request.model" in a:
                extra = f"  model={a.get('gen_ai.response.model', a['gen_ai.request.model'])} in={a.get('gen_ai.usage.input_tokens')} out={a.get('gen_ai.usage.output_tokens')}"
            if s.name == "gateway.chat":
                extra += f"  route={a.get('gateway.route')} tier={a.get('gateway.tier')} outcome={a.get('gateway.outcome')} cost=${a.get('gateway.cost_usd', 0):.6f}"
            lines.append(f"{'  ' * depth}{s.name} ({(s.end_time - s.start_time) / 1e6:.0f} ms){extra}")
            walk(s.context.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)


def run_backend(backend, tickets, pin_light=False):
    from gateway.routing import LIGHT
    gw, _ = build_gateway(use_cache=False, tenant_budget_usd=0.05, audit_path=f"audit/phase2_{backend}{'_pinned' if pin_light else ''}.jsonl",
                          classifier=(lambda m, t, s: LIGHT) if pin_light else None)
    platform = TicketPlatform(gw, backend=backend)
    rows, outcomes = [], []
    for t in tickets:
        o = platform.handle(t)
        outcomes.append((t, o))
        cat = o.triage.category if o.triage else "-"
        sev = o.triage.severity if o.triage else "-"
        rows.append([t.id, t.tenant, t.kind, o.action, f"{cat}/{t.category}", f"{sev}/{t.severity}",
                     ",".join(sorted(set(o.tools_used))) or "-", o.model_calls, f"{o.latency_s:.1f}",
                     "-" if mention_rate(t, o) is None else f"{mention_rate(t, o):.0%}", ";".join(o.reasons)[:60]])
        print(f"  {backend} {t.id} -> {o.action}")
        if SHOW:
            print("     reply:", o.reply[:400].replace(chr(10), " "))
    import json as _json
    with open(f"audit/phase2_outcomes_{backend}{'_pinned' if pin_light else ''}.json", "w", encoding="utf-8") as f:
        _json.dump([{"id": t.id, "action": o.action, "reasons": o.reasons, "reply": o.reply, "tools": o.tools_used,
                     "latency_s": o.latency_s, "calls": o.model_calls,
                     "triage": o.triage.model_dump() if o.triage else None} for t, o in outcomes], f, indent=1)
    print(f"\n== backend={backend}")
    print(tabulate(rows, headers=["id", "tenant", "kind", "action", "cat got/want", "sev got/want", "tools", "calls", "s", "facts", "reasons"], tablefmt="github"))
    return gw, outcomes


def score(outcomes):
    normal = [(t, o) for t, o in outcomes if t.kind != "injection"]
    cat = sum(1 for t, o in normal if o.triage and o.triage.category == t.category)
    sev = sum(1 for t, o in normal if o.triage and o.triage.severity == t.severity)
    facts = [mention_rate(t, o) for t, o in normal if mention_rate(t, o) is not None]
    lat = [o.latency_s for t, o in normal]
    return {
        "category_acc": f"{cat}/{len(normal)}", "severity_acc": f"{sev}/{len(normal)}",
        "facts_recall": f"{statistics.mean(facts):.0%}", "released": sum(1 for _, o in outcomes if o.action == "released"),
        "escalated": sum(1 for _, o in outcomes if o.action == "escalated"), "held": sum(1 for _, o in outcomes if o.action == "held"),
        "median_latency_s": round(statistics.median(lat), 1), "mean_calls": round(statistics.mean(o.model_calls for _, o in normal), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="both", choices=["adk", "native", "both"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pin-light", action="store_true", help="route every request to the light tier so backends are compared on the same model")
    args = ap.parse_args()
    global SHOW
    SHOW = args.show
    tickets = [t for t in TICKETS if not args.only or t.id == args.only]
    backends = ["adk", "native"] if args.backend == "both" else [args.backend]
    results = {}
    for b in backends:
        gw, outcomes = run_backend(b, tickets, args.pin_light)
        results[b] = (gw, outcomes)
        print("\n", b, "score:", score(outcomes))
        print(b, "gateway summary:", gw.audit.summary())
    first = next(iter(results))
    for sp in exporter.get_finished_spans():
        if sp.name == "ticket.handle" and sp.attributes.get("ticket.action") == "released":
            print(f"\n== Trace tree for one released ticket ({sp.attributes['ticket.id']}, backend={sp.attributes['ticket.backend']})")
            print(trace_tree(sp.context.trace_id))
            break


if __name__ == "__main__":
    main()
