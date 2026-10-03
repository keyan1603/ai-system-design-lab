"""Routes and the policy that picks a tier for a request.

A *route* is one concrete provider+model the gateway may call. Routes are
grouped into tiers ("light" = cheap and fast, "heavy" = more capable,
"degraded" = last-resort local model). The policy only decides the *tier*;
which route inside a tier goes first, and what happens when it fails, is the
gateway's resilience logic, not the policy's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from requisite.core.cost_limiter import CostFn
from requisite.core.interfaces import Message
from requisite.providers.base import BaseProvider

from gateway.resilience import CircuitBreaker

LIGHT, HEAVY, DEGRADED = "light", "heavy", "degraded"


@dataclass
class Route:
    name: str
    provider: BaseProvider
    tier: str
    cost_fn: CostFn
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


Classifier = Callable[[Sequence[Message], bool, bool], str]


def default_classifier(messages: Sequence[Message], has_tools: bool, has_schema: bool) -> str:
    """Cheap, explainable default: long prompts and tool use go heavy, everything else light.

    This is deliberately a heuristic you can read in ten seconds. Real routers
    range from this up to a trained classifier; the value of the gateway is
    that swapping one for the other is a one-line change here, not an edit to
    every caller.
    """
    chars = sum(len(m.content or "") for m in messages)
    if has_tools or chars > 1500:
        return HEAVY
    return LIGHT


def fallback_order(routes: Sequence[Route], tier: str) -> list[Route]:
    """Routes to try, in order, for a request classified as ``tier``.

    Same tier first (in declaration order), then escalate to a heavier tier
    (a stronger model rescuing a failed cheap one), then degrade to lighter
    tiers, with the degraded tier always last. A request is never silently
    dropped to a weaker model before every equal-or-stronger option is spent.
    """
    same = [r for r in routes if r.tier == tier]
    if tier == LIGHT:
        order_rest = [HEAVY, DEGRADED]
    elif tier == HEAVY:
        order_rest = [LIGHT, DEGRADED]
    else:
        order_rest = [LIGHT, HEAVY]
    rest = [r for t in order_rest for r in routes if r.tier == t]
    return same + rest
