"""TenantProvider: pins every call through the gateway to one tenant.

Frameworks forward their own arguments to a provider, not arbitrary gateway
kwargs, and orchestrator backends differ in what they forward. Wrapping the
gateway per tenant makes tenant identity part of the provider object, so
routing, cache scope, budgets and audit see the right tenant no matter which
orchestrator drives the agents.
"""

from __future__ import annotations

from typing import Any

from requisite.providers.base import BaseProvider

from gateway.provider import GatewayProvider


class TenantProvider(BaseProvider):
    def __init__(self, gateway: GatewayProvider, tenant: str) -> None:
        super().__init__(api_key="tenant", model=gateway.model)
        self.gateway, self.tenant = gateway, tenant

    name = property(lambda self: self.gateway.name)
    model = property(lambda self: self.gateway.model)

    def validate_config(self) -> None:
        self.gateway.validate_config()

    def chat(self, messages, **kw: Any):
        kw.setdefault("tenant", self.tenant)
        return self.gateway.chat(messages, **kw)

    async def achat(self, messages, **kw: Any):
        kw.setdefault("tenant", self.tenant)
        return await self.gateway.achat(messages, **kw)

    def stream(self, messages, **kw: Any):
        kw.setdefault("tenant", self.tenant)
        return self.gateway.stream(messages, **kw)

    def astream(self, messages, **kw: Any):
        kw.setdefault("tenant", self.tenant)
        return self.gateway.astream(messages, **kw)
