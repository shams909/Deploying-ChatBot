"""Backend health tracking, Node 01 inference semaphore, and Node 02 circuit breaker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from app.config import settings
from app.router import ModelTier


@dataclass
class CircuitState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    healthy: bool = True
    queue_depth: int = 0


class BackendManager:
    def __init__(self) -> None:
        self.node1_semaphore = asyncio.Semaphore(settings.node1_max_concurrent)
        self.node2_state = CircuitState()
        self.tier_health: dict[ModelTier, bool] = {
            ModelTier.FAST: True,
            ModelTier.REASONING: True,
            ModelTier.PRO: True,
        }
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    def url_for_tier(self, tier: ModelTier) -> str:
        if tier == ModelTier.FAST:
            return settings.tier1_url
        if tier == ModelTier.REASONING:
            return settings.tier2_url
        return settings.tier3_url

    def is_node2_available(self) -> bool:
        now = time.monotonic()
        if now < self.node2_state.open_until:
            return False
        return self.node2_state.healthy

    def record_node2_success(self) -> None:
        self.node2_state.consecutive_failures = 0
        self.node2_state.open_until = 0.0
        self.node2_state.healthy = True

    def record_node2_failure(self) -> None:
        self.node2_state.consecutive_failures += 1
        if self.node2_state.consecutive_failures >= settings.circuit_breaker_failures:
            self.node2_state.open_until = time.monotonic() + settings.circuit_breaker_cooldown_seconds
            self.node2_state.healthy = False

    async def poll_health_once(self) -> dict[str, bool]:
        for tier in ModelTier:
            url = self.url_for_tier(tier)
            try:
                resp = await self._client.get(f"{url}/health", timeout=2.5)
                ok = resp.status_code == 200
            except Exception:
                ok = False
            self.tier_health[tier] = ok
            if tier == ModelTier.PRO:
                if ok and time.monotonic() >= self.node2_state.open_until:
                    self.node2_state.healthy = True
                elif not ok:
                    self.node2_state.healthy = False
        return {t.value: v for t, v in self.tier_health.items()}

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client


backend_manager = BackendManager()
