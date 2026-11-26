"""Redis Exact + Semantic Prompt Cache and Per-User Sliding Token Quota."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from app.config import settings
from app.router import semantic_scorer

logger = logging.getLogger("labai.cache")

_EXACT_PREFIX = "labai:cache:exact:"
_SEM_LIST_KEY = "labai:cache:semantic_ring"
_QUOTA_PREFIX = "labai:quota:"
_ACTIVE_PREFIX = "labai:active:"


class CacheAndQuotaManager:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        try:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Connected to Redis at %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); caching and quotas disabled.", exc)
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    @staticmethod
    def _exact_key(model_tier: str, prompt: str) -> str:
        digest = hashlib.sha256(f"{model_tier}::{prompt.strip().lower()}".encode("utf-8")).hexdigest()
        return f"{_EXACT_PREFIX}{digest}"

    async def lookup_response(
        self,
        model_tier: str,
        prompt: str,
        is_single_turn: bool,
        temperature: float,
    ) -> tuple[str | None, str]:
        if self._redis is None or not is_single_turn or temperature > settings.cache_max_temperature or not prompt.strip():
            return None, "BYPASS"

        try:
            exact = await self._redis.get(self._exact_key(model_tier, prompt))
            if exact:
                return exact, "HIT_EXACT"

            query_vec = semantic_scorer.embed(prompt[:800])
            if query_vec is None:
                return None, "MISS"

            raw_items = await self._redis.lrange(_SEM_LIST_KEY, 0, settings.semantic_cache_max_entries - 1)
            best_sim = -1.0
            best_answer: str | None = None
            for raw in raw_items:
                entry = json.loads(raw)
                if entry.get("tier") != model_tier:
                    continue
                vec = np.array(entry["vec"], dtype=np.float32)
                sim = float(np.dot(query_vec, vec))
                if sim > best_sim:
                    best_sim = sim
                    best_answer = entry["response"]

            if best_answer is not None and best_sim >= settings.semantic_cache_threshold:
                return best_answer, f"HIT_SEMANTIC({best_sim:.2f})"
        except Exception as exc:
            logger.debug("Cache lookup error: %s", exc)

        return None, "MISS"

    async def store_response(
        self,
        model_tier: str,
        prompt: str,
        response_text: str,
        is_single_turn: bool,
        temperature: float,
    ) -> None:
        if (
            self._redis is None
            or not is_single_turn
            or temperature > settings.cache_max_temperature
            or not prompt.strip()
            or len(response_text) < 8
        ):
            return
        try:
            await self._redis.set(
                self._exact_key(model_tier, prompt),
                response_text,
                ex=settings.semantic_cache_ttl_seconds,
            )
            vec = semantic_scorer.embed(prompt[:800])
            if vec is not None:
                payload = json.dumps(
                    {
                        "tier": model_tier,
                        "vec": [round(float(x), 5) for x in vec.tolist()],
                        "response": response_text,
                    }
                )
                await self._redis.lpush(_SEM_LIST_KEY, payload)
                await self._redis.ltrim(_SEM_LIST_KEY, 0, settings.semantic_cache_max_entries - 1)
        except Exception as exc:
            logger.debug("Cache store error: %s", exc)

    async def check_and_acquire_user_slot(self, user_id: str) -> tuple[bool, str | None]:
        if self._redis is None:
            return True, None
        try:
            quota_key = f"{_QUOTA_PREFIX}{user_id}"
            used = int(await self._redis.get(quota_key) or 0)
            if used >= settings.user_token_quota:
                ttl = await self._redis.ttl(quota_key)
                return False, f"Token quota reached ({used}/{settings.user_token_quota}). Resets in {max(ttl, 1)}s."

            active_key = f"{_ACTIVE_PREFIX}{user_id}"
            acquired = await self._redis.set(active_key, str(time.time()), nx=True, ex=180)
            if not acquired:
                return False, "You already have an active generation in progress. Please wait for it to finish."
            return True, None
        except Exception as exc:
            logger.debug("Quota check error: %s", exc)
            return True, None

    async def release_user_slot_and_record_tokens(self, user_id: str, output_tokens: int) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(f"{_ACTIVE_PREFIX}{user_id}")
            if output_tokens > 0:
                quota_key = f"{_QUOTA_PREFIX}{user_id}"
                pipe = self._redis.pipeline()
                pipe.incrby(quota_key, output_tokens)
                pipe.expire(quota_key, settings.quota_window_seconds, nx=True)
                await pipe.execute()
        except Exception as exc:
            logger.debug("Quota release error: %s", exc)


cache_manager = CacheAndQuotaManager()
