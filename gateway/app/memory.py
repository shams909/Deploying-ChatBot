"""Cross-chat semantic long-term user memory stored in Redis with explicit user controls."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

import numpy as np

from app.backends import backend_manager
from app.cache import cache_manager
from app.config import settings
from app.router import ModelTier, semantic_scorer

logger = logging.getLogger("labai.memory")

_MEM_KEY_PREFIX = "labai:memory:"
_FORGET_PHRASE = re.compile(r"^(?:please\s+)?forget\s+(?:everything|all(?:\s+memories)?)\s*(?:you\s+remember\s+about\s+me)?[.!]?$", re.IGNORECASE)


def _mem_key(user_id: str) -> str:
    return f"{_MEM_KEY_PREFIX}{user_id}"


async def list_memories(user_id: str) -> list[dict[str, Any]]:
    redis = cache_manager.redis
    if redis is None:
        return []
    try:
        raw_items = await redis.lrange(_mem_key(user_id), 0, settings.memory_max_items_per_user - 1)
        return [json.loads(r) for r in raw_items]
    except Exception as exc:
        logger.debug("Memory list error: %s", exc)
        return []


async def delete_memories(user_id: str, chat_id: str | None = None) -> int:
    redis = cache_manager.redis
    if redis is None:
        return 0
    key = _mem_key(user_id)
    if chat_id is None:
        items = await list_memories(user_id)
        await redis.delete(key)
        return len(items)

    items = await list_memories(user_id)
    kept = [it for it in items if it.get("chat_id") != chat_id]
    removed = len(items) - len(kept)
    if removed > 0:
        await redis.delete(key)
        if kept:
            await redis.rpush(key, *(json.dumps(it) for it in kept))
    return removed


async def handle_memory_command(user_id: str, user_text: str) -> str | None:
    """Handle /memory commands or explicit forget requests directly without calling the LLM."""
    text = user_text.strip()
    lower = text.lower()

    if lower in ("/memory", "/memory list"):
        items = await list_memories(user_id)
        if not items:
            return "No cross-chat memories are saved for your account yet."
        lines = [f"{idx}. {it['fact']}" for idx, it in enumerate(items, start=1)]
        return "**Saved Cross-Chat Memories:**\n" + "\n".join(lines) + "\n\n*Use `/memory delete <number>` or `/memory clear` anytime.*"

    if lower in ("/memory clear", "/memory reset") or _FORGET_PHRASE.match(text):
        count = await delete_memories(user_id)
        return f"Cleared {count} saved cross-chat {'memory' if count == 1 else 'memories'}."

    if lower.startswith("/memory delete "):
        arg = text.split(None, 2)[-1].strip()
        if not arg.isdigit():
            return "Usage: `/memory delete <number>` (run `/memory` first to see numbers)."
        idx = int(arg) - 1
        items = await list_memories(user_id)
        if idx < 0 or idx >= len(items):
            return f"Memory #{arg} not found."
        removed = items.pop(idx)
        redis = cache_manager.redis
        if redis is not None:
            await redis.delete(_mem_key(user_id))
            if items:
                await redis.rpush(_mem_key(user_id), *(json.dumps(it) for it in items))
        return f"Deleted memory: `{removed['fact']}`"

    return None


async def recall_memories(user_id: str, query_text: str) -> str | None:
    if not settings.enable_long_term_memory or not query_text.strip():
        return None
    items = await list_memories(user_id)
    if not items:
        return None

    query_vec = semantic_scorer.embed(query_text[:600])
    selected: list[str] = []
    if query_vec is not None:
        scored: list[tuple[float, str]] = []
        for it in items:
            vec_list = it.get("vec")
            if not vec_list:
                continue
            sim = float(np.dot(query_vec, np.array(vec_list, dtype=np.float32)))
            if sim >= 0.32:
                scored.append((sim, it["fact"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [fact for _, fact in scored[: settings.memory_recall_top_k]]
    else:
        selected = [it["fact"] for it in items[: settings.memory_recall_top_k]]

    if not selected:
        return None
    return "Relevant long-term context remembered from past chats (do not mention unless helpful):\n" + "\n".join(
        f"- {fact}" for fact in selected
    )


async def extract_and_save_memories(user_id: str, chat_id: str | None, user_text: str) -> None:
    """Background task using Tier 1 (Fast 4B) to extract durable user facts after a substantive turn."""
    redis = cache_manager.redis
    if (
        not settings.enable_long_term_memory
        or redis is None
        or len(user_text.strip()) < 25
        or user_text.strip().startswith("/")
    ):
        return

    extraction_prompt = (
        "Extract at most 2 concise, durable facts about the user (e.g., role, skills, ongoing project, "
        "preferred programming language, or preferences) from the message below. "
        "Ignore greetings, one-off questions, or generic trivia. "
        'Return ONLY a JSON array of short strings (e.g. ["User works with Python on Linux"]), or [] if none.\n\n'
        f"User message: {user_text[:600]}\n/no_think"
    )
    payload = {
        "model": ModelTier.FAST.value,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 120,
    }
    try:
        async with backend_manager.node1_semaphore:
            resp = await backend_manager.client.post(f"{settings.tier1_url}/v1/chat/completions", json=payload, timeout=15.0)
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"].get("content", "").strip()
    except Exception as exc:
        logger.debug("Background memory extraction skipped: %s", exc)
        return

    match = re.search(r"\[.*\]", raw_content, re.DOTALL)
    if not match:
        return
    try:
        facts = json.loads(match.group(0))
    except json.JSONDecodeError:
        return
    if not isinstance(facts, list):
        return

    existing = await list_memories(user_id)
    existing_vecs = [np.array(it["vec"], dtype=np.float32) for it in existing if it.get("vec")]
    existing_lower = {it["fact"].lower() for it in existing}

    for fact in facts[:2]:
        if not isinstance(fact, str):
            continue
        clean = fact.strip()
        if len(clean) < 8 or clean.lower() in existing_lower:
            continue
        vec = semantic_scorer.embed(clean)
        if vec is not None and any(float(np.dot(vec, ev)) >= 0.85 for ev in existing_vecs):
            continue
        entry = {
            "id": uuid.uuid4().hex[:10],
            "fact": clean,
            "chat_id": chat_id or "",
            "vec": [round(float(x), 5) for x in vec.tolist()] if vec is not None else None,
            "created_at": int(time.time()),
        }
        await redis.lpush(_mem_key(user_id), json.dumps(entry))
        await redis.ltrim(_mem_key(user_id), 0, settings.memory_max_items_per_user - 1)
        existing_lower.add(clean.lower())
        if vec is not None:
            existing_vecs.append(vec)
