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
_MEM_DISABLED_PREFIX = "labai:memory:disabled:"
_PERSONA_PREFIX = "labai:persona:"
_TEMP_MODE_PREFIX = "labai:temp_mode:"
_FORGET_PHRASE = re.compile(r"^(?:please\s+)?forget\s+(?:everything|all(?:\s+memories)?)\s*(?:you\s+remember\s+about\s+me)?[.!]?$", re.IGNORECASE)


def _mem_key(user_id: str) -> str:
    return f"{_MEM_KEY_PREFIX}{user_id}"


def _mem_disabled_key(user_id: str) -> str:
    return f"{_MEM_DISABLED_PREFIX}{user_id}"


def _persona_key(user_id: str) -> str:
    return f"{_PERSONA_PREFIX}{user_id}"


def _temp_mode_key(user_id: str) -> str:
    return f"{_TEMP_MODE_PREFIX}{user_id}"


async def get_persona(user_id: str) -> str | None:
    """Return user's saved personalization instructions if set."""
    redis = cache_manager.redis
    if redis is None:
        return None
    try:
        val = await redis.get(_persona_key(user_id))
        return val.strip() if isinstance(val, str) and val.strip() else None
    except Exception:
        return None


async def is_temp_chat(user_id: str, chat_id: str | None, has_webui_user: bool = False) -> bool:
    """Return True when OpenWebUI Temporary Chat is active or user toggled temp mode on."""
    if has_webui_user and (not chat_id or str(chat_id).startswith("local:")):
        return True
    redis = cache_manager.redis
    if redis is None:
        return False
    try:
        return bool(await redis.get(_temp_mode_key(user_id)))
    except Exception:
        return False


async def is_memory_enabled(user_id: str) -> bool:
    """Return True unless the user has explicitly paused long-term memory."""
    if not settings.enable_long_term_memory:
        return False
    redis = cache_manager.redis
    if redis is None:
        return True
    try:
        return not bool(await redis.get(_mem_disabled_key(user_id)))
    except Exception:
        return True


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
    """Handle memory commands (/memory, !memory, or plain 'memory ...') directly without calling the LLM."""
    text = user_text.strip()
    lower = text.lower()
    norm = lower[1:] if lower.startswith(("/", "!")) else lower

    if norm in ("memory", "memory list"):
        enabled = await is_memory_enabled(user_id)
        status = "ON" if enabled else "OFF (paused)"
        items = await list_memories(user_id)
        if not items:
            return (
                f"No cross-chat memories are saved for your account yet (Memory status: **{status}**).\n\n"
                "*Commands: `memory`, `memory clear`, `memory off`, `memory on`*"
            )
        lines = [f"{idx}. {it['fact']}" for idx, it in enumerate(items, start=1)]
        return (
            f"**Saved Cross-Chat Memories ({status}):**\n"
            + "\n".join(lines)
            + "\n\n*Use `memory delete <number>`, `memory clear`, `memory off`, or `memory on` anytime.*"
        )

    if norm in ("memory clear", "memory reset", "clear memory", "clear memories", "reset memory") or _FORGET_PHRASE.match(text):
        count = await delete_memories(user_id)
        return f"Cleared {count} saved cross-chat {'memory' if count == 1 else 'memories'}."

    if norm in ("memory off", "memory disable"):
        redis = cache_manager.redis
        if redis is not None:
            await redis.set(_mem_disabled_key(user_id), "1")
        return "Cross-chat memory is now **OFF** for your account (`memory on` to re-enable, `memory clear` to wipe saved items)."

    if norm in ("memory on", "memory enable"):
        redis = cache_manager.redis
        if redis is not None:
            await redis.delete(_mem_disabled_key(user_id))
        return "Cross-chat memory is now **ON** for your account (`memory off` to pause anytime)."

    if norm in ("temp", "temp status", "incognito"):
        active = await is_temp_chat(user_id, None, has_webui_user=False)
        state = "ON (no memories saved or recalled; personalization still active)" if active else "OFF"
        return f"Temporary chat mode is **{state}**.\n\n*Commands: `temp on`, `temp off`, `persona <style>`*"

    if norm in ("temp on", "incognito on"):
        redis = cache_manager.redis
        if redis is not None:
            await redis.set(_temp_mode_key(user_id), "1")
        return "Temporary chat mode is now **ON** (no chat memories recalled or saved; your `persona` stays active; `temp off` to exit)."

    if norm in ("temp off", "incognito off"):
        redis = cache_manager.redis
        if redis is not None:
            await redis.delete(_temp_mode_key(user_id))
        return "Temporary chat mode is now **OFF**."

    if norm in ("persona", "persona show"):
        current = await get_persona(user_id)
        if not current:
            return (
                "No custom personalization saved yet.\n\n"
                "*Set one anytime with `persona <how you want me to talk & behave>`, or `persona clear` to remove.*"
            )
        return f"**Saved Personalization:**\n> {current}\n\n*Update with `persona <new instructions>` or clear with `persona clear`.*"

    if norm in ("persona clear", "persona reset", "persona off"):
        redis = cache_manager.redis
        if redis is not None:
            await redis.delete(_persona_key(user_id))
        return "Cleared your custom personalization."

    if norm.startswith(("persona ", "persona:")):
        body = text.split(":", 1)[1].strip() if norm.startswith("persona:") else text.split(None, 1)[1].strip()
        if not body:
            return "Usage: `persona <how you want me to behave>`"
        redis = cache_manager.redis
        if redis is not None:
            await redis.set(_persona_key(user_id), body[:800])
        return f"Saved your personalization (applies in both regular and Temporary chats):\n> {body[:800]}"

    if norm.startswith("memory delete "):
        arg = text.split(None, 2)[-1].strip()
        if not arg.isdigit():
            return "Usage: `memory delete <number>` (send `memory` first to see numbers)."
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


_MEMORY_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
    "have", "has", "had", "not", "but", "can", "will", "just", "like", "what", "when",
    "where", "who", "how", "why", "from", "about", "into", "they", "them", "then",
})
_MEMORY_QUERY_PATTERN = re.compile(
    r"\b(?:what\s+do\s+(?:you|u)\s+(?:remem?ber|know)\s+(?:about|bout)\s+me"
    r"|do\s+(?:you|u)\s+remem?ber\s+(?:me|anything)"
    r"|what\s+(?:have|do)\s+(?:you|u)\s+(?:saved|stored)\s+(?:about|bout)\s+me)\b",
    re.IGNORECASE,
)


async def recall_memories(user_id: str, query_text: str) -> str | None:
    if not query_text.strip():
        return None
    is_meta_recall = bool(_MEMORY_QUERY_PATTERN.search(query_text))
    if not await is_memory_enabled(user_id):
        if is_meta_recall:
            return (
                "Memory status: Cross-chat memory is currently turned OFF for this user. "
                "Answer honestly that nothing is saved or remembered from past chats—do NOT invent shared memories."
            )
        return None
    items = await list_memories(user_id)
    if not items:
        if is_meta_recall:
            return (
                "Memory status: No personal facts are saved about this user from past chats yet. "
                "Answer honestly that nothing is saved yet—do NOT invent fake past memories or stories."
            )
        return None
    if is_meta_recall:
        return "Facts saved about this user from past chats:\n" + "\n".join(
            f"- {it['fact']}" for it in items[: settings.memory_recall_top_k]
        )

    query_vec = semantic_scorer.embed(query_text[:600])
    selected: list[str] = []
    if query_vec is not None:
        scored: list[tuple[float, str]] = []
        for it in items:
            vec_list = it.get("vec")
            if not vec_list:
                continue
            sim = float(np.dot(query_vec, np.array(vec_list, dtype=np.float32)))
            # bge-small baseline for unrelated sentences sits around 0.45-0.55; require >= 0.62
            if sim >= 0.62:
                scored.append((sim, it["fact"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [fact for _, fact in scored[: settings.memory_recall_top_k]]
    else:
        q_tokens = {w for w in re.findall(r"[a-z0-9]{3,}", query_text.lower()) if w not in _MEMORY_STOPWORDS}
        if q_tokens:
            for it in items:
                f_tokens = {w for w in re.findall(r"[a-z0-9]{3,}", it["fact"].lower()) if w not in _MEMORY_STOPWORDS}
                if q_tokens & f_tokens:
                    selected.append(it["fact"])
                    if len(selected) >= settings.memory_recall_top_k:
                        break

    if not selected:
        return None
    return "Optional background about the user (ignore completely unless directly relevant to the current message):\n" + "\n".join(
        f"- {fact}" for fact in selected
    )


async def extract_and_save_memories(user_id: str, chat_id: str | None, user_text: str) -> None:
    """Background task using Tier 1 (Fast 4B) to extract durable user facts after a substantive turn."""
    redis = cache_manager.redis
    if (
        redis is None
        or len(user_text.strip()) < 25
        or user_text.strip().startswith(("/", "!"))
        or not await is_memory_enabled(user_id)
    ):
        return

    extraction_prompt = (
        "Extract at most 2 concise, durable facts about the user (e.g., role, skills, ongoing project, "
        "preferred programming language, or preferences) from the message below. "
        "Ignore greetings, one-off questions, or generic trivia. "
        'Return ONLY a JSON array of short strings (e.g. ["User works with Python on Linux"]), or [] if none.\n\n'
        f"User message: {user_text[:600]}"
    )
    payload = {
        "model": ModelTier.FAST.value,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 120,
        "chat_template_kwargs": {"enable_thinking": False},
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
