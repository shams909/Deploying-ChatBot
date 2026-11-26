"""Unit tests for Frontier prompt steering, auto web browsing, and cross-chat memory."""

from datetime import datetime, timezone

from app.config import settings
from app.memory import delete_memories, handle_memory_command, list_memories
from app.router import ModelTier, apply_dynamic_steering, route_request
from app.web_search import extract_clean_text, needs_image_search, needs_web_search


def test_frontier_prompt_injects_date_and_skips_slang_and_clipped_greetings(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "yo sup"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.is_casual_chat is True
    fixed_dt = datetime(2026, 11, 26, 22, 45, tzinfo=timezone.utc)
    steered = apply_dynamic_steering([{"role": "user", "content": "yo sup"}], decision, now=fixed_dt)
    sys_msg = steered[0]["content"]
    assert "2026-11-26" in sys_msg
    assert "homie" not in sys_msg.lower()
    assert "Answer concisely and directly" not in sys_msg


def test_web_search_and_image_intent_detection():
    assert needs_web_search("What is the latest Python release in 2026?") is True
    assert needs_web_search("Search online for today's weather in Dhaka") is True
    assert needs_web_search("Write a Python function to reverse a string") is False
    assert needs_web_search("hey") is False

    assert needs_image_search("Show me a photo of a server rack") is True
    assert needs_image_search("Draw an architecture diagram of Redis") is False


def test_html_text_extractor_strips_scripts_and_tags():
    html = "<html><head><script>alert(1)</script><style>body{}</style></head><body><h1>Hello</h1><p>World 2026</p></body></html>"
    assert extract_clean_text(html) == "Hello World 2026"


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.store.get(key, [])
        return items[start : end + 1 if end >= 0 else None]

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def rpush(self, key: str, *values: str) -> None:
        self.store.setdefault(key, []).extend(values)


def test_memory_commands_and_chat_scoped_deletion(monkeypatch):
    import asyncio
    import json
    from app import cache as cache_mod

    fake_redis = _FakeRedis()
    monkeypatch.setattr(cache_mod.cache_manager, "_redis", fake_redis)

    async def run() -> None:
        key = "labai:memory:u1"
        await fake_redis.rpush(
            key,
            json.dumps({"id": "1", "fact": "User studies CS", "chat_id": "c1", "vec": None, "created_at": 1}),
            json.dumps({"id": "2", "fact": "User prefers Linux", "chat_id": "c2", "vec": None, "created_at": 2}),
        )
        listed = await handle_memory_command("u1", "/memory")
        assert listed is not None and "User studies CS" in listed

        removed_c1 = await delete_memories("u1", chat_id="c1")
        assert removed_c1 == 1
        remaining = await list_memories("u1")
        assert len(remaining) == 1 and remaining[0]["fact"] == "User prefers Linux"

        cleared = await handle_memory_command("u1", "/memory clear")
        assert cleared is not None and "Cleared 1" in cleared
        assert await list_memories("u1") == []

    asyncio.run(run())
