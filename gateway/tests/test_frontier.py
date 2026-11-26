"""Unit tests for Frontier prompt steering, auto web browsing, and cross-chat memory."""

from datetime import datetime, timezone

from app.config import settings
from app.memory import delete_memories, handle_memory_command, list_memories
from app.router import (
    ModelTier,
    apply_dynamic_steering,
    has_image_attachments,
    route_request,
    strip_image_attachments,
)
from app.web_search import (
    _special_url_candidates,
    clean_image_query,
    execute_tool_calls,
    extract_clean_text,
    extract_urls,
    is_safe_public_url,
    needs_image_search,
    needs_web_search,
)


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
    assert "22:45" not in sys_msg  # Static YYYY-MM-DD prefix preserves KV cache across minutes
    assert "homie" not in sys_msg.lower()
    assert "Answer concisely and directly" not in sys_msg

    time_decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "what time is it right now?"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    time_steered = apply_dynamic_steering(
        [{"role": "user", "content": "what time is it right now?"}],
        time_decision,
        now=fixed_dt,
    )
    assert "22:45" in time_steered[0]["content"]


def test_web_search_and_image_intent_detection():
    assert needs_web_search("What is the latest Python release in 2026?") is True
    assert needs_web_search("Search online for today's weather in Dhaka") is True
    assert needs_web_search("What is today's date?") is False
    assert needs_web_search("What is the current time right now?") is False
    assert needs_web_search("Write a Python function to reverse a string") is False
    assert needs_web_search("hey") is False

    assert needs_image_search("Show me a photo of a server rack") is True
    assert needs_image_search("an image of an tesla car") is True
    assert needs_image_search("show me the face of mark zuck") is True
    assert needs_image_search("Draw an architecture diagram of Redis") is False
    assert clean_image_query("show me an image of a tesla") == "tesla"
    assert clean_image_query("an image of a server rack") == "server rack"


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


def test_execute_tool_calls_dispatches_multiple_subjects(monkeypatch):
    import asyncio
    import json
    from app import web_search as ws_mod

    seen_queries: list[str] = []

    async def fake_images(client, query: str, limit: int = 2):
        seen_queries.append(query)
        return [{"title": query, "image_url": f"https://img.example/{query.replace(' ', '_')}.jpg", "source": ""}]

    async def fake_pages(client, query: str, max_results=None):
        seen_queries.append(query)
        return [{"title": "Latest Release", "url": "https://example.com", "content": "Python 3.14 released"}]

    monkeypatch.setattr(ws_mod, "search_verified_images", fake_images)
    monkeypatch.setattr(ws_mod, "search_web_pages", fake_pages)

    tool_calls = [
        {"id": "call_1", "function": {"name": "image_search", "arguments": json.dumps({"query": "Mark Zuckerberg"})}},
        {"id": "call_2", "function": {"name": "image_search", "arguments": json.dumps({"query": "Jensen Huang"})}},
        {"id": "call_3", "function": {"name": "web_search", "arguments": json.dumps({"query": "Python 2026 release"})}},
    ]

    async def run() -> None:
        msgs, imgs = await execute_tool_calls(None, tool_calls)  # type: ignore[arg-type]
        assert len(msgs) == 3
        assert len(imgs) == 2
        assert seen_queries == ["Mark Zuckerberg", "Jensen Huang", "Python 2026 release"]

    asyncio.run(run())


def test_url_extraction_ssrf_guard_and_github_raw_mapping():
    assert is_safe_public_url("http://127.0.0.1:8081/health") is False
    assert is_safe_public_url("http://192.168.9.135:8083") is False
    assert is_safe_public_url("http://node01/admin") is False
    assert is_safe_public_url("https://github.com/shams909/Research/tree/main/CPU%20Cluster") is True

    urls = extract_urls("analyze https://github.com/shams909/Research/tree/main/CPU%20Cluster and http://127.0.0.1:8000")
    assert urls == ["https://github.com/shams909/Research/tree/main/CPU%20Cluster"]
    raw = _special_url_candidates(urls[0])
    assert raw == ["https://raw.githubusercontent.com/shams909/Research/main/CPU%20Cluster/README.md"]
    tweet_oembed = _special_url_candidates("https://x.com/OpenAI/status/123456")
    assert len(tweet_oembed) == 1 and "publish.twitter.com/oembed" in tweet_oembed[0]


def test_html_meta_og_fallback_for_js_pages():
    html = '<html><head><meta property="og:title" content="Breaking Tech News"><meta property="og:description" content="Summary of article."></head><body><div id="app"></div></body></html>'
    extracted = extract_clean_text(html)
    assert "Breaking Tech News" in extracted and "Summary of article." in extracted


def test_multimodal_image_guard_and_vision_routing(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Turn this diagram into clean Mermaid"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]
    assert has_image_attachments(msgs) is True
    stripped = strip_image_attachments(msgs)
    assert stripped[0]["content"] == "Turn this diagram into clean Mermaid"

    monkeypatch.setattr(settings, "vision_enabled", True)
    decision = route_request("labai-auto", msgs, None, True, 0)
    assert decision.target_tier == ModelTier.REASONING
    assert decision.reason == "auto-vision-reasoning"

    deep_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Architect a fault-tolerant distributed consensus system from this diagram with trade-off analysis"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]
    deep_decision = route_request("labai-auto", deep_msgs, None, True, 0)
    assert deep_decision.target_tier == ModelTier.PRO
    assert deep_decision.reason == "auto-vision-pro"
