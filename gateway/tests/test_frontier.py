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
    assert clean_image_query("give me another sunset pic") == "sunset"
    assert needs_image_search("give me another sunset pic") is True
    assert needs_image_search("well dear u making me falling for u like the most beloved person") is False
    assert needs_web_search("well dear u making me falling for u like the most beloved person") is False


def test_html_text_extractor_strips_scripts_and_tags():
    html = "<html><head><script>alert(1)</script><style>body{}</style></head><body><h1>Hello</h1><p>World 2026</p></body></html>"
    assert extract_clean_text(html) == "Hello World 2026"


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}
        self.kv: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set(self, key: str, value: str) -> None:
        self.kv[key] = value

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.store.get(key, [])
        return items[start : end + 1 if end >= 0 else None]

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.kv.pop(key, None)

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

        # Unrelated queries must not leak saved technical memories into fresh casual chats
        await fake_redis.rpush(
            key,
            json.dumps({"id": "3", "fact": "User is designing a Redis diagram", "chat_id": "c3", "vec": None, "created_at": 3}),
        )
        from app.memory import recall_memories
        from app.router import semantic_scorer

        monkeypatch.setattr(semantic_scorer, "embed", lambda _txt: None)
        assert await recall_memories("u1", "hey babe miss you") is None
        recalled = await recall_memories("u1", "update my redis diagram")
        assert recalled is not None and "Redis diagram" in recalled

        # Plain-text and bang memory commands avoid OpenWebUI's '/' prompt popup and can pause memory
        assert "Redis diagram" in (await handle_memory_command("u1", "memory") or "")
        off_msg = await handle_memory_command("u1", "memory off")
        assert off_msg is not None and "OFF" in off_msg
        assert await recall_memories("u1", "update my redis diagram") is None
        on_msg = await handle_memory_command("u1", "!memory on")
        assert on_msg is not None and "ON" in on_msg
        assert await recall_memories("u1", "update my redis diagram") is not None
        assert "Cleared 1" in (await handle_memory_command("u1", "/clear memory") or "")
        assert await list_memories("u1") == []
        await fake_redis.rpush(
            key,
            json.dumps({"id": "3", "fact": "User is designing a Redis diagram", "chat_id": "c3", "vec": None, "created_at": 3}),
        )

        # Explicit recall questions ("what do u remember bout me") surface saved facts or honest empty status
        meta_saved = await recall_memories("u1", "what do u remeber bout me")
        assert meta_saved is not None and "Redis diagram" in meta_saved
        await delete_memories("u1")
        meta_empty = await recall_memories("u1", "what do u remeber bout me")
        assert meta_empty is not None and "No personal facts are saved" in meta_empty

        # Personalization + Temporary chat mode
        from app.memory import get_persona, is_temp_chat

        p_reply = await handle_memory_command("u1", "persona call me boss and keep banter playful")
        assert p_reply is not None and "call me boss" in p_reply
        assert await get_persona("u1") == "call me boss and keep banter playful"
        assert await is_temp_chat("u1", "local:temp-123", has_webui_user=True) is True
        assert await is_temp_chat("u1", "chat-uuid-1", has_webui_user=True) is False
        await handle_memory_command("u1", "temp on")
        assert await is_temp_chat("u1", "chat-uuid-1", has_webui_user=True) is True
        assert await get_persona("u1") == "call me boss and keep banter playful"
        await handle_memory_command("u1", "temp off")
        await handle_memory_command("u1", "persona clear")
        assert await get_persona("u1") is None

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


def test_image_search_safesearch_off_and_dedup_pagination(monkeypatch):
    import asyncio
    from app import web_search as ws_mod

    seen_params: list[dict[str, str]] = []

    class _FakeResp:
        def __init__(self, items: list[dict[str, str]]) -> None:
            self._items = items

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"results": self._items}

    class _FakeClient:
        async def get(self, url: str, params: dict[str, str] | None = None, timeout: float = 4.0):
            assert params is not None
            seen_params.append(params)
            if params.get("pageno") == "1":
                return _FakeResp([{"title": "Pic 1", "img_src": "https://img.example/1.jpg", "url": "https://example.com/1"}])
            return _FakeResp([{"title": "Pic 2", "img_src": "https://img.example/2.jpg", "url": "https://example.com/2"}])

    async def _always_ok(_client, _url: str) -> bool:
        return True

    monkeypatch.setattr(ws_mod, "_verify_image_url", _always_ok)

    prior = [{"role": "assistant", "content": "Here:\n![Pic 1](https://img.example/1.jpg)"}]
    prior_urls = ws_mod.extract_prior_image_urls(prior)
    assert prior_urls == {"https://img.example/1.jpg"}

    results = asyncio.run(ws_mod.search_verified_images(_FakeClient(), "send me a photo of mountains", limit=1, exclude_urls=prior_urls))
    assert len(results) == 1
    assert results[0]["image_url"] == "https://img.example/2.jpg"
    assert all(p["safesearch"] == "0" for p in seen_params)
