"""LabAI Control Plane Gateway: OpenAI-compatible SSE streaming, routing, caching & telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.backends import backend_manager
from app.cache import cache_manager
from app.config import settings
from app.memory import (
    delete_memories,
    extract_and_save_memories,
    handle_memory_command,
    list_memories,
    recall_memories,
)
from app.router import (
    PUBLIC_MODELS,
    BudgetClass,
    ModelTier,
    RoutingDecision,
    apply_dynamic_steering,
    extract_last_user_text,
    has_image_attachments,
    route_request,
    strip_image_attachments,
)
from app.web_search import (
    TOOLS_SCHEMA,
    execute_tool_calls,
    extract_urls,
    fetch_prompt_urls,
    gather_web_context,
    might_need_tools,
    search_verified_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("labai.gateway")

# Prometheus Metrics
REQ_COUNTER = Counter("labai_requests_total", "Total chat completion requests", ["requested_model", "target_tier", "cache"])
CACHE_COUNTER = Counter("labai_cache_events_total", "Cache lookup outcomes", ["outcome"])
TOKEN_COUNTER = Counter("labai_tokens_generated_total", "Total output tokens streamed", ["tier"])
WEB_SEARCH_COUNTER = Counter("labai_web_searches_total", "Auto-triggered web searches")
MEMORY_RECALL_COUNTER = Counter("labai_memory_recalls_total", "Requests enriched with cross-chat memory")
TTFT_HIST = Histogram("labai_ttft_seconds", "Time to first token in seconds", ["tier"], buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0))
TPOT_HIST = Histogram("labai_tpot_seconds", "Time per output token in seconds", ["tier"], buckets=(0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.8))
BACKEND_UP = Gauge("labai_backend_up", "Backend health status (1=up, 0=down)", ["tier"])
NODE2_QUEUE_GAUGE = Gauge("labai_node2_queue_depth", "Active/queued requests on Node 02")


async def _health_loop() -> None:
    while True:
        statuses = await backend_manager.poll_health_once()
        for tier_name, is_up in statuses.items():
            BACKEND_UP.labels(tier=tier_name).set(1.0 if is_up else 0.0)
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await cache_manager.connect()
    task = asyncio.create_task(_health_loop())
    try:
        yield
    finally:
        task.cancel()
        await backend_manager.close()
        await cache_manager.close()


app = FastAPI(title="LabAI Control Plane Gateway", version="1.0.0", lifespan=lifespan)


class ChatCompletionRequest(BaseModel):
    model: str = "labai-auto"
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = Field(default=None)
    user: str | None = None


def _sse_chunk(chunk_id: str, model_name: str, content: str | None = None, finish_reason: str | None = None) -> str:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "backends": {k.value: v for k, v in backend_manager.tier_health.items()}}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    any_up = any(backend_manager.tier_health.values())
    return JSONResponse(
        status_code=200 if any_up else 503,
        content={"ready": any_up, "backends": {k.value: v for k, v in backend_manager.tier_health.items()}},
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": now,
                "owned_by": m["owned_by"],
                "description": m["description"],
            }
            for m in PUBLIC_MODELS
        ],
    }


@app.get("/v1/tools/image-search")
async def image_search(q: str = Query(..., min_length=2), limit: int = Query(default=4, ge=1, le=8)) -> dict[str, Any]:
    """Query self-hosted SearXNG for verified internet image URLs."""
    verified = await search_verified_images(backend_manager.client, q, limit=limit)
    return {"query": q, "results": verified}


@app.get("/v1/memory")
async def get_user_memories(
    request: Request,
    user_id: str | None = Query(default=None),
    x_openwebui_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    uid = user_id or x_openwebui_user_id or (request.client.host if request.client else "anonymous")
    items = await list_memories(uid)
    return {
        "user_id": uid,
        "count": len(items),
        "memories": [{"id": i["id"], "fact": i["fact"], "chat_id": i.get("chat_id", ""), "created_at": i.get("created_at")} for i in items],
    }


@app.delete("/v1/memory")
async def clear_user_memories(
    request: Request,
    chat_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    x_openwebui_user_id: str | None = Header(default=None),
) -> dict[str, Any]:
    uid = user_id or x_openwebui_user_id or (request.client.host if request.client else "anonymous")
    removed = await delete_memories(uid, chat_id=chat_id)
    return {"user_id": uid, "chat_id": chat_id, "removed": removed}


def _direct_reply(chunk_id: str, model_name: str, text: str, stream: bool):
    if stream:
        async def direct_stream() -> AsyncIterator[str]:
            yield _sse_chunk(chunk_id, model_name, content=text)
            yield _sse_chunk(chunk_id, model_name, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(direct_stream(), media_type="text/event-stream")
    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def _routing_badge(decision: RoutingDecision, cache_status: str) -> str:
    if not settings.enable_routing_badge or decision.is_internal_task:
        return ""
    tier_label = {
        ModelTier.FAST: "Fast (4B)",
        ModelTier.REASONING: "Reasoning (8B)",
        ModelTier.PRO: "Pro (14B)",
    }[decision.target_tier]
    fb = " • Fallback" if decision.fallback_applied else ""
    return f"> *LabAI → `{tier_label}` • Budget: `{decision.budget_class.value}` (`{decision.max_tokens}`) • Cache: `{cache_status}`{fb}*\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    x_openwebui_user_id: str | None = Header(default=None),
    x_openwebui_chat_id: str | None = Header(default=None),
):
    decision = route_request(
        requested_model=req.model,
        messages=req.messages,
        explicit_max_tokens=req.max_tokens,
        node2_healthy=backend_manager.is_node2_available(),
        node2_queue_depth=backend_manager.node2_state.queue_depth,
    )

    user_id = req.user or x_openwebui_user_id or (request.client.host if request.client else "anonymous")
    user_prompt = extract_last_user_text(req.messages)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not decision.is_internal_task:
        mem_cmd_reply = await handle_memory_command(user_id, user_prompt)
        if mem_cmd_reply is not None:
            return _direct_reply(chunk_id, req.model, mem_cmd_reply, req.stream)

        allowed, deny_reason = await cache_manager.check_and_acquire_user_slot(user_id)
        if not allowed:
            raise HTTPException(status_code=429, detail=deny_reason)

    if (
        settings.enable_llm_router_judge
        and req.model == "labai-auto"
        and not decision.is_internal_task
        and not decision.is_casual_chat
        and 0.42 <= decision.complexity_score < 0.68
        and backend_manager.is_node2_available()
        and backend_manager.node2_state.queue_depth <= settings.node2_queue_limit
    ):
        decision = await _refine_borderline_decision(decision, user_prompt, req.max_tokens)

    user_turns = sum(1 for m in req.messages if m.get("role") == "user")
    is_single_turn = user_turns == 1 and not decision.is_internal_task
    has_images = has_image_attachments(req.messages)

    want_web = False
    want_img = False
    has_urls = False
    recalled_memory: str | None = None
    url_context: str | None = None

    if not decision.is_internal_task:
        want_web, want_img = might_need_tools(user_prompt)
        has_urls = bool(extract_urls(user_prompt))
        recalled_memory, url_context = await asyncio.gather(
            recall_memories(user_id, user_prompt),
            fetch_prompt_urls(backend_manager.client, user_prompt) if has_urls else asyncio.sleep(0, result=None),
        )
        if recalled_memory:
            MEMORY_RECALL_COUNTER.inc()
        if url_context:
            WEB_SEARCH_COUNTER.inc()

    is_cacheable = is_single_turn and not (want_web or want_img or has_urls or has_images) and not recalled_memory
    cached_text, cache_status = await cache_manager.lookup_response(
        model_tier=decision.target_tier.value,
        prompt=user_prompt,
        is_single_turn=is_cacheable,
        temperature=req.temperature,
    )
    CACHE_COUNTER.labels(outcome=cache_status.split("(")[0]).inc()
    REQ_COUNTER.labels(requested_model=req.model, target_tier=decision.target_tier.value, cache=cache_status.split("(")[0]).inc()

    if cached_text is not None:
        if not decision.is_internal_task:
            await cache_manager.release_user_slot_and_record_tokens(user_id, 0)
        badge = _routing_badge(decision, cache_status)
        return _direct_reply(chunk_id, req.model, f"{badge}{cached_text}", req.stream)

    extra_blocks = [b for b in (recalled_memory, url_context) if b]
    steered_messages = apply_dynamic_steering(req.messages, decision, extra_context_blocks=extra_blocks)
    if has_images and (decision.target_tier != ModelTier.FAST or not settings.tier1_vision_enabled):
        steered_messages = await _handle_image_on_text_tier(steered_messages, user_prompt)
    verified_images: list[dict[str, str]] = []
    target_url = f"{backend_manager.url_for_tier(decision.target_tier)}/v1/chat/completions"

    if (want_web or want_img) and not url_context and not decision.is_internal_task:
        planned_msg, tool_images, direct_reply = await _run_tool_planning(
            target_url=target_url,
            tier=decision.target_tier,
            steered_messages=steered_messages,
            user_prompt=user_prompt,
            want_web=want_web,
            want_img=want_img,
        )
        if direct_reply is not None and decision.target_tier == ModelTier.FAST:
            await cache_manager.release_user_slot_and_record_tokens(user_id, max(1, len(direct_reply) // 4))
            badge = _routing_badge(decision, cache_status)
            return _direct_reply(chunk_id, req.model, f"{badge}{direct_reply}", req.stream)
        steered_messages = planned_msg
        verified_images = tool_images
        WEB_SEARCH_COUNTER.inc()

    upstream_payload = {
        "model": decision.target_tier.value,
        "messages": steered_messages,
        "stream": req.stream,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": decision.max_tokens,
    }

    if req.stream:
        return StreamingResponse(
            _stream_upstream(
                target_url=target_url,
                upstream_payload=upstream_payload,
                decision=decision,
                req_model=req.model,
                chunk_id=chunk_id,
                user_id=user_id,
                chat_id=x_openwebui_chat_id,
                user_prompt=user_prompt,
                is_cacheable=is_cacheable,
                temperature=req.temperature,
                cache_status=cache_status,
                verified_images=verified_images,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming path
    output_tokens = 0
    try:
        data = await _call_non_streaming(target_url, upstream_payload, decision.target_tier)
        content = data["choices"][0]["message"].get("content", "")
        content = _append_missing_images(content, verified_images)
        output_tokens = int(data.get("usage", {}).get("completion_tokens", max(1, len(content) // 4)))
        TOKEN_COUNTER.labels(tier=decision.target_tier.value).inc(output_tokens)
        await cache_manager.store_response(decision.target_tier.value, user_prompt, content, is_cacheable, req.temperature)
        if not decision.is_internal_task and not decision.is_casual_chat:
            asyncio.create_task(extract_and_save_memories(user_id, x_openwebui_chat_id, user_prompt))
        data["choices"][0]["message"]["content"] = f"{_routing_badge(decision, cache_status)}{content}"
        data["model"] = req.model
        return data
    finally:
        if not decision.is_internal_task:
            await cache_manager.release_user_slot_and_record_tokens(user_id, output_tokens)


async def _handle_image_on_text_tier(steered_messages: list[dict[str, Any]], user_prompt: str) -> list[dict[str, Any]]:
    """Prevent HTTP 500 on text-only tiers; transcribe via Tier 1 Vision if enabled."""
    vision_note = (
        "Note: The user attached an image/screenshot, but vision is not enabled on this cluster yet. "
        "Politely let them know you cannot see uploaded images directly and ask them to describe it or paste text/Mermaid code."
    )
    if settings.tier1_vision_enabled:
        image_msgs = [m for m in steered_messages if isinstance(m.get("content"), list)]
        if image_msgs:
            vision_payload = {
                "model": ModelTier.FAST.value,
                "messages": [
                    {
                        "role": "system",
                        "content": "Describe all boxes, arrows, labels, text, and layout in this uploaded image accurately in detail.\n/no_think",
                    },
                    image_msgs[-1],
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 700,
            }
            try:
                data = await _call_non_streaming(f"{settings.tier1_url}/v1/chat/completions", vision_payload, ModelTier.FAST)
                desc = data["choices"][0]["message"].get("content", "").strip()
                if desc:
                    vision_note = f"Visual description extracted from the user's attached image:\n{desc}"
            except Exception as exc:
                logger.warning("Tier 1 Vision transcription failed (%s)", exc)

    cleaned = strip_image_attachments(steered_messages)
    if cleaned and cleaned[0].get("role") == "system":
        cleaned[0]["content"] = f"{cleaned[0]['content']}\n\n{vision_note}"
    else:
        cleaned.insert(0, {"role": "system", "content": vision_note})
    return cleaned


async def _refine_borderline_decision(
    decision: RoutingDecision,
    user_prompt: str,
    explicit_max_tokens: int | None,
) -> RoutingDecision:
    """1-token LLM difficulty judge on Fast (4B) for gray-zone prompts across any domain."""
    judge_prompt = (
        "Classify the difficulty of this user request into ONE word: SIMPLE, MEDIUM, or HARD.\n"
        "- SIMPLE: basic facts, greetings, short definitions, casual chat.\n"
        "- MEDIUM: standard coding, debugging, comparisons, explanations.\n"
        "- HARD: deep multi-step architecture, formal proofs, advanced multi-domain synthesis, complex audits.\n\n"
        f"Request: {user_prompt[:350]}\nLabel:\n/no_think"
    )
    payload = {
        "model": ModelTier.FAST.value,
        "messages": [{"role": "user", "content": judge_prompt}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 4,
    }
    try:
        async with backend_manager.node1_semaphore:
            resp = await backend_manager.client.post(
                f"{settings.tier1_url}/v1/chat/completions",
                json=payload,
                timeout=2.5,
            )
            resp.raise_for_status()
            label = resp.json()["choices"][0]["message"].get("content", "").strip().upper()
    except Exception:
        return decision

    if "HARD" in label:
        decision.target_tier = ModelTier.PRO
        decision.budget_class = BudgetClass.DEEP
        if explicit_max_tokens is None:
            decision.max_tokens = settings.budget_deep_tokens
        decision.reason = "auto-judge-pro"
    elif "SIMPLE" in label:
        decision.target_tier = ModelTier.FAST
        decision.reason = "auto-judge-fast"
    return decision


async def _run_tool_planning(
    target_url: str,
    tier: ModelTier,
    steered_messages: list[dict[str, Any]],
    user_prompt: str,
    want_web: bool,
    want_img: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str | None]:
    """Let Qwen3 decide tool calls (web_search/image_search) or fall back cleanly to SearXNG."""
    plan_messages = [dict(m) for m in steered_messages]
    tool_hint = (
        "You have live tools: web_search(query) and image_search(query). "
        "If the user asks for photos/faces/images of multiple subjects, call image_search once per subject. "
        "Never claim you cannot display images.\n/no_think"
    )
    if plan_messages and plan_messages[0].get("role") == "system":
        plan_messages[0]["content"] = f"{plan_messages[0]['content']}\n{tool_hint}"
    else:
        plan_messages.insert(0, {"role": "system", "content": tool_hint})

    plan_payload = {
        "model": ModelTier.FAST.value,
        "messages": plan_messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 250,
    }
    fast_url = f"{settings.tier1_url}/v1/chat/completions"
    try:
        # Always run fast tool planning on Tier 1 (4B) instead of stalling Tier 3 (14B)
        plan_data = await _call_non_streaming(fast_url, plan_payload, ModelTier.FAST)
        assistant_msg = plan_data["choices"][0]["message"]
        tool_calls = assistant_msg.get("tool_calls") or []
        if tool_calls:
            tool_msgs, verified_images = await execute_tool_calls(backend_manager.client, tool_calls)
            updated = list(plan_messages) + [assistant_msg] + tool_msgs
            updated[0]["content"] = (
                f"{updated[0]['content']}\nUse the tool outputs to answer accurately. "
                "Cite web sources as [1](url) and embed verified images with Markdown ![caption](url)."
            )
            return updated, verified_images, None

        direct_content = (assistant_msg.get("content") or "").strip()
        refused_images = "can't display images" in direct_content.lower() or "cannot display images" in direct_content.lower()
        if direct_content and not refused_images and not want_img:
            return steered_messages, [], (direct_content if tier == ModelTier.FAST else None)
    except Exception as exc:
        logger.warning("Native tool planning failed (%s); using direct SearXNG fallback.", exc)

    # Fallback: direct SearXNG lookup if model skipped/refused tool call
    web_ctx, fallback_images, _ = await gather_web_context(backend_manager.client, user_prompt)
    if web_ctx:
        updated = [dict(m) for m in steered_messages]
        if updated and updated[0].get("role") == "system":
            updated[0]["content"] = f"{updated[0]['content']}\n\n{web_ctx}"
        else:
            updated.insert(0, {"role": "system", "content": web_ctx})
        return updated, fallback_images, None
    return steered_messages, [], None


async def _call_non_streaming(url: str, payload: dict[str, Any], tier: ModelTier) -> dict[str, Any]:
    if tier in (ModelTier.FAST, ModelTier.REASONING):
        async with backend_manager.node1_semaphore:
            resp = await backend_manager.client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    backend_manager.node2_state.queue_depth += 1
    NODE2_QUEUE_GAUGE.set(backend_manager.node2_state.queue_depth)
    try:
        resp = await backend_manager.client.post(url, json=payload)
        resp.raise_for_status()
        backend_manager.record_node2_success()
        return resp.json()
    except Exception as exc:
        backend_manager.record_node2_failure()
        raise HTTPException(status_code=502, detail=f"Worker node02 error: {exc}")
    finally:
        backend_manager.node2_state.queue_depth = max(0, backend_manager.node2_state.queue_depth - 1)
        NODE2_QUEUE_GAUGE.set(backend_manager.node2_state.queue_depth)


def _format_image_gallery(verified_images: list[dict[str, str]]) -> str:
    if not verified_images:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for im in verified_images:
        url = im["image_url"]
        if url in seen:
            continue
        seen.add(url)
        lines.append(f"![{im['title']}]({url})")
        if len(lines) >= 6:
            break
    return "\n\n" + "\n\n".join(lines)


def _append_missing_images(content: str, verified_images: list[dict[str, str]]) -> str:
    if not verified_images or "![" in content:
        return content
    return content + _format_image_gallery(verified_images)


async def _stream_upstream(
    target_url: str,
    upstream_payload: dict[str, Any],
    decision: RoutingDecision,
    req_model: str,
    chunk_id: str,
    user_id: str,
    chat_id: str | None,
    user_prompt: str,
    is_cacheable: bool,
    temperature: float,
    cache_status: str,
    verified_images: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    start_ts = time.perf_counter()
    first_token_ts: float | None = None
    token_count = 0
    collected_parts: list[str] = []

    badge = _routing_badge(decision, cache_status)
    if badge:
        yield _sse_chunk(chunk_id, req_model, content=badge)

    async def _run_stream(url: str, payload: dict[str, Any], tier: ModelTier) -> AsyncIterator[str]:
        nonlocal first_token_ts, token_count
        async with backend_manager.client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    full_so_far = "".join(collected_parts)
                    if verified_images and "![" not in full_so_far:
                        gallery = _format_image_gallery(verified_images)
                        collected_parts.append(gallery)
                        yield _sse_chunk(chunk_id, req_model, content=gallery)
                    yield "data: [DONE]\n\n"
                    break
                try:
                    obj = json.loads(raw)
                    obj["model"] = req_model
                    delta_text = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta_text:
                        now = time.perf_counter()
                        if first_token_ts is None:
                            first_token_ts = now
                            TTFT_HIST.labels(tier=tier.value).observe(now - start_ts)
                        token_count += 1
                        collected_parts.append(delta_text)
                    yield f"data: {json.dumps(obj)}\n\n"
                except json.JSONDecodeError:
                    continue

    try:
        if decision.target_tier in (ModelTier.FAST, ModelTier.REASONING):
            async with backend_manager.node1_semaphore:
                async for out in _run_stream(target_url, upstream_payload, decision.target_tier):
                    yield out
        else:
            backend_manager.node2_state.queue_depth += 1
            NODE2_QUEUE_GAUGE.set(backend_manager.node2_state.queue_depth)
            try:
                async for out in _run_stream(target_url, upstream_payload, decision.target_tier):
                    yield out
                backend_manager.record_node2_success()
            except Exception:
                backend_manager.record_node2_failure()
                raise
            finally:
                backend_manager.node2_state.queue_depth = max(0, backend_manager.node2_state.queue_depth - 1)
                NODE2_QUEUE_GAUGE.set(backend_manager.node2_state.queue_depth)

        if first_token_ts is not None and token_count > 1:
            tpot = (time.perf_counter() - first_token_ts) / (token_count - 1)
            TPOT_HIST.labels(tier=decision.target_tier.value).observe(tpot)
        TOKEN_COUNTER.labels(tier=decision.target_tier.value).inc(token_count)
        await cache_manager.store_response(
            decision.target_tier.value,
            user_prompt,
            "".join(collected_parts),
            is_cacheable,
            temperature,
        )
        if not decision.is_internal_task and not decision.is_casual_chat:
            asyncio.create_task(extract_and_save_memories(user_id, chat_id, user_prompt))
    except Exception as exc:
        logger.error("Upstream stream failed on %s: %s", decision.target_tier.value, exc)
        yield _sse_chunk(chunk_id, req_model, content=f"\n\n[Gateway Error: {exc}]", finish_reason="stop")
        yield "data: [DONE]\n\n"
    finally:
        if not decision.is_internal_task:
            await cache_manager.release_user_slot_and_record_tokens(user_id, token_count)
