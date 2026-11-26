"""Semantic + heuristic intent detection, SearXNG web/image search, and native Qwen3 tool execution."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx
import numpy as np

from app.config import settings
from app.router import semantic_scorer

logger = logging.getLogger("labai.web_search")

_WEB_INTENT_PATTERN = re.compile(
    r"\b(today|yesterday|tonight|this week|this month|this year|latest|newest|current|recent|currently|"
    r"news|price|weather|forecast|score|standings|stock|release date|version of|who won|what happened|"
    r"search the web|search online|look up|browse|find online|2026|2027)\b",
    re.IGNORECASE,
)
_IMAGE_INTENT_PATTERN = re.compile(
    r"\b(show\s+me\s+(?:a|an|the)?\s*(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces|snap)|"
    r"find\s+(?:a|an|the)?\s*(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces)|"
    r"drop\s+(?:a|an)?\s*(?:pic|pics|photo|snap|image)|"
    r"(?:a|an|the)?\s*(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces)\s+of|"
    r"what\s+does\s+.+\s+look\s+like)\b",
    re.IGNORECASE,
)
_DIAGRAM_EXCLUSION = re.compile(
    r"\b(diagram|flowchart|sequence diagram|class diagram|er diagram|mermaid|mindmap)\b",
    re.IGNORECASE,
)
_CLOCK_ONLY_PATTERN = re.compile(
    r"^[\w\s,'?!.]*\b(what('?s| is)?\s+(the\s+)?(today('?s)?|current\s+)?(date|time|day)|today('?s)?\s+date)\b[\w\s,'?!.]*$",
    re.IGNORECASE,
)

_IMAGE_PREFIX_STRIP = re.compile(
    r"^(?:please\s+)?(?:show\s+me|find|give\s+me|get\s+me|drop)?\s*"
    r"(?:a|an|the)?\s*(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces|snap)\s+of\s+",
    re.IGNORECASE,
)
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

_INTENT_PROTOTYPES = {
    "live_web": [
        "Who won the football match last night and what are the current standings?",
        "Did any tech company announce a major AI release this week?",
        "What is the current stock price or weather forecast right now?",
        "Who is currently the Prime Minister or President?",
    ],
    "image_lookup": [
        "Can you show me what Jensen Huang looks like?",
        "Drop a pic of a Tesla Cybertruck or an RTX 5090.",
        "Show me photos of the Eiffel Tower and Mount Fuji.",
        "I want to see a picture of a server rack.",
    ],
    "static_chat": [
        "Write a Python function to reverse a linked list.",
        "Explain how consistent hashing works in distributed systems.",
        "Hey, how is it going today?",
        "Draw a Mermaid architecture diagram for an API gateway.",
    ],
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live internet for up-to-date facts, news, releases, prices, weather, sports, or current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concise search query for the web search engine."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Search the web for real verified photographs or images of people, places, objects, or products. Call once per distinct subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Subject name to find an image for, e.g. 'Mark Zuckerberg' or 'Tesla Model S'."}
                },
                "required": ["query"],
            },
        },
    },
]


class _SemanticIntentDetector:
    """Zero-extra-RAM intent classifier reusing the gateway's ONNX embedder."""

    def __init__(self) -> None:
        self._centroids: dict[str, np.ndarray] = {}
        self._ready = False

    def _ensure_centroids(self) -> bool:
        if self._ready:
            return bool(self._centroids)
        self._ready = True
        if not settings.enable_semantic_embedder:
            return False
        try:
            for label, samples in _INTENT_PROTOTYPES.items():
                vecs = [v for v in (semantic_scorer.embed(s) for s in samples) if v is not None]
                if not vecs:
                    return False
                centroid = np.mean(np.vstack(vecs), axis=0)
                self._centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-9)
            return True
        except Exception as exc:
            logger.debug("Intent centroid init failed: %s", exc)
            return False

    def classify(self, text: str) -> tuple[bool, bool]:
        if not self._ensure_centroids():
            return False, False
        qvec = semantic_scorer.embed(text[:500])
        if qvec is None:
            return False, False
        sims = {k: float(np.dot(qvec, c)) for k, c in self._centroids.items()}
        web_hit = sims["live_web"] >= 0.48 and sims["live_web"] > sims["static_chat"] + 0.04
        img_hit = sims["image_lookup"] >= 0.46 and sims["image_lookup"] > sims["static_chat"] + 0.03
        return web_hit, img_hit


semantic_intent = _SemanticIntentDetector()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "nav", "footer", "header"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def extract_clean_text(html_body: str, max_chars: int = 900) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html_body)
        parser.close()
    except Exception:
        return ""
    return parser.get_text()[:max_chars]


def _is_clock_only(user_text: str) -> bool:
    return bool(_CLOCK_ONLY_PATTERN.match(user_text.strip())) and not re.search(
        r"\b(news|weather|price|stock|release|version|won)\b", user_text, re.IGNORECASE
    )


def needs_web_search(user_text: str) -> bool:
    if not settings.enable_auto_web_search or not user_text or len(user_text.strip()) < 6:
        return False
    if _is_clock_only(user_text):
        return False
    if _WEB_INTENT_PATTERN.search(user_text):
        return True
    sem_web, _ = semantic_intent.classify(user_text)
    return sem_web


def needs_image_search(user_text: str) -> bool:
    if not settings.enable_auto_web_search or not user_text or len(user_text.strip()) < 6:
        return False
    if _DIAGRAM_EXCLUSION.search(user_text):
        return False
    if _IMAGE_INTENT_PATTERN.search(user_text):
        return True
    _, sem_img = semantic_intent.classify(user_text)
    return sem_img


def might_need_tools(user_text: str) -> tuple[bool, bool]:
    """Return (want_web, want_image) using regex + ONNX semantic intent."""
    return needs_web_search(user_text), needs_image_search(user_text)


def clean_image_query(user_text: str) -> str:
    cleaned = _IMAGE_PREFIX_STRIP.sub("", user_text.strip())
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:150].strip() or user_text[:150].strip()


async def _fetch_page_excerpt(client: httpx.AsyncClient, url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        resp = await client.get(url, headers=_BROWSER_HEADERS, timeout=2.5, follow_redirects=True)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "text/html" in ctype:
            return extract_clean_text(resp.text, max_chars=900)
    except Exception:
        return ""
    return ""


async def _verify_image_url(client: httpx.AsyncClient, url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        head = await client.head(url, headers=_BROWSER_HEADERS, timeout=2.5, follow_redirects=True)
        if head.status_code == 200 and head.headers.get("content-type", "").lower().startswith("image/"):
            return True
    except Exception:
        pass
    try:
        async with client.stream("GET", url, headers=_BROWSER_HEADERS, timeout=3.0, follow_redirects=True) as resp:
            if resp.status_code == 200 and resp.headers.get("content-type", "").lower().startswith("image/"):
                return True
    except Exception:
        pass
    return False


async def search_web_pages(client: httpx.AsyncClient, query: str, max_results: int | None = None) -> list[dict[str, str]]:
    limit = max_results or settings.web_search_max_results
    try:
        resp = await client.get(
            f"{settings.searxng_url}/search",
            params={"q": query[:200], "categories": "general", "format": "json"},
            timeout=5.0,
        )
        resp.raise_for_status()
        raw_results: list[dict[str, Any]] = resp.json().get("results", [])[:limit]
    except Exception as exc:
        logger.warning("Web search failed for %r: %s", query, exc)
        return []

    excerpts = await asyncio.gather(*(_fetch_page_excerpt(client, str(r.get("url", ""))) for r in raw_results))
    formatted: list[dict[str, str]] = []
    for item, excerpt in zip(raw_results, excerpts):
        title = str(item.get("title") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        body = excerpt or snippet
        if body:
            formatted.append({"title": title, "url": url, "content": body})
    return formatted


async def search_verified_images(client: httpx.AsyncClient, query: str, limit: int = 3) -> list[dict[str, str]]:
    clean_q = clean_image_query(query)
    try:
        resp = await client.get(
            f"{settings.searxng_url}/search",
            params={"q": clean_q, "categories": "images", "format": "json"},
            timeout=5.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:
        logger.debug("Image search failed: %s", exc)
        return []

    verified: list[dict[str, str]] = []
    for item in results[:10]:
        if len(verified) >= limit:
            break
        for candidate in (str(item.get("img_src") or ""), str(item.get("thumbnail_src") or "")):
            if await _verify_image_url(client, candidate):
                verified.append({"title": str(item.get("title") or clean_q), "image_url": candidate, "source": str(item.get("url") or "")})
                break
    return verified


async def execute_tool_calls(
    client: httpx.AsyncClient,
    tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Execute Qwen3 tool_calls concurrently against SearXNG and return (tool_messages, verified_images)."""
    async def _run_one(tc: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            args = {}
        query = str(args.get("query", "")).strip()

        imgs: list[dict[str, str]] = []
        if name == "web_search" and query:
            pages = await search_web_pages(client, query)
            content = json.dumps({"results": pages})
        elif name == "image_search" and query:
            imgs = await search_verified_images(client, query, limit=2)
            content = json.dumps({"images": imgs})
        else:
            content = json.dumps({"error": f"Unknown tool or empty query: {name}"})

        msg = {
            "role": "tool",
            "tool_call_id": str(tc.get("id", "")),
            "name": name,
            "content": content,
        }
        return msg, imgs

    results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))
    tool_messages: list[dict[str, Any]] = []
    all_images: list[dict[str, str]] = []
    for msg, imgs in results:
        tool_messages.append(msg)
        all_images.extend(imgs)
    return tool_messages, all_images


async def gather_web_context(client: httpx.AsyncClient, user_text: str) -> tuple[str | None, list[dict[str, str]], bool]:
    """Fallback direct SearXNG lookup when model tool-calling isn't used."""
    do_web, do_img = might_need_tools(user_text)
    if not do_web and not do_img:
        return None, [], False

    sections: list[str] = []
    verified_images: list[dict[str, str]] = []
    if do_web:
        pages = await search_web_pages(client, user_text)
        if pages:
            lines = [f"[{i}] {p['title']} ({p['url']})\n{p['content']}" for i, p in enumerate(pages, start=1)]
            sections.append(
                "Live web search results (use these for up-to-date facts and cite sources inline as [1](url)):\n"
                + "\n\n".join(lines)
            )
    if do_img:
        verified_images = await search_verified_images(client, user_text, limit=3)
        if verified_images:
            img_lines = [f"- {im['title']}: {im['image_url']}" for im in verified_images]
            sections.append(
                "You CAN display images directly in chat using Markdown ![caption](url). "
                "Do NOT say you cannot display images. Embed relevant verified image URLs from this list:\n"
                + "\n".join(img_lines)
            )

    if not sections:
        return None, [], False
    return "\n\n".join(sections), verified_images, True
