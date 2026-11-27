"""Semantic + heuristic intent detection, SearXNG web/image search, and native Qwen3 tool execution."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

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
    r"\b((?:show|send|give|find|drop)\s+(?:me\s+)?(?:another|more|a|an|the)?\s*(?:[\w'-]+\s+){0,3}?(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces|snap)|"
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
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)]+", re.IGNORECASE)
_MD_IMAGE_URL_PATTERN = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_INTERNAL_HOSTS = {
    "localhost",
    "node01",
    "node02",
    "gateway",
    "redis",
    "searxng",
    "open-webui",
    "prometheus",
    "grafana",
    "caddy",
    "host.docker.internal",
}

_IMAGE_LEAD_VERB = re.compile(
    r"^(?:please\s+)?(?:show|send|give|find|get|drop)\s+(?:me\s+)?(?:another|more|one\s+more)?\s*",
    re.IGNORECASE,
)
_IMAGE_PREFIX_STRIP = re.compile(
    r"^(?:another|more|a|an|the)?\s*(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces|snap)\s*(?:of\s+)?",
    re.IGNORECASE,
)
_IMAGE_SUFFIX_STRIP = re.compile(
    r"\s+(?:photo|photos|picture|pictures|image|images|pic|pics|face|faces|snap)$",
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
        web_hit = sims["live_web"] >= 0.58 and sims["live_web"] > sims["static_chat"] + 0.08
        img_hit = sims["image_lookup"] >= 0.66 and sims["image_lookup"] > sims["static_chat"] + 0.14
        return web_hit, img_hit


semantic_intent = _SemanticIntentDetector()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._meta_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "meta":
            attr_map = {k.lower(): (v or "") for k, v in attrs}
            key = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = attr_map.get("content", "").strip()
            if key in {"og:title", "og:description", "description", "twitter:title", "twitter:description"} and content:
                if content not in self._meta_parts:
                    self._meta_parts.append(content)
        if lower in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
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
        body = re.sub(r"\s+", " ", " ".join(self._chunks)).strip()
        if len(body) >= 120:
            return body
        meta = " — ".join(self._meta_parts).strip()
        return f"{meta}\n{body}".strip() if meta else body


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
    cleaned = _IMAGE_LEAD_VERB.sub("", user_text.strip())
    cleaned = _IMAGE_PREFIX_STRIP.sub("", cleaned)
    cleaned = _IMAGE_SUFFIX_STRIP.sub("", cleaned)
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned[:150].strip() or user_text[:150].strip()


async def _fetch_page_excerpt(client: httpx.AsyncClient, url: str, max_chars: int = 900, timeout: float = 2.5) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        resp = await client.get(url, headers=_BROWSER_HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        ctype = resp.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            return extract_clean_text(resp.text, max_chars=max_chars)
        if "text/plain" in ctype or "markdown" in ctype or "json" in ctype:
            return resp.text.strip()[:max_chars]
    except Exception:
        return ""
    return ""


def is_safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in _INTERNAL_HOSTS or host.endswith((".local", ".internal")):
        return False
    try:
        addr = ipaddress.ip_address(host)
        return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved)
    except ValueError:
        pass
    try:
        for info in socket.getaddrinfo(host, None):
            ip_str = info[4][0]
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved:
                return False
    except Exception:
        # Allow DNS failure here; HTTP client will fail cleanly if unreachable
        return True
    return True


def extract_urls(text: str, limit: int = 2) -> list[str]:
    found: list[str] = []
    for raw in _URL_PATTERN.findall(text):
        cleaned = raw.rstrip(".,;:!?")
        if cleaned not in found and is_safe_public_url(cleaned):
            found.append(cleaned)
        if len(found) >= limit:
            break
    return found


def _special_url_candidates(url: str) -> list[str]:
    """Return friendly raw/oEmbed endpoints for GitHub and public X/Twitter status URLs."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"} and "/status/" in parsed.path:
        from urllib.parse import quote
        return [f"https://publish.twitter.com/oembed?url={quote(url, safe='')}"]
    if host != "github.com":
        return []
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 4 and parts[2] == "blob":
        owner, repo, _, branch = parts[:4]
        rest = "/".join(parts[4:])
        return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}"]
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1]
        if len(parts) >= 4 and parts[2] == "tree":
            branch = parts[3]
            sub = "/".join(parts[4:])
            prefix = f"{sub}/" if sub else ""
            return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{prefix}README.md"]
        return [f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"]
    return []


async def fetch_prompt_urls(client: httpx.AsyncClient, user_text: str) -> str | None:
    urls = extract_urls(user_text, limit=2)
    if not urls:
        return None

    async def _fetch_one(url: str) -> str | None:
        targets = _special_url_candidates(url) + [url]
        chunks: list[str] = []
        for target in targets:
            text = await _fetch_page_excerpt(client, target, max_chars=2400, timeout=4.0)
            if text:
                if "publish.twitter.com/oembed" in target:
                    try:
                        oembed = json.loads(text)
                        author = oembed.get("author_name", "")
                        tweet_text = extract_clean_text(str(oembed.get("html", "")), max_chars=1500)
                        text = f"Post by {author}: {tweet_text}".strip()
                    except Exception:
                        pass
                chunks.append(f"[{target}]\n{text}")
                if len("\n".join(chunks)) >= 2800:
                    break
        if chunks:
            return "\n\n".join(chunks)

        # Fallback to SearXNG indexed snippets if site blocked direct HTTP (login wall / JS-only)
        pages = await search_web_pages(client, url, max_results=2)
        if pages:
            snippets = "\n".join(f"- {p['title']} ({p['url']}): {p['content']}" for p in pages)
            return f"[Indexed search snippets for {url}]\n{snippets}"
        return f"[{url}] Direct fetch blocked by site (login wall, JS-only, or anti-bot protection)."

    fetched = [c for c in await asyncio.gather(*(_fetch_one(u) for u in urls)) if c]
    if not fetched:
        return None
    return (
        "Live content fetched from the URL(s) pasted by the user "
        "(use this to analyze/answer; if a site blocked direct access, state that honestly instead of guessing):\n\n"
        + "\n\n---\n\n".join(fetched)
    )


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


def extract_prior_image_urls(messages: list[dict[str, Any]]) -> set[str]:
    """Collect image URLs already shown in this conversation so repeat requests fetch fresh ones."""
    seen: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            seen.update(_MD_IMAGE_URL_PATTERN.findall(content))
    return seen


async def _pick_first_valid_candidate(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    clean_q: str,
    exclude_urls: set[str] | None = None,
) -> dict[str, str] | None:
    for candidate in (str(item.get("img_src") or ""), str(item.get("thumbnail_src") or "")):
        if not candidate or (exclude_urls and candidate in exclude_urls):
            continue
        if await _verify_image_url(client, candidate):
            return {"title": str(item.get("title") or clean_q), "image_url": candidate, "source": str(item.get("url") or "")}
    return None


async def search_verified_images(
    client: httpx.AsyncClient,
    query: str,
    limit: int = 3,
    exclude_urls: set[str] | None = None,
) -> list[dict[str, str]]:
    clean_q = clean_image_query(query)
    if not clean_q:
        return []

    collected: list[dict[str, str]] = []
    seen_here: set[str] = set(exclude_urls or ())
    max_pages = 3 if exclude_urls else 1

    for pageno in range(1, max_pages + 1):
        try:
            resp = await client.get(
                f"{settings.searxng_url}/search",
                params={
                    "q": clean_q,
                    "categories": "images",
                    "format": "json",
                    "safesearch": "0",
                    "pageno": str(pageno),
                },
                timeout=4.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:16]
        except Exception as exc:
            logger.debug("Image search failed (page %d): %s", pageno, exc)
            break

        if not results:
            break

        checked = await asyncio.gather(
            *(_pick_first_valid_candidate(client, item, clean_q, exclude_urls=seen_here) for item in results)
        )
        for c in checked:
            if c is None or c["image_url"] in seen_here:
                continue
            seen_here.add(c["image_url"])
            collected.append(c)
            if len(collected) >= limit:
                return collected

    return collected


async def execute_tool_calls(
    client: httpx.AsyncClient,
    tool_calls: list[dict[str, Any]],
    exclude_urls: set[str] | None = None,
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
            if exclude_urls:
                imgs = await search_verified_images(client, query, limit=2, exclude_urls=exclude_urls)
            else:
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


async def gather_web_context(
    client: httpx.AsyncClient,
    user_text: str,
    exclude_urls: set[str] | None = None,
) -> tuple[str | None, list[dict[str, str]], bool]:
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
        if exclude_urls:
            verified_images = await search_verified_images(client, user_text, limit=3, exclude_urls=exclude_urls)
        else:
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
