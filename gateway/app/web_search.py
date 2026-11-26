"""Auto-triggered live internet search, webpage extraction, and verified image lookup via SearXNG."""

from __future__ import annotations

import asyncio
import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("labai.web_search")

_WEB_INTENT_PATTERN = re.compile(
    r"\b(today|yesterday|tonight|this week|this month|this year|latest|newest|current|recent|currently|"
    r"news|price|weather|forecast|score|standings|stock|release date|version of|who won|what happened|"
    r"search the web|search online|look up|browse|find online|2026|2027)\b",
    re.IGNORECASE,
)
_IMAGE_INTENT_PATTERN = re.compile(
    r"\b(show me (a |an )?(photo|picture|image|pic)|find (a |an )?(photo|picture|image)|"
    r"what does .+ look like|photos of|pictures of|images of)\b",
    re.IGNORECASE,
)
_DIAGRAM_EXCLUSION = re.compile(
    r"\b(diagram|flowchart|sequence diagram|class diagram|er diagram|mermaid|mindmap)\b",
    re.IGNORECASE,
)


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


def needs_web_search(user_text: str) -> bool:
    if not settings.enable_auto_web_search or not user_text or len(user_text.strip()) < 6:
        return False
    return bool(_WEB_INTENT_PATTERN.search(user_text))


def needs_image_search(user_text: str) -> bool:
    if not settings.enable_auto_web_search or not user_text:
        return False
    if _DIAGRAM_EXCLUSION.search(user_text):
        return False
    return bool(_IMAGE_INTENT_PATTERN.search(user_text))


async def _fetch_page_excerpt(client: httpx.AsyncClient, url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        resp = await client.get(url, timeout=2.5, follow_redirects=True)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "text/html" in ctype:
            return extract_clean_text(resp.text, max_chars=900)
    except Exception:
        return ""
    return ""


async def search_verified_images(client: httpx.AsyncClient, query: str, limit: int = 3) -> list[dict[str, str]]:
    try:
        resp = await client.get(
            f"{settings.searxng_url}/search",
            params={"q": query, "categories": "images", "format": "json"},
            timeout=5.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:
        logger.debug("Image search failed: %s", exc)
        return []

    verified: list[dict[str, str]] = []
    for item in results:
        if len(verified) >= limit:
            break
        img_url = item.get("img_src") or item.get("url")
        if not img_url or not str(img_url).startswith(("http://", "https://")):
            continue
        try:
            head = await client.head(img_url, timeout=2.0, follow_redirects=True)
            ctype = head.headers.get("content-type", "")
            if head.status_code == 200 and ctype.startswith("image/"):
                verified.append({"title": str(item.get("title") or query), "image_url": str(img_url), "source": str(item.get("url") or "")})
        except Exception:
            continue
    return verified


async def gather_web_context(client: httpx.AsyncClient, user_text: str) -> tuple[str | None, bool]:
    """Run SearXNG web and/or image lookup if triggered; returns (context_block, used_web)."""
    do_web = needs_web_search(user_text)
    do_img = needs_image_search(user_text)
    if not do_web and not do_img:
        return None, False

    sections: list[str] = []
    if do_web:
        try:
            resp = await client.get(
                f"{settings.searxng_url}/search",
                params={"q": user_text[:200], "categories": "general", "format": "json"},
                timeout=5.0,
            )
            resp.raise_for_status()
            raw_results: list[dict[str, Any]] = resp.json().get("results", [])[: settings.web_search_max_results]
            excerpts = await asyncio.gather(*(_fetch_page_excerpt(client, str(r.get("url", ""))) for r in raw_results))
            lines: list[str] = []
            for idx, (item, excerpt) in enumerate(zip(raw_results, excerpts), start=1):
                title = str(item.get("title") or "Untitled").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("content") or "").strip()
                body = excerpt or snippet
                if body:
                    lines.append(f"[{idx}] {title} ({url})\n{body}")
            if lines:
                sections.append(
                    "Live web search results (use these for up-to-date facts and cite sources inline as [1](url)):\n"
                    + "\n\n".join(lines)
                )
        except Exception as exc:
            logger.warning("Auto web search failed: %s", exc)

    if do_img:
        images = await search_verified_images(client, user_text[:150], limit=2)
        if images:
            img_lines = [f"- {im['title']}: {im['image_url']}" for im in images]
            sections.append(
                "Verified image URLs you may embed directly using Markdown ![caption](url):\n" + "\n".join(img_lines)
            )

    if not sections:
        return None, False
    return "\n\n".join(sections), True
