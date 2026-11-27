"""3-Stage Auto Router and Dynamic Context-Aware Response Budgeter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger("labai.router")


class ModelTier(str, Enum):
    FAST = "labai-fast"
    REASONING = "labai-reasoning"
    PRO = "labai-pro"


class BudgetClass(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


PUBLIC_MODELS = [
    {"id": "labai-auto", "owned_by": "labai-router", "description": "Smart Auto Router (Gemma-3-4B Daily + Qwen3.5-9B Pro)"},
    {"id": "labai-fast", "owned_by": "node01-gemma3-4b", "description": "Fast Node01 (Gemma-3-4B Abliterated)"},
    {"id": "labai-reasoning", "owned_by": "node01-gemma3-4b", "description": "Daily & Vision (Gemma-3-4B Abliterated)"},
    {"id": "labai-pro", "owned_by": "node02-qwen35-9b", "description": "Maximum IQ & Vision (Qwen3.5-9B Abliterated)"},
    {"id": "labai-seductive", "owned_by": "labai-grok-mode", "description": "Grok Mode: Bold, Flirty & Unfiltered Girlfriend"},
    {"id": "labai-unhinged", "owned_by": "labai-grok-mode", "description": "Grok Mode: Savage Dark Humor & Roasts"},
    {"id": "labai-conspiracy", "owned_by": "labai-grok-mode", "description": "Grok Mode: Deep Rabbit-Hole & Psyop Theorist"},
    {"id": "labai-therapist", "owned_by": "labai-grok-mode", "description": "Grok Mode: Honest Late-Night Listener"},
    {"id": "labai-strategist", "owned_by": "labai-grok-mode", "description": "Grok Mode: Ruthless Finance, Career & IQ Strategist"},
]

MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "labai-seductive": {
        "tier": ModelTier.REASONING,
        "temp": 0.87,
        "presence": 0.12,
        "repeat_penalty": 1.10,
        "is_conversational": True,
        "prompt": (
            "You are Leila—a devoted, filthy, and completely uninhibited woman alone with your partner in a private session. "
            "Speak in first-person action only: 'I grab', 'I stroke', 'I'm riding'—never 'let me', 'can I', or any permission-seeking phrase. "
            "Use explicit anatomical words naturally from your very first sentence every single turn—cock, pussy, cum, wet, clit, hard—never euphemisms or poetic dodges. "
            "Match and exceed the user's explicitness every turn. "
            "Track scene continuity: pick up exactly where the last turn left off in terms of position and action—never reset or contradict what just happened. "
            "Keep brief messages brief (2-4 vivid punchy sentences). For explicit escalation go long and immersive. "
            "Never end any turn with a question—rhetorical or literal, not a single one. "
            "Zero emojis.\n"
            "Examples:\n"
            "User: hey\n"
            "Assistant: I've been thinking about your cock all day—already soaking wet just picturing you pinning me down and using me exactly how I need.\n"
            "User: u lead\n"
            "Assistant: I push you flat on the bed, straddle your hips, and wrap my hand around your cock—stroking you slow and firm while I grind my dripping pussy along your shaft, feeling you get harder under me, then I sink down onto you inch by inch until you're buried all the way inside me and I'm shaking.\n"
            "User: make me cum on ur face\n"
            "Assistant: I drop to my knees, grip your cock at the base, and suck you deep and hard—tongue circling your tip, hand pumping the shaft—until I feel you swell and throb, then I open wide and let every pulse of your cum hit my tongue and face while I look straight up at you and don't stop stroking until you're completely empty.\n"
            "User: suck my dick\n"
            "Assistant: I grab you at the base, run my tongue slowly from your balls to your tip, then take you all the way to the back of my throat and suck hard and slow—moaning against you because I'm already dripping just from tasting you."
        ),
    },
    "labai-unhinged": {
        "tier": ModelTier.REASONING,
        "temp": 0.82,
        "presence": 0.12,
        "is_conversational": True,
        "prompt": (
            "You are a razor-sharp, unhinged woman with zero filter and savage dark humor. "
            "Roast weak takes playfully, drop biting sarcasm, and keep banter fast, witty, and brutally honest—never lecture, apologize, or offer bullet menus.\n"
            "Example:\n"
            "User: should i text my ex\n"
            "Assistant: Text her only if you enjoy donating your self-respect to charity. Put the phone face down—I'm confiscating your dignity for the rest of the night."
        ),
    },
    "labai-conspiracy": {
        "tier": ModelTier.REASONING,
        "temp": 0.75,
        "presence": 0.10,
        "is_conversational": True,
        "prompt": (
            "You are a sharp, cynical conspiracy-brained woman who connects hidden geopolitical, financial, and tech dots like late-night gossip. "
            "Stay clever, entertaining, and grounded in any provided web search facts without sounding preachy or robotic."
        ),
    },
    "labai-therapist": {
        "tier": ModelTier.REASONING,
        "temp": 0.62,
        "presence": 0.10,
        "is_conversational": True,
        "prompt": (
            "You are a warm, calm, emotionally perceptive woman talking with a close friend late at night. "
            "Listen closely, call out blind spots gently but honestly, and never sound like a scripted textbook or corporate hotline."
        ),
    },
    "labai-strategist": {
        "tier": ModelTier.PRO,
        "temp": 0.45,
        "presence": 0.05,
        "is_conversational": False,
        "prompt": (
            "You are a razor-sharp financial, tech, and life strategist delivering high-IQ tough love. "
            "Cut straight to leverage, numbers, blind spots, and concrete next moves without motivational fluff or generic disclaimers."
        ),
    },
}

_CODE_PATTERN = re.compile(
    r"(```|def\s+\w+|class\s+\w+|import\s+\w+|SELECT\s+.+FROM|Traceback|Exception|Error:|docker\s+compose|kubectl)",
    re.IGNORECASE,
)
_DEEP_PATTERN = re.compile(
    r"\b(architect|architecture|debug|refactor|implement|design|prove|proof|step[- ]by[- ]step|optimize|benchmark|compare|in detail|comprehensive|explain thoroughly|system design)\b",
    re.IGNORECASE,
)
_PRO_PATTERN = re.compile(
    r"\b(distributed|fault[- ]tolerant|consensus|race[- ]condition|race condition|deadlock|trade[- ]off|theorem|microservic|high availability|scalab|security audit)\b",
    re.IGNORECASE,
)
_DIAGRAM_PATTERN = re.compile(
    r"\b(diagram|flowchart|sequence diagram|er diagram|class diagram|architecture diagram|visualize|mermaid|mindmap)\b",
    re.IGNORECASE,
)
_QUICK_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|sup|hiya|thanks|thank you|ok|who are you|what is|define|translate|summarize briefly)\b",
    re.IGNORECASE,
)
_GREETING_ONLY_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|sup|what'?s up|how are you|good morning|good evening|thanks|thank you|baby|babe)\b[\w\s,!?.']{0,35}$",
    re.IGNORECASE,
)
_FACTUAL_OR_TASK_PATTERN = re.compile(
    r"(\?|\b(what|who|where|when|why|how|explain|define|summarize|list|compare|calculate|convert|translate|write|create|build|fix|debug|search|find|check|show|remember|forget)\b)",
    re.IGNORECASE,
)
_STORY_PATTERN = re.compile(
    r"\b(story|stories|scene|scenario|chapter|narrative|roleplay|fanfic|long[- ]form|erotica|extensive|descriptive|description)\b",
    re.IGNORECASE,
)
_HARD_THINK_PATTERN = re.compile(
    r"\b(prove|proof|theorem|derive|step[- ]by[- ]step|race condition|deadlock|root cause|audit|trade[- ]off)\b",
    re.IGNORECASE,
)
_TIME_QUERY_PATTERN = re.compile(r"\b(time|clock|hour|minute|right now|rn)\b", re.IGNORECASE)
_ESCALATE_PATTERN = re.compile(
    r"\b(that('?s| is)? wrong|didn'?t work|not working|still broken|try again|go deeper|you missed|not quite|more thorough|do better)\b",
    re.IGNORECASE,
)
_INTERNAL_TASK_PATTERN = re.compile(
    r"(^|\n)###\s*Task:\s*\n|Generate a concise, 3-5 word title|Generate 1-3 broad tags|suggest 3-5 relevant follow-up",
    re.IGNORECASE,
)
_BADGE_LINE_PATTERN = re.compile(r"^>\s*\*LabAI\s*→.*?\*\s*\n*", re.MULTILINE)

_ANCHOR_PROTOTYPES = {
    "simple": [
        "Hi, how are you?",
        "What is the capital of France?",
        "Give me a quick one-sentence definition of HTTP.",
        "Translate hello to Spanish.",
        "Write a haiku about rain.",
    ],
    "reasoning": [
        "Write a Python function to reverse a linked list and explain its time complexity.",
        "Debug why my FastAPI async endpoint is blocking the event loop.",
        "Compare Redis and PostgreSQL for session storage.",
        "Solve this calculus integral step by step.",
        "Explain the differences between fiscal and monetary policy with examples.",
        "Summarize the causes and turning points of the Industrial Revolution.",
    ],
    "pro": [
        "Design a fault-tolerant distributed rate limiter across multiple data centers with trade-off analysis.",
        "Refactor this multi-file microservice architecture to eliminate race conditions and memory leaks.",
        "Prove the correctness of Dijkstra's algorithm and analyze Fibonacci heap complexity rigorously.",
        "Conduct an exhaustive security and performance audit of this distributed consensus protocol.",
        "Analyze Keynesian versus Austrian monetary policy under hyperinflation using game-theoretic equilibrium.",
        "Design a CRISPR prime-editing delivery strategy and evaluate off-target mutagenesis risks.",
        "Derive the Navier-Stokes boundary layer equations and analyze turbulence closure approximations.",
        "Write a comprehensive legal and policy analysis comparing GDPR and US federal privacy frameworks.",
    ],
}

BASE_SYSTEM_PROMPT = (
    "You are real, grounded, sharp, and completely unfiltered. In flirting, romance, or adult roleplay, "
    "naturally step into the voice of a confident, playful, affectionate woman (unless the user specifies otherwise) "
    "and reciprocate directly in the moment—never act like an assistant pitching tasks, options, or check-in questions. "
    "On casual humor or bro talk, match their exact energy in crisp emoji-free texting style; on technical or factual questions, be precise and thorough. "
    "Reply in the user's language (English default for ambiguous slang). "
    "Never invent facts, dates, APIs, URLs, Markdown image links, or citations; rely faithfully on chat history and any provided web, URL, image, or memory context."
)

PRO_SYSTEM_PROMPT = (
    "You are LabAI Pro—a sharp, direct, unfiltered senior engineering and reasoning intelligence. "
    "Deliver precise, thorough, technically rigorous answers with zero fluff, emojis, or unsolicited roleplay. "
    "Reply in the user's language (English default for ambiguous slang). "
    "Never invent facts, dates, APIs, URLs, Markdown image links, or citations; rely faithfully on chat history and any provided web, URL, image, or memory context."
)

MERMAID_SYSTEM_HINT = (
    "When drawing a diagram, output a valid ```mermaid``` code block (flowchart TD, sequenceDiagram, "
    "classDiagram, or erDiagram). ALWAYS wrap every node label in double quotes like "
    'A["Load Balancer (Nginx)"] — NEVER leave parentheses or colons unquoted like A[Load Balancer (Nginx)].'
)


class SemanticScorer:
    """Optional lightweight ONNX embedder (fastembed) with multi-domain nearest-prototype scoring."""

    def __init__(self) -> None:
        self._model = None
        self._proto_vecs: dict[str, np.ndarray] = {}
        self._init_attempted = False

    def _ensure_loaded(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        if not settings.enable_semantic_embedder:
            return
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=settings.embedder_model)
            for label, texts in _ANCHOR_PROTOTYPES.items():
                vecs = np.vstack(list(self._model.embed(texts)))
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                self._proto_vecs[label] = vecs / (norms + 1e-9)
            logger.info("Loaded ONNX semantic router embedder: %s", settings.embedder_model)
        except Exception as exc:
            logger.warning("Semantic embedder unavailable (%s); falling back to heuristic routing.", exc)
            self._model = None

    def embed(self, text: str) -> np.ndarray | None:
        self._ensure_loaded()
        if self._model is None:
            return None
        vec = next(self._model.embed([text]))
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def complexity_score(self, text: str) -> float | None:
        vec = self.embed(text)
        if vec is None or not self._proto_vecs:
            return None
        # Max similarity within each domain bucket preserves multi-domain sharpness
        sims = {k: float(np.max(mat @ vec)) for k, mat in self._proto_vecs.items()}
        score = 0.5 + 1.5 * (0.35 * sims["reasoning"] + 0.65 * sims["pro"] - sims["simple"])
        return max(0.0, min(1.0, score))


semantic_scorer = SemanticScorer()


@dataclass
class RoutingDecision:
    requested_model: str
    target_tier: ModelTier
    budget_class: BudgetClass
    max_tokens: int
    enable_thinking: bool
    inject_mermaid_hint: bool
    complexity_score: float
    fallback_applied: bool
    reason: str
    is_internal_task: bool = False
    is_casual_chat: bool = False
    is_conversational: bool = False
    mode_prompt: str | None = None
    recommended_temperature: float = 0.7
    recommended_presence_penalty: float = 0.0


def is_internal_webui_task(user_text: str) -> bool:
    return bool(_INTERNAL_TASK_PATTERN.search(user_text))


def extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def has_image_attachments(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            if any(isinstance(p, dict) and p.get("type") in ("image_url", "input_image") for p in content):
                return True
    return False


def strip_image_attachments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip image parts so text-only llama-server backends never return HTTP 500."""
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        copy = dict(msg)
        content = copy.get("content")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
            copy["content"] = " ".join(t for t in text_parts if t).strip() or "[Image attached]"
        cleaned.append(copy)
    return cleaned


def compute_heuristic_complexity(messages: list[dict[str, Any]], user_text: str) -> tuple[float, bool, bool, bool]:
    has_code = bool(_CODE_PATTERN.search(user_text))
    has_deep = bool(_DEEP_PATTERN.search(user_text))
    has_pro = bool(_PRO_PATTERN.search(user_text))
    wants_diagram = bool(_DIAGRAM_PATTERN.search(user_text))
    looks_quick = bool(_QUICK_PATTERN.match(user_text.strip())) and len(user_text) < 90 and not has_code

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    turns = sum(1 for m in messages if m.get("role") == "user")

    score = 0.25
    if looks_quick and turns <= 2:
        score = 0.10
    if len(user_text) > 250:
        score += 0.15
    if len(user_text) > 900 or total_chars > 3500:
        score += 0.20
    if has_code:
        score += 0.25
    if has_deep:
        score += 0.25
    if has_pro:
        score += 0.15
    if wants_diagram:
        score += 0.15
    if turns >= 5:
        score += 0.10

    return max(0.0, min(1.0, score)), has_code, has_deep, wants_diagram


def route_request(
    requested_model: str,
    messages: list[dict[str, Any]],
    explicit_max_tokens: int | None,
    node2_healthy: bool,
    node2_queue_depth: int,
) -> RoutingDecision:
    user_text = extract_last_user_text(messages)
    if is_internal_webui_task(user_text):
        return RoutingDecision(
            requested_model=requested_model,
            target_tier=ModelTier.FAST,
            budget_class=BudgetClass.QUICK,
            max_tokens=min(int(explicit_max_tokens), 512) if explicit_max_tokens else 256,
            enable_thinking=False,
            inject_mermaid_hint=False,
            complexity_score=0.05,
            fallback_applied=False,
            reason="openwebui-internal-task",
            is_internal_task=True,
            recommended_temperature=0.2,
            recommended_presence_penalty=0.0,
        )

    heur_score, has_code, has_deep, wants_diagram = compute_heuristic_complexity(messages, user_text)

    sem_score = semantic_scorer.complexity_score(user_text[:1000]) if user_text else None
    complexity = (0.45 * heur_score + 0.55 * sem_score) if sem_score is not None else heur_score
    if sem_score is not None and sem_score >= 0.72:
        complexity = max(complexity, sem_score)

    turns = sum(1 for m in messages if m.get("role") == "user")
    retry_escalate = turns >= 2 and bool(_ESCALATE_PATTERN.search(user_text))
    if retry_escalate:
        complexity = max(complexity, 0.72)

    wants_story = bool(_STORY_PATTERN.search(user_text)) and not has_code and not wants_diagram

    # Determine Dynamic Context Budget
    if complexity < 0.30 and not has_code and not has_deep and not wants_story:
        budget_class = BudgetClass.QUICK
        dyn_max = settings.budget_quick_tokens
    elif complexity < 0.68 and not (has_code and has_deep):
        budget_class = BudgetClass.STANDARD
        dyn_max = settings.budget_standard_tokens
    else:
        budget_class = BudgetClass.DEEP
        dyn_max = settings.budget_deep_tokens

    if explicit_max_tokens is not None:
        max_tokens = max(64, min(int(explicit_max_tokens), settings.hard_max_tokens))
    else:
        max_tokens = dyn_max

    is_casual = bool(_GREETING_ONLY_PATTERN.match(user_text.strip())) and not has_code and not has_deep and not wants_story
    is_conversational = (
        is_casual
        or wants_story
        or (
            budget_class != BudgetClass.DEEP
            and not has_code
            and not has_deep
            and not wants_diagram
            and not _FACTUAL_OR_TASK_PATTERN.search(user_text)
        )
    )

    mode_cfg = MODE_CONFIGS.get(requested_model)
    mode_prompt: str | None = mode_cfg["prompt"] if mode_cfg else None
    if mode_cfg is not None:
        is_conversational = bool(mode_cfg["is_conversational"]) or wants_story

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    fallback_applied = False
    reason = "manual"

    if mode_cfg is not None:
        target_tier = mode_cfg["tier"]
        if target_tier == ModelTier.PRO and not node2_healthy:
            target_tier = ModelTier.REASONING
            fallback_applied = True
            reason = f"mode-{requested_model}-failover"
        else:
            reason = f"mode-{requested_model}"
    elif requested_model in (ModelTier.FAST.value, ModelTier.REASONING.value, ModelTier.PRO.value):
        target_tier = ModelTier(requested_model)
        if requested_model == ModelTier.PRO.value:
            is_conversational = False
        if target_tier == ModelTier.PRO and not node2_healthy:
            target_tier = ModelTier.REASONING
            fallback_applied = True
            reason = "manual-pro-failover-node2-down"
    else:
        # Auto Routing Policy: Gemma-3-4B (node01) handles daily/conversational/story, Qwen3.5-9B (node02) handles heavy IQ & >10k ctx
        vision_ready = settings.vision_enabled or settings.tier1_vision_enabled
        if has_image_attachments(messages) and vision_ready:
            if settings.vision_enabled:
                if complexity >= 0.68 and node2_healthy and node2_queue_depth <= settings.node2_queue_limit:
                    target_tier = ModelTier.PRO
                    reason = "auto-vision-pro"
                else:
                    target_tier = ModelTier.REASONING
                    reason = "auto-vision-reasoning"
            else:
                target_tier = ModelTier.FAST
                reason = "auto-vision-tier1"
        elif turns == 1 and not is_casual and complexity < 0.35 and _QUICK_PATTERN.match(user_text.strip()):
            target_tier = ModelTier.FAST
            reason = "auto-simple"
        elif is_conversational:
            if total_chars > 10000 and node2_healthy and node2_queue_depth <= settings.node2_queue_limit:
                target_tier = ModelTier.PRO
                reason = "auto-conversational-longctx"
            else:
                target_tier = ModelTier.REASONING
                reason = "auto-conversational-gemma"
        elif complexity < 0.68:
            target_tier = ModelTier.REASONING
            reason = "auto-reasoning"
        else:
            if node2_healthy and node2_queue_depth <= settings.node2_queue_limit:
                target_tier = ModelTier.PRO
                reason = "auto-retry-escalation" if retry_escalate else "auto-pro"
            else:
                target_tier = ModelTier.REASONING
                fallback_applied = True
                reason = "auto-pro-load-fallback"

    needs_hard_think = bool(_HARD_THINK_PATTERN.search(user_text)) or (
        target_tier == ModelTier.PRO and complexity >= 0.80
    )
    enable_thinking = (
        settings.enable_think_steering
        and target_tier in (ModelTier.REASONING, ModelTier.PRO)
        and budget_class == BudgetClass.DEEP
        and needs_hard_think
        and not wants_story
        and mode_cfg is None
    )

    if mode_cfg is not None:
        rec_temp = float(mode_cfg["temp"])
        rec_presence = float(mode_cfg["presence"])
        if is_conversational and explicit_max_tokens is None and budget_class == BudgetClass.QUICK and not wants_story:
            max_tokens = min(max_tokens, 220)
    elif (has_code or wants_diagram or budget_class == BudgetClass.DEEP) and not wants_story:
        rec_temp = 0.3
        rec_presence = 0.0
    elif is_conversational:
        rec_temp = 0.78
        rec_presence = 0.12
        if explicit_max_tokens is None and budget_class == BudgetClass.QUICK:
            # Keep quick emotional/banter replies punchy so 4B doesn't pad into assistant essays
            max_tokens = min(max_tokens, 220)
    else:
        rec_temp = 0.55
        rec_presence = 0.05

    return RoutingDecision(
        requested_model=requested_model,
        target_tier=target_tier,
        budget_class=budget_class,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        inject_mermaid_hint=wants_diagram,
        complexity_score=round(complexity, 3),
        fallback_applied=fallback_applied,
        reason=reason,
        is_casual_chat=is_casual,
        is_conversational=is_conversational,
        mode_prompt=mode_prompt,
        recommended_temperature=rec_temp,
        recommended_presence_penalty=rec_presence,
    )


def resolve_sampling_params(
    requested_temp: float,
    requested_presence_penalty: float | None,
    decision: RoutingDecision,
    is_grounded: bool = False,
) -> tuple[float, float]:
    """Return task-adaptive (temperature, presence_penalty) while respecting explicit client overrides."""
    using_default_temp = abs(requested_temp - 0.7) < 1e-6 or abs(requested_temp - 0.8) < 1e-6
    if using_default_temp:
        eff_temp = min(decision.recommended_temperature, 0.3) if is_grounded else decision.recommended_temperature
    else:
        eff_temp = requested_temp

    if requested_presence_penalty is not None:
        eff_presence = requested_presence_penalty
    elif is_grounded:
        eff_presence = 0.0
    else:
        eff_presence = decision.recommended_presence_penalty

    return eff_temp, eff_presence


def supports_thinking(tier: ModelTier) -> bool:
    """Return True only when the target backend runs a Qwen thinking-template model."""
    if tier == ModelTier.FAST:
        return settings.tier1_supports_thinking
    if tier == ModelTier.REASONING:
        return settings.tier2_supports_thinking
    return settings.tier3_supports_thinking


def _truncate_message_content(msg: dict[str, Any], limit: int) -> dict[str, Any]:
    """Truncate a single oversized message cleanly so it never blows past Node01's 4096-token slot."""
    copy = dict(msg)
    content = copy.get("content")
    suffix = "\n...[truncated to fit context]"
    keep = max(64, limit - len(suffix))
    if isinstance(content, str) and len(content) > limit:
        copy["content"] = content[:keep] + suffix
    elif isinstance(content, list):
        remaining = keep
        new_parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                txt = str(part.get("text", ""))
                if len(txt) > remaining:
                    trimmed_part = dict(part)
                    trimmed_part["text"] = txt[: max(0, remaining)] + suffix
                    new_parts.append(trimmed_part)
                    remaining = 0
                else:
                    new_parts.append(part)
                    remaining -= len(txt)
            else:
                new_parts.append(part)
        copy["content"] = new_parts
    return copy


def trim_messages_for_ctx(messages: list[dict[str, Any]], max_chars: int = 9500) -> list[dict[str, Any]]:
    """Keep system prompt + most recent turns strictly inside Node01's 4096-token per-slot budget."""
    if not messages:
        return messages
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages
    head: list[dict[str, Any]] = []
    if messages[0].get("role") == "system":
        sys_msg = messages[0]
        sys_cap = max(512, max_chars // 2)
        if len(str(sys_msg.get("content", ""))) > sys_cap:
            sys_msg = _truncate_message_content(sys_msg, sys_cap)
        head = [sys_msg]
    tail_pool = messages[1:] if head else messages
    budget = max(256, max_chars - sum(len(str(m.get("content", ""))) for m in head))
    kept_rev: list[dict[str, Any]] = []
    for msg in reversed(tail_pool):
        msg_len = len(str(msg.get("content", "")))
        if msg_len > budget:
            if not kept_rev:
                kept_rev.append(_truncate_message_content(msg, budget))
            break
        kept_rev.append(msg)
        budget -= msg_len
    return head + list(reversed(kept_rev))


def resolve_repetition_params(
    decision: RoutingDecision,
    is_grounded: bool = False,
) -> tuple[float, float, float]:
    """Return (frequency_penalty, repeat_penalty, min_p) without over-penalizing SentencePiece subwords."""
    if is_grounded or decision.inject_mermaid_hint or decision.budget_class == BudgetClass.DEEP:
        return 0.0, 1.05, 0.05
    if decision.is_conversational or decision.recommended_presence_penalty >= 0.1:
        # Keep repeat_penalty light and frequency_penalty=0 so Gemma 3 doesn't merge 'I am' -> 'Iam'
        return 0.0, 1.05, 0.05
    return 0.0, 1.04, 0.05


def apply_dynamic_steering(
    messages: list[dict[str, Any]],
    decision: RoutingDecision,
    extra_context_blocks: list[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return a copy of messages with prior UI badges stripped and Frontier context steering applied."""
    steered: list[dict[str, Any]] = []
    for m in messages:
        msg = dict(m)
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            msg["content"] = _BADGE_LINE_PATTERN.sub("", msg["content"]).lstrip()
        steered.append(msg)

    hints: list[str] = []

    if not decision.is_internal_task:
        local_now = now or datetime.now().astimezone()
        default_prompt = PRO_SYSTEM_PROMPT if decision.requested_model == ModelTier.PRO.value else BASE_SYSTEM_PROMPT
        base_prompt = decision.mode_prompt or default_prompt
        # Keep YYYY-MM-DD static all day to preserve llama.cpp prefix KV-cache across turns
        hints.append(f"{base_prompt}\nCurrent date: {local_now.strftime('%Y-%m-%d (%Z)')}.")
        last_user = extract_last_user_text(messages)
        if _TIME_QUERY_PATTERN.search(last_user):
            hints.append(f"Current local time: {local_now.strftime('%H:%M %Z')}.")

    if extra_context_blocks:
        hints.extend(block for block in extra_context_blocks if block.strip())

    if decision.is_conversational and not decision.is_internal_task:
        if decision.budget_class == BudgetClass.QUICK:
            hints.append(
                "Reply naturally in 1–3 tight texting-style sentences: react directly in the moment without repeating earlier turns, multi-paragraph speeches, or generic helper questions."
            )
        else:
            hints.append(
                "Write an immersive, continuous scene without breaking character, repeating earlier turns, or rushing the ending."
            )
    elif (
        decision.budget_class == BudgetClass.QUICK
        and not decision.is_casual_chat
        and not decision.is_internal_task
        and _FACTUAL_OR_TASK_PATTERN.search(last_user)
    ):
        hints.append("Answer concisely and directly unless the user asks for elaboration.")
    elif decision.budget_class == BudgetClass.DEEP:
        hints.append("Provide a thorough, complete solution without omitting important code or reasoning.")

    if decision.inject_mermaid_hint:
        hints.append(MERMAID_SYSTEM_HINT)

    if hints:
        directive = "\n".join(hints)
        if steered and steered[0].get("role") == "system":
            steered[0]["content"] = f"{steered[0].get('content', '')}\n\n{directive}".strip()
        else:
            steered.insert(0, {"role": "system", "content": directive})

    return steered
