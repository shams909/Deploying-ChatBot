"""3-Stage Auto Router and Dynamic Context-Aware Response Budgeter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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
    {"id": "labai-auto", "owned_by": "labai-router", "description": "Smart Auto Router (Fast / Reasoning / Pro)"},
    {"id": "labai-fast", "owned_by": "node01-qwen3-4b", "description": "Fast Everyday Chat (Qwen3-4B)"},
    {"id": "labai-reasoning", "owned_by": "node01-qwen3-8b", "description": "Coding & Reasoning (Qwen3-8B)"},
    {"id": "labai-pro", "owned_by": "node02-qwen3-14b", "description": "Maximum Quality (Qwen3-14B)"},
]

_CODE_PATTERN = re.compile(
    r"(```|def\s+\w+|class\s+\w+|import\s+\w+|SELECT\s+.+FROM|Traceback|Exception|Error:|docker\s+compose|kubectl)",
    re.IGNORECASE,
)
_DEEP_PATTERN = re.compile(
    r"\b(architect|architecture|debug|refactor|implement|design|prove|proof|step[- ]by[- ]step|optimize|benchmark|compare|in detail|comprehensive|explain thoroughly|system design)\b",
    re.IGNORECASE,
)
_DIAGRAM_PATTERN = re.compile(
    r"\b(diagram|flowchart|sequence diagram|er diagram|class diagram|architecture diagram|visualize|mermaid|mindmap)\b",
    re.IGNORECASE,
)
_QUICK_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|who are you|what is|define|translate|summarize briefly)\b",
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
    ],
    "reasoning": [
        "Write a Python function to reverse a linked list and explain its time complexity.",
        "Debug why my FastAPI async endpoint is blocking the event loop.",
        "Compare Redis and PostgreSQL for session storage.",
        "Solve this calculus integral step by step.",
    ],
    "pro": [
        "Design a fault-tolerant distributed rate limiter across multiple data centers with trade-off analysis.",
        "Refactor this multi-file microservice architecture to eliminate race conditions and memory leaks.",
        "Prove the correctness of Dijkstra's algorithm and analyze Fibonacci heap complexity rigorously.",
        "Conduct an exhaustive security and performance audit of this distributed consensus protocol.",
    ],
}

MERMAID_SYSTEM_HINT = (
    "When drawing a diagram, output a valid ```mermaid``` code block (flowchart TD, sequenceDiagram, "
    "classDiagram, or erDiagram). Quote node labels containing parentheses or brackets."
)


class SemanticScorer:
    """Optional lightweight ONNX embedder (fastembed) with safe offline fallback."""

    def __init__(self) -> None:
        self._model = None
        self._centroids: dict[str, np.ndarray] = {}
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
                vecs = list(self._model.embed(texts))
                centroid = np.mean(np.vstack(vecs), axis=0)
                norm = np.linalg.norm(centroid)
                self._centroids[label] = centroid / (norm + 1e-9)
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
        if vec is None or not self._centroids:
            return None
        sims = {k: float(np.dot(vec, c)) for k, c in self._centroids.items()}
        # Map relative affinity toward reasoning/pro vs simple into [0, 1]
        score = 0.5 + 0.8 * (0.4 * sims["reasoning"] + 0.6 * sims["pro"] - sims["simple"])
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


def compute_heuristic_complexity(messages: list[dict[str, Any]], user_text: str) -> tuple[float, bool, bool, bool]:
    has_code = bool(_CODE_PATTERN.search(user_text))
    has_deep = bool(_DEEP_PATTERN.search(user_text))
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
        )

    heur_score, has_code, has_deep, wants_diagram = compute_heuristic_complexity(messages, user_text)

    sem_score = semantic_scorer.complexity_score(user_text[:1000]) if user_text else None
    complexity = (0.6 * heur_score + 0.4 * sem_score) if sem_score is not None else heur_score

    # Determine Dynamic Context Budget
    if complexity < 0.30 and not has_code and not has_deep:
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

    fallback_applied = False
    reason = "manual"

    if requested_model in (ModelTier.FAST.value, ModelTier.REASONING.value, ModelTier.PRO.value):
        target_tier = ModelTier(requested_model)
        if target_tier == ModelTier.PRO and not node2_healthy:
            target_tier = ModelTier.REASONING
            fallback_applied = True
            reason = "manual-pro-failover-node2-down"
    else:
        # Auto Routing Policy
        if complexity < 0.35:
            target_tier = ModelTier.FAST
            reason = "auto-simple"
        elif complexity < 0.72:
            target_tier = ModelTier.REASONING
            reason = "auto-reasoning"
        else:
            if node2_healthy and node2_queue_depth <= settings.node2_queue_limit:
                target_tier = ModelTier.PRO
                reason = "auto-pro"
            else:
                target_tier = ModelTier.REASONING
                fallback_applied = True
                reason = "auto-pro-load-fallback"

    enable_thinking = (
        settings.enable_think_steering
        and target_tier in (ModelTier.REASONING, ModelTier.PRO)
        and budget_class == BudgetClass.DEEP
    )

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
    )


def apply_dynamic_steering(messages: list[dict[str, Any]], decision: RoutingDecision) -> list[dict[str, Any]]:
    """Return a copy of messages with prior UI badges stripped and dynamic steering applied."""
    steered: list[dict[str, Any]] = []
    for m in messages:
        msg = dict(m)
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            msg["content"] = _BADGE_LINE_PATTERN.sub("", msg["content"]).lstrip()
        steered.append(msg)

    hints: list[str] = []

    if decision.budget_class == BudgetClass.QUICK:
        hints.append("Answer concisely and directly unless the user asks for elaboration.")
    elif decision.budget_class == BudgetClass.DEEP:
        hints.append("Provide a thorough, complete solution without omitting important code or reasoning.")

    if decision.inject_mermaid_hint:
        hints.append(MERMAID_SYSTEM_HINT)

    if settings.enable_think_steering:
        hints.append("/think" if decision.enable_thinking else "/no_think")

    if hints:
        directive = "\n".join(hints)
        if steered and steered[0].get("role") == "system":
            steered[0]["content"] = f"{steered[0].get('content', '')}\n\n{directive}".strip()
        else:
            steered.insert(0, {"role": "system", "content": directive})

    return steered
