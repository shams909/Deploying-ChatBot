"""Unit tests for Auto Router, Dynamic Budgeting, and Circuit Breaker Fallback."""

from app.config import settings
from app.router import BudgetClass, ModelTier, apply_dynamic_steering, route_request


def test_quick_question_routes_to_fast_and_concise_budget(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "What is DNS?"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.target_tier == ModelTier.FAST
    assert decision.budget_class == BudgetClass.QUICK
    assert decision.max_tokens == settings.budget_quick_tokens
    assert decision.enable_thinking is False


def test_code_debug_routes_to_reasoning_or_pro_with_deep_budget(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    prompt = "Debug this FastAPI race condition and prove the fix step by step:\n```python\nasync def f(): pass\n```"
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": prompt}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.target_tier in (ModelTier.REASONING, ModelTier.PRO)
    assert decision.budget_class == BudgetClass.DEEP
    assert decision.max_tokens == settings.budget_deep_tokens
    assert decision.enable_thinking is True


def test_node2_unhealthy_falls_back_to_reasoning(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    prompt = "Architect a fault-tolerant distributed consensus system in detail with proof and code."
    decision = route_request(
        requested_model="labai-pro",
        messages=[{"role": "user", "content": prompt}],
        explicit_max_tokens=None,
        node2_healthy=False,
        node2_queue_depth=0,
    )
    assert decision.target_tier == ModelTier.REASONING
    assert decision.fallback_applied is True


def test_diagram_intent_injects_mermaid_rule(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "Draw an architecture diagram for a microservices gateway"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.inject_mermaid_hint is True
    steered = apply_dynamic_steering([{"role": "user", "content": "Draw an architecture diagram"}], decision)
    assert steered[0]["role"] == "system"
    assert "mermaid" in steered[0]["content"].lower()


def test_openwebui_internal_task_routes_to_fast_without_badge(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    prompt = "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history."
    decision = route_request(
        requested_model="labai-pro",
        messages=[{"role": "user", "content": prompt}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.is_internal_task is True
    assert decision.target_tier == ModelTier.FAST
    assert decision.enable_thinking is False


def test_apply_dynamic_steering_strips_prior_routing_badges(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "hey"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    history = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "> *LabAI → `Fast (4B)` • Budget: `quick` (`768`) • Cache: `HIT_EXACT`*\n\nHi!"},
        {"role": "user", "content": "yo"},
    ]
    steered = apply_dynamic_steering(history, decision)
    assistant_msg = next(m for m in steered if m["role"] == "assistant")
    assert "LabAI →" not in assistant_msg["content"]
    assert assistant_msg["content"] == "Hi!"
