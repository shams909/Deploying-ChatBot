"""Unit tests for Auto Router, Dynamic Budgeting, and Circuit Breaker Fallback."""

from app.config import settings
from app.router import (
    BudgetClass,
    ModelTier,
    apply_dynamic_steering,
    resolve_repetition_params,
    resolve_sampling_params,
    route_request,
)


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


def test_distributed_architecture_prompt_escalates_to_pro(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    prompt = (
        "Architect a fault-tolerant distributed rate limiter using Redis and Kafka "
        "with race-condition handling, trade-off analysis, and a Mermaid sequence diagram"
    )
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": prompt}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.target_tier == ModelTier.PRO


def test_retry_frustration_escalates_follow_up_to_pro(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    history = [
        {"role": "user", "content": "Explain how this works"},
        {"role": "assistant", "content": "Basic summary..."},
        {"role": "user", "content": "that's wrong, go deeper"},
    ]
    decision = route_request(
        requested_model="labai-auto",
        messages=history,
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.target_tier == ModelTier.PRO
    assert decision.reason == "auto-retry-escalation"


def test_dynamic_sampling_coding_and_diagram_prompts(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "Write a Python FastAPI endpoint and Mermaid diagram"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    temp, penalty = resolve_sampling_params(0.7, None, decision, is_grounded=False)
    assert temp == 0.3
    assert penalty == 0.0


def test_dynamic_sampling_casual_and_grounded_prompts(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    for text in ("hey", "hug me"):
        casual = route_request(
            requested_model="labai-auto",
            messages=[{"role": "user", "content": text}],
            explicit_max_tokens=None,
            node2_healthy=True,
            node2_queue_depth=0,
        )
        temp, penalty = resolve_sampling_params(0.7, None, casual, is_grounded=False)
        assert temp == 0.78
        assert penalty == 0.12

    factual = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "What is DNS?"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    f_temp, f_penalty = resolve_sampling_params(0.7, None, factual, is_grounded=False)
    assert f_temp == 0.55
    assert f_penalty == 0.05

    # Grounded web/URL requests clamp back down for factual adherence
    grounded_temp, grounded_penalty = resolve_sampling_params(0.8, None, casual, is_grounded=True)
    assert grounded_temp == 0.3
    assert grounded_penalty == 0.0


def test_dynamic_sampling_respects_explicit_client_overrides(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    decision = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "hey"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    temp, penalty = resolve_sampling_params(0.15, 0.4, decision, is_grounded=True)
    assert temp == 0.15
    assert penalty == 0.4


def test_multiturn_conversational_chat_promotes_to_reasoning(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    history = [
        {"role": "user", "content": "baby"},
        {"role": "assistant", "content": "hey, missed you"},
        {"role": "user", "content": "babe i miss u"},
    ]
    decision = route_request(
        requested_model="labai-auto",
        messages=history,
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert decision.target_tier == ModelTier.REASONING
    assert decision.reason == "auto-conversational-gemma"
    assert decision.recommended_temperature == 0.78
    assert decision.recommended_presence_penalty == 0.12
    freq, rep, min_p = resolve_repetition_params(decision, is_grounded=False)
    assert freq == 0.0
    assert rep == 1.05
    assert min_p == 0.05
    assert decision.is_conversational is True
    assert decision.max_tokens <= 220
    steered = apply_dynamic_steering(history, decision)
    assert "texting-style" in steered[0]["content"]

    long_history = history + [{"role": "user", "content": "x" * 10500}]
    long_decision = route_request(
        requested_model="labai-auto",
        messages=long_history,
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert long_decision.target_tier == ModelTier.PRO


def test_story_prompt_unlocks_standard_budget_without_thinking(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    story = route_request(
        requested_model="labai-auto",
        messages=[{"role": "user", "content": "Write a bold roleplay story scene set in a rainy penthouse"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert story.target_tier == ModelTier.REASONING
    assert story.budget_class == BudgetClass.STANDARD
    assert story.max_tokens == settings.budget_standard_tokens
    assert story.enable_thinking is False
    assert story.is_conversational is True
    assert story.recommended_temperature == 0.78
    steered = apply_dynamic_steering([{"role": "user", "content": "Write a bold roleplay story scene"}], story)
    assert "immersive" in steered[0]["content"]


def test_grok_modes_route_and_steer_cleanly(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_embedder", False)
    seductive = route_request(
        requested_model="labai-seductive",
        messages=[{"role": "user", "content": "miss u tonight"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert seductive.target_tier == ModelTier.REASONING
    assert seductive.reason == "mode-labai-seductive"
    assert seductive.recommended_temperature == 0.78
    assert seductive.enable_thinking is False
    steered = apply_dynamic_steering([{"role": "user", "content": "miss u tonight"}], seductive)
    assert "horny woman" in steered[0]["content"]
    assert "tell me what you'd do" not in steered[0]["content"].lower()

    desc = route_request(
        requested_model="labai-seductive",
        messages=[{"role": "user", "content": "like with extensive description naming everything"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert desc.budget_class == BudgetClass.STANDARD
    assert desc.max_tokens == settings.budget_standard_tokens

    strategist = route_request(
        requested_model="labai-strategist",
        messages=[{"role": "user", "content": "How should I invest 5k?"}],
        explicit_max_tokens=None,
        node2_healthy=True,
        node2_queue_depth=0,
    )
    assert strategist.target_tier == ModelTier.PRO
    assert strategist.recommended_temperature == 0.45


def test_ctx_trimmer_and_thinking_tier_guard():
    from app.router import supports_thinking, trim_messages_for_ctx

    assert supports_thinking(ModelTier.FAST) is False
    assert supports_thinking(ModelTier.REASONING) is False
    assert supports_thinking(ModelTier.PRO) is True

    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + ("x" * 1000)}
        for i in range(20)
    ]
    trimmed = trim_messages_for_ctx(msgs, max_chars=4500)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"].startswith("turn-19")
    assert sum(len(m["content"]) for m in trimmed) <= 4500
