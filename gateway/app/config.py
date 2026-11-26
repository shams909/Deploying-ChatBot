"""Application configuration for the LabAI Control Plane Gateway."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LABAI_", env_file=".env", extra="ignore")

    tier1_url: str = "http://127.0.0.1:8081"
    tier2_url: str = "http://127.0.0.1:8082"
    tier3_url: str = "http://192.168.9.135:8083"

    redis_url: str = "redis://127.0.0.1:6379/0"
    searxng_url: str = "http://127.0.0.1:8080"

    # Routing & Caching
    enable_semantic_embedder: bool = True
    enable_llm_router_judge: bool = True
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    semantic_cache_threshold: float = 0.88
    cache_max_temperature: float = 0.85
    semantic_cache_ttl_seconds: int = 86400
    semantic_cache_max_entries: int = 250

    # Dynamic Token Budgeting
    budget_quick_tokens: int = 768
    budget_standard_tokens: int = 2048
    budget_deep_tokens: int = 4096
    hard_max_tokens: int = 4096

    # Concurrency, Quotas & Circuit Breaker
    node1_max_concurrent: int = 1
    node2_queue_limit: int = 2
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_seconds: int = 30
    user_token_quota: int = 10000
    quota_window_seconds: int = 900

    # UX Telemetry & Frontier Features
    enable_routing_badge: bool = False
    enable_think_steering: bool = True
    enable_auto_web_search: bool = True
    enable_long_term_memory: bool = True
    web_search_max_results: int = 3
    memory_max_items_per_user: int = 50
    memory_recall_top_k: int = 4


settings = Settings()
