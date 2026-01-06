from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openai.types.responses.response_usage import ResponseUsage


@dataclass(frozen=True)
class PricingPer1M:
    input_usd: float = 1.75
    cached_input_usd: float = 0.175
    output_usd: float = 14.0


@dataclass(frozen=True)
class WebSearchPricing:
    per_1k_calls_usd: float = 10.0


@dataclass(frozen=True)
class GeminiPricing:
    input_usd_per_1m: float = 0.50
    output_usd_per_1m: float = 3.00
    search_usd_per_1k: float = 35.00


def compute_cost_usd(usage: ResponseUsage, pricing: PricingPer1M) -> float:
    """Compute USD cost from token usage using per-1M token pricing."""
    cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    non_cached = max(0, input_tokens - cached)
    return (
        (non_cached / 1_000_000.0) * pricing.input_usd
        + (cached / 1_000_000.0) * pricing.cached_input_usd
        + (output_tokens / 1_000_000.0) * pricing.output_usd
    )


def compute_web_search_tool_cost_usd(web_search_calls: int, pricing: WebSearchPricing) -> float:
    calls = max(0, int(web_search_calls))
    return calls * (pricing.per_1k_calls_usd / 1000.0)


def compute_gemini_cost_usd(usage: Any, pricing: GeminiPricing, *, search_queries: int = 0) -> float:
    """Compute Gemini cost from usage metadata + number of grounded search queries.

    Notes:
    - Token usage comes from `usage_metadata`.
    - Grounding cost is billed per query/request (provider-specific). We approximate
      using `search_queries` extracted from grounding metadata.
    """
    # usage is likely google.genai.types.GenerateContentResponseUsageMetadata
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0

    token_cost = (
        (prompt_tokens / 1_000_000.0) * pricing.input_usd_per_1m
        + (candidates_tokens / 1_000_000.0) * pricing.output_usd_per_1m
    )
    
    search_cost = (max(0, int(search_queries)) * (pricing.search_usd_per_1k / 1_000.0)) if pricing.search_usd_per_1k else 0.0

    return token_cost + search_cost


def pricing_from_env(env: dict, default: Optional[PricingPer1M] = None) -> PricingPer1M:
    """Read OpenAI pricing config from environment."""
    base = default or PricingPer1M()

    def _get_float(key: str, fallback: float) -> float:
        v = env.get(key)
        if v is None or str(v).strip() == "":
            return fallback
        return float(str(v).strip())

    return PricingPer1M(
        input_usd=_get_float("MANUAV_PRICE_INPUT_PER_1M", base.input_usd),
        cached_input_usd=_get_float("MANUAV_PRICE_CACHED_INPUT_PER_1M", base.cached_input_usd),
        output_usd=_get_float("MANUAV_PRICE_OUTPUT_PER_1M", base.output_usd),
    )


def web_search_pricing_from_env(env: dict, default: Optional[WebSearchPricing] = None) -> WebSearchPricing:
    base = default or WebSearchPricing()

    v = env.get("MANUAV_PRICE_WEB_SEARCH_PER_1K")
    if v is None or str(v).strip() == "":
        return base
    return WebSearchPricing(per_1k_calls_usd=float(str(v).strip()))


def gemini_pricing_from_env(env: dict) -> GeminiPricing:
    """Read Gemini pricing config from environment."""
    def _get_float(key: str, fallback: float) -> float:
        v = env.get(key)
        if v is None or str(v).strip() == "":
            return fallback
        return float(str(v).strip())

    return GeminiPricing(
        input_usd_per_1m=_get_float("GEMINI_PRICE_INPUT_PER_1M", 0.50),
        output_usd_per_1m=_get_float("GEMINI_PRICE_OUTPUT_PER_1M", 3.00),
        search_usd_per_1k=_get_float("GEMINI_PRICE_SEARCH_PER_1K", 35.00),
    )
