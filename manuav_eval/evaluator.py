from __future__ import annotations

import json
import time
from typing import Any, Dict
import random

from openai import OpenAI
from openai.types.responses.response_usage import ResponseUsage
import hashlib

from .rubric_loader import load_rubric_text
from .schema import OUTPUT_SCHEMA_WITH_SOURCES, json_schema_text_config


BASE_SYSTEM_PROMPT = """\
You are a specialized evaluation assistant for Manuav, a B2B cold outbound (phone outreach) and lead-generation agency.

You will be given:
- a company website URL
- a rubric (below)

Your job:
- research the company using the web search tool (this is required)
- apply the rubric
- return ONLY valid JSON matching the provided schema (no extra keys, no markdown)

Evidence discipline:
- Do not hallucinate. If something is unknown, say so, lower confidence, and be conservative.

Research process (required):
- Use the web search tool to:
  - visit/review the company website (home, product, pricing, cases, careers, legal/imprint/contact)
  - search the web for corroborating third-party evidence
- Use the web search tool strategically.
  - If you have a limited tool-call/search budget, prioritize validating the rubric’s hard lines and the biggest unknowns first.
- Prefer primary sources first, then reputable third-party sources. Prioritize DACH-relevant signals.
- You do NOT need to output a sources list in JSON. Keep the output compact.

Entity disambiguation (guideline):
- Be mindful of same-name/lookalike companies. Use your judgment to sanity-check that a source is actually about the company behind the provided website URL.
- Helpful identity signals include:
  - domain consistency and links from the official site
  - legal entity name and imprint/registration details
  - headquarters/location and language/market focus
  - product description, ICP, and branding match
  - the official LinkedIn/company page referenced by the website
- If attribution is uncertain, either avoid relying on the source or briefly note the uncertainty in your reasoning.
"""


def _extract_json_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text

    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass

    raise RuntimeError("Could not extract text output from OpenAI response.")


def evaluate_company(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    second_query_on_uncertainty: bool = False,
) -> Dict[str, Any]:
    result, _usage, _web_search_calls = _evaluate_company_raw(
        url,
        model,
        rubric_file=rubric_file,
        max_tool_calls=max_tool_calls,
        reasoning_effort=reasoning_effort,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
        service_tier=service_tier,
        timeout_seconds=timeout_seconds,
        flex_max_retries=flex_max_retries,
        flex_fallback_to_auto=flex_fallback_to_auto,
        second_query_on_uncertainty=second_query_on_uncertainty,
    )
    return result


def evaluate_company_with_usage(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    second_query_on_uncertainty: bool = False,
) -> tuple[Dict[str, Any], ResponseUsage]:
    result, usage, _web_search_calls = _evaluate_company_raw(
        url,
        model,
        rubric_file=rubric_file,
        max_tool_calls=max_tool_calls,
        reasoning_effort=reasoning_effort,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
        service_tier=service_tier,
        timeout_seconds=timeout_seconds,
        flex_max_retries=flex_max_retries,
        flex_fallback_to_auto=flex_fallback_to_auto,
        second_query_on_uncertainty=second_query_on_uncertainty,
    )
    if usage is None:
        raise RuntimeError("OpenAI response did not include usage; cannot compute cost.")
    return result, usage


def evaluate_company_with_usage_and_web_search_calls(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    second_query_on_uncertainty: bool = False,
) -> tuple[Dict[str, Any], ResponseUsage, int]:
    result, usage, ws_stats = _evaluate_company_raw(
        url,
        model,
        rubric_file=rubric_file,
        max_tool_calls=max_tool_calls,
        reasoning_effort=reasoning_effort,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
        service_tier=service_tier,
        timeout_seconds=timeout_seconds,
        flex_max_retries=flex_max_retries,
        flex_fallback_to_auto=flex_fallback_to_auto,
        second_query_on_uncertainty=second_query_on_uncertainty,
    )
    if usage is None:
        raise RuntimeError("OpenAI response did not include usage; cannot compute cost.")
    return result, usage, _billable_web_search_calls(ws_stats)


def _count_web_search_calls(resp: Any) -> int:
    """Count how many *billable* web search tool calls happened in the response.

    OpenAI billing/dashboard appears to count only *query* web searches (not page open/visit calls).
    We therefore count only status='completed' calls classified as kind='query'.
    """
    ws = _web_search_call_debug(resp)
    return _billable_web_search_calls(ws)


def _billable_web_search_calls(ws_stats: Dict[str, Any]) -> int:
    """Return the best-available estimate of billable web-search calls.

    We treat kind='query' as billable. If kind breakdown is missing, fall back to total completed.
    """
    by_kind_completed = ws_stats.get("by_kind_completed") or {}
    if isinstance(by_kind_completed, dict) and "query" in by_kind_completed:
        try:
            return int(by_kind_completed.get("query", 0) or 0)
        except Exception:
            return 0
    try:
        return int(ws_stats.get("completed", 0) or 0)
    except Exception:
        return 0


def _web_search_call_debug(resp: Any) -> Dict[str, Any]:
    """Extract debug info about web search tool usage from a Responses API response."""
    output = getattr(resp, "output", []) or []
    output_item_types = [getattr(it, "type", None) for it in output]

    def _safe_model_dump(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        md = getattr(obj, "model_dump", None)
        if callable(md):
            try:
                d = md()
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        # Fall back to shallow attribute extraction for likely fields.
        out: Dict[str, Any] = {}
        for k in ("id", "type", "status", "query", "url", "input", "arguments", "action", "name"):
            try:
                v = getattr(obj, k, None)
            except Exception:
                v = None
            if v is not None:
                out[k] = v
        return out

    def _classify_call(it: Any) -> Dict[str, Any]:
        """Best-effort classification of a web_search_call as query vs open/visit."""
        raw = _safe_model_dump(it)

        # Try common shapes: top-level query/url, or nested under input/arguments.
        action = raw.get("action") or raw.get("name") or ""
        inp = raw.get("input") or raw.get("arguments") or {}
        if not isinstance(inp, dict):
            inp = {}

        query = raw.get("query") or inp.get("query") or inp.get("q") or inp.get("search_query") or inp.get("searchTerm")
        url = raw.get("url") or inp.get("url") or inp.get("link") or inp.get("target_url")

        # Normalize action string.
        action_s = str(action or "").strip().lower()

        kind = "unknown"
        if isinstance(query, str) and query.strip():
            kind = "query"
        elif isinstance(url, str) and url.strip():
            kind = "open"
        else:
            # Heuristics: action/name hints.
            if any(tok in action_s for tok in ("search", "query")):
                kind = "query"
            elif any(tok in action_s for tok in ("open", "visit", "fetch", "browse")):
                kind = "open"

        # Only include compact, useful fields.
        out: Dict[str, Any] = {
            "id": getattr(it, "id", None),
            "status": getattr(it, "status", None) or "unknown",
            "kind": kind,
        }
        if isinstance(query, str) and query.strip():
            out["query"] = query.strip()
        if isinstance(url, str) and url.strip():
            out["url"] = url.strip()
        if action_s:
            out["action_hint"] = action_s
        return out

    def _extract_url_citations() -> list[dict[str, str]]:
        citations: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for c in getattr(item, "content", []) or []:
                anns = getattr(c, "annotations", None) or []
                for ann in anns:
                    ann_type = getattr(ann, "type", None)
                    if ann_type != "url_citation":
                        continue
                    uc = getattr(ann, "url_citation", None)
                    url = getattr(uc, "url", None) if uc is not None else None
                    title = getattr(uc, "title", None) if uc is not None else None
                    if isinstance(url, str) and url and url not in seen:
                        citations.append({"url": url, "title": title or ""})
                        seen.add(url)
        return citations

    calls = []
    by_status: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    by_kind_completed: Dict[str, int] = {}
    total = 0
    completed = 0

    for it in output:
        if getattr(it, "type", None) != "web_search_call":
            continue
        total += 1
        status = getattr(it, "status", None) or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        if status == "completed":
            completed += 1
        c = _classify_call(it)
        kind = c.get("kind") or "unknown"
        by_kind[str(kind)] = by_kind.get(str(kind), 0) + 1
        if status == "completed":
            by_kind_completed[str(kind)] = by_kind_completed.get(str(kind), 0) + 1
        calls.append(c)

    return {
        "output_item_types": output_item_types,
        "total": total,
        "completed": completed,
        "by_status": by_status,
        "by_kind": by_kind,
        "by_kind_completed": by_kind_completed,
        "calls": calls,
        "url_citations": _extract_url_citations(),
    }


def evaluate_company_with_usage_and_web_search_debug(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    include_sources: bool = False,
    extra_user_instructions: str | None = None,
    second_query_on_uncertainty: bool = False,
) -> tuple[Dict[str, Any], ResponseUsage, Dict[str, Any]]:
    """Returns model JSON + usage + debug info about web_search_call items."""
    result, usage, ws_stats = _evaluate_company_raw(
        url,
        model,
        rubric_file=rubric_file,
        max_tool_calls=max_tool_calls,
        reasoning_effort=reasoning_effort,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
        service_tier=service_tier,
        timeout_seconds=timeout_seconds,
        flex_max_retries=flex_max_retries,
        flex_fallback_to_auto=flex_fallback_to_auto,
        include_sources=include_sources,
        extra_user_instructions=extra_user_instructions,
        second_query_on_uncertainty=second_query_on_uncertainty,
    )
    if usage is None:
        raise RuntimeError("OpenAI response did not include usage; cannot compute cost.")
    return result, usage, ws_stats


def evaluate_company_with_usage_and_web_search_artifacts(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    second_query_on_uncertainty: bool = False,
) -> tuple[Dict[str, Any], ResponseUsage, int, list[dict[str, str]]]:
    """
    Returns:
    - parsed JSON result (compact, no sources list)
    - token usage
    - billable web search tool calls (query-type)
    - URL citations extracted from response annotations (when available)
    """
    result, usage, ws_stats = _evaluate_company_raw(
        url,
        model,
        rubric_file=rubric_file,
        max_tool_calls=max_tool_calls,
        reasoning_effort=reasoning_effort,
        prompt_cache=prompt_cache,
        prompt_cache_retention=prompt_cache_retention,
        service_tier=service_tier,
        timeout_seconds=timeout_seconds,
        flex_max_retries=flex_max_retries,
        flex_fallback_to_auto=flex_fallback_to_auto,
        second_query_on_uncertainty=second_query_on_uncertainty,
    )
    if usage is None:
        raise RuntimeError("OpenAI response did not include usage; cannot compute cost.")
    web_search_calls = _billable_web_search_calls(ws_stats)
    citations = ws_stats.get("url_citations") or []
    return result, usage, web_search_calls, citations


def _normalize_service_tier(service_tier: str | None) -> str | None:
    st = (service_tier or "").strip().lower()
    if not st or st == "auto":
        return None
    return st


def _is_resource_unavailable_429(exc: Exception) -> bool:
    # Flex may return 429 "Resource Unavailable" (not charged).
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status != 429:
        return False
    msg = str(exc).lower()
    return "resource unavailable" in msg


def _is_prompt_cache_retention_unsupported_400(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status != 400:
        return False
    msg = str(exc).lower()
    return "prompt_cache_retention" in msg and "not supported" in msg


def _evaluate_company_raw(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    service_tier: str | None = None,
    timeout_seconds: float | None = None,
    flex_max_retries: int | None = None,
    flex_fallback_to_auto: bool | None = None,
    include_sources: bool = False,
    extra_user_instructions: str | None = None,
    second_query_on_uncertainty: bool = False,
) -> tuple[Dict[str, Any], ResponseUsage | None, Dict[str, Any]]:
    client = OpenAI()
    rubric_path, rubric_text = load_rubric_text(rubric_file)
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nRubric file: {rubric_path}\n\n{rubric_text}\n"

    normalized_url = url.strip()
    if normalized_url and not normalized_url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    # Put dynamic content (URL) at the end so more of the prompt prefix can be cached.
    tool_budget_line = (
        f"- Tool-call budget: you can make at most {max_tool_calls} web search tool call(s). Use them wisely.\n"
        if max_tool_calls is not None
        else ""
    )

    sources_instruction = ""
    text_cfg = json_schema_text_config()
    if include_sources:
        sources_instruction = (
            "- Include a short sources list in JSON under key `sources` (max 8 items).\n"
            "  - Each item: {url, title (optional), note (very short)}.\n"
            "  - URLs are allowed ONLY inside `sources`, not in `reasoning`.\n"
        )
        text_cfg = json_schema_text_config(schema=OUTPUT_SCHEMA_WITH_SOURCES)

    extra_instruction_block = ""
    if extra_user_instructions and extra_user_instructions.strip():
        extra_instruction_block = f"\nExtra instructions (debug):\n{extra_user_instructions.strip()}\n"
    elif second_query_on_uncertainty:
        # Production-safe toggle: do NOT force a second search, but allow it for sticky cases.
        extra_instruction_block = (
            "\nExtra instructions:\n"
            "- Default to ONE web search query.\n"
            "- If (and only if) the first query does NOT yield trustworthy evidence about the company behind the provided domain "
            "(e.g., domain seems inactive/parked, results point to different entities, or multiple similarly named companies appear), "
            "you SHOULD run exactly ONE additional disambiguation query.\n"
            "- Do NOT use a second query just to gather extra detail (e.g., pricing/ARPU) when the company is already clearly identified.\n"
            "- Do not use more than two queries total.\n"
        )

    user_prompt = f"""\
Evaluate this company for Manuav using web research and the Manuav Fit logic.

Instructions:
- Use the web search tool to research:
  - the company website itself (product/service, ICP, pricing, cases, careers, legal/imprint)
  - and the broader web for each rubric category (DACH presence, operational status, TAM, competition, innovation, economics, onboarding, pitchability, risk).
{tool_budget_line}- Be conservative when evidence is missing.
{extra_instruction_block}- In the JSON output:
  - set input_url exactly to the Company website URL below
  - keep reasoning SHORT (max ~600 characters, 2-4 sentences). Focus on the top 2-4 drivers and the biggest unknown.
{sources_instruction}  - do NOT include URLs in `reasoning`.

Company website URL: {normalized_url}
"""

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [{"type": "web_search_preview"}],
        "text": text_cfg,
    }
    st = _normalize_service_tier(service_tier)
    if st is not None:
        create_kwargs["service_tier"] = st
    if max_tool_calls is not None:
        create_kwargs["max_tool_calls"] = max_tool_calls
    if reasoning_effort:
        create_kwargs["reasoning"] = {"effort": reasoning_effort}
    # Prompt caching: allows repeated static input (rubric + system instructions) to be billed at cached rate.
    if prompt_cache:
        h = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
        create_kwargs["prompt_cache_key"] = f"manuav:{model}:{h}"
        if prompt_cache_retention:
            create_kwargs["prompt_cache_retention"] = prompt_cache_retention

    # Flex processing may be slower; allow a larger timeout and retries on 429 Resource Unavailable.
    call_client = client
    if timeout_seconds is not None and hasattr(client, "with_options"):
        call_client = client.with_options(timeout=float(timeout_seconds))

    max_retries = int(flex_max_retries) if flex_max_retries is not None else 0
    fallback = bool(flex_fallback_to_auto) if flex_fallback_to_auto is not None else False

    # Track Flex retry behavior for observability in large runs.
    retry_meta: Dict[str, Any] = {
        "service_tier_requested": (st or "auto"),
        "service_tier_used": (st or "auto"),
        "attempts": 0,
        "retries": 0,
        "sleep_seconds_total": 0.0,
        "fallback_used": False,
    }

    for attempt in range(max_retries + 1):
        try:
            retry_meta["attempts"] += 1
            resp = call_client.responses.create(**create_kwargs)
            break
        except Exception as e:  # pragma: no cover (SDK exception types vary)
            # Some models don't support prompt_cache_retention even if prompt caching is enabled.
            # If we hit that, retry once without the retention parameter.
            if create_kwargs.get("prompt_cache_retention") is not None and _is_prompt_cache_retention_unsupported_400(e):
                create_kwargs.pop("prompt_cache_retention", None)
                retry_meta["attempts"] += 1
                resp = call_client.responses.create(**create_kwargs)
                break

            if st != "flex" or not _is_resource_unavailable_429(e):
                raise

            # If Flex is unavailable and we've exhausted retries, optionally fall back to standard processing.
            if attempt >= max_retries:
                if fallback:
                    create_kwargs.pop("service_tier", None)
                    retry_meta["fallback_used"] = True
                    retry_meta["service_tier_used"] = "auto"
                    retry_meta["attempts"] += 1
                    resp = call_client.responses.create(**create_kwargs)
                    break
                raise

            # Exponential backoff with jitter.
            retry_meta["retries"] += 1
            base = 1.0
            delay = min(60.0, base * (2**attempt))
            delay = delay * (0.8 + 0.4 * random.random())
            retry_meta["sleep_seconds_total"] = float(retry_meta["sleep_seconds_total"]) + float(delay)
            time.sleep(delay)

    text = _extract_json_text(resp)
    result = json.loads(text)
    ws_stats = _web_search_call_debug(resp)
    ws_stats["flex"] = retry_meta
    return result, resp.usage, ws_stats


