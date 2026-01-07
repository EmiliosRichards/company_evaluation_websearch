import argparse
import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from manuav_eval import evaluate_company as core_evaluate_company
from manuav_eval import evaluate_company_with_usage_and_web_search_artifacts, evaluate_company_with_usage_and_web_search_debug
from manuav_eval.costing import compute_cost_usd, compute_web_search_tool_cost_usd, pricing_from_env, web_search_pricing_from_env
from manuav_eval.rubric_loader import DEFAULT_RUBRIC_FILE


def _extract_json_text(resp: Any) -> str:
    # Newer SDKs expose a convenience accessor.
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text

    # Fallback: attempt to traverse the structured output.
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


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Single-call Manuav company evaluator (URL -> score).")
    parser.add_argument("url", help="Company website URL (e.g., https://example.com)")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model (default: env OPENAI_MODEL or gpt-4.1-mini)",
    )
    parser.add_argument(
        "--rubric-file",
        default=os.environ.get("MANUAV_RUBRIC_FILE", str(DEFAULT_RUBRIC_FILE)),
        help="Path to rubric file (default: env MANUAV_RUBRIC_FILE or the default rubric)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=int(os.environ["MANUAV_MAX_TOOL_CALLS"]) if os.environ.get("MANUAV_MAX_TOOL_CALLS") else None,
        help="Optional cap on tool calls (web searches) within the single LLM call. Env: MANUAV_MAX_TOOL_CALLS",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("MANUAV_REASONING_EFFORT") or None,
        help="Optional reasoning effort override: none/minimal/low/medium/high/xhigh. Default: auto (unset). Env: MANUAV_REASONING_EFFORT",
    )
    parser.add_argument(
        "--prompt-cache",
        action="store_true",
        default=(os.environ.get("MANUAV_PROMPT_CACHE", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Enable prompt caching for repeated static input (rubric + system prompt). Env: MANUAV_PROMPT_CACHE=1",
    )
    parser.add_argument(
        "--prompt-cache-retention",
        default=os.environ.get("MANUAV_PROMPT_CACHE_RETENTION") or None,
        help="Prompt cache retention: in-memory or 24h. Env: MANUAV_PROMPT_CACHE_RETENTION",
    )
    parser.add_argument(
        "--service-tier",
        default=os.environ.get("MANUAV_SERVICE_TIER", "auto"),
        help="OpenAI service tier: auto (default) or flex. Env: MANUAV_SERVICE_TIER",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ["MANUAV_OPENAI_TIMEOUT_SECONDS"]) if os.environ.get("MANUAV_OPENAI_TIMEOUT_SECONDS") else None,
        help="Request timeout in seconds. For flex, you may want ~900s. Env: MANUAV_OPENAI_TIMEOUT_SECONDS",
    )
    parser.add_argument(
        "--flex-max-retries",
        type=int,
        default=int(os.environ.get("MANUAV_FLEX_MAX_RETRIES", "5")),
        help="Retries (with exponential backoff) on 429 Resource Unavailable when service-tier is flex. Env: MANUAV_FLEX_MAX_RETRIES",
    )
    parser.add_argument(
        "--flex-fallback-to-auto",
        action="store_true",
        default=(os.environ.get("MANUAV_FLEX_FALLBACK_TO_AUTO", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="If flex is unavailable after retries, retry once with standard processing (auto). Env: MANUAV_FLEX_FALLBACK_TO_AUTO=1",
    )
    parser.add_argument(
        "--no-cost",
        action="store_true",
        help="Do not print estimated USD cost to stderr (JSON output remains unchanged).",
    )
    parser.add_argument(
        "--debug-web-search",
        action="store_true",
        default=(os.environ.get("MANUAV_DEBUG_WEB_SEARCH", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Print debug info about web_search_call items to stderr. Env: MANUAV_DEBUG_WEB_SEARCH=1",
    )
    parser.add_argument(
        "--second-query-on-uncertainty",
        action="store_true",
        default=(os.environ.get("MANUAV_SECOND_QUERY_ON_UNCERTAINTY", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Allow a second web-search query only for ambiguous/low-confidence cases (does not force it). Env: MANUAV_SECOND_QUERY_ON_UNCERTAINTY=1",
    )
    parser.add_argument(
        "--retry-disambiguation-on-low-confidence",
        action="store_true",
        default=(
            os.environ.get("MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE", "").strip()
            in ("1", "true", "TRUE", "yes", "YES")
        ),
        help=(
            "If the model returns confidence=low, re-run the evaluation once with stronger disambiguation instructions. "
            "This is a second model call (extra tokens + extra web-search queries if used). "
            "Env: MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE=1"
        ),
    )
    parser.add_argument(
        "--retry-max-tool-calls",
        type=int,
        default=int(os.environ.get("MANUAV_RETRY_MAX_TOOL_CALLS", "3") or 3),
        help="max_tool_calls to use for the retry call (if retry is triggered). Default: 3. Env: MANUAV_RETRY_MAX_TOOL_CALLS",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    # Flex can be slower; default to a larger timeout if not set explicitly.
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None and (args.service_tier or "").strip().lower() == "flex":
        timeout_seconds = 900.0

    def _do_eval_once(
        *,
        max_tool_calls: int | None,
        extra_user_instructions: str | None,
        second_query_on_uncertainty: bool,
        use_debug: bool,
    ) -> Dict[str, Any]:
        if use_debug:
            res, use, ws = evaluate_company_with_usage_and_web_search_debug(
                args.url,
                args.model,
                rubric_file=args.rubric_file,
                max_tool_calls=max_tool_calls,
                reasoning_effort=args.reasoning_effort,
                prompt_cache=args.prompt_cache,
                prompt_cache_retention=args.prompt_cache_retention,
                service_tier=args.service_tier,
                timeout_seconds=timeout_seconds,
                flex_max_retries=args.flex_max_retries,
                flex_fallback_to_auto=args.flex_fallback_to_auto,
                include_sources=False,
                extra_user_instructions=extra_user_instructions,
                second_query_on_uncertainty=second_query_on_uncertainty,
            )
            by_kind_completed = ws.get("by_kind_completed") or {}
            billed_q = int(by_kind_completed.get("query", 0) or 0)
            return {"result": res, "usage": use, "ws_debug": ws, "web_search_calls": billed_q}

        res, use, billed_q, _citations = evaluate_company_with_usage_and_web_search_artifacts(
            args.url,
            args.model,
            rubric_file=args.rubric_file,
            max_tool_calls=max_tool_calls,
            reasoning_effort=args.reasoning_effort,
            prompt_cache=args.prompt_cache,
            prompt_cache_retention=args.prompt_cache_retention,
            service_tier=args.service_tier,
            timeout_seconds=timeout_seconds,
            flex_max_retries=args.flex_max_retries,
            flex_fallback_to_auto=args.flex_fallback_to_auto,
            second_query_on_uncertainty=second_query_on_uncertainty,
        )
        return {"result": res, "usage": use, "ws_debug": None, "web_search_calls": int(billed_q)}

    # Attempt 1
    a1 = _do_eval_once(
        max_tool_calls=args.max_tool_calls,
        extra_user_instructions=None,
        second_query_on_uncertainty=bool(args.second_query_on_uncertainty),
        use_debug=bool(args.debug_web_search),
    )
    attempts = [a1]
    selected = a1

    # Optional retry on confidence=low
    conf1 = str((a1.get("result") or {}).get("confidence") or "").strip().lower()
    if args.retry_disambiguation_on_low_confidence and conf1 == "low":
        retry_max = int(args.retry_max_tool_calls)
        if args.max_tool_calls is not None:
            retry_max = max(int(args.max_tool_calls), retry_max)
        disambig_prompt = (
            "Perform TWO distinct web searches before scoring.\n"
            "1) Search using the provided domain/company name (e.g., '<domain>').\n"
            "2) Search again with a disambiguation query that adds legal-entity/location hints, e.g. "
            "'<name> GmbH impressum', '<name> Munich impressum', '<name> HRB'.\n"
            "If results are ambiguous/conflicting, prefer sources that match the provided domain and DACH legal/imprint details; "
            "otherwise explicitly note uncertainty.\n"
        )
        a2 = _do_eval_once(
            max_tool_calls=retry_max,
            extra_user_instructions=disambig_prompt,
            second_query_on_uncertainty=False,
            # Retry needs debug mode to inject extra instructions.
            use_debug=True,
        )
        attempts.append(a2)
        conf2 = str((a2.get("result") or {}).get("confidence") or "").strip().lower()
        if conf2 and conf2 != "low":
            selected = a2

    result = selected["result"]
    usage = selected["usage"]
    web_search_calls = sum(int(a.get("web_search_calls", 0) or 0) for a in attempts)

    if args.debug_web_search:
        ws_debug = selected.get("ws_debug") or {}
        citations = ws_debug.get("url_citations") or []
        print(f"web_search_debug={json.dumps(ws_debug, ensure_ascii=False)}", file=sys.stderr)
        print(f"url_citations={json.dumps(citations, ensure_ascii=False)}", file=sys.stderr)
    if not args.no_cost:
        pricing = pricing_from_env(os.environ)
        token_cost_raw = sum(compute_cost_usd(a["usage"], pricing) for a in attempts)
        flex_discount = float(os.environ.get("MANUAV_FLEX_TOKEN_DISCOUNT", "0.5") or 0.5)
        token_cost = (token_cost_raw * flex_discount) if (args.service_tier or "").strip().lower() == "flex" else token_cost_raw
        tool_pricing = web_search_pricing_from_env(os.environ)
        web_search_cost = compute_web_search_tool_cost_usd(web_search_calls, tool_pricing)
        cost = token_cost + web_search_cost
        input_total = sum(int(a["usage"].input_tokens) for a in attempts)
        output_total = sum(int(a["usage"].output_tokens) for a in attempts)
        cached_total = sum(int(getattr(getattr(a["usage"], "input_tokens_details", None), "cached_tokens", 0) or 0) for a in attempts)
        print(
            f"Estimated cost_usd={cost:.6f} (service_tier={args.service_tier}, tokens={token_cost:.6f}, web_search_calls={web_search_calls}, web_search_tool_cost={web_search_cost:.6f}, input={input_total}, cached={cached_total}, output={output_total}, attempts={len(attempts)})",
            file=sys.stderr,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


