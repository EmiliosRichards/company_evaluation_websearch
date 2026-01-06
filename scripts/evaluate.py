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
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    # By default, print cost to stderr (JSON stays on stdout).
    if args.debug_web_search:
        result, usage, ws_debug = evaluate_company_with_usage_and_web_search_debug(
            args.url,
            args.model,
            rubric_file=args.rubric_file,
            max_tool_calls=args.max_tool_calls,
            reasoning_effort=args.reasoning_effort,
            prompt_cache=args.prompt_cache,
            prompt_cache_retention=args.prompt_cache_retention,
        )
        web_search_calls = int(ws_debug.get("completed", 0))
        citations = ws_debug.get("url_citations") or []
        print(f"web_search_debug={json.dumps(ws_debug, ensure_ascii=False)}", file=sys.stderr)
        print(f"url_citations={json.dumps(citations, ensure_ascii=False)}", file=sys.stderr)
    else:
        result, usage, web_search_calls, _citations = evaluate_company_with_usage_and_web_search_artifacts(
            args.url,
            args.model,
            rubric_file=args.rubric_file,
            max_tool_calls=args.max_tool_calls,
            reasoning_effort=args.reasoning_effort,
            prompt_cache=args.prompt_cache,
            prompt_cache_retention=args.prompt_cache_retention,
        )
    if not args.no_cost:
        pricing = pricing_from_env(os.environ)
        token_cost = compute_cost_usd(usage, pricing)
        tool_pricing = web_search_pricing_from_env(os.environ)
        web_search_cost = compute_web_search_tool_cost_usd(web_search_calls, tool_pricing)
        cost = token_cost + web_search_cost
        print(
            f"Estimated cost_usd={cost:.6f} (tokens={token_cost:.6f}, web_search_calls={web_search_calls}, web_search_tool_cost={web_search_cost:.6f}, input={usage.input_tokens}, cached={usage.input_tokens_details.cached_tokens}, output={usage.output_tokens})",
            file=sys.stderr,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


