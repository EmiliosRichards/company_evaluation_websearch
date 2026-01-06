import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from manuav_eval import (
    evaluate_company_with_usage_and_web_search_artifacts,
    evaluate_company_with_usage_and_web_search_debug,
)
from manuav_eval.costing import (
    compute_cost_usd,
    compute_web_search_tool_cost_usd,
    pricing_from_env,
    web_search_pricing_from_env,
)
from manuav_eval.rubric_loader import DEFAULT_RUBRIC_FILE


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _mae(pairs: List[Tuple[float, float]]) -> float:
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else 0.0


def _run_stamp() -> str:
    # Filesystem-friendly local timestamp.
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _suffix_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Keep it filename-friendly.
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
    out = "".join(safe).strip("_")
    return out


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the one-call evaluator on Irene's 9-row sample and compare scores.")
    parser.add_argument("--sample", default="data/irene_sample_9.csv", help="Path to sample CSV created by make_irene_sample.py")
    parser.add_argument("--out", default=None, help="Where to write JSONL results (default: outputs/<timestamp>[_suffix].jsonl)")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Where to write CSV results (default: outputs/<timestamp>[_suffix].csv)",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default="",
        help="Optional suffix added to output filenames (default: none). Example: -s baseline",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), help="OpenAI model")
    parser.add_argument(
        "--rubric-file",
        default=os.environ.get("MANUAV_RUBRIC_FILE", str(DEFAULT_RUBRIC_FILE)),
        help="Path to rubric file (default: env MANUAV_RUBRIC_FILE or rubrics/manuav_rubric_v4_en.md)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=int(os.environ["MANUAV_MAX_TOOL_CALLS"]) if os.environ.get("MANUAV_MAX_TOOL_CALLS") else None,
        help="Optional cap on tool calls (web searches) within each single LLM call. Env: MANUAV_MAX_TOOL_CALLS",
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
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between calls")
    parser.add_argument(
        "--debug-web-search",
        action="store_true",
        default=(os.environ.get("MANUAV_DEBUG_WEB_SEARCH", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Include OpenAI web_search_call debug info in JSONL records. Env: MANUAV_DEBUG_WEB_SEARCH=1",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    sample_path = Path(args.sample)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = _run_stamp()
    suffix = _suffix_slug(args.suffix)
    stem = stamp if not suffix else f"{stamp}_{suffix}"

    out_path = Path(args.out) if args.out else (out_dir / f"{stem}.jsonl")
    out_csv_path = Path(args.out_csv) if args.out_csv else (out_dir / f"{stem}.csv")

    rows: List[Dict[str, str]] = []
    with sample_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    results: List[Dict[str, Any]] = []
    pairs: List[Tuple[float, float]] = []

    pricing = pricing_from_env(os.environ)
    tool_pricing = web_search_pricing_from_env(os.environ)

    csv_fieldnames = [
        "run_id",
        "bucket",
        "firma",
        "website",
        "irene_score",
        "model_score",
        "company_name",
        "input_url",
        "confidence",
        "reasoning",
        "url_citations_json",
        "rubric_file",
        "model",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
        "token_cost_usd",
        "web_search_calls",
        "web_search_tool_cost_usd",
        "price_input_per_1m",
        "price_cached_input_per_1m",
        "price_output_per_1m",
        "price_web_search_per_1k",
    ]

    with out_path.open("w", encoding="utf-8") as out, out_csv_path.open("w", encoding="utf-8", newline="") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, r in enumerate(rows, start=1):
            website = (r.get("Website") or "").strip()
            irene_score = _to_float(r.get("Manuav-Score", ""))
            if not website or irene_score is None:
                continue

            print(f"[{i}/{len(rows)}] Evaluating: {r.get('Firma','').strip()} | {website}")
            if args.debug_web_search:
                model_result, usage, ws_debug = evaluate_company_with_usage_and_web_search_debug(
                    website,
                    args.model,
                    rubric_file=args.rubric_file,
                    max_tool_calls=args.max_tool_calls,
                    reasoning_effort=args.reasoning_effort,
                    prompt_cache=args.prompt_cache,
                    prompt_cache_retention=args.prompt_cache_retention,
                )
                web_search_calls = int(ws_debug.get("completed", 0))
                url_citations = ws_debug.get("url_citations") or []
            else:
                model_result, usage, web_search_calls, url_citations = evaluate_company_with_usage_and_web_search_artifacts(
                    website,
                    args.model,
                    rubric_file=args.rubric_file,
                    max_tool_calls=args.max_tool_calls,
                    reasoning_effort=args.reasoning_effort,
                    prompt_cache=args.prompt_cache,
                    prompt_cache_retention=args.prompt_cache_retention,
                )
            model_score = float(model_result.get("manuav_fit_score", 0.0))
            token_cost_usd = compute_cost_usd(usage, pricing)
            web_search_tool_cost_usd = compute_web_search_tool_cost_usd(web_search_calls, tool_pricing)
            cost_usd = token_cost_usd + web_search_tool_cost_usd

            cached_tokens = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
            reasoning_tokens = int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)

            record = {
                "bucket": (r.get("bucket") or "").strip(),
                "firma": (r.get("Firma") or "").strip(),
                "website": website,
                "irene_score": irene_score,
                "model_score": model_score,
                "model_confidence": model_result.get("confidence"),
                "reasoning": model_result.get("reasoning"),
                "url_citations": url_citations,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "cost_usd": round(cost_usd, 6),
                "token_cost_usd": round(token_cost_usd, 6),
                "web_search_calls": int(web_search_calls),
                "web_search_tool_cost_usd": round(web_search_tool_cost_usd, 6),
                "web_search_debug": ws_debug if args.debug_web_search else None,
                "raw": model_result,
            }
            results.append(record)
            pairs.append((irene_score, model_score))

            # JSONL (full raw)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            # CSV (flattened)
            sources = url_citations or []
            writer.writerow(
                {
                    "run_id": stem,
                    "bucket": record["bucket"],
                    "firma": record["firma"],
                    "website": record["website"],
                    "irene_score": record["irene_score"],
                    "model_score": record["model_score"],
                    "company_name": model_result.get("company_name"),
                    "input_url": model_result.get("input_url"),
                    "confidence": model_result.get("confidence"),
                    "reasoning": model_result.get("reasoning"),
                    "url_citations_json": json.dumps(sources, ensure_ascii=False),
                    "rubric_file": args.rubric_file,
                    "model": args.model,
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost_usd": round(cost_usd, 6),
                    "token_cost_usd": round(token_cost_usd, 6),
                    "web_search_calls": int(web_search_calls),
                    "web_search_tool_cost_usd": round(web_search_tool_cost_usd, 6),
                    "price_input_per_1m": pricing.input_usd,
                    "price_cached_input_per_1m": pricing.cached_input_usd,
                    "price_output_per_1m": pricing.output_usd,
                    "price_web_search_per_1k": tool_pricing.per_1k_calls_usd,
                }
            )
            out_csv.flush()

            time.sleep(max(0.0, args.sleep))

    print(f"\nWrote results (jsonl): {out_path}")
    print(f"Wrote results (csv):   {out_csv_path}")
    print(f"Compared {len(pairs)} rows. MAE={_mae(pairs):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


