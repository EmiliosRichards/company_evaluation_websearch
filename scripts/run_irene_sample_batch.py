import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from manuav_eval.costing import (
    PricingPer1M,
    WebSearchPricing,
    compute_web_search_tool_cost_usd,
    pricing_from_env,
    web_search_pricing_from_env,
)
from manuav_eval.openai_batch import build_irene_batch_lines, parse_batch_output_jsonl, write_batch_input_jsonl


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def cmd_create(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    if args.enable_web_search:
        print(
            "Batch API error: Web search tools are not supported in the OpenAI Batch API (endpoint /v1/responses). "
            "Use scripts.evaluate_list (sync) for web-search runs, or re-run with --no-web-search to create a batch "
            "that does not use the web search tool.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = _run_stamp()
    input_path = out_dir / f"{stamp}_{args.suffix}_batch_input.jsonl"

    lines = build_irene_batch_lines(
        Path(args.sample),
        model=args.model,
        rubric_file=args.rubric_file,
        max_tool_calls=args.max_tool_calls,
        reasoning_effort=args.reasoning_effort,
        prompt_cache=args.prompt_cache,
        prompt_cache_retention=args.prompt_cache_retention,
        custom_id_prefix=args.custom_id_prefix,
        enable_web_search=args.enable_web_search,
    )
    write_batch_input_jsonl(lines, input_path)

    client = OpenAI()
    fobj = client.files.create(file=open(input_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=fobj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": args.metadata or "manuav irene batch"},
    )

    meta_path = out_dir / f"{stamp}_{args.suffix}_batch_meta.json"
    meta = {
        "batch_id": batch.id,
        "input_file_id": fobj.id,
        "endpoint": "/v1/responses",
        "model": args.model,
        "sample": args.sample,
        "rubric_file": args.rubric_file,
        "max_tool_calls": args.max_tool_calls,
        "reasoning_effort": args.reasoning_effort,
        "prompt_cache": bool(args.prompt_cache),
        "prompt_cache_retention": args.prompt_cache_retention,
        "created_at": stamp,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote batch input: {input_path}")
    print(f"Wrote batch meta:  {meta_path}")
    print(f"Batch created: {batch.id} (status={batch.status})")
    return 0


def cmd_status(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2
    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    print(json.dumps(batch.model_dump(), indent=2, ensure_ascii=False))
    return 0


def _token_cost_from_usage_dict(usage: dict, pricing: PricingPer1M) -> float:
    inp = float(usage.get("input_tokens") or 0)
    out = float(usage.get("output_tokens") or 0)
    cached = float((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    non_cached = max(0.0, inp - cached)
    return (non_cached / 1_000_000.0) * pricing.input_usd + (cached / 1_000_000.0) * pricing.cached_input_usd + (out / 1_000_000.0) * pricing.output_usd


def cmd_fetch(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp()

    client = OpenAI()
    batch = client.batches.retrieve(args.batch_id)
    if not batch.output_file_id:
        print(f"Batch {args.batch_id} has no output_file_id yet (status={batch.status})", file=sys.stderr)
        return 1

    output_path = out_dir / f"{stamp}_{args.suffix}_batch_output.jsonl"
    content = client.files.content(batch.output_file_id).text
    output_path.write_text(content, encoding="utf-8")

    # Parse and write flat CSV of results
    csv_path = out_dir / f"{stamp}_{args.suffix}_batch_results.csv"
    jsonl_path = out_dir / f"{stamp}_{args.suffix}_batch_results.jsonl"

    pricing = pricing_from_env(os.environ)
    tool_pricing = web_search_pricing_from_env(os.environ)
    batch_discount = float(os.environ.get("MANUAV_BATCH_DISCOUNT", "0.5") or 0.5) if args.apply_discount else 1.0

    import csv

    fieldnames = [
        "batch_id",
        "custom_id",
        "status_code",
        "input_url",
        "company_name",
        "manuav_fit_score",
        "confidence",
        "reasoning",
        "web_search_calls",
        "url_citations_json",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "token_cost_usd",
        "web_search_tool_cost_usd",
        "total_cost_usd",
        "batch_discount_multiplier",
        "model",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as cf, jsonl_path.open("w", encoding="utf-8") as jf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()

        for rec in parse_batch_output_jsonl(output_path):
            model_result = rec.model_result or {}
            usage = rec.usage or {}

            cached = float((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
            tok_cost = _token_cost_from_usage_dict(usage, pricing) * batch_discount
            ws_cost = compute_web_search_tool_cost_usd(rec.web_search_calls, tool_pricing) * batch_discount
            total = tok_cost + ws_cost

            row = {
                "batch_id": args.batch_id,
                "custom_id": rec.custom_id,
                "status_code": rec.status_code,
                "input_url": model_result.get("input_url"),
                "company_name": model_result.get("company_name"),
                "manuav_fit_score": model_result.get("manuav_fit_score"),
                "confidence": model_result.get("confidence"),
                "reasoning": model_result.get("reasoning"),
                "web_search_calls": rec.web_search_calls,
                "url_citations_json": json.dumps(rec.url_citations, ensure_ascii=False),
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": cached,
                "output_tokens": usage.get("output_tokens"),
                "token_cost_usd": round(tok_cost, 6),
                "web_search_tool_cost_usd": round(ws_cost, 6),
                "total_cost_usd": round(total, 6),
                "batch_discount_multiplier": batch_discount,
                "model": batch.model,
            }
            w.writerow(row)
            jf.write(json.dumps({"row": row, "error": rec.error}, ensure_ascii=False) + "\n")

    print(f"Wrote batch raw output: {output_path}")
    print(f"Wrote batch results:    {csv_path}")
    print(f"Wrote batch results:    {jsonl_path}")
    return 0


def main() -> int:
    load_dotenv(override=False)
    _ensure_utf8()

    parser = argparse.ArgumentParser(description="Run Irene sample via OpenAI Batch API (/v1/responses).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a batch job from a sample CSV.")
    p_create.add_argument("--sample", default="data/irene_sample_9.csv")
    p_create.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))
    p_create.add_argument("--rubric-file", default=os.environ.get("MANUAV_RUBRIC_FILE"))
    p_create.add_argument("--max-tool-calls", type=int, default=3)
    p_create.add_argument("--reasoning-effort", default=os.environ.get("MANUAV_REASONING_EFFORT") or None)
    p_create.add_argument("--prompt-cache", action="store_true", default=(os.environ.get("MANUAV_PROMPT_CACHE", "") == "1"))
    p_create.add_argument("--prompt-cache-retention", default=os.environ.get("MANUAV_PROMPT_CACHE_RETENTION") or None)
    p_create.add_argument("--custom-id-prefix", default="irene")
    p_create.add_argument("--suffix", default="batch")
    p_create.add_argument("--metadata", default="")
    p_create.add_argument(
        "--no-web-search",
        dest="enable_web_search",
        action="store_false",
        help="Create a batch that does NOT use the web search tool (Batch API currently rejects web search tools).",
    )
    p_create.set_defaults(enable_web_search=True)
    p_create.set_defaults(func=cmd_create)

    p_status = sub.add_parser("status", help="Check batch status.")
    p_status.add_argument("batch_id")
    p_status.set_defaults(func=cmd_status)

    p_fetch = sub.add_parser("fetch", help="Download output and parse results.")
    p_fetch.add_argument("batch_id")
    p_fetch.add_argument("--suffix", default="batch")
    p_fetch.add_argument("--apply-discount", action="store_true", default=True, help="Apply 50% discount multiplier when estimating costs (default true).")
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


