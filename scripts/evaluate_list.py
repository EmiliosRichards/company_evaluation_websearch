import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

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


def _is_probably_url_list_file(path: Path) -> bool:
    # Heuristic: .txt/.list is treated as a URL list; .csv is treated as CSV.
    suffix = path.suffix.lower()
    return suffix in {".txt", ".list", ".urls"}


def _normalize_for_dedupe(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    # Strip scheme and trailing slash; keep path/query as-is (so distinct URLs stay distinct).
    if u.lower().startswith(("http://", "https://")):
        try:
            pu = urlparse(u)
            u = pu.netloc + (pu.path or "")
            if pu.query:
                u += "?" + pu.query
        except Exception:
            u = u
    u = u.strip().rstrip("/")
    return u.lower()


def _load_url_list(path: Path) -> List[Dict[str, str]]:
    # Returns rows shaped like DictReader rows for downstream compatibility.
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append({"Website": s})
    return rows


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_rows(
    input_path: Path,
    *,
    input_format: str | None,
) -> List[Dict[str, str]]:
    fmt = (input_format or "auto").strip().lower()
    if fmt not in {"auto", "csv", "txt"}:
        raise SystemExit(f"Unsupported --input-format {input_format!r}. Use: auto/csv/txt.")

    if fmt == "auto":
        fmt = "txt" if _is_probably_url_list_file(input_path) else "csv"

    if fmt == "txt":
        return _load_url_list(input_path)
    return _load_csv_rows(input_path)


def _iter_processed_websites_from_jsonl(jsonl_path: Path) -> Iterable[str]:
    if not jsonl_path.exists():
        return []

    def gen() -> Iterable[str]:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                w = (rec.get("website") or "").strip()
                if w:
                    yield w

    return gen()


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a list of companies using the one-call evaluator.\n\n"
            "Inputs:\n"
            "- CSV (default): uses columns like Website/Firma/Manuav-Score (Irene file)\n"
            "- TXT: one URL per line\n\n"
            "By default this script writes JSONL + a flattened CSV to outputs/ and prints MAE when a score column is present."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/irene_sample_9.csv",
        help="(Deprecated) Path to sample CSV created by make_irene_sample.py. Prefer --input or env MANUAV_INPUT_PATH.",
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("MANUAV_INPUT_PATH") or None,
        help="Path to input file (CSV or TXT). Env: MANUAV_INPUT_PATH. Overrides --sample if set.",
    )
    parser.add_argument(
        "--input-format",
        default=os.environ.get("MANUAV_INPUT_FORMAT") or "auto",
        help="Input format: auto (default), csv, txt. Env: MANUAV_INPUT_FORMAT",
    )
    parser.add_argument(
        "--url-column",
        default=os.environ.get("MANUAV_URL_COLUMN", "Website"),
        help="CSV column name that contains the URL. Ignored for TXT. Default: Website. Env: MANUAV_URL_COLUMN",
    )
    parser.add_argument(
        "--name-column",
        default=os.environ.get("MANUAV_NAME_COLUMN", "Firma"),
        help="CSV column name for display name. Default: Firma. Env: MANUAV_NAME_COLUMN",
    )
    parser.add_argument(
        "--score-column",
        default=os.environ.get("MANUAV_SCORE_COLUMN", "Manuav-Score"),
        help="CSV column name with reference score (for MAE). Set to empty to disable. Default: Manuav-Score. Env: MANUAV_SCORE_COLUMN",
    )
    parser.add_argument(
        "--bucket-column",
        default=os.environ.get("MANUAV_BUCKET_COLUMN", "bucket"),
        help="Optional CSV column for bucket labels (low/mid/high). Default: bucket. Env: MANUAV_BUCKET_COLUMN",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=(os.environ.get("MANUAV_DEDUPE", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Dedupe rows by normalized URL before evaluating. Env: MANUAV_DEDUPE=1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ["MANUAV_LIMIT"]) if os.environ.get("MANUAV_LIMIT") else None,
        help="Optional cap on number of rows to evaluate after filtering/dedupe. Env: MANUAV_LIMIT",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=(os.environ.get("MANUAV_RESUME", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Resume from existing --out JSONL: skip URLs already present and append to outputs. Env: MANUAV_RESUME=1",
    )
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

    # Flex can be slower; default to a larger timeout if not set explicitly.
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None and (args.service_tier or "").strip().lower() == "flex":
        timeout_seconds = 900.0

    input_path = Path(args.input) if args.input else Path(args.sample)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = _run_stamp()
    suffix = _suffix_slug(args.suffix)
    stem = stamp if not suffix else f"{stamp}_{suffix}"

    out_path = Path(args.out) if args.out else (out_dir / f"{stem}.jsonl")
    out_csv_path = Path(args.out_csv) if args.out_csv else (out_dir / f"{stem}.csv")

    rows = _load_rows(input_path, input_format=args.input_format)

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
        "service_tier",
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

    processed = set()
    if args.resume:
        processed = {_normalize_for_dedupe(u) for u in _iter_processed_websites_from_jsonl(out_path)}
        processed.discard("")

    # Optional dedupe: keep the first occurrence.
    filtered_rows: List[Dict[str, str]] = []
    seen = set()
    for r in rows:
        website = (r.get(args.url_column) or r.get("Website") or "").strip()
        if not website:
            continue
        key = _normalize_for_dedupe(website) if args.dedupe or args.resume else website
        if not key:
            continue
        if args.resume and key in processed:
            continue
        if args.dedupe:
            if key in seen:
                continue
            seen.add(key)
        filtered_rows.append(r)

    if args.limit is not None:
        filtered_rows = filtered_rows[: max(0, args.limit)]

    # If resuming and output files already exist, append; else create new.
    out_mode = "a" if (args.resume and out_path.exists()) else "w"
    csv_mode = "a" if (args.resume and out_csv_path.exists()) else "w"

    with out_path.open(out_mode, encoding="utf-8") as out, out_csv_path.open(csv_mode, encoding="utf-8", newline="") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=csv_fieldnames, extrasaction="ignore")
        if csv_mode == "w":
            writer.writeheader()

        for i, r in enumerate(filtered_rows, start=1):
            website = (r.get(args.url_column) or r.get("Website") or "").strip()
            name = (r.get(args.name_column) or r.get("Firma") or "").strip()
            score_col = (args.score_column or "").strip()
            irene_score = _to_float(r.get(score_col, "")) if score_col else None
            if not website:
                continue

            print(f"[{i}/{len(filtered_rows)}] Evaluating: {name} | {website}")
            if args.debug_web_search:
                model_result, usage, ws_debug = evaluate_company_with_usage_and_web_search_debug(
                    website,
                    args.model,
                    rubric_file=args.rubric_file,
                    max_tool_calls=args.max_tool_calls,
                    reasoning_effort=args.reasoning_effort,
                    prompt_cache=args.prompt_cache,
                    prompt_cache_retention=args.prompt_cache_retention,
                    service_tier=args.service_tier,
                    timeout_seconds=timeout_seconds,
                    flex_max_retries=args.flex_max_retries,
                    flex_fallback_to_auto=args.flex_fallback_to_auto,
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
                    service_tier=args.service_tier,
                    timeout_seconds=timeout_seconds,
                    flex_max_retries=args.flex_max_retries,
                    flex_fallback_to_auto=args.flex_fallback_to_auto,
                )
            model_score = float(model_result.get("manuav_fit_score", 0.0))
            token_cost_usd = compute_cost_usd(usage, pricing)
            web_search_tool_cost_usd = compute_web_search_tool_cost_usd(web_search_calls, tool_pricing)
            cost_usd = token_cost_usd + web_search_tool_cost_usd

            cached_tokens = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
            reasoning_tokens = int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0)

            record = {
                "bucket": (r.get(args.bucket_column) or r.get("bucket") or "").strip(),
                "firma": name,
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
            if irene_score is not None:
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
                    "irene_score": record["irene_score"] if record["irene_score"] is not None else "",
                    "model_score": record["model_score"],
                    "company_name": model_result.get("company_name"),
                    "input_url": model_result.get("input_url"),
                    "confidence": model_result.get("confidence"),
                    "reasoning": model_result.get("reasoning"),
                    "url_citations_json": json.dumps(sources, ensure_ascii=False),
                    "rubric_file": args.rubric_file,
                    "model": args.model,
                    "service_tier": args.service_tier,
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
    if pairs:
        print(f"Compared {len(pairs)} rows. MAE={_mae(pairs):.2f}")
    else:
        print("No reference score column values found; MAE not computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


