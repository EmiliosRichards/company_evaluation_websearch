import argparse
import csv
import json
import os
import random
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
    # Materialize all rows (OK for smaller files).
    return list(_iter_csv_rows(path, csv_delimiter=None))


def _detect_csv_delimiter(sample: str) -> str:
    # Prefer using the header line only (data lines may contain lots of commas inside fields).
    header = ""
    for line in (sample or "").splitlines():
        if line.strip():
            header = line
            break

    # Heuristic: pick the delimiter with the highest count in the header.
    if header:
        counts = {d: header.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        if counts[best] > 0:
            return best

    # Best-effort: try Sniffer as a fallback.
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        if getattr(dialect, "delimiter", None):
            return dialect.delimiter
    except Exception:
        pass

    return ","


def _iter_csv_rows(path: Path, csv_delimiter: str | None) -> Iterable[Dict[str, str]]:
    # Use utf-8-sig to handle BOM-prefixed CSV headers.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # Auto-detect delimiter unless explicitly provided.
        delim = (csv_delimiter or "").strip()
        if not delim:
            head = f.read(4096)
            f.seek(0)
            delim = _detect_csv_delimiter(head)
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            yield r


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


def _iter_rows(
    input_path: Path,
    *,
    input_format: str | None,
    csv_delimiter: str | None,
) -> Iterable[Dict[str, str]]:
    fmt = (input_format or "auto").strip().lower()
    if fmt not in {"auto", "csv", "txt"}:
        raise SystemExit(f"Unsupported --input-format {input_format!r}. Use: auto/csv/txt.")

    if fmt == "auto":
        fmt = "txt" if _is_probably_url_list_file(input_path) else "csv"

    if fmt == "txt":
        return _load_url_list(input_path)
    return _iter_csv_rows(input_path, csv_delimiter=csv_delimiter)


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
        "--csv-delimiter",
        default=os.environ.get("MANUAV_CSV_DELIMITER") or None,
        help="CSV delimiter override (e.g. ';'). Default: auto-detect. Env: MANUAV_CSV_DELIMITER",
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
        help="CSV column name with reference score (for MAE). Use '-' (or set env empty) to disable. Default: Manuav-Score. Env: MANUAV_SCORE_COLUMN",
    )
    parser.add_argument(
        "--bucket-column",
        default=os.environ.get("MANUAV_BUCKET_COLUMN", "bucket"),
        help="Optional CSV column for bucket labels (low/mid/high). Use '-' to disable. Default: bucket. Env: MANUAV_BUCKET_COLUMN",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=int(os.environ["MANUAV_RANDOM_SAMPLE"]) if os.environ.get("MANUAV_RANDOM_SAMPLE") else None,
        help="Randomly sample N unique rows (URL present) from the input before evaluating. Env: MANUAV_RANDOM_SAMPLE",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("MANUAV_SAMPLE_SEED", "42")),
        help="RNG seed for --random-sample (default 42). Env: MANUAV_SAMPLE_SEED",
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
        help=(
            "Optional cap on tool calls (web searches) within each single LLM call. "
            "This is a cost guardrail; the model may use fewer. Env: MANUAV_MAX_TOOL_CALLS"
        ),
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=(os.environ.get("MANUAV_CONTINUE_ON_ERROR", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Continue the run if an evaluation errors; write an error record instead of aborting. Env: MANUAV_CONTINUE_ON_ERROR=1",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("MANUAV_PROGRESS_EVERY", "25")),
        help="Print progress/ETA every N completed evaluations. Env: MANUAV_PROGRESS_EVERY (default 25)",
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

    rows_iter = _iter_rows(input_path, input_format=args.input_format, csv_delimiter=args.csv_delimiter)

    results: List[Dict[str, Any]] = []
    pairs: List[Tuple[float, float]] = []

    pricing = pricing_from_env(os.environ)
    tool_pricing = web_search_pricing_from_env(os.environ)

    # Flex discount: apply a multiplier to token-cost estimates only.
    # Web search tool usage is billed separately (typically $0.01 per query) and is not discounted the same way.
    flex_discount = float(os.environ.get("MANUAV_FLEX_TOKEN_DISCOUNT", "0.5") or 0.5)
    apply_flex_discount = (args.service_tier or "").strip().lower() == "flex"

    bucket_col_raw = (args.bucket_column or "").strip()
    bucket_col = "" if bucket_col_raw.lower() in {"", "-", "none", "null"} else bucket_col_raw
    include_bucket = bool(bucket_col)

    score_col_raw = (args.score_column or "").strip()
    score_col = "" if score_col_raw.lower() in {"", "-", "none", "null"} else score_col_raw
    include_score = bool(score_col)
    csv_fieldnames = [
        "run_id",
        "firma",
        "website",
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
        "web_search_tool_calls_total",
        "web_search_tool_cost_usd",
        "web_search_calls_query",
        "web_search_calls_open",
        "web_search_calls_unknown",
        "price_input_per_1m",
        "price_cached_input_per_1m",
        "price_output_per_1m",
        "price_web_search_per_1k",
        "duration_seconds",
        "flex_attempts",
        "flex_retries",
        "flex_sleep_seconds_total",
        "flex_fallback_used",
        "retry_used",
        "retry_selected",
        "error",
    ]
    if include_score:
        # Keep the legacy column name for compatibility when a reference score is provided.
        csv_fieldnames.insert(4, "irene_score")
    if include_bucket:
        csv_fieldnames.insert(1, "bucket")

    processed = set()
    if args.resume:
        processed = {_normalize_for_dedupe(u) for u in _iter_processed_websites_from_jsonl(out_path)}
        processed.discard("")

    # (score_col already normalized above)

    # Build the evaluation set:
    # - if --random-sample is set, reservoir-sample unique URLs while streaming the input
    # - else, filter/dedupe/resume over all rows
    filtered_rows: List[Dict[str, str]] = []
    seen = set()

    def _row_website(r: Dict[str, str]) -> str:
        return (r.get(args.url_column) or r.get("Website") or "").strip()

    if args.random_sample is not None:
        k = max(0, int(args.random_sample))
        rng = random.Random(int(args.seed))
        reservoir: List[Dict[str, str]] = []
        unique_seen: set[str] = set()
        n_unique = 0
        for r in rows_iter:
            website = _row_website(r)
            if not website:
                continue
            key = _normalize_for_dedupe(website)
            if not key:
                continue
            if args.resume and key in processed:
                continue
            if key in unique_seen:
                continue
            unique_seen.add(key)
            n_unique += 1
            if len(reservoir) < k:
                reservoir.append(r)
            else:
                j = rng.randrange(n_unique)
                if j < k:
                    reservoir[j] = r
        filtered_rows = reservoir
    else:
        # Optional dedupe: keep the first occurrence.
        for r in rows_iter:
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

    # If we used random sampling, write out the sampled URLs for reproducibility.
    if args.random_sample is not None:
        sample_path = out_dir / f"{stem}_sample_urls.txt"
        with sample_path.open("w", encoding="utf-8") as sf:
            for r in filtered_rows:
                website = _row_website(r)
                name = (r.get(args.name_column) or r.get("Firma") or "").strip()
                if name:
                    sf.write(f"{website}\t{name}\n")
                else:
                    sf.write(f"{website}\n")
        print(f"Wrote sampled URL list: {sample_path}", flush=True)

    # If resuming and output files already exist, append; else create new.
    out_mode = "a" if (args.resume and out_path.exists()) else "w"
    csv_mode = "a" if (args.resume and out_csv_path.exists()) else "w"

    run_started_at = time.monotonic()
    started_wall = datetime.now()
    completed_ok = 0
    completed_err = 0

    with out_path.open(out_mode, encoding="utf-8") as out, out_csv_path.open(csv_mode, encoding="utf-8", newline="") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=csv_fieldnames, extrasaction="ignore")
        if csv_mode == "w":
            writer.writeheader()

        for i, r in enumerate(filtered_rows, start=1):
            website = (r.get(args.url_column) or r.get("Website") or "").strip()
            name = (r.get(args.name_column) or r.get("Firma") or "").strip()
            irene_score = _to_float(r.get(score_col, "")) if include_score else None
            if not website:
                continue

            print(f"[{i}/{len(filtered_rows)}] Evaluating: {name} | {website}", flush=True)
            t0 = time.monotonic()
            ws_debug: Dict[str, Any] | None = None
            error: str | None = None
            retry_used = False
            retry_selected = "first"
            # Totals may include multiple attempts if retry is enabled.
            usage_input_tokens = 0
            usage_output_tokens = 0
            usage_total_tokens = 0
            cached_tokens = 0
            reasoning_tokens = 0
            token_cost_usd_raw = 0.0
            token_cost_usd = 0.0
            flex_attempts = 0
            flex_retries = 0
            flex_sleep = 0.0
            flex_fallback_used = False

            try:
                def _do_eval_once(
                    *,
                    max_tool_calls: int | None,
                    extra_user_instructions: str | None,
                    second_query_on_uncertainty: bool,
                ) -> Dict[str, Any]:
                    use_debug = bool(args.debug_web_search) or bool(extra_user_instructions and extra_user_instructions.strip())
                    if use_debug:
                        res, use, ws = evaluate_company_with_usage_and_web_search_debug(
                            website,
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
                        citations = (ws or {}).get("url_citations") or []
                        by_kind_completed = (ws or {}).get("by_kind_completed") or {}
                        q = int(by_kind_completed.get("query", 0) or 0)
                        o = int(by_kind_completed.get("open", 0) or 0)
                        u = int(by_kind_completed.get("unknown", 0) or 0)
                        billed_q = int(q)
                        tool_total = int((ws or {}).get("completed", 0) or 0)
                        flex_meta_local = (ws or {}).get("flex") if isinstance(ws, dict) else {}
                        return {
                            "model_result": res,
                            "usage": use,
                            "ws_debug": ws,
                            "url_citations": citations,
                            "web_search_calls": billed_q,
                            "web_search_calls_query": q,
                            "web_search_calls_open": o,
                            "web_search_calls_unknown": u,
                            "web_search_tool_calls_total": tool_total,
                            "flex_meta": flex_meta_local or {},
                        }

                    res, use, billed_q, citations = evaluate_company_with_usage_and_web_search_artifacts(
                        website,
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
                    return {
                        "model_result": res,
                        "usage": use,
                        "ws_debug": None,
                        "url_citations": citations,
                        "web_search_calls": int(billed_q),
                        "web_search_calls_query": 0,
                        "web_search_calls_open": 0,
                        "web_search_calls_unknown": 0,
                        "web_search_tool_calls_total": int(billed_q),
                        "flex_meta": {},
                    }

                a1 = _do_eval_once(
                    max_tool_calls=args.max_tool_calls,
                    extra_user_instructions=None,
                    second_query_on_uncertainty=bool(args.second_query_on_uncertainty),
                )
                attempts: list[Dict[str, Any]] = [a1]

                selected = a1
                conf1 = str((a1.get("model_result") or {}).get("confidence") or "").strip().lower()
                if args.retry_disambiguation_on_low_confidence and conf1 == "low":
                    retry_used = True
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
                    )
                    attempts.append(a2)
                    conf2 = str((a2.get("model_result") or {}).get("confidence") or "").strip().lower()
                    if conf2 and conf2 != "low":
                        selected = a2
                        retry_selected = "retry"

                model_result = selected["model_result"]
                usage = selected["usage"]
                url_citations = selected["url_citations"]
                ws_debug = selected["ws_debug"] if args.debug_web_search else None

                # Aggregate tool usage across attempts (billing happens for both calls).
                web_search_calls = sum(int(a.get("web_search_calls", 0) or 0) for a in attempts)
                web_search_calls_query = sum(int(a.get("web_search_calls_query", 0) or 0) for a in attempts)
                web_search_calls_open = sum(int(a.get("web_search_calls_open", 0) or 0) for a in attempts)
                web_search_calls_unknown = sum(int(a.get("web_search_calls_unknown", 0) or 0) for a in attempts)
                web_search_tool_calls_total = sum(int(a.get("web_search_tool_calls_total", 0) or 0) for a in attempts)

                # Aggregate usage + token cost across attempts.
                usage_input_tokens = sum(int(a["usage"].input_tokens) for a in attempts)
                usage_output_tokens = sum(int(a["usage"].output_tokens) for a in attempts)
                usage_total_tokens = sum(int(a["usage"].total_tokens) for a in attempts)
                cached_tokens = sum(
                    int(getattr(getattr(a["usage"], "input_tokens_details", None), "cached_tokens", 0) or 0) for a in attempts
                )
                reasoning_tokens = sum(
                    int(getattr(getattr(a["usage"], "output_tokens_details", None), "reasoning_tokens", 0) or 0) for a in attempts
                )
                token_cost_usd_raw = sum(compute_cost_usd(a["usage"], pricing) for a in attempts)
                token_cost_usd = (token_cost_usd_raw * flex_discount) if apply_flex_discount else token_cost_usd_raw

                # Flex stats (if available) are best-effort aggregates (debug path only).
                flex_attempts = sum(int((a.get("flex_meta") or {}).get("attempts", 0) or 0) for a in attempts)
                flex_retries = sum(int((a.get("flex_meta") or {}).get("retries", 0) or 0) for a in attempts)
                flex_sleep = sum(float((a.get("flex_meta") or {}).get("sleep_seconds_total", 0.0) or 0.0) for a in attempts)
                flex_fallback_used = any(bool((a.get("flex_meta") or {}).get("fallback_used", False)) for a in attempts)
            except Exception as e:
                if not args.continue_on_error:
                    raise
                error = f"{type(e).__name__}: {e}"
                model_result = {
                    "input_url": website if website.startswith("http") else f"https://{website}",
                    "company_name": "",
                    "manuav_fit_score": 0.0,
                    "confidence": "low",
                    "reasoning": "",
                }
                usage = type(
                    "_Usage",
                    (),
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "input_tokens_details": type("X", (), {"cached_tokens": 0})(),
                        "output_tokens_details": type("Y", (), {"reasoning_tokens": 0})(),
                    },
                )()
                web_search_calls = 0
                url_citations = []
                ws_debug = {"flex": {"attempts": 0, "retries": 0, "sleep_seconds_total": 0.0, "fallback_used": False}}
                web_search_calls_query = 0
                web_search_calls_open = 0
                web_search_calls_unknown = 0
                web_search_tool_calls_total = 0

            duration_seconds = time.monotonic() - t0

            model_score = float(model_result.get("manuav_fit_score", 0.0))
            web_search_tool_cost_usd = compute_web_search_tool_cost_usd(web_search_calls, tool_pricing)
            cost_usd = token_cost_usd + web_search_tool_cost_usd

            record = {
                "firma": name,
                "website": website,
                "model_score": model_score,
                "model_confidence": model_result.get("confidence"),
                "reasoning": model_result.get("reasoning"),
                "url_citations": url_citations,
                "duration_seconds": duration_seconds,
                "flex": {
                    "attempts": flex_attempts,
                    "retries": flex_retries,
                    "sleep_seconds_total": flex_sleep,
                    "fallback_used": flex_fallback_used,
                },
                "error": error,
                "usage": {
                    "input_tokens": usage_input_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": usage_output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": usage_total_tokens,
                },
                "cost_usd": round(cost_usd, 6),
                "token_cost_usd": round(token_cost_usd, 6),
                "web_search_calls": int(web_search_calls),
                "web_search_tool_calls_total": int(web_search_tool_calls_total),
                "web_search_tool_cost_usd": round(web_search_tool_cost_usd, 6),
                "web_search_calls_query": int(web_search_calls_query),
                "web_search_calls_open": int(web_search_calls_open),
                "web_search_calls_unknown": int(web_search_calls_unknown),
                "web_search_debug": ws_debug if args.debug_web_search else None,
                "retry": {"used": bool(retry_used), "selected": retry_selected},
                "raw": model_result,
            }
            if include_bucket:
                record["bucket"] = (r.get(bucket_col) or r.get("bucket") or "").strip()
            if include_score:
                record["irene_score"] = irene_score
            results.append(record)
            if irene_score is not None:
                pairs.append((irene_score, model_score))

            if error:
                completed_err += 1
            else:
                completed_ok += 1

            if args.progress_every and ((completed_ok + completed_err) % max(1, args.progress_every) == 0):
                elapsed = time.monotonic() - run_started_at
                done = completed_ok + completed_err
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = max(0, len(filtered_rows) - done)
                eta = (remaining / rate) if rate > 0 else float("inf")
                print(
                    f"Progress: {done}/{len(filtered_rows)} (ok={completed_ok}, err={completed_err}) "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )

            # JSONL (full raw)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            # CSV (flattened)
            sources = url_citations or []
            row_out: Dict[str, Any] = {
                "run_id": stem,
                "firma": record["firma"],
                "website": record["website"],
                "model_score": record["model_score"],
                "company_name": model_result.get("company_name"),
                "input_url": model_result.get("input_url"),
                "confidence": model_result.get("confidence"),
                "reasoning": model_result.get("reasoning"),
                "url_citations_json": json.dumps(sources, ensure_ascii=False),
                "rubric_file": args.rubric_file,
                "model": args.model,
                "service_tier": args.service_tier,
                "input_tokens": usage_input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": usage_output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": usage_total_tokens,
                "cost_usd": round(cost_usd, 6),
                "token_cost_usd": round(token_cost_usd, 6),
                "web_search_calls": int(web_search_calls),
                "web_search_tool_calls_total": int(web_search_tool_calls_total),
                "web_search_tool_cost_usd": round(web_search_tool_cost_usd, 6),
                "web_search_calls_query": int(web_search_calls_query),
                "web_search_calls_open": int(web_search_calls_open),
                "web_search_calls_unknown": int(web_search_calls_unknown),
                "price_input_per_1m": pricing.input_usd,
                "price_cached_input_per_1m": pricing.cached_input_usd,
                "price_output_per_1m": pricing.output_usd,
                "price_web_search_per_1k": tool_pricing.per_1k_calls_usd,
                "duration_seconds": round(duration_seconds, 3),
                "flex_attempts": flex_attempts,
                "flex_retries": flex_retries,
                "flex_sleep_seconds_total": round(flex_sleep, 3),
                "flex_fallback_used": int(flex_fallback_used),
                "retry_used": int(bool(retry_used)),
                "retry_selected": retry_selected,
                "error": error or "",
            }
            if include_bucket:
                row_out["bucket"] = record.get("bucket", "")
            if include_score:
                row_out["irene_score"] = irene_score if irene_score is not None else ""
            writer.writerow(row_out)
            out_csv.flush()

            time.sleep(max(0.0, args.sleep))

    print(f"\nWrote results (jsonl): {out_path}", flush=True)
    print(f"Wrote results (csv):   {out_csv_path}", flush=True)
    total_elapsed = time.monotonic() - run_started_at
    print(f"Run time: {total_elapsed/60:.1f} minutes (started {started_wall.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    print(f"Completed: ok={completed_ok}, err={completed_err}, total={completed_ok + completed_err}", flush=True)
    if pairs:
        print(f"Compared {len(pairs)} rows. MAE={_mae(pairs):.2f}", flush=True)
    else:
        print("No reference score column values found; MAE not computed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


