from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

from manuav_eval import evaluate_company_with_usage_and_web_search_debug
from manuav_eval.rubric_loader import DEFAULT_RUBRIC_FILE


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _suffix_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
    return "".join(safe).strip("_")


def _detect_csv_delimiter(sample: str) -> str:
    header = ""
    for line in (sample or "").splitlines():
        if line.strip():
            header = line
            break
    if header:
        counts = {d: header.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        if counts[best] > 0:
            return best
    return ","


def _iter_csv_rows(path: Path, delimiter: str | None) -> Iterable[Dict[str, str]]:
    # Use utf-8-sig to handle BOM-prefixed CSV headers.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        delim = (delimiter or "").strip()
        if not delim:
            head = f.read(4096)
            f.seek(0)
            delim = _detect_csv_delimiter(head)
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            yield r


def _iter_url_list(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield {"Website": s}


def _normalize_for_dedupe(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = u.rstrip("/")
    return u.lower()


def _extract_queries_and_opens(ws_debug: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (queries, opens) in call order."""
    queries: List[str] = []
    opens: List[str] = []
    calls = ws_debug.get("calls") or []
    for c in calls:
        if not isinstance(c, dict):
            continue
        kind = (c.get("kind") or "").strip().lower()
        if kind == "query":
            q = (c.get("query") or "").strip()
            if q:
                queries.append(q)
            else:
                # Best-effort: query sometimes lives inside action_hint.
                ah = str(c.get("action_hint") or "")
                m = re.search(r"'query'\s*:\s*'([^']+)'", ah)
                if m:
                    queries.append(m.group(1))
        elif kind == "open":
            u = (c.get("url") or "").strip()
            if u:
                opens.append(u)
            else:
                ah = str(c.get("action_hint") or "")
                m = re.search(r"'url'\s*:\s*'([^']+)'", ah)
                if m:
                    opens.append(m.group(1))
    return queries, opens


def _reservoir_sample_unique(
    rows: Iterable[Dict[str, str]],
    *,
    url_column: str,
    k: int,
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(int(seed))
    reservoir: List[Dict[str, str]] = []
    seen: set[str] = set()
    n_unique = 0
    for r in rows:
        url = (r.get(url_column) or r.get("Website") or "").strip()
        if not url:
            continue
        key = _normalize_for_dedupe(url)
        if not key or key in seen:
            continue
        seen.add(key)
        n_unique += 1
        if len(reservoir) < k:
            reservoir.append(r)
        else:
            j = rng.randrange(n_unique)
            if j < k:
                reservoir[j] = r
    return reservoir


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description=(
            "Trace and summarize OpenAI web search tool behavior (queries vs open/visit) per evaluation.\n\n"
            "This script is for understanding how thorough the model is at your current tool budget.\n"
            "It records the sequence of web_search_call actions (query/open), counts, and timing."
        )
    )
    ap.add_argument("--input", required=True, help="CSV or TXT input file")
    ap.add_argument("--input-format", default="auto", help="auto/csv/txt (default auto)")
    ap.add_argument("--csv-delimiter", default=None, help="CSV delimiter override (e.g. ';')")
    ap.add_argument("--url-column", default=os.environ.get("MANUAV_URL_COLUMN", "Website"), help="URL column (CSV)")
    ap.add_argument("--name-column", default=os.environ.get("MANUAV_NAME_COLUMN", "Firma"), help="Name column (CSV)")
    ap.add_argument("--sample", type=int, default=10, help="How many companies to trace (default 10)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    ap.add_argument("--suffix", default="trace", help="Output filename suffix")
    ap.add_argument(
        "--include-sources",
        action="store_true",
        default=False,
        help="Ask the model to include a compact `sources` list in its JSON (debug/audit only; increases output tokens).",
    )
    ap.add_argument(
        "--force-second-query",
        action="store_true",
        default=False,
        help=(
            "Debug: ask the model to run a second disambiguation query if the first results are ambiguous. "
            "Useful for tricky same-name entities."
        ),
    )

    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), help="OpenAI model")
    ap.add_argument(
        "--rubric-file",
        default=os.environ.get("MANUAV_RUBRIC_FILE", str(DEFAULT_RUBRIC_FILE)),
        help="Rubric file",
    )
    ap.add_argument(
        "--max-tool-calls",
        type=int,
        default=int(os.environ.get("MANUAV_MAX_TOOL_CALLS") or 3),
        help="Max tool calls for the evaluation (default: env MANUAV_MAX_TOOL_CALLS or 3)",
    )
    ap.add_argument("--reasoning-effort", default=os.environ.get("MANUAV_REASONING_EFFORT") or None)
    ap.add_argument(
        "--prompt-cache",
        action="store_true",
        default=(os.environ.get("MANUAV_PROMPT_CACHE", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
    )
    ap.add_argument("--prompt-cache-retention", default=os.environ.get("MANUAV_PROMPT_CACHE_RETENTION") or None)
    ap.add_argument("--service-tier", default=os.environ.get("MANUAV_SERVICE_TIER", "auto"))
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ["MANUAV_OPENAI_TIMEOUT_SECONDS"]) if os.environ.get("MANUAV_OPENAI_TIMEOUT_SECONDS") else None,
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    fmt = (args.input_format or "auto").strip().lower()
    if fmt == "auto":
        fmt = "txt" if input_path.suffix.lower() in {".txt", ".list", ".urls"} else "csv"
    if fmt not in {"csv", "txt"}:
        raise SystemExit("input-format must be auto/csv/txt")

    if fmt == "csv":
        rows_iter = _iter_csv_rows(input_path, args.csv_delimiter)
    else:
        rows_iter = _iter_url_list(input_path)

    k = max(1, int(args.sample))
    rows = _reservoir_sample_unique(rows_iter, url_column=args.url_column, k=k, seed=int(args.seed))
    if not rows:
        print("No rows with URL found.", file=sys.stderr)
        return 2

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp()
    suffix = _suffix_slug(args.suffix)
    stem = f"{stamp}_{suffix}" if suffix else stamp
    out_jsonl = out_dir / f"{stem}.jsonl"
    out_csv = out_dir / f"{stem}.csv"

    csv_fields = [
        "run_id",
        "website",
        "name",
        "model",
        "service_tier",
        "max_tool_calls",
        "duration_seconds",
        "tool_calls_total",
        "query_calls",
        "open_calls",
        "unknown_calls",
        "queries_json",
        "opens_json",
        "company_name",
        "manuav_fit_score",
        "confidence",
        "reasoning",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "error",
    ]

    q_counts: List[int] = []
    o_counts: List[int] = []
    total_counts: List[int] = []
    durations: List[float] = []

    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()

        for i, r in enumerate(rows, start=1):
            website = (r.get(args.url_column) or r.get("Website") or "").strip()
            name = (r.get(args.name_column) or r.get("Firma") or "").strip()
            if not website:
                continue

            print(f"[{i}/{len(rows)}] Tracing: {name} | {website}", flush=True)
            t0 = time.monotonic()
            error = None
            try:
                extra = None
                if args.force_second_query:
                    # Keep generic: ask for a second query that includes 'impressum', 'gmbh', and a location hint if relevant.
                    # This is intentionally light-touch and only used in the tracer.
                    extra = (
                        "DEBUG REQUIREMENT: Perform TWO distinct web searches before scoring.\n"
                        "1) Search using the domain/company name (e.g., '<domain>').\n"
                        "2) Search again using a disambiguation query that adds legal-entity/location hints, e.g. "
                        "'<name> GmbH impressum', '<name> Munich impressum', '<name> HRB'.\n"
                        "If the two searches surface conflicting/similarly-named entities, prefer sources that match the provided domain and "
                        "DACH legal/imprint details; otherwise explicitly note uncertainty."
                    )
                result, usage, ws_debug = evaluate_company_with_usage_and_web_search_debug(
                    website,
                    args.model,
                    rubric_file=args.rubric_file,
                    max_tool_calls=args.max_tool_calls,
                    reasoning_effort=args.reasoning_effort,
                    prompt_cache=args.prompt_cache,
                    prompt_cache_retention=args.prompt_cache_retention,
                    service_tier=args.service_tier,
                    timeout_seconds=args.timeout_seconds,
                    include_sources=bool(args.include_sources),
                    extra_user_instructions=extra,
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                result = {"input_url": website, "company_name": "", "manuav_fit_score": 0, "confidence": "low", "reasoning": ""}
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
                ws_debug = {"completed": 0, "by_kind_completed": {}, "calls": []}

            dt = time.monotonic() - t0

            by_kind_completed = ws_debug.get("by_kind_completed") or {}
            qn = int(by_kind_completed.get("query", 0) or 0)
            on = int(by_kind_completed.get("open", 0) or 0)
            un = int(by_kind_completed.get("unknown", 0) or 0)
            total = int(ws_debug.get("completed", 0) or 0)

            queries, opens = _extract_queries_and_opens(ws_debug)

            q_counts.append(qn)
            o_counts.append(on)
            total_counts.append(total)
            durations.append(dt)

            record = {
                "run_id": stem,
                "website": website,
                "name": name,
                "model": args.model,
                "service_tier": args.service_tier,
                "max_tool_calls": int(args.max_tool_calls),
                "duration_seconds": dt,
                "web_search_debug": ws_debug,
                "queries": queries,
                "opens": opens,
                "result": result,
                "usage": {
                    "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                    "cached_input_tokens": int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0),
                    "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                    "reasoning_tokens": int(getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                },
                "error": error,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()

            w.writerow(
                {
                    "run_id": stem,
                    "website": website,
                    "name": name,
                    "model": args.model,
                    "service_tier": args.service_tier,
                    "max_tool_calls": int(args.max_tool_calls),
                    "duration_seconds": round(dt, 3),
                    "tool_calls_total": total,
                    "query_calls": qn,
                    "open_calls": on,
                    "unknown_calls": un,
                    "queries_json": json.dumps(queries, ensure_ascii=False),
                    "opens_json": json.dumps(opens, ensure_ascii=False),
                    "company_name": result.get("company_name"),
                    "manuav_fit_score": result.get("manuav_fit_score"),
                    "confidence": result.get("confidence"),
                    "reasoning": result.get("reasoning"),
                    "input_tokens": record["usage"]["input_tokens"],
                    "cached_input_tokens": record["usage"]["cached_input_tokens"],
                    "output_tokens": record["usage"]["output_tokens"],
                    "reasoning_tokens": record["usage"]["reasoning_tokens"],
                    "total_tokens": record["usage"]["total_tokens"],
                    "error": error or "",
                }
            )
            cf.flush()
            time.sleep(max(0.0, float(args.sleep)))

    # Summary
    print(f"\nWrote trace JSONL: {out_jsonl}", flush=True)
    print(f"Wrote trace CSV:   {out_csv}", flush=True)
    print("\nSummary:", flush=True)
    print(f"- companies: {len(total_counts)}", flush=True)
    print(f"- avg duration: {mean(durations):.2f}s (median {median(durations):.2f}s)", flush=True)
    print(f"- avg query calls: {mean(q_counts):.2f} (distribution {dict(sorted(Counter(q_counts).items()))})", flush=True)
    print(f"- avg open calls:  {mean(o_counts):.2f} (distribution {dict(sorted(Counter(o_counts).items()))})", flush=True)
    print(f"- avg total tool calls: {mean(total_counts):.2f} (distribution {dict(sorted(Counter(total_counts).items()))})", flush=True)
    full_budget = sum(1 for t in total_counts if t >= int(args.max_tool_calls))
    print(f"- pct hitting tool budget (>= {args.max_tool_calls}): {100.0*full_budget/len(total_counts):.1f}%", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


