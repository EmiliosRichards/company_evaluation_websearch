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

from manuav_eval.costing import compute_gemini_cost_usd, gemini_pricing_from_env
from manuav_eval.gemini_evaluator import evaluate_company_gemini
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
    out = "".join(safe).strip("_")
    return out


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run the Gemini evaluator on Irene's 9-row sample and compare scores.")
    parser.add_argument("--sample", default="data/irene_sample_9.csv", help="Path to sample CSV created by make_irene_sample.py")
    parser.add_argument("--out", default=None, help="Where to write JSONL results (default: outputs/<timestamp>[_suffix].jsonl)")
    parser.add_argument("--out-csv", default=None, help="Where to write CSV results (default: outputs/<timestamp>[_suffix].csv)")
    parser.add_argument("-s", "--suffix", default="", help="Optional suffix added to output filenames (default: none). Example: -s gemini")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"), help="Gemini model")
    parser.add_argument(
        "--rubric-file",
        default=os.environ.get("MANUAV_RUBRIC_FILE", str(DEFAULT_RUBRIC_FILE)),
        help="Path to rubric file (default: env MANUAV_RUBRIC_FILE or rubrics/manuav_rubric_v4_en.md)",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between calls")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Missing GEMINI_API_KEY env var.")

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

    pairs: List[Tuple[float, float]] = []
    pricing = gemini_pricing_from_env(os.environ)

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
        "sources_visited_json",
        "rubric_file",
        "model",
        "input_tokens",
        "output_tokens",
        "search_queries",
        "cost_usd",
        "price_input_per_1m",
        "price_output_per_1m",
        "price_search_per_1k",
    ]

    with out_path.open("w", encoding="utf-8") as out, out_csv_path.open("w", encoding="utf-8", newline="") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, r in enumerate(rows, start=1):
            website = (r.get("Website") or "").strip()
            irene_score = _to_float(r.get("Manuav-Score", ""))
            if not website or irene_score is None:
                continue

            print(f"[{i}/{len(rows)}] Evaluating (Gemini): {r.get('Firma','').strip()} | {website}")
            model_result, usage, search_queries = evaluate_company_gemini(
                website,
                model_name=args.model,
                rubric_file=args.rubric_file,
            )

            model_score = float(model_result.get("manuav_fit_score", 0.0))
            pairs.append((irene_score, model_score))

            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            cost_usd = compute_gemini_cost_usd(usage, pricing, search_queries=search_queries)

            record = {
                "bucket": (r.get("bucket") or "").strip(),
                "firma": (r.get("Firma") or "").strip(),
                "website": website,
                "irene_score": irene_score,
                "model_score": model_score,
                "model_confidence": model_result.get("confidence"),
                "reasoning": model_result.get("reasoning"),
                "sources_visited": model_result.get("sources_visited"),
                "usage": {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "search_queries": search_queries,
                },
                "cost_usd": round(cost_usd, 6),
                "raw": model_result,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            sources = model_result.get("sources_visited") or []
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
                    "sources_visited_json": json.dumps(sources, ensure_ascii=False),
                    "rubric_file": args.rubric_file,
                    "model": args.model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "search_queries": search_queries,
                    "cost_usd": round(cost_usd, 6),
                    "price_input_per_1m": pricing.input_usd_per_1m,
                    "price_output_per_1m": pricing.output_usd_per_1m,
                    "price_search_per_1k": pricing.search_usd_per_1k,
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


