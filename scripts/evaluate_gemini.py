import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from manuav_eval.gemini_evaluator import evaluate_company_gemini, evaluate_company_gemini_with_debug
from manuav_eval.rubric_loader import DEFAULT_RUBRIC_FILE
from manuav_eval.costing import compute_gemini_cost_usd, gemini_pricing_from_env


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Single-call Manuav company evaluator using Google Gemini.")
    parser.add_argument("url", help="Company website URL (e.g., https://example.com)")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        help="Gemini model (default: env GEMINI_MODEL or gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--rubric-file",
        default=os.environ.get("MANUAV_RUBRIC_FILE", str(DEFAULT_RUBRIC_FILE)),
        help="Path to rubric file (default: env MANUAV_RUBRIC_FILE or the default rubric)",
    )
    parser.add_argument(
        "--no-cost",
        action="store_true",
        help="Do not print estimated USD cost to stderr (JSON output remains unchanged).",
    )
    parser.add_argument(
        "--debug-grounding",
        action="store_true",
        help="Write grounding debug info (web_search_queries and grounding_chunks) to outputs/<timestamp>_gemini_grounding_debug.json",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Missing GEMINI_API_KEY env var.", file=sys.stderr)
        return 2

    try:
        if args.debug_grounding:
            result, usage, search_queries, debug = evaluate_company_gemini_with_debug(
                args.url, model_name=args.model, rubric_file=args.rubric_file
            )
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = out_dir / f"{stamp}_gemini_grounding_debug.json"
            debug_payload = {
                "model": args.model,
                "input_url": result.get("input_url"),
                "search_queries_count": search_queries,
                "web_search_queries": debug.get("web_search_queries", []),
                "grounding_chunks": debug.get("grounding_chunks", []),
            }
            debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote grounding debug: {debug_path}", file=sys.stderr)
        else:
            result, usage, search_queries = evaluate_company_gemini(args.url, model_name=args.model, rubric_file=args.rubric_file)
        
        if not args.no_cost:
            pricing = gemini_pricing_from_env(os.environ)
            cost = compute_gemini_cost_usd(usage, pricing, search_queries=search_queries)
            
            in_tok = getattr(usage, "prompt_token_count", 0)
            out_tok = getattr(usage, "candidates_token_count", 0)
            
            print(
                f"Estimated cost_usd={cost:.6f} (input={in_tok}, output={out_tok}, search_queries={search_queries})",
                file=sys.stderr,
            )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"Error evaluating company: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
