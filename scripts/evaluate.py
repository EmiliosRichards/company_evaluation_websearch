import argparse
import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from manuav_eval import evaluate_company as core_evaluate_company
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
        help="Path to rubric file (default: env MANUAV_RUBRIC_FILE or rubrics/manuav_rubric_v1.md)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 2

    result = core_evaluate_company(args.url, args.model, rubric_file=args.rubric_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


