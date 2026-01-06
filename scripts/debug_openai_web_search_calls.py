import argparse
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from manuav_eval.evaluator import BASE_SYSTEM_PROMPT, _extract_json_text  # type: ignore
from manuav_eval.rubric_loader import load_rubric_text
from manuav_eval.schema import json_schema_text_config


def _safe(obj: Any) -> Any:
    """Best-effort conversion of SDK objects to plain JSON-serializable data."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> int:
    load_dotenv(override=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Debug OpenAI Responses output to count web_search_call items.")
    p.add_argument("url", help="Company website URL")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07"))
    p.add_argument("--max-tool-calls", type=int, default=3)
    p.add_argument("--rubric-file", default=os.environ.get("MANUAV_RUBRIC_FILE"))
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    rubric_path, rubric_text = load_rubric_text(args.rubric_file)
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nRubric file: {rubric_path}\n\n{rubric_text}\n"

    normalized_url = args.url.strip()
    if normalized_url and not normalized_url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    tool_budget_line = f"- Tool-call budget: you can make at most {args.max_tool_calls} web search tool call(s). Use them wisely.\n"

    user_prompt = f"""\
Evaluate this company for Manuav using web research and the Manuav Fit logic.

Instructions:
- Use the web search tool to research:
  - the company website itself (product/service, ICP, pricing, cases, careers, legal/imprint)
  - and the broader web for each rubric category (DACH presence, operational status, TAM, competition, innovation, economics, onboarding, pitchability, risk).
{tool_budget_line}- Be conservative when evidence is missing.
- In the JSON output:
  - set input_url exactly to the Company website URL below
  - provide a concise reasoning narrative for the score (why it is high/low; mention key rubric drivers and any critical unknowns)
  - provide sources_visited as the list of URLs you relied on (with titles). Include primary sources where possible.

Company website URL: {normalized_url}
"""

    client = OpenAI()
    resp = client.responses.create(
        model=args.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search_preview"}],
        max_tool_calls=args.max_tool_calls,
        text=json_schema_text_config(),
    )

    output_items = getattr(resp, "output", None) or []
    types_list = [getattr(it, "type", None) for it in output_items]
    ws_items = [it for it in output_items if getattr(it, "type", None) == "web_search_call"]

    print(f"model={args.model}")
    print(f"requested max_tool_calls={args.max_tool_calls}")
    print(f"response.output items={len(output_items)}")
    print(f"output types={types_list}")
    print(f"web_search_call count={len(ws_items)}")

    # Show minimal details per web_search_call item.
    for i, it in enumerate(ws_items, start=1):
        dumped = _safe(it)
        # Keep it small: show only a subset of keys if possible.
        if isinstance(dumped, dict):
            subset: Dict[str, Any] = {}
            for k in ("id", "status", "type", "query", "q", "search_query"):
                if k in dumped:
                    subset[k] = dumped[k]
            subset["keys"] = sorted(dumped.keys())
            print(f"\nweb_search_call[{i}] {json.dumps(subset, ensure_ascii=False, indent=2)}")
        else:
            print(f"\nweb_search_call[{i}] {dumped}")

    # Confirm we can still parse the model JSON.
    try:
        parsed = json.loads(_extract_json_text(resp))
        print("\nparsed_json_ok=true")
        print(f"parsed_input_url={parsed.get('input_url')}")
    except Exception as e:
        print("\nparsed_json_ok=false")
        print(f"parse_error={e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


