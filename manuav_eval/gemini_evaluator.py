import json
import os
from typing import Any, Dict, Tuple

from google import genai
from google.genai import types

from .rubric_loader import load_rubric_text
from .schema import OUTPUT_SCHEMA


def _gemini_schema(schema: Any) -> Any:
    """
    Gemini's response_schema is not full JSON Schema.
    In particular, it rejects fields like `additionalProperties`.
    We strip unsupported keys recursively while keeping the core shape constraints.
    """
    if isinstance(schema, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in schema.items():
            if k in ("additionalProperties",):
                continue
            cleaned[k] = _gemini_schema(v)
        return cleaned
    if isinstance(schema, list):
        return [_gemini_schema(x) for x in schema]
    return schema


def _extract_grounding_debug(response: Any) -> Dict[str, Any]:
    """Extract grounding details (queries + chunks) from a Gemini response."""
    out: Dict[str, Any] = {"web_search_queries": [], "grounding_chunks": []}
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return out

    gm = getattr(candidates[0], "grounding_metadata", None)
    if gm is None:
        return out

    # Best-effort: pydantic model_dump (google-genai uses pydantic models).
    if hasattr(gm, "model_dump"):
        try:
            dumped = gm.model_dump()
            out["web_search_queries"] = dumped.get("web_search_queries") or []
            out["grounding_chunks"] = dumped.get("grounding_chunks") or []
            return out
        except Exception:
            pass

    # Fallback: attribute access.
    if hasattr(gm, "web_search_queries"):
        try:
            out["web_search_queries"] = list(getattr(gm, "web_search_queries") or [])
        except Exception:
            out["web_search_queries"] = []
    if hasattr(gm, "grounding_chunks"):
        try:
            out["grounding_chunks"] = list(getattr(gm, "grounding_chunks") or [])
        except Exception:
            out["grounding_chunks"] = []
    return out

# Shared base logic (same as OpenAI evaluator)
BASE_SYSTEM_PROMPT = """\
You are a specialized evaluation assistant for Manuav, a B2B cold outbound (phone outreach) and lead-generation agency.

You will be given:
- a company website URL
- a rubric (below)

Your job:
- research the company using the Google Search tool (this is required)
- apply the rubric
- return ONLY valid JSON matching the provided schema (no extra keys, no markdown)

Evidence discipline:
- Do not hallucinate. If something is unknown, say so, lower confidence, and be conservative.
- Prefer primary sources first, then reputable third-party sources. Prioritize DACH-relevant signals.

Entity disambiguation (guideline):
- Be mindful of same-name/lookalike companies. Use your judgment to sanity-check that a source is actually about the company behind the provided website URL.
- If attribution is uncertain, either avoid relying on the source or briefly note the uncertainty in your reasoning.
"""


def evaluate_company_gemini_with_debug(
    url: str,
    model_name: str = "gemini-3-flash-preview",
    *,
    rubric_file: str | None = None,
    api_key: str | None = None,
) -> Tuple[Dict[str, Any], Any, int, Dict[str, Any]]:
    """Like evaluate_company_gemini, but also returns grounding debug info."""
    result, usage, search_queries, debug = _evaluate_company_gemini_raw(
        url=url,
        model_name=model_name,
        rubric_file=rubric_file,
        api_key=api_key,
        debug_grounding=True,
    )
    return result, usage, search_queries, debug


def evaluate_company_gemini(
    url: str,
    model_name: str = "gemini-3-flash-preview",
    *,
    rubric_file: str | None = None,
    api_key: str | None = None,
) -> Tuple[Dict[str, Any], Any, int]:
    """
    Evaluates a company using the new Google Gen AI SDK (v1.0+)
    with Google Search grounding.
    
    Returns:
        (result_dict, usage_metadata, search_queries_count)
    """
    result, usage, search_queries, _debug = _evaluate_company_gemini_raw(
        url=url,
        model_name=model_name,
        rubric_file=rubric_file,
        api_key=api_key,
        debug_grounding=False,
    )
    return result, usage, search_queries


def _evaluate_company_gemini_raw(
    *,
    url: str,
    model_name: str,
    rubric_file: str | None,
    api_key: str | None,
    debug_grounding: bool,
) -> Tuple[Dict[str, Any], Any, int, Dict[str, Any]]:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=key)

    # Load rubric
    rubric_path, rubric_text = load_rubric_text(rubric_file)
    system_instruction = f"{BASE_SYSTEM_PROMPT}\n\nRubric file: {rubric_path}\n\n{rubric_text}\n"

    normalized_url = url.strip()
    if normalized_url and not normalized_url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    user_prompt = f"""\
Evaluate this company for Manuav using web research and the Manuav Fit logic.

Instructions:
- Use the Google Search tool to research:
  - the company website itself (product/service, ICP, pricing, cases, careers, legal/imprint)
  - and the broader web for each rubric category (DACH presence, operational status, TAM, competition, innovation, economics, onboarding, pitchability, risk).
- Be conservative when evidence is missing.
- In the JSON output:
  - set input_url exactly to the Company website URL below
  - keep reasoning SHORT (max ~600 characters, 2-4 sentences). Focus on the top 2-4 drivers and the biggest unknown.

Company website URL: {normalized_url}
"""

    # Using the new SDK syntax for tools (google_search) and JSON schema
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_schema=_gemini_schema(OUTPUT_SCHEMA),
            temperature=0.0,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned empty text.")

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback cleanup just in case
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())

    # Estimate number of web search queries from grounding metadata (when available).
    search_queries_count = 0
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        gm = getattr(candidates[0], "grounding_metadata", None)
        if gm is not None and hasattr(gm, "web_search_queries"):
            try:
                search_queries_count = len(gm.web_search_queries or [])
            except Exception:
                search_queries_count = 0

    debug = _extract_grounding_debug(response) if debug_grounding else {"web_search_queries": [], "grounding_chunks": []}
    return data, response.usage_metadata, search_queries_count, debug
