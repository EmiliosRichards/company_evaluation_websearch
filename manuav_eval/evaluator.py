from __future__ import annotations

import json
from typing import Any, Dict

from openai import OpenAI

from .rubric_loader import load_rubric_text
from .schema import json_schema_text_config


BASE_SYSTEM_PROMPT = """\
You are a specialized evaluation assistant for Manuav, a B2B cold outbound (phone outreach) and lead-generation agency.

You will be given:
- a company website URL
- a rubric (below)

Your job:
- research the company using the web search tool (this is required)
- apply the rubric
- return ONLY valid JSON matching the provided schema (no extra keys, no markdown)

Evidence discipline:
- Do not hallucinate. If something is unknown, say so, lower confidence, and be conservative.

Research process (required):
- Use the web search tool to:
  - visit/review the company website (home, product, pricing, cases, careers, legal/imprint/contact)
  - search the web for corroborating third-party evidence
- For each category/section in the rubric, run at least one targeted search query and use the results to support your reasoning.
- Prefer primary sources first, then reputable third-party sources. Prioritize DACH-relevant signals.
- Record every page you relied on in sources_visited (title + URL).

Entity disambiguation (guideline):
- Be mindful of same-name/lookalike companies. Use your judgment to sanity-check that a source is actually about the company behind the provided website URL.
- Helpful identity signals include:
  - domain consistency and links from the official site
  - legal entity name and imprint/registration details
  - headquarters/location and language/market focus
  - product description, ICP, and branding match
  - the official LinkedIn/company page referenced by the website
- If attribution is uncertain, either avoid relying on the source or briefly note the uncertainty in your reasoning.
"""


def _extract_json_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text

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


def evaluate_company(
    url: str,
    model: str,
    *,
    rubric_file: str | None = None,
) -> Dict[str, Any]:
    client = OpenAI()
    rubric_path, rubric_text = load_rubric_text(rubric_file)
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nRubric file: {rubric_path}\n\n{rubric_text}\n"

    normalized_url = url.strip()
    if normalized_url and not normalized_url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    user_prompt = f"""\
Evaluate this company for Manuav using web research and the Manuav Fit logic:

Company website URL: {normalized_url}

Instructions:
- Use the web search tool to research:
  - the company website itself (product/service, ICP, pricing, cases, careers, legal/imprint)
  - and the broader web for each rubric category (DACH presence, operational status, TAM, competition, innovation, economics, onboarding, pitchability, risk).
- Be conservative when evidence is missing.
- In the JSON output:
  - set input_url exactly to: {normalized_url}
  - provide a concise reasoning narrative for the score (why it is high/low; mention key rubric drivers and any critical unknowns)
  - provide sources_visited as the list of URLs you relied on (with titles). Include primary sources where possible.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search_preview"}],
        text=json_schema_text_config(),
    )

    text = _extract_json_text(resp)
    return json.loads(text)


