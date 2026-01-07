from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .evaluator import BASE_SYSTEM_PROMPT
from .rubric_loader import load_rubric_text
from .schema import json_schema_text_config


@dataclass(frozen=True)
class BatchRequestLine:
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]


def build_responses_body(
    company_url: str,
    model: str,
    *,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    enable_web_search: bool = True,
) -> Dict[str, Any]:
    """Build a /v1/responses body matching our single-call evaluator settings."""
    rubric_path, rubric_text = load_rubric_text(rubric_file)
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nRubric file: {rubric_path}\n\n{rubric_text}\n"

    normalized_url = (company_url or "").strip()
    if normalized_url and not normalized_url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    tool_budget_line = (
        f"- Tool-call budget: you can make at most {max_tool_calls} web search tool call(s). Use them wisely.\n"
        if max_tool_calls is not None
        else ""
    )

    # Keep dynamic URL at the end to help caching.
    user_prompt = f"""\
Evaluate this company for Manuav using web research and the Manuav Fit logic.

Instructions:
- Use the web search tool to research:
  - the company website itself (product/service, ICP, pricing, cases, careers, legal/imprint)
  - and the broader web for each rubric category (DACH presence, operational status, TAM, competition, innovation, economics, onboarding, pitchability, risk).
{tool_budget_line}- Be conservative when evidence is missing.
- In the JSON output:
  - set input_url exactly to the Company website URL below
  - keep reasoning SHORT (max ~600 characters, 2-4 sentences). Focus on the top 2-4 drivers and the biggest unknown.
  - do NOT include URLs or a sources list in JSON.

Company website URL: {normalized_url}
"""

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": json_schema_text_config(),
    }
    if enable_web_search:
        body["tools"] = [{"type": "web_search_preview"}]
    if max_tool_calls is not None:
        body["max_tool_calls"] = max_tool_calls
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if prompt_cache:
        # Batch requests can include prompt caching parameters if supported for the model.
        if prompt_cache_key:
            body["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention:
            body["prompt_cache_retention"] = prompt_cache_retention

    return body


def write_batch_input_jsonl(lines: Iterable[BatchRequestLine], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(
                json.dumps(
                    {
                        "custom_id": line.custom_id,
                        "method": line.method,
                        "url": line.url,
                        "body": line.body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_irene_batch_lines(
    sample_csv_path: Path,
    *,
    model: str,
    rubric_file: str | None = None,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    prompt_cache: bool | None = None,
    prompt_cache_retention: str | None = None,
    custom_id_prefix: str = "irene",
    enable_web_search: bool = True,
) -> List[BatchRequestLine]:
    rows: List[Dict[str, str]] = []
    with sample_csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    lines: List[BatchRequestLine] = []
    for idx, row in enumerate(rows, start=1):
        website = (row.get("Website") or "").strip()
        score = (row.get("Manuav-Score") or "").strip()
        if not website or not score:
            continue

        body = build_responses_body(
            website,
            model,
            rubric_file=rubric_file,
            max_tool_calls=max_tool_calls,
            reasoning_effort=reasoning_effort,
            prompt_cache=prompt_cache,
            prompt_cache_retention=prompt_cache_retention,
            enable_web_search=enable_web_search,
        )
        firma = (row.get("Firma") or "").strip().replace(" ", "_")
        custom_id = f"{custom_id_prefix}-{idx}-{firma}" if firma else f"{custom_id_prefix}-{idx}"
        lines.append(BatchRequestLine(custom_id=custom_id, method="POST", url="/v1/responses", body=body))

    return lines


def _extract_json_text_from_body(body: Dict[str, Any]) -> str:
    # Prefer output_text convenience field.
    out_text = body.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text

    # Fallback traverse output message content.
    parts: List[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for c in item.get("content", []) or []:
            if isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("Could not extract text output from batch response body.")


def _count_completed_web_search_calls_from_body(body: Dict[str, Any]) -> int:
    n = 0
    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search_call":
            continue
        if item.get("status") == "completed":
            n += 1
    return n


def _extract_url_citations_from_body(body: Dict[str, Any]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if not isinstance(c, dict):
                continue
            for ann in c.get("annotations", []) or []:
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") != "url_citation":
                    continue
                uc = ann.get("url_citation") or {}
                url = uc.get("url")
                title = uc.get("title", "")
                if isinstance(url, str) and url and url not in seen:
                    citations.append({"url": url, "title": title or ""})
                    seen.add(url)
    return citations


def _usage_from_body(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


@dataclass(frozen=True)
class BatchParsedResult:
    custom_id: str
    status_code: int
    model_result: Optional[Dict[str, Any]]
    usage: Dict[str, Any]
    web_search_calls: int
    url_citations: List[Dict[str, str]]
    error: Optional[Dict[str, Any]]


def parse_batch_output_jsonl(path: Path) -> Iterator[BatchParsedResult]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            resp = obj.get("response")
            err = obj.get("error")
            if not resp:
                yield BatchParsedResult(
                    custom_id=custom_id,
                    status_code=0,
                    model_result=None,
                    usage={},
                    web_search_calls=0,
                    url_citations=[],
                    error=err if isinstance(err, dict) else None,
                )
                continue
            status_code = int(resp.get("status_code", 0) or 0)
            body = resp.get("body") or {}
            if not isinstance(body, dict):
                body = {}

            if status_code != 200:
                yield BatchParsedResult(
                    custom_id=custom_id,
                    status_code=status_code,
                    model_result=None,
                    usage=_usage_from_body(body),
                    web_search_calls=_count_completed_web_search_calls_from_body(body),
                    url_citations=_extract_url_citations_from_body(body),
                    error=err if isinstance(err, dict) else None,
                )
                continue

            text = _extract_json_text_from_body(body)
            model_result = json.loads(text)
            yield BatchParsedResult(
                custom_id=custom_id,
                status_code=status_code,
                model_result=model_result,
                usage=_usage_from_body(body),
                web_search_calls=_count_completed_web_search_calls_from_body(body),
                url_citations=_extract_url_citations_from_body(body),
                error=None,
            )


