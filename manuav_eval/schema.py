from __future__ import annotations

from typing import Any, Dict


OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "input_url": {"type": "string"},
        "company_name": {"type": "string"},
        "manuav_fit_score": {"type": "number", "minimum": 0, "maximum": 10},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        # Keep short to reduce output tokens. If more detail is needed, rely on logs / citations.
        "reasoning": {"type": "string", "maxLength": 600},
    },
    "required": [
        "input_url",
        "company_name",
        "manuav_fit_score",
        "confidence",
        "reasoning",
    ],
}


def json_schema_text_config(
    *,
    name: str = "manuav_company_fit",
    strict: bool = True,
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": strict,
            "schema": schema or OUTPUT_SCHEMA,
        }
    }


