from __future__ import annotations

from manuav_eval.schema import OUTPUT_SCHEMA, json_schema_text_config


def test_schema_has_expected_keys_only() -> None:
    props = OUTPUT_SCHEMA["properties"]
    required = set(OUTPUT_SCHEMA["required"])

    expected = {
        "input_url",
        "company_name",
        "manuav_fit_score",
        "confidence",
        "reasoning",
        "sources_visited",
    }
    assert set(props.keys()) == expected
    assert required == expected

    assert OUTPUT_SCHEMA["additionalProperties"] is False


def test_schema_text_config_shape() -> None:
    cfg = json_schema_text_config()
    assert cfg["format"]["type"] == "json_schema"
    assert cfg["format"]["name"] == "manuav_company_fit"
    assert cfg["format"]["strict"] is True
    assert cfg["format"]["schema"] == OUTPUT_SCHEMA


