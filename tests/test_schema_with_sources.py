from __future__ import annotations

from manuav_eval.schema import OUTPUT_SCHEMA_WITH_SOURCES


def test_sources_schema_is_strict_and_required_keys_complete() -> None:
    # Root required keys include sources + all base fields.
    required = set(OUTPUT_SCHEMA_WITH_SOURCES["required"])
    assert "sources" in required
    props = OUTPUT_SCHEMA_WITH_SOURCES["properties"]
    assert required.issuperset(set(props.keys()))

    # Sources item schema: required must include all properties for OpenAI strict schema.
    sources = props["sources"]
    items = sources["items"]
    item_props = set(items["properties"].keys())
    item_required = set(items["required"])
    assert item_required == item_props


