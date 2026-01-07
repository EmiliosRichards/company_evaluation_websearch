from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import scripts.evaluate_list as runner


def test_evaluate_list_debug_web_search_counts_query_vs_open(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Firma", "Website"])
        w.writeheader()
        w.writerow({"Firma": "A", "Website": "a.com"})

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MANUAV_WEB_SEARCH_PRICE_PER_1K", "10.0")  # $0.01 per query
    monkeypatch.setenv("MANUAV_PRICE_INPUT_PER_1M", "0")
    monkeypatch.setenv("MANUAV_PRICE_CACHED_INPUT_PER_1M", "0")
    monkeypatch.setenv("MANUAV_PRICE_OUTPUT_PER_1M", "0")

    # Stub the evaluator.
    class _Usage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_debug(*_a, **_kw):
        return (
            {
                "input_url": "https://a.com",
                "company_name": "A",
                "manuav_fit_score": 5.0,
                "confidence": "low",
                "reasoning": "r",
            },
            _Usage(),
            {
                "completed": 2,  # total tool calls
                "by_kind_completed": {"query": 1, "open": 1},
                "url_citations": [],
                "calls": [],
                "output_item_types": [],
            },
        )

    monkeypatch.setattr(runner, "evaluate_company_with_usage_and_web_search_debug", _fake_debug)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_list.py",
            "--input",
            str(sample),
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--debug-web-search",
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )

    rc = runner.main()
    assert rc == 0

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    # web_search_calls should represent billed query count (1), not total tool calls (2)
    assert rec["web_search_calls"] == 1
    assert rec["web_search_tool_calls_total"] == 2
    assert rec["web_search_calls_query"] == 1
    assert rec["web_search_calls_open"] == 1
    assert rec["web_search_tool_cost_usd"] == 0.01


