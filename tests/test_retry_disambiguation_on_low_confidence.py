from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import scripts.evaluate_list as runner


def test_retry_disambiguation_on_low_confidence_aggregates_cost_and_calls(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Firma", "Website"])
        w.writeheader()
        w.writerow({"Firma": "A", "Website": "a.com"})

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MANUAV_PRICE_INPUT_PER_1M", "0")
    monkeypatch.setenv("MANUAV_PRICE_CACHED_INPUT_PER_1M", "0")
    monkeypatch.setenv("MANUAV_PRICE_OUTPUT_PER_1M", "0")
    monkeypatch.setenv("MANUAV_WEB_SEARCH_PRICE_PER_1K", "10.0")  # $0.01 per query

    calls: list[dict] = []

    class _Usage1:
        input_tokens = 10
        output_tokens = 0
        total_tokens = 10
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    class _Usage2:
        input_tokens = 20
        output_tokens = 0
        total_tokens = 20
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_debug(*_a, **kw):
        # First call: low confidence -> triggers retry. Second call: high confidence.
        calls.append(dict(kw))
        if len(calls) == 1:
            return (
                {"input_url": "https://a.com", "company_name": "A", "manuav_fit_score": 1.0, "confidence": "low", "reasoning": "r1"},
                _Usage1(),
                {"completed": 1, "by_kind_completed": {"query": 1}, "url_citations": [], "calls": []},
            )
        return (
            {"input_url": "https://a.com", "company_name": "A", "manuav_fit_score": 7.0, "confidence": "high", "reasoning": "r2"},
            _Usage2(),
            {"completed": 2, "by_kind_completed": {"query": 2}, "url_citations": [], "calls": []},
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
            "--retry-disambiguation-on-low-confidence",
            "--retry-max-tool-calls",
            "3",
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )

    assert runner.main() == 0
    assert len(calls) == 2

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["retry"]["used"] is True
    assert rec["retry"]["selected"] == "retry"
    # web_search_calls is billed query count, aggregated across attempts: 1 + 2 = 3
    assert rec["web_search_calls"] == 3
    assert rec["usage"]["input_tokens"] == 30

    row = next(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert int(row["retry_used"]) == 1
    assert row["retry_selected"] == "retry"
    assert int(row["web_search_calls"]) == 3
    assert int(row["input_tokens"]) == 30


