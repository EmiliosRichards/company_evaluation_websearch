from __future__ import annotations

import csv
import sys
from pathlib import Path

import scripts.evaluate as eval_one
import scripts.evaluate_list as eval_list


def test_evaluate_list_applies_flex_discount_to_token_cost_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MANUAV_FLEX_TOKEN_DISCOUNT", "0.5")
    monkeypatch.setenv("MANUAV_PRICE_INPUT_PER_1M", "1.0")
    monkeypatch.setenv("MANUAV_PRICE_CACHED_INPUT_PER_1M", "0.0")
    monkeypatch.setenv("MANUAV_PRICE_OUTPUT_PER_1M", "0.0")
    monkeypatch.setenv("MANUAV_WEB_SEARCH_PRICE_PER_1K", "10.0")

    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Firma", "Website"])
        w.writeheader()
        w.writerow({"Firma": "A", "Website": "a.com"})

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    class _Usage:
        # 1M input tokens, no cache, no output => raw token cost would be $1.0.
        input_tokens = 1_000_000
        output_tokens = 0
        total_tokens = 1_000_000
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_eval(*_a, **_kw):
        return (
            {"input_url": "https://a.com", "company_name": "A", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            0,  # no web search
            [],
        )

    monkeypatch.setattr(eval_list, "evaluate_company_with_usage_and_web_search_artifacts", _fake_eval)

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
            "--service-tier",
            "flex",
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )
    assert eval_list.main() == 0

    row = next(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert float(row["token_cost_usd"]) == 0.5
    assert float(row["web_search_tool_cost_usd"]) == 0.0
    assert float(row["cost_usd"]) == 0.5


def test_evaluate_single_applies_flex_discount_to_token_cost_only(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MANUAV_FLEX_TOKEN_DISCOUNT", "0.5")
    monkeypatch.setenv("MANUAV_PRICE_INPUT_PER_1M", "1.0")
    monkeypatch.setenv("MANUAV_PRICE_CACHED_INPUT_PER_1M", "0.0")
    monkeypatch.setenv("MANUAV_PRICE_OUTPUT_PER_1M", "0.0")
    monkeypatch.setenv("MANUAV_WEB_SEARCH_PRICE_PER_1K", "10.0")

    class _Usage:
        input_tokens = 1_000_000
        output_tokens = 0
        total_tokens = 1_000_000
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_eval(*_a, **_kw):
        return (
            {"input_url": "https://a.com", "company_name": "A", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            0,
            [],
        )

    monkeypatch.setattr(eval_one, "evaluate_company_with_usage_and_web_search_artifacts", _fake_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "https://a.com",
            "--service-tier",
            "flex",
        ],
    )
    assert eval_one.main() == 0
    out = capsys.readouterr()
    # stderr includes cost line; token cost should be discounted to 0.5
    assert "tokens=0.500000" in (out.err + out.out)


