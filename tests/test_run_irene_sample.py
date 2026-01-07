from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import scripts.evaluate_list as runner


def test_run_irene_sample_main_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    # Create a tiny sample CSV (not the real 9-row file).
    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "Firma", "Website", "Manuav-Score"])
        w.writeheader()
        w.writerow({"bucket": "low", "Firma": "A", "Website": "a.com", "Manuav-Score": "2"})
        w.writerow({"bucket": "high", "Firma": "B", "Website": "b.com", "Manuav-Score": "8"})

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    # Ensure the script does not early-exit on missing key.
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Stub the evaluator so no network calls happen.
    class _Usage:
        input_tokens = 100
        output_tokens = 200
        total_tokens = 300
        input_tokens_details = type("X", (), {"cached_tokens": 25})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 50})()

    def _fake_evaluate_company_with_usage_and_web_search_artifacts(
        url: str,
        model: str,
        *,
        rubric_file=None,
        max_tool_calls=None,
        reasoning_effort=None,
        prompt_cache=None,
        prompt_cache_retention=None,
        service_tier=None,
        timeout_seconds=None,
        flex_max_retries=None,
        flex_fallback_to_auto=None,
    ):
        score = 2.0 if "a.com" in url else 8.0
        return (
            {
                "input_url": url if url.startswith("http") else f"https://{url}",
                "company_name": "X",
                "manuav_fit_score": score,
                "confidence": "low",
                "reasoning": "r",
            },
            _Usage(),
            3,
            [{"url": "https://example.com", "title": "t"}],
        )

    monkeypatch.setattr(
        runner,
        "evaluate_company_with_usage_and_web_search_artifacts",
        _fake_evaluate_company_with_usage_and_web_search_artifacts,
    )

    def _fake_evaluate_company_with_usage_and_web_search_debug(
        url: str,
        model: str,
        *,
        rubric_file=None,
        max_tool_calls=None,
        reasoning_effort=None,
        prompt_cache=None,
        prompt_cache_retention=None,
        service_tier=None,
        timeout_seconds=None,
        flex_max_retries=None,
        flex_fallback_to_auto=None,
    ):
        score = 2.0 if "a.com" in url else 8.0
        return (
            {
                "input_url": url if url.startswith("http") else f"https://{url}",
                "company_name": "X",
                "manuav_fit_score": score,
                "confidence": "low",
                "reasoning": "r",
            },
            _Usage(),
            {
                "completed": 3,
                "total": 3,
                "by_status": {"completed": 3},
                "calls": [],
                "output_item_types": [],
                "url_citations": [{"url": "https://example.com", "title": "t"}],
            },
        )

    monkeypatch.setattr(
        runner,
        "evaluate_company_with_usage_and_web_search_debug",
        _fake_evaluate_company_with_usage_and_web_search_debug,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_list.py",
            "--sample",
            str(sample),
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--model",
            "gpt-test",
            "--sleep",
            "0",
        ],
    )

    rc = runner.main()
    assert rc == 0
    assert out.exists()
    assert out_csv.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert "raw" in rec
    assert "url_citations" in rec


