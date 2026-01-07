from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.evaluate_list as runner


def test_semicolon_csv_with_bom_and_commas_in_fields_parses_correctly(tmp_path: Path, monkeypatch) -> None:
    # Semicolon CSV with BOM-prefixed header, and a field containing many commas.
    # If delimiter detection is wrong (comma), Website will be missing and no evaluations will run.
    sample = tmp_path / "companies.csv"
    content = (
        "\ufeffCompany;Website;Short Description\n"
        'Acme GmbH;acme.example;"a, b, c, d, e"\n'
    )
    sample.write_text(content, encoding="utf-8")

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Stub evaluator (no network).
    class _Usage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    seen = {"url": None, "name": None}

    def _fake_eval(url: str, *_a, **_kw):
        seen["url"] = url
        return (
            {"input_url": "https://acme.example", "company_name": "Acme GmbH", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            1,
            [],
        )

    monkeypatch.setattr(runner, "evaluate_company_with_usage_and_web_search_artifacts", _fake_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_list.py",
            "--input",
            str(sample),
            "--input-format",
            "csv",
            "--url-column",
            "Website",
            "--name-column",
            "Company",
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--limit",
            "1",
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
    assert seen["url"] == "acme.example"

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["firma"] == "Acme GmbH"
    assert rec["website"] == "acme.example"


