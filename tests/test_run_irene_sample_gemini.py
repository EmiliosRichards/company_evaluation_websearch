import csv
import sys
from pathlib import Path

import scripts.run_irene_sample_gemini as runner


def test_run_irene_sample_gemini_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    # Tiny synthetic sample
    sample = tmp_path / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "Firma", "Website", "Manuav-Score"])
        w.writeheader()
        w.writerow({"bucket": "low", "Firma": "A", "Website": "a.com", "Manuav-Score": "2"})

    out = tmp_path / "out.jsonl"
    out_csv = tmp_path / "out.csv"

    monkeypatch.setenv("GEMINI_API_KEY", "test")

    class _Usage:
        prompt_token_count = 100
        candidates_token_count = 50

    def _fake_eval(url: str, model_name: str, *, rubric_file=None, api_key=None):
        return (
            {
                "input_url": url if url.startswith("http") else f"https://{url}",
                "company_name": "X",
                "manuav_fit_score": 2.0,
                "confidence": "low",
                "reasoning": "r",
                "sources_visited": [{"title": "t", "url": "u"}],
            },
            _Usage(),
            3,
        )

    monkeypatch.setattr(runner, "evaluate_company_gemini", _fake_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_irene_sample_gemini.py",
            "--sample",
            str(sample),
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--model",
            "gemini-3-flash-preview",
            "--sleep",
            "0",
        ],
    )

    rc = runner.main()
    assert rc == 0
    assert out.exists()
    assert out_csv.exists()


