from __future__ import annotations

import csv
import sys
from pathlib import Path

import scripts.evaluate_list as runner


def test_random_sample_is_deterministic_and_unique(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Stable run stamp so the sampled URL file path is deterministic.
    monkeypatch.setattr(runner, "_run_stamp", lambda: "20990101_000000")

    sample = tmp_path / "in.csv"
    with sample.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Company", "Website"])
        w.writeheader()
        # Include a duplicate URL to ensure uniqueness logic works.
        w.writerow({"Company": "A", "Website": "a.com"})
        w.writerow({"Company": "A-dup", "Website": "a.com"})
        w.writerow({"Company": "B", "Website": "b.com"})
        w.writerow({"Company": "C", "Website": "c.com"})
        w.writerow({"Company": "D", "Website": "d.com"})
        w.writerow({"Company": "E", "Website": "e.com"})

    # Stub evaluator.
    class _Usage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_eval(*_a, **_kw):
        return (
            {"input_url": "https://x", "company_name": "X", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            0,
            [],
        )

    monkeypatch.setattr(runner, "evaluate_company_with_usage_and_web_search_artifacts", _fake_eval)

    out = tmp_path / "o.jsonl"
    out_csv = tmp_path / "o.csv"

    # First run (seed=1).
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
            "--random-sample",
            "4",
            "--seed",
            "1",
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )
    assert runner.main() == 0
    sample_urls_1 = (tmp_path / "outputs" / "20990101_000000_sample_urls.txt").read_text(encoding="utf-8")
    urls_1 = [line.split("\t")[0].strip() for line in sample_urls_1.strip().splitlines() if line.strip()]
    assert len(urls_1) == len(set(urls_1))  # unique

    # Second run with same seed should produce identical sample.
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
            "--random-sample",
            "4",
            "--seed",
            "1",
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )
    assert runner.main() == 0
    sample_urls_1b = (tmp_path / "outputs" / "20990101_000000_sample_urls.txt").read_text(encoding="utf-8")
    assert sample_urls_1b == sample_urls_1

    # Different seed should usually produce a different sample.
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
            "--random-sample",
            "4",
            "--seed",
            "2",
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )
    assert runner.main() == 0
    sample_urls_2 = (tmp_path / "outputs" / "20990101_000000_sample_urls.txt").read_text(encoding="utf-8")
    assert sample_urls_2 != sample_urls_1


def test_resume_skips_already_processed_websites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    inp = tmp_path / "in.csv"
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Company", "Website"])
        w.writeheader()
        w.writerow({"Company": "A", "Website": "a.com"})
        w.writerow({"Company": "B", "Website": "b.com"})

    out = tmp_path / "outputs" / "resume.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"website":"a.com"}\n', encoding="utf-8")

    out_csv = tmp_path / "outputs" / "resume.csv"

    calls: list[str] = []

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_eval(url: str, *_a, **_kw):
        calls.append(url)
        return (
            {"input_url": f"https://{url}", "company_name": "X", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            0,
            [],
        )

    monkeypatch.setattr(runner, "evaluate_company_with_usage_and_web_search_artifacts", _fake_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_list.py",
            "--input",
            str(inp),
            "--input-format",
            "csv",
            "--url-column",
            "Website",
            "--name-column",
            "Company",
            "--resume",
            "--out",
            str(out),
            "--out-csv",
            str(out_csv),
            "--score-column",
            "-",
            "--bucket-column",
            "-",
            "--sleep",
            "0",
        ],
    )
    assert runner.main() == 0

    # a.com should be skipped due to resume file.
    assert calls == ["b.com"]


