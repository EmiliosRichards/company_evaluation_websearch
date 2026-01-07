from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import scripts.analyze_run as analyze
import scripts.trace_web_search as trace


def test_trace_web_search_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Stable stamp for deterministic filenames.
    monkeypatch.setattr(trace, "_run_stamp", lambda: "20990101_000000")

    url_list = tmp_path / "urls.txt"
    url_list.write_text("a.com\nb.com\n", encoding="utf-8")

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_tokens_details = type("X", (), {"cached_tokens": 0})()
        output_tokens_details = type("Y", (), {"reasoning_tokens": 0})()

    def _fake_debug(*_a, **_kw):
        return (
            {"input_url": "https://a.com", "company_name": "A", "manuav_fit_score": 5, "confidence": "low", "reasoning": "r"},
            _Usage(),
            {"completed": 1, "by_kind_completed": {"query": 1}, "calls": [{"kind": "query", "query": "a.com"}]},
        )

    monkeypatch.setattr(trace, "evaluate_company_with_usage_and_web_search_debug", _fake_debug)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trace_web_search.py",
            "--input",
            str(url_list),
            "--input-format",
            "txt",
            "--sample",
            "1",
            "--seed",
            "1",
            "--suffix",
            "t",
            "--sleep",
            "0",
        ],
    )
    assert trace.main() == 0

    out_csv = tmp_path / "outputs" / "20990101_000000_t.csv"
    out_jsonl = tmp_path / "outputs" / "20990101_000000_t.jsonl"
    assert out_csv.exists()
    assert out_jsonl.exists()

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert int(rows[0]["query_calls"]) == 1

    rec = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert rec["web_search_debug"]["completed"] == 1


def test_analyze_run_reads_csv_and_prints_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    p = tmp_path / "run.csv"
    p.write_text(
        "run_id,firma,website,cost_usd,token_cost_usd,web_search_tool_cost_usd,web_search_calls,web_search_calls_query,web_search_calls_open,web_search_calls_unknown,duration_seconds,url_citations_json\n"
        "r,A,a.com,0.02,0.01,0.01,1,1,0,0,1.0,[]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["analyze_run.py", "--csv", str(p)])
    assert analyze.main() == 0
    out = capsys.readouterr()
    assert "rows: 1" in (out.out + out.err)
    assert "cost_usd" in (out.out + out.err)


