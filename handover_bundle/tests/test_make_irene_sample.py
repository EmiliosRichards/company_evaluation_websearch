from __future__ import annotations

import csv
import sys
from pathlib import Path

import scripts.make_irene_sample as sampler


def test_parse_score_handles_commas_and_blanks() -> None:
    assert sampler._parse_score("") is None
    assert sampler._parse_score("  ") is None
    assert sampler._parse_score("2") == 2.0
    assert sampler._parse_score("7,5") == 7.5


def test_bucket_fixed() -> None:
    assert sampler._bucket_fixed(2) == "low"
    assert sampler._bucket_fixed(4) == "mid"
    assert sampler._bucket_fixed(6) == "mid"
    assert sampler._bucket_fixed(7) == "high"


def test_make_sample_main_creates_9_rows(tmp_path: Path, monkeypatch) -> None:
    # Create a small synthetic "Irene" CSV with enough rows per bucket.
    src = tmp_path / "irene.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Firma", "Website", "Manuav-Score", "Kurzurteil"])
        w.writeheader()
        for i in range(3):
            w.writerow({"Firma": f"L{i}", "Website": f"low{i}.com", "Manuav-Score": "2", "Kurzurteil": ""})
        for i in range(3):
            w.writerow({"Firma": f"M{i}", "Website": f"mid{i}.com", "Manuav-Score": "5", "Kurzurteil": ""})
        for i in range(3):
            w.writerow({"Firma": f"H{i}", "Website": f"high{i}.com", "Manuav-Score": "8", "Kurzurteil": ""})

    out = tmp_path / "sample.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        ["make_irene_sample.py", "--input", str(src), "--output", str(out), "--seed", "123"],
    )
    rc = sampler.main()
    assert rc == 0
    assert out.exists()

    with out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9


