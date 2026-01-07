from __future__ import annotations

import os
import sys
from pathlib import Path

import scripts.run_irene_sample_batch as batch_cli


def test_batch_create_blocks_web_search_by_default(tmp_path: Path, monkeypatch) -> None:
    # Ensure we don't hit the real API.
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # Provide a fake sample path; create should bail out before reading it.
    sample = tmp_path / "sample.csv"
    sample.write_text("bucket,Firma,Website,Manuav-Score\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_irene_sample_batch.py",
            "create",
            "--sample",
            str(sample),
            "--suffix",
            "t",
        ],
    )

    rc = batch_cli.main()
    assert rc == 2


