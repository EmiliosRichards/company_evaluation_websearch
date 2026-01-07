from __future__ import annotations

from pathlib import Path


def test_rubric_v4_exists_and_has_expected_sections() -> None:
    p = Path("rubrics/manuav_rubric_v4_en.md")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "## What Manuav is looking for" in text
    assert "### Score bands" in text


def test_rubric_v4_does_not_contain_removed_overall_scoring_block() -> None:
    text = Path("rubrics/manuav_rubric_v4_en.md").read_text(encoding="utf-8")
    assert "## Overall scoring" not in text
    assert "## Score selection rules" not in text
    assert "## Confidence" not in text


