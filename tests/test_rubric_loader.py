from __future__ import annotations

from pathlib import Path

from manuav_eval.rubric_loader import DEFAULT_RUBRIC_FILE, load_rubric_text


def test_load_default_rubric_text_is_non_empty() -> None:
    path_str, text = load_rubric_text(None)
    assert Path(path_str).as_posix().endswith(DEFAULT_RUBRIC_FILE.as_posix())
    assert "Manuav" in text
    assert len(text) > 1000


def test_load_custom_rubric_text(tmp_path: Path) -> None:
    p = tmp_path / "rubric.md"
    p.write_text("hello rubric", encoding="utf-8")
    path_str, text = load_rubric_text(str(p))
    assert Path(path_str) == p
    assert text == "hello rubric"


