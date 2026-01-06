from __future__ import annotations

from pathlib import Path


DEFAULT_RUBRIC_FILE = Path("rubrics/manuav_rubric_v4_en.md")


def load_rubric_text(rubric_file: str | None) -> tuple[str, str]:
    """Returns (rubric_path_str, rubric_text)."""
    path = Path(rubric_file) if rubric_file else DEFAULT_RUBRIC_FILE
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Rubric file is empty: {path}")
    return str(path), text


