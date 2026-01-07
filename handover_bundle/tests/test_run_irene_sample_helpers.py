from __future__ import annotations

import re
from pathlib import Path

import scripts.evaluate_list as runner


def test_suffix_slug() -> None:
    assert runner._suffix_slug("") == ""
    assert runner._suffix_slug("  ") == ""
    assert runner._suffix_slug("baseline") == "baseline"
    assert runner._suffix_slug("my run 01") == "my_run_01"
    assert runner._suffix_slug("weird*&^%name") == "weirdname"
    assert runner._suffix_slug("__a__b__") == "a__b"


def test_run_stamp_format() -> None:
    stamp = runner._run_stamp()
    assert re.fullmatch(r"\d{8}_\d{6}", stamp)


def test_load_url_list(tmp_path: Path) -> None:
    p = tmp_path / "urls.txt"
    p.write_text(
        "# comment\n\nhttps://example.com\nexample.org/path\n  www.test.com  \n",
        encoding="utf-8",
    )
    rows = runner._load_rows(p, input_format="txt")
    assert rows == [{"Website": "https://example.com"}, {"Website": "example.org/path"}, {"Website": "www.test.com"}]


def test_normalize_for_dedupe() -> None:
    assert runner._normalize_for_dedupe("HTTPS://Example.COM/") == "example.com"
    assert runner._normalize_for_dedupe("example.com/") == "example.com"
    assert runner._normalize_for_dedupe("https://example.com/a?b=c") == "example.com/a?b=c"
    assert runner._normalize_for_dedupe("") == ""

