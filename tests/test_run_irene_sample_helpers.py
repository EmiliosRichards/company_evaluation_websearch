from __future__ import annotations

import re

import scripts.run_irene_sample as runner


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


