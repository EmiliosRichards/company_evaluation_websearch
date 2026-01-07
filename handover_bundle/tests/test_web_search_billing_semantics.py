from __future__ import annotations

from manuav_eval import evaluator


def test_billable_web_search_calls_prefers_query_count() -> None:
    ws_stats = {"completed": 5, "by_kind_completed": {"query": 2, "open": 3}}
    assert evaluator._billable_web_search_calls(ws_stats) == 2


def test_billable_web_search_calls_falls_back_to_completed_when_no_breakdown() -> None:
    ws_stats = {"completed": 4}
    assert evaluator._billable_web_search_calls(ws_stats) == 4


def test_billable_web_search_calls_handles_bad_inputs() -> None:
    assert evaluator._billable_web_search_calls({}) == 0
    assert evaluator._billable_web_search_calls({"by_kind_completed": {"query": "not-an-int"}}) == 0


