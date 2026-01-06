from __future__ import annotations

from typing import Any, Optional

import manuav_eval.evaluator as evaluator


def test_system_prompt_mentions_search_and_budget_guidance(monkeypatch: Any) -> None:
    # Ensure the prompt keeps the behavioral contract: search tool use + budget-aware guidance.
    def _fake_load_rubric_text(_: Optional[str]) -> tuple[str, str]:
        return ("rubrics/test.md", "RUBRIC_BODY")

    monkeypatch.setattr(evaluator, "load_rubric_text", _fake_load_rubric_text)

    class _FakeResponses:
        def __init__(self) -> None:
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            # Minimal valid JSON for schema.
            class _R:
                output_text = (
                    '{"input_url":"https://example.com","company_name":"X","manuav_fit_score":5,'
                    '"confidence":"low","reasoning":"r"}'
                )
                usage = type(
                    "U",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": type("X", (), {"cached_tokens": 0})(),
                        "output_tokens_details": type("Y", (), {"reasoning_tokens": 0})(),
                    },
                )()

            return _R()

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    fake = _FakeClient()
    monkeypatch.setattr(evaluator, "OpenAI", lambda: fake)

    evaluator.evaluate_company("example.com", "gpt-test")
    system_msg = fake.responses.kwargs["input"][0]["content"]

    assert "web search tool (this is required)" in system_msg
    assert "Use the web search tool strategically" in system_msg
    assert "limited tool-call/search budget" in system_msg


