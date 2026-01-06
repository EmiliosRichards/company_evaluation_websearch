from __future__ import annotations

from typing import Any, Optional

import manuav_eval.evaluator as evaluator


def test_system_prompt_requires_search_per_rubric_category(monkeypatch: Any) -> None:
    # Ensure the prompt keeps the behavioral contract: search tool use per rubric category.
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
                    '"confidence":"low","reasoning":"r","sources_visited":[{"title":"t","url":"u"}]}'
                )

            return _R()

    class _FakeClient:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

    fake = _FakeClient()
    monkeypatch.setattr(evaluator, "OpenAI", lambda: fake)

    evaluator.evaluate_company("example.com", "gpt-test")
    system_msg = fake.responses.kwargs["input"][0]["content"]

    assert "web search tool (this is required)" in system_msg
    assert "For each category/section in the rubric, run at least one targeted search query" in system_msg


