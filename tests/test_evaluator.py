from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import manuav_eval.evaluator as evaluator


@dataclass
class _FakeContent:
    text: str


@dataclass
class _FakeOutputItem:
    content: List[_FakeContent]


class _FakeResponse:
    def __init__(
        self,
        *,
        output_text: str | None = None,
        output: list[_FakeOutputItem] | None = None,
    ) -> None:
        self.output_text = output_text
        self.output = output or []


class _FakeResponses:
    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] | None = None
        self.response_to_return: _FakeResponse | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        assert self.response_to_return is not None
        return self.response_to_return


class _FakeOpenAI:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_extract_json_text_prefers_output_text() -> None:
    resp = _FakeResponse(output_text='{"a": 1}')
    assert evaluator._extract_json_text(resp) == '{"a": 1}'


def test_extract_json_text_fallback_traversal() -> None:
    resp = _FakeResponse(
        output=[
            _FakeOutputItem(content=[_FakeContent(text='{"a": '), _FakeContent(text="1}")]),
        ]
    )
    assert evaluator._extract_json_text(resp) == '{"a": \n1}'


def test_evaluate_company_builds_prompt_and_normalizes_url(monkeypatch: Any) -> None:
    fake_client = _FakeOpenAI()

    def _fake_openai_ctor() -> _FakeOpenAI:
        return fake_client

    monkeypatch.setattr(evaluator, "OpenAI", _fake_openai_ctor)

    # Make rubric deterministic and small.
    def _fake_load_rubric_text(_: Optional[str]) -> tuple[str, str]:
        return ("rubrics/test.md", "RUBRIC_BODY")

    monkeypatch.setattr(evaluator, "load_rubric_text", _fake_load_rubric_text)

    payload = {
        "input_url": "https://example.com",
        "company_name": "Example",
        "manuav_fit_score": 5.0,
        "confidence": "low",
        "reasoning": "because",
        "sources_visited": [{"title": "t", "url": "https://example.com"}],
    }
    fake_client.responses.response_to_return = _FakeResponse(output_text=json.dumps(payload))

    result = evaluator.evaluate_company("example.com", "gpt-test")
    assert result["input_url"] == "https://example.com"

    # Verify the request structure.
    assert fake_client.responses.last_kwargs is not None
    kwargs = fake_client.responses.last_kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["tools"] == [{"type": "web_search_preview"}]

    # System prompt includes rubric reference and body.
    system_msg = kwargs["input"][0]["content"]
    assert "Rubric file: rubrics/test.md" in system_msg
    assert "RUBRIC_BODY" in system_msg

    # User prompt includes normalized URL.
    user_msg = kwargs["input"][1]["content"]
    assert "Company website URL: https://example.com" in user_msg


