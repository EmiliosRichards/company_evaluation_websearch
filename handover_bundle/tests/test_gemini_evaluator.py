import json
from unittest.mock import MagicMock

from manuav_eval import gemini_evaluator
from google.genai import types


def test_evaluate_company_gemini(monkeypatch):
    # Mock the new genai.Client
    mock_client_cls = MagicMock()
    mock_client_instance = MagicMock()
    mock_response = MagicMock()

    monkeypatch.setattr(gemini_evaluator.genai, "Client", mock_client_cls)
    mock_client_cls.return_value = mock_client_instance
    mock_client_instance.models.generate_content.return_value = mock_response

    # Stub response with usage metadata
    fake_result = {
        "input_url": "https://example.com",
        "company_name": "Example Inc",
        "manuav_fit_score": 7.5,
        "confidence": "high",
        "reasoning": "Good fit.",
    }
    mock_response.text = json.dumps(fake_result)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 50

    # Grounding metadata with 2 search queries
    gm = MagicMock()
    gm.web_search_queries = ["q1", "q2"]
    cand = MagicMock()
    cand.grounding_metadata = gm
    mock_response.candidates = [cand]

    # Stub rubric loader
    def _fake_load_rubric_text(_):
        return ("test_rubric.md", "Rubric content")

    monkeypatch.setattr(gemini_evaluator, "load_rubric_text", _fake_load_rubric_text)

    # Run
    result, usage, search_queries = gemini_evaluator.evaluate_company_gemini(
        "example.com", model_name="gemini-3-flash-preview", api_key="test-key"
    )

    # Assertions
    assert result["manuav_fit_score"] == 7.5
    assert result["input_url"] == "https://example.com"
    assert usage.prompt_token_count == 100
    assert search_queries == 2


def test_evaluate_company_gemini_with_debug(monkeypatch):
    # Mock the new genai.Client
    mock_client_cls = MagicMock()
    mock_client_instance = MagicMock()
    mock_response = MagicMock()

    monkeypatch.setattr(gemini_evaluator.genai, "Client", mock_client_cls)
    mock_client_cls.return_value = mock_client_instance
    mock_client_instance.models.generate_content.return_value = mock_response

    fake_result = {
        "input_url": "https://example.com",
        "company_name": "Example Inc",
        "manuav_fit_score": 7.5,
        "confidence": "high",
        "reasoning": "Good fit.",
    }
    mock_response.text = json.dumps(fake_result)
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 50

    gm = MagicMock()
    gm.web_search_queries = ["q1", "q2"]
    gm.model_dump.return_value = {"web_search_queries": ["q1", "q2"], "grounding_chunks": [{"title": "T", "uri": "U"}]}
    cand = MagicMock()
    cand.grounding_metadata = gm
    mock_response.candidates = [cand]

    def _fake_load_rubric_text(_):
        return ("test_rubric.md", "Rubric content")

    monkeypatch.setattr(gemini_evaluator, "load_rubric_text", _fake_load_rubric_text)

    result, usage, search_queries, debug = gemini_evaluator.evaluate_company_gemini_with_debug(
        "example.com", model_name="gemini-3-flash-preview", api_key="test-key"
    )
    assert result["input_url"] == "https://example.com"
    assert search_queries == 2
    assert debug["web_search_queries"] == ["q1", "q2"]
    assert debug["grounding_chunks"] == [{"title": "T", "uri": "U"}]
    
    # Check client init
    mock_client_cls.assert_called_with(api_key="test-key")
    
    # Check generate call
    mock_client_instance.models.generate_content.assert_called_once()
    kwargs = mock_client_instance.models.generate_content.call_args[1]
    
    assert kwargs["model"] == "gemini-3-flash-preview"
    assert "https://example.com" in kwargs["contents"]
    
    config = kwargs["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.response_mime_type == "application/json"
    # Ensure google_search tool is present
    assert len(config.tools) == 1
    assert config.tools[0].google_search is not None
