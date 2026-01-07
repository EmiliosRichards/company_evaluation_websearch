from __future__ import annotations

import json
from pathlib import Path

from manuav_eval.openai_batch import (
    BatchRequestLine,
    build_irene_batch_lines,
    parse_batch_output_jsonl,
    write_batch_input_jsonl,
)


def test_write_batch_input_jsonl(tmp_path: Path, monkeypatch) -> None:
    # Stub rubric loader to avoid reading real rubric file.
    import manuav_eval.openai_batch as ob

    monkeypatch.setattr(ob, "load_rubric_text", lambda _: ("rubrics/test.md", "RUBRIC"))

    lines = [
        BatchRequestLine(custom_id="c1", method="POST", url="/v1/responses", body={"model": "x"}),
        BatchRequestLine(custom_id="c2", method="POST", url="/v1/responses", body={"model": "x"}),
    ]
    out = tmp_path / "in.jsonl"
    write_batch_input_jsonl(lines, out)
    txt = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(txt) == 2
    obj = json.loads(txt[0])
    assert obj["custom_id"] == "c1"
    assert obj["url"] == "/v1/responses"


def test_parse_batch_output_jsonl_success(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    line = {
        "id": "batch_req_1",
        "custom_id": "req-1",
        "response": {
            "status_code": 200,
            "body": {
                "output_text": "{\"input_url\":\"https://x\",\"company_name\":\"X\",\"manuav_fit_score\":5,\"confidence\":\"low\",\"reasoning\":\"r\"}",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "input_tokens_details": {"cached_tokens": 0},
                },
                "output": [
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "x",
                                "annotations": [
                                    {"type": "url_citation", "url_citation": {"url": "https://a", "title": "A"}}
                                ],
                            }
                        ],
                    },
                ],
            },
        },
        "error": None,
    }
    out.write_text(json.dumps(line) + "\n", encoding="utf-8")
    recs = list(parse_batch_output_jsonl(out))
    assert len(recs) == 1
    r = recs[0]
    assert r.custom_id == "req-1"
    assert r.status_code == 200
    assert r.model_result and r.model_result["manuav_fit_score"] == 5
    assert r.web_search_calls == 1
    assert r.url_citations == [{"url": "https://a", "title": "A"}]


