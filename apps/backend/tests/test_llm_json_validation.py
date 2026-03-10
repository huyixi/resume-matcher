import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.llm import LLMConfig, _check_resume_json_truncation, complete_json
from app.services.improver import extract_job_keywords, improve_resume


def _mock_json_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message={"content": json.dumps(payload, ensure_ascii=False)}
            )
        ]
    )


def test_extract_job_keywords_accepts_valid_non_resume_json(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    response = _mock_json_response(
        {
            "required_skills": ["Python"],
            "preferred_skills": "Kubernetes",
            "keywords": ["backend"],
            "personalInfo": {"unexpected": True},
        }
    )

    with (
        patch(
            "app.llm.get_llm_config",
            return_value=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test-key",
            ),
        ),
        patch("app.llm.litellm.acompletion", new=AsyncMock(return_value=response)),
    ):
        result = asyncio.run(extract_job_keywords("Need a Python backend engineer"))

    assert result["required_skills"] == ["Python"]
    assert result["preferred_skills"] == ["Kubernetes"]
    assert result["keywords"] == ["backend"]
    assert "personalInfo" not in result
    assert not any("personalInfo" in record.message for record in caplog.records)
    assert not any(
        "Parsed JSON appears truncated" in record.message for record in caplog.records
    )


def test_complete_json_retries_when_resume_payload_looks_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    response = _mock_json_response({"summary": "Tailored summary"})
    completion = AsyncMock(side_effect=[response, response, response])

    with patch("app.llm.litellm.acompletion", new=completion):
        result = asyncio.run(
            complete_json(
                prompt="Return resume json",
                config=LLMConfig(
                    provider="openai",
                    model="gpt-4o-mini",
                    api_key="test-key",
                ),
                truncation_checker=_check_resume_json_truncation,
            )
        )

    assert result == {"summary": "Tailored summary"}
    assert completion.await_count == 3
    assert any(
        "missing required section 'personalInfo'" in record.message
        for record in caplog.records
    )


def test_improve_resume_raises_for_truncated_resume_result() -> None:
    with patch(
        "app.services.improver.complete_json",
        AsyncMock(return_value={"summary": "Missing personal info"}),
    ):
        with pytest.raises(ValueError, match="Missing required section: personalInfo"):
            asyncio.run(
                improve_resume(
                    original_resume="Original resume",
                    job_description="Backend role",
                    job_keywords={"required_skills": ["Python"]},
                    prompt_id="keywords",
                )
            )
