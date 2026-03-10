import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.llm import AIServiceError, LLMConfig, complete_json
from app.main import app
from app.routers import resumes as resumes_router


class TestLlmErrorClassification(unittest.IsolatedAsyncioTestCase):
    async def test_complete_json_raises_api_key_missing_when_not_configured(self) -> None:
        with patch(
            "app.llm.get_llm_config",
            return_value=LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key="",
            ),
        ):
            with self.assertRaises(AIServiceError) as exc_info:
                await complete_json("prompt")

        self.assertEqual(exc_info.exception.error_code, "api_key_missing")
        self.assertEqual(exc_info.exception.status_code, 400)


class TestImproveErrorResponses(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.resume = {
            "resume_id": "resume-1",
            "content": "Resume content",
            "processed_data": {"personalInfo": {"name": "Test User"}},
        }
        self.job = {
            "job_id": "job-1",
            "content": "Job content",
        }

    def test_preview_returns_classified_ai_error(self) -> None:
        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "extract_job_keywords",
                AsyncMock(
                    side_effect=AIServiceError(
                        error_code="rate_limited",
                        status_code=429,
                        detail="The LLM provider rate limit was reached. Please try again shortly.",
                    )
                ),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = self.job

            response = self.client.post(
                "/api/v1/resumes/improve/preview",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {
                "detail": "The LLM provider rate limit was reached. Please try again shortly.",
                "error_code": "rate_limited",
            },
        )

    def test_improve_returns_classified_ai_error(self) -> None:
        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "extract_job_keywords",
                AsyncMock(
                    side_effect=AIServiceError(
                        error_code="auth_failed",
                        status_code=401,
                        detail="LLM authentication failed. Please check your API key and provider settings.",
                    )
                ),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = self.job

            response = self.client.post(
                "/api/v1/resumes/improve",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json(),
            {
                "detail": "LLM authentication failed. Please check your API key and provider settings.",
                "error_code": "auth_failed",
            },
        )

    def test_preview_keeps_generic_internal_error_for_non_ai_failures(self) -> None:
        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "extract_job_keywords",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = self.job

            response = self.client.post(
                "/api/v1/resumes/improve/preview",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Failed to preview resume. Please try again."},
        )

    def test_preview_returns_controlled_500_when_response_finalization_fails(self) -> None:
        job = {
            "job_id": "job-1",
            "content": "Job content",
            "job_keywords": {
                "required_skills": ["Python"],
                "preferred_skills": [],
                "keywords": [],
            },
            "job_keywords_hash": resumes_router._hash_job_content("Job content"),
        }

        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "improve_resume",
                AsyncMock(return_value={"personalInfo": {"name": "Test User"}}),
            ),
            patch.object(
                resumes_router,
                "_calculate_diff_from_resume",
                return_value=(None, None, None),
            ),
            patch.object(
                resumes_router,
                "generate_improvements",
                return_value=[],
            ),
            patch.object(
                resumes_router,
                "_finalize_improve_response",
                side_effect=ValueError("preview_response is not JSON serializable"),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = job
            resumes_router.db.get_master_resume.return_value = None
            resumes_router.db.update_job.return_value = job

            response = self.client.post(
                "/api/v1/resumes/improve/preview",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Failed to preview resume. Please try again."},
        )

    def test_preview_logs_key_stages_on_success(self) -> None:
        job = {
            "job_id": "job-1",
            "content": "Job content",
            "job_keywords": {
                "required_skills": ["Python"],
                "preferred_skills": [],
                "keywords": [],
            },
            "job_keywords_hash": resumes_router._hash_job_content("Job content"),
        }

        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "improve_resume",
                AsyncMock(return_value={"personalInfo": {"name": "Test User"}}),
            ),
            patch.object(
                resumes_router,
                "_calculate_diff_from_resume",
                return_value=(None, None, None),
            ),
            patch.object(
                resumes_router,
                "generate_improvements",
                return_value=[],
            ),
            patch.object(resumes_router, "_log_improve_stage") as stage_logger,
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = job
            resumes_router.db.get_master_resume.return_value = None
            resumes_router.db.update_job.return_value = job

            response = self.client.post(
                "/api/v1/resumes/improve/preview",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["data"]["markdownOriginal"])
        self.assertIsNone(payload["data"]["markdownImproved"])
        logged_stages = [call.args[1] for call in stage_logger.call_args_list]
        self.assertEqual(
            logged_stages,
            [
                "load_job_keywords",
                "improve_resume",
                "refine_resume",
                "serialize_improved_data",
                "hash_improved_data",
                "persist_preview_hash",
                "calculate_diff",
                "generate_improvements",
                "build_response_model",
                "validate_response_json",
                "return_response",
            ],
        )

    def test_improve_returns_json_success_payload(self) -> None:
        job = {
            "job_id": "job-1",
            "content": "Job content",
        }
        tailored_resume = {
            "resume_id": "tailored-1",
            "content": "{}",
        }

        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "extract_job_keywords",
                AsyncMock(
                    return_value={
                        "required_skills": ["Python"],
                        "preferred_skills": [],
                        "keywords": [],
                    }
                ),
            ),
            patch.object(
                resumes_router,
                "improve_resume",
                AsyncMock(return_value={"personalInfo": {"name": "Test User"}}),
            ),
            patch.object(
                resumes_router,
                "_calculate_diff_from_resume",
                return_value=(None, None, None),
            ),
            patch.object(
                resumes_router,
                "generate_improvements",
                return_value=[],
            ),
            patch.object(
                resumes_router,
                "_generate_auxiliary_messages",
                AsyncMock(return_value=(None, None, "Tailored Title", [])),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = job
            resumes_router.db.get_master_resume.return_value = None
            resumes_router.db.create_resume.return_value = tailored_resume
            resumes_router.db.create_improvement.return_value = {"request_id": "imp-1"}

            response = self.client.post(
                "/api/v1/resumes/improve",
                json={"resume_id": "resume-1", "job_id": "job-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/json")
        payload = response.json()
        self.assertEqual(payload["data"]["resume_id"], "tailored-1")
        self.assertEqual(payload["data"]["job_id"], "job-1")

    def test_confirm_returns_success_payload(self) -> None:
        improved_data = resumes_router.ResumeData.model_validate(
            {"personalInfo": {"name": "Test User"}}
        ).model_dump()
        job = {
            "job_id": "job-1",
            "content": "Job content",
            "preview_hashes": {
                "keywords": resumes_router._hash_improved_data(improved_data),
            },
        }
        tailored_resume = {
            "resume_id": "tailored-1",
            "content": "{}",
        }

        with (
            patch.object(resumes_router, "db", MagicMock()),
            patch.object(
                resumes_router,
                "_calculate_diff_from_resume",
                return_value=(None, None, None),
            ),
            patch.object(
                resumes_router,
                "_generate_auxiliary_messages",
                AsyncMock(return_value=(None, None, "Tailored Title", [])),
            ),
        ):
            resumes_router.db.get_resume.return_value = self.resume
            resumes_router.db.get_job.return_value = job
            resumes_router.db.create_resume.return_value = tailored_resume
            resumes_router.db.create_improvement.return_value = {"request_id": "imp-1"}

            response = self.client.post(
                "/api/v1/resumes/improve/confirm",
                json={
                    "resume_id": "resume-1",
                    "job_id": "job-1",
                    "improved_data": improved_data,
                    "improvements": [],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/json")
        payload = response.json()
        self.assertEqual(payload["data"]["resume_id"], "tailored-1")
        self.assertEqual(payload["data"]["job_id"], "job-1")


class TestImproveResponseSafety(unittest.TestCase):
    def test_sanitize_json_value_coerces_non_finite_floats(self) -> None:
        payload = {
            "stats": {
                "initial": float("nan"),
                "final": float("inf"),
                "negative": float("-inf"),
            }
        }

        sanitized = resumes_router._sanitize_json_value(payload, "response_payload")

        self.assertEqual(
            sanitized,
            {
                "stats": {
                    "initial": 0.0,
                    "final": 0.0,
                    "negative": 0.0,
                }
            },
        )
