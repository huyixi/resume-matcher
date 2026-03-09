import unittest
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks

from app import llm
from app.routers import config as config_router
from app.schemas import LLMConfigRequest


class TestLlmConfigResolution(unittest.TestCase):
    def test_get_llm_config_uses_provider_scoped_api_key(self) -> None:
        stored_config = {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "api_keys": {"google": "google-secret"},
        }

        with patch.object(llm, "_load_stored_config", return_value=stored_config):
            config = llm.get_llm_config()

        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-3-flash-preview")
        self.assertEqual(config.api_key, "google-secret")


class TestLlmConfigEndpoints(unittest.IsolatedAsyncioTestCase):
    async def test_get_llm_config_endpoint_masks_provider_scoped_api_key(self) -> None:
        stored_config = {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "api_keys": {"google": "secret-key-1234"},
        }

        with patch.object(config_router, "_load_config", return_value=stored_config):
            response = await config_router.get_llm_config_endpoint()

        self.assertEqual(response.provider, "gemini")
        self.assertEqual(response.model, "gemini-3-flash-preview")
        self.assertTrue(response.api_key.startswith("secr"))
        self.assertTrue(response.api_key.endswith("1234"))

    async def test_update_llm_config_syncs_provider_scoped_key_store(self) -> None:
        stored_config = {"provider": "gemini", "model": "gemini-3-flash-preview"}
        request = LLMConfigRequest(api_key="secret-key-1234")
        background_tasks = BackgroundTasks()

        with (
            patch.object(config_router, "_load_config", return_value=stored_config),
            patch.object(config_router, "_save_config") as save_config,
            patch.object(config_router, "_log_llm_health_check", AsyncMock()),
        ):
            response = await config_router.update_llm_config(request, background_tasks)

        saved = save_config.call_args.args[0]
        self.assertEqual(saved["api_key"], "secret-key-1234")
        self.assertEqual(saved["api_keys"]["google"], "secret-key-1234")
        self.assertTrue(response.api_key.startswith("secr"))
        self.assertTrue(response.api_key.endswith("1234"))
