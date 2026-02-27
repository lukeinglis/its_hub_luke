"""Tests for backend/config.py."""

import pytest

from backend.config import MODEL_REGISTRY, get_api_key, get_model_config


class TestGetModelConfig:
    """Tests for get_model_config()."""

    def test_valid_model_returns_config(self):
        config = get_model_config("gpt-4o")
        assert config["model_name"] == "gpt-4o"
        assert config["provider"] == "openai"

    def test_valid_model_has_required_fields(self):
        config = get_model_config("gpt-4o-mini")
        assert "base_url" in config
        assert "api_key_env_var" in config
        assert "model_name" in config

    def test_invalid_model_raises_valueerror(self):
        with pytest.raises(ValueError, match="not found in registry"):
            get_model_config("nonexistent-model-xyz")

    def test_all_registry_models_retrievable(self):
        for model_id in MODEL_REGISTRY:
            config = get_model_config(model_id)
            assert config is not None
            assert "model_name" in config


class TestGetApiKey:
    """Tests for get_api_key()."""

    def test_returns_key_when_set(self, mock_env_api_key):
        key = get_api_key("gpt-4o")
        assert key == "sk-test-key-12345"

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key not found"):
            get_api_key("gpt-4o")

    def test_error_message_includes_env_var_name(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_api_key("gpt-4o")


class TestModelRegistry:
    """Tests for MODEL_REGISTRY structure and completeness."""

    def test_registry_has_models(self):
        assert len(MODEL_REGISTRY) > 0

    def test_all_models_have_api_key_env_var(self):
        for model_id, config in MODEL_REGISTRY.items():
            assert "api_key_env_var" in config, f"Model '{model_id}' missing 'api_key_env_var'"

    def test_all_models_have_description(self):
        for model_id, config in MODEL_REGISTRY.items():
            assert "description" in config, f"Model '{model_id}' missing 'description'"

    def test_known_providers(self):
        expected_providers = {"openai", "openrouter", "vertex_ai", "local"}
        actual_providers = {c.get("provider") for c in MODEL_REGISTRY.values() if "provider" in c}
        assert actual_providers <= expected_providers, (
            f"Unexpected providers: {actual_providers - expected_providers}"
        )
