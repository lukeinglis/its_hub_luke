"""Shared fixtures for demo_ui backend tests."""

import os
import sys

import pytest

# Ensure demo_ui package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_model_config():
    """A typical model config dict for testing."""
    return {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "description": "GPT-4o Mini",
        "provider": "openai",
        "size": "Small",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "supports_tools": True,
    }


@pytest.fixture
def zero_cost_config():
    """Model config with no pricing."""
    return {
        "base_url": "http://localhost:8000/v1",
        "api_key_env_var": "LOCAL_KEY",
        "model_name": "local-model",
        "description": "Local Model",
        "provider": "local",
        "size": "Custom",
    }


@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Set a mock OPENAI_API_KEY in the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
