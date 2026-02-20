"""
Configuration for the demo backend.

Model registry maps model_id to connection details.
"""

import os
from typing import Dict, TypedDict


class ModelConfig(TypedDict, total=False):
    """Configuration for a single model."""
    base_url: str
    api_key_env_var: str
    model_name: str
    description: str
    provider: str  # Optional: 'openai', 'vertex_ai', etc. Defaults to 'openai'
    vertex_project: str  # Optional: for Vertex AI
    vertex_location: str  # Optional: for Vertex AI


# Model registry
# Maps model_id -> { base_url, api_key_env_var, model_name }
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # GPT-4o models (latest)
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4o",
        "description": "OpenAI GPT-4o (Most capable)",
    },
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini (Fast & affordable)",
    },

    # GPT-4 Turbo
    "gpt-4-turbo": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4-turbo",
        "description": "OpenAI GPT-4 Turbo",
    },

    # GPT-4
    "gpt-4": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4",
        "description": "OpenAI GPT-4",
    },

    # GPT-3.5
    "gpt-3.5-turbo": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-3.5-turbo",
        "description": "OpenAI GPT-3.5 Turbo",
    },

    # o1 models (reasoning models)
    "o1-preview": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "o1-preview",
        "description": "OpenAI o1-preview (Reasoning)",
    },
    "o1-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "o1-mini",
        "description": "OpenAI o1-mini (Fast reasoning)",
    },

    # Claude via Vertex AI (Google Cloud)
    # Note: Only models you have access to are included here
    # To request access to more models, visit:
    # https://console.cloud.google.com/vertex-ai/publishers/anthropic
    "claude-haiku-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "claude-3-5-haiku@20241022",
        "description": "Claude 3.5 Haiku (Vertex AI)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
    },

    # Uncomment these once you have access (check with check_vertex_models.py)
    # "claude-sonnet-vertex": {
    #     "base_url": "",
    #     "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
    #     "model_name": "claude-3-5-sonnet-v2@20241022",
    #     "description": "Claude 3.5 Sonnet v2 (Vertex AI)",
    #     "provider": "vertex_ai",
    #     "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
    #     "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
    # },
    # "claude-opus-vertex": {
    #     "base_url": "",
    #     "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
    #     "model_name": "claude-3-opus@20240229",
    #     "description": "Claude 3 Opus (Vertex AI)",
    #     "provider": "vertex_ai",
    #     "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
    #     "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
    # },

    # Claude via Anthropic API (direct)
    "claude-sonnet": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "model_name": "claude-3-5-sonnet-20241022",
        "description": "Claude 3.5 Sonnet (Anthropic)",
        "provider": "anthropic",
    },
    "claude-opus": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "model_name": "claude-3-opus-20240229",
        "description": "Claude 3 Opus (Anthropic)",
        "provider": "anthropic",
    },
    "claude-haiku": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "model_name": "claude-3-5-haiku-20241022",
        "description": "Claude 3.5 Haiku (Anthropic)",
        "provider": "anthropic",
    },

    # Local vLLM
    "local-vllm": {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1"),
        "api_key_env_var": "VLLM_API_KEY",
        "model_name": os.getenv("VLLM_MODEL_NAME", "your-model-name"),
        "description": "Local vLLM server",
    },
}


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a model by ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def get_api_key(model_id: str) -> str:
    """Get API key for a model from environment variables."""
    config = get_model_config(model_id)
    api_key = os.getenv(config["api_key_env_var"])
    if not api_key:
        raise ValueError(
            f"API key not found for model '{model_id}'. "
            f"Please set environment variable '{config['api_key_env_var']}'"
        )
    return api_key
