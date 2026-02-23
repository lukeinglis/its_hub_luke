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
    size: str  # Optional: Model size (e.g., "175B", "70B", "7B")
    input_cost_per_1m: float  # Optional: Cost per 1M input tokens in USD
    output_cost_per_1m: float  # Optional: Cost per 1M output tokens in USD


# Model registry
# Maps model_id -> { base_url, api_key_env_var, model_name }
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # ========================================================================
    # OPENAI MODELS (Native OpenAI API)
    # ========================================================================

    # === Frontier Models 🏆 ===
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4o",
        "description": "🏆 GPT-4o (Frontier)",
        "provider": "openai",
        "size": "Large",
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
    },
    "gpt-4-turbo": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4-turbo",
        "description": "🏆 GPT-4 Turbo (Frontier)",
        "provider": "openai",
        "size": "Large",
        "input_cost_per_1m": 10.00,
        "output_cost_per_1m": 30.00,
    },

    # === Small/Fast Models ⚡ ===
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "description": "⚡ GPT-4o Mini (Small, fast)",
        "provider": "openai",
        "size": "Small",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "gpt-3.5-turbo": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "model_name": "gpt-3.5-turbo",
        "description": "⚡ GPT-3.5 Turbo (Small)",
        "provider": "openai",
        "size": "Small",
        "input_cost_per_1m": 0.50,
        "output_cost_per_1m": 1.50,
    },

    # ========================================================================
    # VERTEX AI MODELS (Google Cloud)
    # ========================================================================
    # Setup: https://console.cloud.google.com/vertex-ai
    # Authentication: gcloud auth application-default login
    # Or set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path

    # === Claude Models (via Vertex AI) ===
    # Frontier Models 🏆
    "claude-sonnet-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "claude-3-5-sonnet-v2@20241022",
        "description": "🏆 Claude 3.5 Sonnet (Vertex AI, Frontier)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
        "size": "Large",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
    },
    "claude-opus-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "claude-3-opus@20240229",
        "description": "🏆 Claude 3 Opus (Vertex AI, Frontier)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
        "size": "Large",
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 75.00,
    },
    # Small/Fast Models ⚡
    "claude-haiku-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "claude-3-5-haiku@20241022",
        "description": "⚡ Claude 3.5 Haiku (Vertex AI, Small)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
        "size": "Small",
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
    },

    # === Gemini Models (via Vertex AI) ===
    # Frontier Models 🏆
    "gemini-pro-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "gemini-1.5-pro-002",
        "description": "🏆 Gemini 1.5 Pro (Vertex AI, Frontier)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
        "size": "Large",
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 5.00,
    },
    # Small/Fast Models ⚡
    "gemini-flash-vertex": {
        "base_url": "",  # Not used for Vertex AI
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_name": "gemini-1.5-flash-002",
        "description": "⚡ Gemini 1.5 Flash (Vertex AI, Small)",
        "provider": "vertex_ai",
        "vertex_project": os.getenv("VERTEX_PROJECT", "your-gcp-project-id"),
        "vertex_location": os.getenv("VERTEX_LOCATION", "us-east5"),
        "size": "Small",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.30,
    },

    # ========================================================================
    # IBM GRANITE MODELS
    # ========================================================================
    # IBM's open-source Granite models for enterprise AI
    # Latest versions: Granite 4.0 and Granite 3.3
    # Available via Hugging Face or self-hosted via vLLM

    # === Granite 4.0 Models (Latest) ===
    # Note: Requires vLLM server running on GRANITE_BASE_URL
    # Start with: vllm serve ibm-granite/granite-4.0-8b-instruct --port 8100
    "granite-4.0-8b": {
        "base_url": os.getenv("GRANITE_BASE_URL", "http://localhost:8100/v1"),
        "api_key_env_var": "GRANITE_API_KEY",
        "model_name": "ibm-granite/granite-4.0-8b-instruct",
        "description": "🏢 IBM Granite 4.0 8B (Latest, open-source)",
        "provider": "openai",
        "size": "8B",
        "input_cost_per_1m": 0.0,  # Free if self-hosted
        "output_cost_per_1m": 0.0,
    },
    "granite-4.0-3b": {
        "base_url": os.getenv("GRANITE_BASE_URL", "http://localhost:8100/v1"),
        "api_key_env_var": "GRANITE_API_KEY",
        "model_name": "ibm-granite/granite-4.0-3b-instruct",
        "description": "⚡ IBM Granite 4.0 3B (Small, fast)",
        "provider": "openai",
        "size": "3B",
        "input_cost_per_1m": 0.0,  # Free if self-hosted
        "output_cost_per_1m": 0.0,
    },

    # === Granite 3.3 Models ===
    "granite-3.3-8b": {
        "base_url": os.getenv("GRANITE_BASE_URL", "http://localhost:8100/v1"),
        "api_key_env_var": "GRANITE_API_KEY",
        "model_name": "ibm-granite/granite-3.3-8b-instruct",
        "description": "🏢 IBM Granite 3.3 8B (Open-source)",
        "provider": "openai",
        "size": "8B",
        "input_cost_per_1m": 0.0,  # Free if self-hosted
        "output_cost_per_1m": 0.0,
    },

    # ===== LOCAL/CUSTOM MODELS =====
    # For running your own vLLM server with any open-source model
    "local-vllm": {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1"),
        "api_key_env_var": "VLLM_API_KEY",
        "model_name": os.getenv("VLLM_MODEL_NAME", "your-model-name"),
        "description": "🔧 Local vLLM (Your own model)",
        "size": os.getenv("VLLM_MODEL_SIZE", "Custom"),
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
