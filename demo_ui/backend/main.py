"""
FastAPI backend for ITS demo.

Exposes two endpoints:
- GET /health: Health check
- POST /compare: Compare baseline vs ITS
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
logger.info(f"Loading .env file from: {env_path}")
logger.info(f".env file exists: {env_path.exists()}")
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded (log only if exists, don't log the actual key)
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    logger.info(f"OPENAI_API_KEY loaded successfully (length: {len(openai_key)})")
else:
    logger.warning("OPENAI_API_KEY not found in environment!")

from its_hub.lms import OpenAICompatibleLanguageModel, LiteLLMLanguageModel
from its_hub.algorithms import BestOfN, SelfConsistency
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.types import ChatMessage
from its_hub.utils import extract_content_from_lm_response
from its_hub.base import AbstractLanguageModel

from .config import get_model_config, get_api_key, MODEL_REGISTRY
from .vertex_lm import VertexAIClaudeModel
from .models import (
    CompareRequest,
    CompareResponse,
    ResultDetail,
    HealthResponse,
)

# Create FastAPI app
app = FastAPI(
    title="ITS Demo API",
    description="Demo API for comparing baseline vs ITS (Inference-Time Scaling)",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        message="ITS Demo API is running"
    )


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "id": model_id,
                "description": config["description"],
                "model_name": config["model_name"],
            }
            for model_id, config in MODEL_REGISTRY.items()
        ]
    }


def create_language_model(model_id: str) -> AbstractLanguageModel:
    """
    Create a language model instance based on the model configuration.

    Supports OpenAI, Anthropic, and Vertex AI providers.
    """
    model_config = get_model_config(model_id)
    provider = model_config.get("provider", "openai")

    if provider == "vertex_ai":
        # Use native Vertex AI SDK for Claude models
        vertex_project = model_config.get("vertex_project")
        vertex_location = model_config.get("vertex_location")

        if not vertex_project or vertex_project == "your-gcp-project-id":
            raise ValueError(
                "VERTEX_PROJECT not configured. Please set VERTEX_PROJECT "
                "environment variable in your .env file"
            )

        logger.info(
            f"Creating Vertex AI Claude model: {model_config['model_name']} "
            f"(project: {vertex_project}, location: {vertex_location})"
        )

        return VertexAIClaudeModel(
            project_id=vertex_project,
            location=vertex_location,
            model_name=model_config["model_name"],
        )

    elif provider == "anthropic":
        # Use LiteLLM for Anthropic direct API
        api_key = get_api_key(model_id)

        logger.info(f"Creating Anthropic model via LiteLLM: {model_config['model_name']}")

        return LiteLLMLanguageModel(
            model_name=model_config["model_name"],
            api_key=api_key,
            is_async=True,
        )
    else:
        # Use OpenAI-compatible endpoint
        api_key = get_api_key(model_id)

        logger.info(f"Creating OpenAI-compatible model: {model_config['model_name']}")

        return OpenAICompatibleLanguageModel(
            endpoint=model_config["base_url"],
            api_key=api_key,
            model_name=model_config["model_name"],
        )


async def run_baseline(
    lm: OpenAICompatibleLanguageModel,
    question: str
) -> tuple[str, int]:
    """
    Run baseline inference (single completion, no ITS).

    Returns:
        (answer, latency_ms)
    """
    start_time = time.time()

    messages = [ChatMessage(role="user", content=question)]
    response = await lm.agenerate(messages)
    answer = extract_content_from_lm_response(response)

    latency_ms = int((time.time() - start_time) * 1000)

    return answer, latency_ms


async def run_its(
    lm: OpenAICompatibleLanguageModel,
    question: str,
    algorithm: str,
    budget: int,
    api_key: str,
) -> tuple[str, int]:
    """
    Run ITS inference with the specified algorithm.

    Returns:
        (answer, latency_ms)
    """
    start_time = time.time()

    # Create algorithm instance
    if algorithm == "best_of_n":
        # Use LLM judge for Best-of-N
        judge = LLMJudgeRewardModel(
            model="gpt-4o-mini",
            criterion="overall_quality",
            judge_type="pointwise",
            api_key=api_key,
            enable_judge_logging=False,
        )
        alg = BestOfN(judge)
    elif algorithm == "self_consistency":
        # Self-consistency doesn't need a judge
        alg = SelfConsistency()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run inference
    result = await alg.ainfer(lm, question, budget=budget, return_response_only=True)
    answer = extract_content_from_lm_response(result)

    latency_ms = int((time.time() - start_time) * 1000)

    return answer, latency_ms


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    Compare baseline vs ITS inference.

    Input:
        - question: The question to answer
        - model_id: Model identifier from the registry
        - algorithm: ITS algorithm (best_of_n or self_consistency)
        - budget: Computation budget

    Output:
        - baseline: { answer, latency_ms, log_preview }
        - its: { answer, latency_ms, log_preview }
        - meta: { model_id, algorithm, budget, run_id }
    """
    run_id = str(uuid.uuid4())

    logger.info(
        f"[{run_id}] Starting comparison: "
        f"model={request.model_id}, algorithm={request.algorithm}, budget={request.budget}"
    )

    try:
        # Create language model based on provider
        lm = create_language_model(request.model_id)

        # Get API key for judge (always use OpenAI for judge)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for LLM judge")

        # Run baseline and ITS in parallel
        baseline_task = run_baseline(lm, request.question)
        its_task = run_its(
            lm,
            request.question,
            request.algorithm,
            request.budget,
            openai_key,
        )

        (baseline_answer, baseline_latency), (its_answer, its_latency) = await asyncio.gather(
            baseline_task,
            its_task,
        )

        logger.info(
            f"[{run_id}] Comparison complete: "
            f"baseline_latency={baseline_latency}ms, its_latency={its_latency}ms"
        )

        # Build response
        response = CompareResponse(
            baseline=ResultDetail(
                answer=baseline_answer,
                latency_ms=baseline_latency,
                log_preview="",  # Placeholder for future
            ),
            its=ResultDetail(
                answer=its_answer,
                latency_ms=its_latency,
                log_preview="",  # Placeholder for future
            ),
            meta={
                "model_id": request.model_id,
                "algorithm": request.algorithm,
                "budget": request.budget,
                "run_id": run_id,
            }
        )

        return response

    except ValueError as e:
        logger.error(f"[{run_id}] Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{run_id}] Error during comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
