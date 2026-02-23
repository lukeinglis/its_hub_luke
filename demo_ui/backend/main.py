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

from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import (
    BestOfN,
    SelfConsistency,
    BeamSearch,
    ParticleFiltering,
    EntropicParticleFiltering,
    ParticleGibbs,
)
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.types import ChatMessage
from its_hub.utils import extract_content_from_lm_response
from its_hub.base import AbstractLanguageModel

from .config import get_model_config, get_api_key, MODEL_REGISTRY, ModelConfig
from .vertex_lm import VertexAIClaudeModel, VertexAIGeminiModel
from .llm_prm import LLMProcessRewardModel


def calculate_cost(
    model_config: ModelConfig,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Calculate cost in USD based on token usage and model pricing.

    Returns:
        Cost in USD (rounded to 4 decimal places)
    """
    input_cost_per_1m = model_config.get("input_cost_per_1m", 0.0)
    output_cost_per_1m = model_config.get("output_cost_per_1m", 0.0)

    if input_cost_per_1m == 0.0 and output_cost_per_1m == 0.0:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    return round(total_cost, 6)  # Round to 6 decimal places for precision
from .example_questions import get_all_questions, get_questions_by_algorithm
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


def check_server_available(base_url: str, timeout: float = 1.0) -> bool:
    """Check if a server is available by attempting to connect to it."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 8100

        # Try to connect to the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        return result == 0
    except Exception as e:
        logger.debug(f"Server check failed for {base_url}: {e}")
        return False


@app.get("/models")
async def list_models():
    """List available models. Only include models where the server is available."""
    available_models = []

    for model_id, config in MODEL_REGISTRY.items():
        # Check if model requires external server (has non-standard base_url)
        base_url = config.get("base_url", "")

        # Skip server check for standard OpenAI and Vertex AI models
        if (base_url.startswith("https://api.openai.com") or
            config.get("provider") == "vertex_ai" or
            not base_url):
            available_models.append({
                "id": model_id,
                "description": config["description"],
                "model_name": config["model_name"],
                "size": config.get("size", "Unknown"),
            })
            continue

        # For custom endpoints (Granite, local vLLM), check if server is available
        server_available = check_server_available(base_url, timeout=1.0)

        if server_available:
            available_models.append({
                "id": model_id,
                "description": config["description"],
                "model_name": config["model_name"],
                "size": config.get("size", "Unknown"),
            })
        else:
            logger.debug(f"Skipping model {model_id} - server at {base_url} not available")

    return {"models": available_models}


@app.get("/examples")
async def list_examples(algorithm: str | None = None):
    """
    Get example questions.

    Query params:
        algorithm: Optional algorithm to filter questions by (e.g., 'beam_search')
    """
    if algorithm:
        questions = get_questions_by_algorithm(algorithm, limit=10)
    else:
        questions = get_all_questions()

    return {
        "examples": [
            {
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "expected_answer": q["expected_answer"],
                "best_for": q["best_for"],
                "why": q["why"],
                "source": q.get("source", "unknown"),
                "source_id": q.get("source_id", ""),
            }
            for q in questions
        ]
    }


def create_language_model(model_id: str) -> AbstractLanguageModel:
    """
    Create a language model instance based on the model configuration.

    All models use OpenAI-compatible endpoints except Vertex AI models.
    """
    model_config = get_model_config(model_id)
    provider = model_config.get("provider", "openai")

    if provider == "vertex_ai":
        # Use native Vertex AI SDK for Claude and Gemini models (NOT OpenAI-compatible)
        vertex_project = model_config.get("vertex_project")
        vertex_location = model_config.get("vertex_location")

        if not vertex_project or vertex_project == "your-gcp-project-id":
            raise ValueError(
                "VERTEX_PROJECT not configured. Please set VERTEX_PROJECT "
                "environment variable in your .env file"
            )

        model_name = model_config["model_name"]

        # Determine if it's a Claude or Gemini model based on model name
        if "claude" in model_name.lower():
            logger.info(
                f"Creating Vertex AI Claude model: {model_name} "
                f"(project: {vertex_project}, location: {vertex_location})"
            )
            return VertexAIClaudeModel(
                project_id=vertex_project,
                location=vertex_location,
                model_name=model_name,
            )
        elif "gemini" in model_name.lower():
            logger.info(
                f"Creating Vertex AI Gemini model: {model_name} "
                f"(project: {vertex_project}, location: {vertex_location})"
            )
            return VertexAIGeminiModel(
                project_id=vertex_project,
                location=vertex_location,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unknown Vertex AI model type: {model_name}")
    else:
        # All other models use OpenAI-compatible endpoints
        # This includes: OpenAI, OpenRouter (Claude, Gemini), Together AI (open-source), vLLM
        api_key = get_api_key(model_id)

        logger.info(f"Creating OpenAI-compatible model: {model_config['model_name']} via {model_config['base_url']}")

        return OpenAICompatibleLanguageModel(
            endpoint=model_config["base_url"],
            api_key=api_key,
            model_name=model_config["model_name"],
        )


async def run_baseline(
    lm: AbstractLanguageModel,
    question: str
) -> tuple[str, int, int, int]:
    """
    Run baseline inference (single completion, no ITS).

    Returns:
        (answer, latency_ms, input_tokens, output_tokens)
    """
    start_time = time.time()

    messages = [ChatMessage(role="user", content=question)]

    # Get response and try to capture usage information
    input_tokens = 0
    output_tokens = 0

    # For OpenAI-compatible models, try to get usage directly via litellm
    if isinstance(lm, OpenAICompatibleLanguageModel):
        import litellm
        try:
            request_data = lm._prepare_request_data(
                messages,
                stop=None,
                max_tokens=None,
                temperature=None,
                include_stop_str_in_output=None,
                tools=None,
                tool_choice=None,
            )
            full_response = await litellm.acompletion(**request_data)

            # Extract usage from full response
            if hasattr(full_response, 'usage') and full_response.usage:
                input_tokens = getattr(full_response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(full_response.usage, 'completion_tokens', 0)

            # Extract message
            response = full_response.choices[0].message.dict()
        except Exception as e:
            logger.warning(f"Could not capture token usage: {e}")
            response = await lm.agenerate(messages)
    else:
        # For Vertex AI or other models, use standard interface
        response = await lm.agenerate(messages)

    answer = extract_content_from_lm_response(response)
    latency_ms = int((time.time() - start_time) * 1000)

    return answer, latency_ms, input_tokens, output_tokens


async def run_its(
    lm: OpenAICompatibleLanguageModel,
    question: str,
    algorithm: str,
    budget: int,
    api_key: str,
    baseline_input_tokens: int = 0,
    baseline_output_tokens: int = 0,
) -> tuple[str, int, int, int]:
    """
    Run ITS inference with the specified algorithm.

    Returns:
        (answer, latency_ms, input_tokens, output_tokens)
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

    elif algorithm in ["beam_search", "particle_filtering", "entropic_particle_filtering", "particle_gibbs"]:
        # Process-based algorithms need StepGeneration and Process Reward Model

        # Create StepGeneration with step-by-step reasoning
        # Using "\n\n" as step delimiter for reasoning steps
        step_gen = StepGeneration(
            max_steps=8,  # Maximum reasoning steps
            step_token="\n\n",  # Step delimiter
            stop_token=None,  # No explicit stop token
            temperature=0.8,
            include_stop_str_in_output=False,
        )

        # Create LLM-based process reward model
        prm = LLMProcessRewardModel(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
        )

        if algorithm == "beam_search":
            # BeamSearch with beam_width (use budget/2 for reasonable beam width)
            beam_width = max(2, min(4, budget // 2))
            # Adjust budget to be divisible by beam_width
            adjusted_budget = (budget // beam_width) * beam_width
            alg = BeamSearch(sg=step_gen, prm=prm, beam_width=beam_width)
            budget = adjusted_budget

        elif algorithm == "particle_filtering":
            # Basic particle filtering
            alg = ParticleFiltering(
                sg=step_gen,
                prm=prm,
            )

        elif algorithm == "entropic_particle_filtering":
            # Entropic particle filtering with temperature annealing
            alg = EntropicParticleFiltering(
                sg=step_gen,
                prm=prm,
            )

        elif algorithm == "particle_gibbs":
            # Particle Gibbs with multiple iterations
            # Use smaller budget for iterations to avoid timeout
            num_iterations = max(2, min(3, budget // 4))
            alg = ParticleGibbs(
                sg=step_gen,
                prm=prm,
                num_iterations=num_iterations,
            )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run inference
    result = await alg.ainfer(lm, question, budget=budget, return_response_only=True)
    answer = extract_content_from_lm_response(result)

    latency_ms = int((time.time() - start_time) * 1000)

    # Estimate token usage for ITS
    # ITS algorithms typically make multiple calls (roughly proportional to budget)
    # For outcome-based (best_of_n, self_consistency): budget calls
    # For process-based: budget calls spread across steps
    # This is an approximation since we don't have exact tracking within algorithms
    if baseline_input_tokens > 0 and baseline_output_tokens > 0:
        # Estimate: ITS makes ~budget calls, each similar to baseline
        # Input stays similar, output may vary
        estimated_input = baseline_input_tokens * budget
        estimated_output = baseline_output_tokens * budget
    else:
        # Fallback: no estimation available
        estimated_input = 0
        estimated_output = 0

    return answer, latency_ms, estimated_input, estimated_output


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
        # Get API key for judge (always use OpenAI for judge)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for LLM judge")

        # Create language models based on use case
        small_baseline_answer = None
        small_baseline_latency = None
        small_baseline_input_tokens = None
        small_baseline_output_tokens = None

        if request.use_case == "match_frontier":
            # Use Case 2: Small model + ITS vs Large frontier model
            if not request.frontier_model_id:
                raise ValueError("frontier_model_id required for match_frontier use case")

            small_model = create_language_model(request.model_id)
            frontier_model = create_language_model(request.frontier_model_id)

            # Run small baseline first to get token counts for ITS estimation
            small_baseline_answer, small_baseline_latency, small_baseline_input_tokens, small_baseline_output_tokens = await run_baseline(
                small_model, request.question
            )

            # Run ITS and frontier baseline in parallel
            its_task = run_its(
                small_model,
                request.question,
                request.algorithm,
                request.budget,
                openai_key,
                small_baseline_input_tokens,
                small_baseline_output_tokens,
            )
            frontier_baseline_task = run_baseline(frontier_model, request.question)

            (its_answer, its_latency, its_input_tokens, its_output_tokens), (baseline_answer, baseline_latency, baseline_input_tokens, baseline_output_tokens) = await asyncio.gather(
                its_task,
                frontier_baseline_task,
            )
        else:
            # Use Case 1: Same model with/without ITS
            lm = create_language_model(request.model_id)

            # Run baseline first to get token counts for ITS estimation
            baseline_answer, baseline_latency, baseline_input_tokens, baseline_output_tokens = await run_baseline(
                lm, request.question
            )

            # Run ITS with token estimates
            its_answer, its_latency, its_input_tokens, its_output_tokens = await run_its(
                lm,
                request.question,
                request.algorithm,
                request.budget,
                openai_key,
                baseline_input_tokens,
                baseline_output_tokens,
            )

        logger.info(
            f"[{run_id}] Comparison complete: "
            f"baseline_latency={baseline_latency}ms, its_latency={its_latency}ms"
        )

        # Get model configs
        model_config = get_model_config(request.model_id)
        model_size = model_config.get("size", "Unknown")

        frontier_model_config = None
        frontier_model_size = None
        if request.use_case == "match_frontier" and request.frontier_model_id:
            frontier_model_config = get_model_config(request.frontier_model_id)
            frontier_model_size = frontier_model_config.get("size", "Unknown")

        # Calculate costs
        if request.use_case == "match_frontier":
            # Small baseline cost
            small_baseline_cost = calculate_cost(
                model_config,
                small_baseline_input_tokens or 0,
                small_baseline_output_tokens or 0
            )
            # ITS cost (small model)
            its_cost = calculate_cost(
                model_config,
                its_input_tokens or 0,
                its_output_tokens or 0
            )
            # Frontier baseline cost
            baseline_cost = calculate_cost(
                frontier_model_config,
                baseline_input_tokens or 0,
                baseline_output_tokens or 0
            )
        else:
            # Both use same model
            baseline_cost = calculate_cost(
                model_config,
                baseline_input_tokens or 0,
                baseline_output_tokens or 0
            )
            its_cost = calculate_cost(
                model_config,
                its_input_tokens or 0,
                its_output_tokens or 0
            )
            small_baseline_cost = None

        # Build response
        response_data = {
            "baseline": ResultDetail(
                answer=baseline_answer,
                latency_ms=baseline_latency,
                log_preview="",  # Placeholder for future
                model_size=frontier_model_size if request.use_case == "match_frontier" else model_size,
                cost_usd=baseline_cost if baseline_cost > 0 else None,
                input_tokens=baseline_input_tokens if baseline_input_tokens > 0 else None,
                output_tokens=baseline_output_tokens if baseline_output_tokens > 0 else None,
            ),
            "its": ResultDetail(
                answer=its_answer,
                latency_ms=its_latency,
                log_preview="",  # Placeholder for future
                model_size=model_size,
                cost_usd=its_cost if its_cost > 0 else None,
                input_tokens=its_input_tokens if its_input_tokens > 0 else None,
                output_tokens=its_output_tokens if its_output_tokens > 0 else None,
            ),
            "meta": {
                "model_id": request.model_id,
                "algorithm": request.algorithm,
                "budget": request.budget,
                "run_id": run_id,
                "use_case": request.use_case,
            }
        }

        # Add small baseline if match_frontier use case
        if request.use_case == "match_frontier" and small_baseline_answer is not None:
            response_data["small_baseline"] = ResultDetail(
                answer=small_baseline_answer,
                latency_ms=small_baseline_latency,
                log_preview="",
                model_size=model_size,
                cost_usd=small_baseline_cost if small_baseline_cost and small_baseline_cost > 0 else None,
                input_tokens=small_baseline_input_tokens if small_baseline_input_tokens and small_baseline_input_tokens > 0 else None,
                output_tokens=small_baseline_output_tokens if small_baseline_output_tokens and small_baseline_output_tokens > 0 else None,
            )
            response_data["meta"]["frontier_model_id"] = request.frontier_model_id

        response = CompareResponse(**response_data)

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
