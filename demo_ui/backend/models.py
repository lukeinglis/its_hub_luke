"""
Pydantic models for request/response validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request to compare baseline vs ITS."""
    question: str = Field(..., min_length=1, description="The question to answer")
    model_id: str = Field(..., description="Model identifier from the registry (for ITS)")
    algorithm: Literal[
        "best_of_n",
        "self_consistency",
        "beam_search",
        "particle_filtering",
        "entropic_particle_filtering",
        "particle_gibbs"
    ] = Field(
        ..., description="ITS algorithm to use"
    )
    budget: int = Field(..., ge=1, le=32, description="Computation budget (1-32)")
    use_case: Literal["improve_model", "match_frontier"] = Field(
        default="improve_model",
        description="Use case: improve_model (same model +ITS) or match_frontier (small+ITS vs large)"
    )
    frontier_model_id: Optional[str] = Field(
        default=None,
        description="Frontier model to compare against (only for match_frontier use case)"
    )


class ResultDetail(BaseModel):
    """Details of a single result (baseline or ITS)."""
    answer: str = Field(..., description="The final answer text")
    latency_ms: int = Field(..., description="Latency in milliseconds")
    log_preview: str = Field(default="", description="Placeholder for step logs (future)")
    model_size: Optional[str] = Field(default=None, description="Model size (e.g., 'Large', 'Small')")
    cost_usd: Optional[float] = Field(default=None, description="Cost in USD")
    input_tokens: Optional[int] = Field(default=None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(default=None, description="Number of output tokens")


class CompareResponse(BaseModel):
    """Response from comparison endpoint."""
    baseline: ResultDetail
    its: ResultDetail
    meta: dict = Field(..., description="Metadata about the comparison run")
    small_baseline: Optional[ResultDetail] = Field(
        default=None,
        description="Small model baseline (only for match_frontier use case)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
