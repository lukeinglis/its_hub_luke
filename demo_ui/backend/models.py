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


# --- Algorithm Trace Models ---

class CandidateResponse(BaseModel):
    """A single candidate response from an algorithm."""
    index: int = Field(..., description="Candidate index")
    content: str = Field(..., description="Response text content")
    is_selected: bool = Field(default=False, description="Whether this candidate was selected as the winner")


class SelfConsistencyTrace(BaseModel):
    """Trace data for Self-Consistency algorithm."""
    algorithm: Literal["self_consistency"] = "self_consistency"
    candidates: list[CandidateResponse]
    vote_counts: dict[str, int] = Field(..., description="Vote counts per unique answer")
    total_votes: int = Field(..., description="Total number of votes cast")


class BestOfNTrace(BaseModel):
    """Trace data for Best-of-N algorithm."""
    algorithm: Literal["best_of_n"] = "best_of_n"
    candidates: list[CandidateResponse]
    scores: list[float] = Field(..., description="LLM judge scores for each candidate")
    max_score: float
    min_score: float


class BeamSearchTrace(BaseModel):
    """Trace data for Beam Search algorithm."""
    algorithm: Literal["beam_search"] = "beam_search"
    candidates: list[CandidateResponse]
    scores: list[float] = Field(..., description="PRM scores for each beam")
    steps_used: list[int] = Field(..., description="Number of steps used per beam")


class ParticleFilteringTrace(BaseModel):
    """Trace data for Particle Filtering algorithm."""
    algorithm: Literal["particle_filtering"] = "particle_filtering"
    candidates: list[CandidateResponse]
    log_weights: list[float] = Field(..., description="Log-weights for each particle")
    normalized_weights: list[float] = Field(..., description="Normalized weights (sum to 1)")
    steps_used: list[int] = Field(..., description="Number of steps used per particle")


class ParticleGibbsTrace(BaseModel):
    """Trace data for Particle Gibbs algorithm."""
    algorithm: Literal["particle_gibbs"] = "particle_gibbs"
    num_iterations: int = Field(..., description="Number of Gibbs iterations")
    iterations: list[ParticleFilteringTrace] = Field(..., description="Trace per iteration")


class ResultDetail(BaseModel):
    """Details of a single result (baseline or ITS)."""
    answer: str = Field(..., description="The final answer text")
    latency_ms: int = Field(..., description="Latency in milliseconds")
    log_preview: str = Field(default="", description="Placeholder for step logs (future)")
    model_size: Optional[str] = Field(default=None, description="Model size (e.g., 'Large', 'Small')")
    cost_usd: Optional[float] = Field(default=None, description="Cost in USD")
    input_tokens: Optional[int] = Field(default=None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(default=None, description="Number of output tokens")
    trace: Optional[dict] = Field(default=None, description="Algorithm trace data for visualization")


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
