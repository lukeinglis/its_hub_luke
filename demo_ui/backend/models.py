"""
Pydantic models for request/response validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request to compare baseline vs ITS."""
    question: str = Field(..., min_length=1, description="The question to answer")
    model_id: str = Field(..., description="Model identifier from the registry")
    algorithm: Literal["best_of_n", "self_consistency"] = Field(
        ..., description="ITS algorithm to use"
    )
    budget: int = Field(..., ge=1, le=32, description="Computation budget (1-32)")


class ResultDetail(BaseModel):
    """Details of a single result (baseline or ITS)."""
    answer: str = Field(..., description="The final answer text")
    latency_ms: int = Field(..., description="Latency in milliseconds")
    log_preview: str = Field(default="", description="Placeholder for step logs (future)")


class CompareResponse(BaseModel):
    """Response from comparison endpoint."""
    baseline: ResultDetail
    its: ResultDetail
    meta: dict = Field(..., description="Metadata about the comparison run")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
