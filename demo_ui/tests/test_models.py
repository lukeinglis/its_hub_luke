"""Tests for Pydantic models in backend/models.py."""

import pytest
from pydantic import ValidationError

from backend.models import (
    BestOfNTrace,
    CandidateResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ResultDetail,
    SelfConsistencyTrace,
    ToolCall,
    ToolVotingTrace,
)


class TestCompareRequest:
    """Tests for CompareRequest validation."""

    def test_valid_minimal_request(self):
        req = CompareRequest(
            question="What is 2+2?",
            model_id="gpt-4o-mini",
            algorithm="self_consistency",
            budget=6,
        )
        assert req.question == "What is 2+2?"
        assert req.budget == 6
        assert req.use_case == "improve_model"  # default

    def test_valid_full_request(self):
        req = CompareRequest(
            question="What tools to use?",
            model_id="gpt-4o-mini",
            algorithm="best_of_n",
            budget=8,
            use_case="tool_consensus",
            enable_tools=True,
            tool_vote="tool_name",
            question_type="tool_calling",
        )
        assert req.enable_tools is True
        assert req.tool_vote == "tool_name"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="",
                model_id="gpt-4o-mini",
                algorithm="self_consistency",
                budget=6,
            )

    def test_budget_too_low(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="gpt-4o-mini",
                algorithm="self_consistency",
                budget=0,
            )

    def test_budget_too_high(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="gpt-4o-mini",
                algorithm="self_consistency",
                budget=33,
            )

    def test_budget_boundaries(self):
        for valid_budget in [1, 16, 32]:
            req = CompareRequest(
                question="Test",
                model_id="m",
                algorithm="self_consistency",
                budget=valid_budget,
            )
            assert req.budget == valid_budget

    def test_invalid_algorithm_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="m",
                algorithm="invalid_algo",
                budget=6,
            )

    def test_all_valid_algorithms(self):
        valid_algos = [
            "best_of_n",
            "self_consistency",
            "beam_search",
            "particle_filtering",
            "entropic_particle_filtering",
            "particle_gibbs",
        ]
        for algo in valid_algos:
            req = CompareRequest(
                question="Test", model_id="m", algorithm=algo, budget=6
            )
            assert req.algorithm == algo

    def test_invalid_use_case_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="m",
                algorithm="self_consistency",
                budget=6,
                use_case="invalid_case",
            )

    def test_invalid_tool_vote_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="m",
                algorithm="self_consistency",
                budget=6,
                tool_vote="invalid_vote",
            )

    def test_invalid_question_type_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(
                question="Test",
                model_id="m",
                algorithm="self_consistency",
                budget=6,
                question_type="invalid_type",
            )


class TestToolCall:
    """Tests for ToolCall model."""

    def test_valid_tool_call(self):
        tc = ToolCall(name="calculate", arguments={"expression": "2+2"})
        assert tc.name == "calculate"
        assert tc.result is None  # default

    def test_tool_call_with_result(self):
        tc = ToolCall(
            name="web_search",
            arguments={"query": "weather"},
            result="Sunny, 72F",
        )
        assert tc.result == "Sunny, 72F"


class TestCandidateResponse:
    """Tests for CandidateResponse model."""

    def test_valid_candidate(self):
        c = CandidateResponse(index=0, content="The answer is 4")
        assert c.is_selected is False  # default
        assert c.tool_calls is None  # default

    def test_selected_candidate(self):
        c = CandidateResponse(index=1, content="42", is_selected=True)
        assert c.is_selected is True


class TestTraceModels:
    """Tests for algorithm trace models."""

    def _make_candidates(self, n=3):
        return [
            CandidateResponse(index=i, content=f"Answer {i}", is_selected=(i == 0))
            for i in range(n)
        ]

    def test_self_consistency_trace(self):
        trace = SelfConsistencyTrace(
            candidates=self._make_candidates(),
            vote_counts={"4": 2, "5": 1},
            total_votes=3,
        )
        assert trace.algorithm == "self_consistency"
        assert trace.total_votes == 3
        assert trace.tool_voting is None

    def test_self_consistency_with_tool_voting(self):
        tv = ToolVotingTrace(
            tool_vote_type="tool_name",
            tool_counts={"calculate": 2, "web_search": 1},
            winning_tool="calculate",
            total_tool_calls=3,
        )
        trace = SelfConsistencyTrace(
            candidates=self._make_candidates(),
            vote_counts={"4": 3},
            total_votes=3,
            tool_voting=tv,
        )
        assert trace.tool_voting.winning_tool == "calculate"

    def test_best_of_n_trace(self):
        trace = BestOfNTrace(
            candidates=self._make_candidates(),
            scores=[0.8, 0.6, 0.9],
            max_score=0.9,
            min_score=0.6,
        )
        assert trace.algorithm == "best_of_n"
        assert trace.max_score == 0.9


class TestResultDetail:
    """Tests for ResultDetail model."""

    def test_minimal_result(self):
        r = ResultDetail(answer="42", latency_ms=1500)
        assert r.cost_usd is None
        assert r.trace is None

    def test_full_result(self):
        r = ResultDetail(
            answer="The answer is 4",
            latency_ms=2000,
            model_size="Small",
            cost_usd=0.0015,
            input_tokens=100,
            output_tokens=50,
        )
        assert r.cost_usd == 0.0015


class TestCompareResponse:
    """Tests for CompareResponse model."""

    def test_valid_response(self):
        baseline = ResultDetail(answer="3", latency_ms=500)
        its = ResultDetail(answer="4", latency_ms=2000)
        resp = CompareResponse(
            baseline=baseline,
            its=its,
            meta={"run_id": "test-123", "algorithm": "self_consistency"},
        )
        assert resp.small_baseline is None  # default

    def test_match_frontier_response(self):
        baseline = ResultDetail(answer="3", latency_ms=500)
        its = ResultDetail(answer="4", latency_ms=2000)
        small = ResultDetail(answer="2", latency_ms=300)
        resp = CompareResponse(
            baseline=baseline,
            its=its,
            meta={"use_case": "match_frontier"},
            small_baseline=small,
        )
        assert resp.small_baseline is not None


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health(self):
        h = HealthResponse(status="ok", message="Service is healthy")
        assert h.status == "ok"
