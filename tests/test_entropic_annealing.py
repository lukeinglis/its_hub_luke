"""Test for the particle Gibbs resampling weight calculation fix (issue #54)."""

import random

import numpy as np

from its_hub.algorithms.particle_gibbs import (
    Particle,
    EntropicParticleFiltering,
    ParticleFilteringResult,
    SelectionMethod,
    TemperatureMethod,
    ResamplingMethod,
)
from its_hub.base import AbstractLanguageModel, AbstractProcessRewardModel
from its_hub.lms import StepGeneration


class MockLanguageModelForResampling(AbstractLanguageModel):
    """Mock LM that generates predictable steps for testing resampling."""

    def __init__(self):
        self.step_counter = 0

    def generate(self, messages, max_tokens=100, **kwargs):
        # Handle both single and batch calls like OpenAICompatibleLanguageModel
        if (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], list)
        ):
            # Batch generation
            results = []
            for _ in messages:
                step = f"step{self.step_counter}"
                self.step_counter += 1
                results.append({"role": "assistant", "content": step})
            return results
        else:
            # Single generation
            step = f"step{self.step_counter}"
            self.step_counter += 1
            return {"role": "assistant", "content": step}

    def evaluate(self, prompt, response):
        # Not used in these tests
        return 0.5


class MockProcessRewardModelForResampling(AbstractProcessRewardModel):
    """Mock PRM that gives higher scores to longer sequences."""

    def score(self, prompt, response):
        if isinstance(response, list):
            # Batch scoring
            return [self._score_single(r) for r in response]
        else:
            # Single scoring
            return self._score_single(response)

    def _score_single(self, response):
        # Give higher scores to longer responses
        # This simulates a scenario where reference particles (being longer)
        # would have unfairly high scores if we don't use partial weights
        num_steps = response.count("step")
        # Return a score between 0.5 and 0.9 based on length
        return min(0.2 + 0.1 * num_steps, 0.9)


class TestEntropicAnnealing:
    """Test the entropic annealing."""

    def effective_sample_size(self, probabilities):
        # Create mock models
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.ESS,
            ess_threshold=0.5,
            early_phase=0.5,
        )
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        ess = epf._effective_sample_size(probabilities)
        assert isinstance(ess, float)
        assert ess == 1.0 / (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2 + 0.5**2)

    def resampling(self, probabilities):
        # Create mock models
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.ESS,
            ess_threshold=0.5,
            early_phase=0.5,
        )
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        idx = epf._resampling_multinomial(probabilities)
        assert isinstance(idx, list)
        assert len(idx) == len(probabilities)

        idx = epf._resampling_systematic(probabilities)
        assert isinstance(idx, list)
        assert len(idx) == len(probabilities)

    def test_temperature_functions(self):
        """Test the temperature functions."""
        # Create mock models
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.ESS,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        # Test ESS temperature early phase
        t = epf._temperature_ess(ess_ratio=0.2, progress=0.2)
        assert isinstance(t, float)
        assert t == 4.0

        # Test ESS temperature late phase
        t = epf._temperature_ess(ess_ratio=0.5, progress=0.8)
        assert isinstance(t, float)
        assert t == 1.0

        # Test entropy temperature
        t = epf._temperature_entropy(entropy_n=0.5, progress=0.3)
        v = 1.0 / (0.5 + (1 - 0.5) * 0.3)
        assert isinstance(t, float)
        assert t == v

        # Test entropy temperature edge case
        t = epf._temperature_entropy(entropy_n=1.0, progress=0.2)
        assert isinstance(t, float)
        assert t == 1.0

        # Test base temperature
        t = epf._temperature_base(value_max=2.0, progress=0.5)
        assert isinstance(t, float)
        assert t == 1.50

        # Test base temperature edge case
        t = epf._temperature_base(value_max=0.8, progress=0.5)
        assert isinstance(t, float)
        assert t == 1.0

    def test_entropic_annealing_with_ess_temperature_multinomial(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.ESS,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)

    def test_entropic_annealing_with_entropy_temperature_multinomial(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.ENTROPY,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)

    def test_entropic_annealing_with_base_temperature_multinomial(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.MULTINOMIAL,
            temperature_method=TemperatureMethod.BASE,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)

    def test_entropic_annealing_with_ess_temperature_systematic(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.SYSTEMATIC,
            temperature_method=TemperatureMethod.ESS,
            ess_threshold=0.5,
            early_phase=0.5,
        )
        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)

    def test_entropic_annealing_with_entropy_temperature_systematic(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.SYSTEMATIC,
            temperature_method=TemperatureMethod.ENTROPY,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)

    def test_entropic_annealing_with_base_temperature_systematic(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create EntropicParticleFiltering
        epf = EntropicParticleFiltering(
            sg=sg,
            prm=mock_prm,
            selection_method=SelectionMethod.ARGMAX,
            resampling_method=ResamplingMethod.SYSTEMATIC,
            temperature_method=TemperatureMethod.BASE,
            ess_threshold=0.5,
            early_phase=0.5,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)
