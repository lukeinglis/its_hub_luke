"""Tests for tool call handling in scaling algorithms."""

import pytest

from its_hub.algorithms.self_consistency import SelfConsistency, SelfConsistencyResult
from its_hub.types import ChatMessage, ChatMessages
from tests.mocks.language_models import SimpleMockLanguageModel


class ToolCallMockLanguageModel(SimpleMockLanguageModel):
    """Mock language model that returns responses with tool calls."""
    
    def __init__(self):
        super().__init__([])  # No predefined responses
        self.call_count = 0
        
    def generate(self, messages, **kwargs):
        if isinstance(messages[0], list):
            # Batch generation - return responses with tool calls and content
            responses = []
            for i in range(len(messages)):
                if i % 2 == 0:
                    # Response with tool calls and content
                    responses.append({
                        "role": "assistant",
                        "content": "I need to calculate this. The answer is 42.",
                        "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "calculate"}}]
                    })
                else:
                    # Response with only content
                    responses.append({
                        "role": "assistant", 
                        "content": "The answer is 42."
                    })
            return responses
        else:
            # Single generation
            return {
                "role": "assistant",
                "content": "The answer is 42."
            }


class TestToolCallHandling:
    """Test tool call handling across scaling algorithms."""
    
    def test_self_consistency_with_tool_calls(self):
        """Test that self-consistency handles tool calls without breaking."""
        mock_lm = ToolCallMockLanguageModel()
        
        # Simple projection function that extracts numbers
        def extract_number(text):
            import re
            numbers = re.findall(r'\d+', text)
            return numbers[0] if numbers else ""
        
        sc = SelfConsistency(extract_number)
        
        # This should not throw a NotImplementedError and should handle tool calls gracefully
        result = sc.infer(mock_lm, "What is 6 * 7?", budget=4, return_response_only=False)
        
        assert isinstance(result, SelfConsistencyResult)
        # Should vote on content only, ignoring tool calls
        assert result.the_one in ["I need to calculate this. The answer is 42.", "The answer is 42."]
        assert len(result.responses) == 4
        
        # Verify that responses contain the extracted content (not full message dicts)
        for response in result.responses:
            assert isinstance(response, str)
            assert "42" in response
    
    def test_self_consistency_tool_calls_content_voting(self):
        """Test that self-consistency votes on content, not tool calls."""
        mock_lm = ToolCallMockLanguageModel()
        
        # Identity projection function - just returns the content as-is
        def identity_projection(text):
            return text
        
        sc = SelfConsistency(identity_projection)
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)
        
        # Should have 2 responses of each type
        response_counts = {}
        for response in result.responses:
            response_counts[response] = response_counts.get(response, 0) + 1
        
        # Both response types should appear
        assert "I need to calculate this. The answer is 42." in response_counts
        assert "The answer is 42." in response_counts
        
        # The selected response should be one of the content strings
        assert result.the_one in ["I need to calculate this. The answer is 42.", "The answer is 42."]
    
    def test_self_consistency_with_empty_content_and_tool_calls(self):
        """Test self-consistency when responses have tool calls but empty content."""
        class EmptyContentToolCallMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])
                
            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    # Return responses with tool calls but empty/no content
                    responses = []
                    for i in range(len(messages)):
                        responses.append({
                            "role": "assistant",
                            "content": "",  # Empty content
                            "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "search"}}]
                        })
                    return responses
                else:
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_single", "type": "function", "function": {"name": "search"}}]
                    }
        
        mock_lm = EmptyContentToolCallMock()
        
        def identity_projection(text):
            return text
        
        sc = SelfConsistency(identity_projection)
        result = sc.infer(mock_lm, "Search for information", budget=3, return_response_only=False)
        
        # All responses should be empty strings (content extracted from tool call responses)
        assert all(response == "" for response in result.responses)
        assert result.the_one == ""
        assert len(result.responses) == 3


# TODO: Future work - implement tool call voting capability
# These tests will need to be updated when tool call voting is implemented
class TestFutureToolCallVoting:
    """Placeholder tests for future tool call voting functionality."""
    
    @pytest.mark.skip(reason="Tool call voting not yet implemented")
    def test_tool_call_consistency_voting(self):
        """Test voting on tool calls themselves, not just content."""
        # TODO: Implement tool call voting in SelfConsistency algorithm
        # This should vote on the most common tool calls, not just content
        pass
    
    @pytest.mark.skip(reason="Tool call voting not yet implemented") 
    def test_mixed_tool_call_and_content_voting(self):
        """Test voting that considers both tool calls and content."""
        # TODO: Implement hybrid voting that considers both tool calls and content
        pass
    
    @pytest.mark.skip(reason="Tool call voting not yet implemented")
    def test_tool_call_parameter_consistency(self):
        """Test consistency voting on tool call parameters."""
        # TODO: Implement voting on tool call parameters for consistency
        pass