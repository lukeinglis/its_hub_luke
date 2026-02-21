"""
Vertex AI Language Model wrappers for Claude and Gemini models.

Uses the official google-cloud-aiplatform and anthropic[vertex] packages.
"""

import asyncio
import logging
from typing import List, Dict

from anthropic import AnthropicVertex
from vertexai.generative_models import GenerativeModel
import vertexai
from its_hub.base import AbstractLanguageModel
from its_hub.types import ChatMessage

logger = logging.getLogger(__name__)


class VertexAIClaudeModel(AbstractLanguageModel):
    """
    Wrapper for Claude models on Vertex AI using the Anthropic Vertex SDK.

    Implements AbstractLanguageModel interface for compatibility with its_hub.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ):
        """
        Initialize Vertex AI Claude model.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location (e.g., 'us-central1')
            model_name: Model name (e.g., 'claude-3-5-sonnet@20240620')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize Anthropic Vertex client
        self.client = AnthropicVertex(
            project_id=project_id,
            region=location,
        )

        logger.info(
            f"Initialized VertexAI Claude model: {model_name} "
            f"(project: {project_id}, location: {location})"
        )

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage list to Anthropic API format."""
        anthropic_messages = []

        for msg in messages:
            role = msg.role
            # Anthropic uses 'user' and 'assistant', not 'system'
            if role == "system":
                # System messages need special handling - prepend to first user message
                # For now, we'll convert to user message
                role = "user"

            anthropic_messages.append({
                "role": role,
                "content": msg.extract_text_content(),
            })

        return anthropic_messages

    async def agenerate(
        self,
        messages_or_messages_lst: List[ChatMessage] | List[List[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | List[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: List[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> Dict | List[Dict]:
        """Generate response(s) asynchronously."""
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )

        # Prepare parameters
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        # Handle list of temperatures
        if isinstance(temp, list):
            temps = temp
        else:
            temps = [temp] * len(messages_lst)

        async def generate_single(messages: List[ChatMessage], temp: float) -> Dict:
            """Generate a single response."""
            anthropic_messages = self._convert_messages(messages)

            # Run in thread pool to avoid blocking
            def sync_call():
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tok,
                    temperature=temp,
                    messages=anthropic_messages,
                    stop_sequences=[stop] if stop else None,
                )
                return response

            # Execute synchronous call in thread pool
            response = await asyncio.to_thread(sync_call)

            # Convert to OpenAI-compatible format
            return {
                "role": "assistant",
                "content": response.content[0].text,
            }

        # Generate all responses in parallel
        responses = await asyncio.gather(
            *(generate_single(msgs, temp) for msgs, temp in zip(messages_lst, temps))
        )

        return responses[0] if is_single else responses

    def generate(
        self,
        messages_or_messages_lst: List[ChatMessage] | List[List[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | List[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: List[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> Dict | List[Dict]:
        """Generate response(s) synchronously."""
        return asyncio.run(
            self.agenerate(
                messages_or_messages_lst,
                stop,
                max_tokens,
                temperature,
                include_stop_str_in_output,
                tools,
                tool_choice,
            )
        )

    def evaluate(self, prompt: str, generation: str) -> List[float]:
        """Evaluate method (not implemented for Claude)."""
        raise NotImplementedError("Evaluate method not implemented for Vertex AI Claude")


class VertexAIGeminiModel(AbstractLanguageModel):
    """
    Wrapper for Gemini models on Vertex AI.

    Implements AbstractLanguageModel interface for compatibility with its_hub.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ):
        """
        Initialize Vertex AI Gemini model.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location (e.g., 'us-central1')
            model_name: Model name (e.g., 'gemini-1.5-pro-002')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        # Initialize Gemini model
        self.model = GenerativeModel(model_name)

        logger.info(
            f"Initialized VertexAI Gemini model: {model_name} "
            f"(project: {project_id}, location: {location})"
        )

    def _convert_messages(self, messages: List[ChatMessage]) -> str:
        """Convert ChatMessage list to prompt string for Gemini."""
        # Gemini uses a simpler format - we'll concatenate messages
        prompt_parts = []
        for msg in messages:
            role = msg.role
            content = msg.extract_text_content()
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)

        return "\n\n".join(prompt_parts)

    async def agenerate(
        self,
        messages_or_messages_lst: List[ChatMessage] | List[List[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | List[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: List[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> Dict | List[Dict]:
        """Generate response(s) asynchronously."""
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )

        # Prepare parameters
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        # Handle list of temperatures
        if isinstance(temp, list):
            temps = temp
        else:
            temps = [temp] * len(messages_lst)

        async def generate_single(messages: List[ChatMessage], temp: float) -> Dict:
            """Generate a single response."""
            prompt = self._convert_messages(messages)

            # Run in thread pool to avoid blocking
            def sync_call():
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tok,
                        "temperature": temp,
                        "stop_sequences": [stop] if stop else None,
                    }
                )
                return response

            # Execute synchronous call in thread pool
            response = await asyncio.to_thread(sync_call)

            # Convert to OpenAI-compatible format
            return {
                "role": "assistant",
                "content": response.text,
            }

        # Generate all responses in parallel
        responses = await asyncio.gather(
            *(generate_single(msgs, temp) for msgs, temp in zip(messages_lst, temps))
        )

        return responses[0] if is_single else responses

    def generate(
        self,
        messages_or_messages_lst: List[ChatMessage] | List[List[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | List[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: List[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> Dict | List[Dict]:
        """Generate response(s) synchronously."""
        return asyncio.run(
            self.agenerate(
                messages_or_messages_lst,
                stop,
                max_tokens,
                temperature,
                include_stop_str_in_output,
                tools,
                tool_choice,
            )
        )

    def evaluate(self, prompt: str, generation: str) -> List[float]:
        """Evaluate method (not implemented for Gemini)."""
        raise NotImplementedError("Evaluate method not implemented for Vertex AI Gemini")
