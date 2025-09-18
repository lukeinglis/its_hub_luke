"""Type definitions for its_hub."""

from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class ChatMessage:
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatMessages:
    """Unified wrapper for handling both string prompts and conversation history."""

    def __init__(self, str_or_messages: str | list[ChatMessage]):
        self._str_or_messages = str_or_messages
        self._is_string = isinstance(str_or_messages, str)

    @classmethod
    def from_prompt_or_messages(cls, prompt_or_messages: str | list[ChatMessage] | "ChatMessages") -> "ChatMessages":
        """Create ChatMessages from various input formats."""
        if isinstance(prompt_or_messages, ChatMessages):
            return prompt_or_messages
        return cls(prompt_or_messages)

    def to_string(self) -> str:
        """Convert to string representation."""
        if self._is_string:
            return self._str_or_messages
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self._str_or_messages)

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert to list of ChatMessage objects."""
        if self._is_string:
            return [ChatMessage(role="user", content=self._str_or_messages)]
        return self._str_or_messages

    def to_batch(self, size: int) -> list[list[ChatMessage]]:
        """Create a batch of identical chat message lists for parallel generation."""
        chat_messages = self.to_chat_messages()
        return [chat_messages for _ in range(size)]

    @property
    def is_string(self) -> bool:
        """Check if the original input was a string."""
        return self._is_string
