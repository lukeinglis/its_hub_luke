"""Type definitions for its_hub."""

from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class Function:
    """Function definition for tool calls."""
    
    name: str
    description: str | None = None
    parameters: dict | None = None


@dataclass
class ToolCall:
    """A tool call made by the assistant."""
    
    id: str
    type: Literal["function"] = "function"
    function: Function | None = None


@dataclass
class ChatMessage:
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatMessages:
    """Unified wrapper for handling both string prompts and conversation history."""

    def __init__(self, str_or_messages: str | list[ChatMessage]):
        self._str_or_messages = str_or_messages
        self._is_string = isinstance(str_or_messages, str)

    def to_string(self) -> str:
        """Convert to string representation."""
        if self._is_string:
            return self._str_or_messages
        
        lines = []
        for msg in self._str_or_messages:
            if msg.role == "tool":
                # Tool messages: include tool_call_id context
                lines.append(f"tool[{msg.tool_call_id}]: {msg.content}")
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant with tool calls: show tool calls + content if any
                tool_call_strs = []
                for tc in msg.tool_calls:
                    if tc.function:
                        tool_call_strs.append(f"{tc.function.name}()")
                tool_calls_text = ", ".join(tool_call_strs)
                if msg.content:
                    lines.append(f"assistant: {msg.content} [calls: {tool_calls_text}]")
                else:
                    lines.append(f"assistant: [calls: {tool_calls_text}]")
            else:
                # Regular messages
                lines.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(lines)

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
