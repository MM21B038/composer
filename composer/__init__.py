"""Composer — thread-based agents with typed streaming."""

__version__ = "0.1.0"

from .agent import (
    Agent,
    MCPClient,
    MCPPromptInfo,
    MCPResourceInfo,
    StreamEvent,
    StreamEventKind,
    StreamEventProcessor,
    ThinkingEvent,
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolCall,
    ToolResult,
    combine_tools,
    extract_thinking,
    extract_assistant_text,
)
from .tools import run_tool_call
from .vector import Vector
from .image import ImageAttach
from .thread import (
    Thread,
    HumanMessage,
    ImageMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    CompressedMessage,
    Message,
    EncoderType,
    TokenCalculator,
)
from .tool_hide import (
    ToolResultHideRule,
    HideMode,
    get_original_content,
    is_hidden_for_model,
    message_matches_rule,
    resolve_full_tool_name,
    restore_hidden_tool_messages,
)

__all__ = [
    "Agent",
    "Vector",
    "MCPClient",
    "MCPPromptInfo",
    "MCPResourceInfo",
    "Thread",
    "HumanMessage",
    "ImageMessage",
    "ImageAttach",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "CompressedMessage",
    "Message",
    "EncoderType",
    "TokenCalculator",
    "HideMode",
    "ToolResultHideRule",
    "get_original_content",
    "is_hidden_for_model",
    "message_matches_rule",
    "resolve_full_tool_name",
    "restore_hidden_tool_messages",
    "ThinkingEvent",
    "AssistantEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StreamEvent",
    "StreamEventKind",
    "StreamEventProcessor",
    "ToolCall",
    "ToolResult",
    "combine_tools",
    "run_tool_call",
    "extract_thinking",
    "extract_assistant_text",
    "ChatProject",
    "ChatSession",
]

_LAZY_IMPORTS = {"ChatProject", "ChatSession"}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        from .persistence import ChatProject, ChatSession

        return {"ChatProject": ChatProject, "ChatSession": ChatSession}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
