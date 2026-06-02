"""Backward-compatible re-exports — prefer paneer.stream."""

from .stream import (
    AssistantEvent,
    StreamEvent,
    StreamEventKind,
    StreamEventProcessor,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    extract_assistant_text,
    extract_thinking,
    parse_langgraph_chunk,
)

__all__ = [
    "AssistantEvent",
    "StreamEvent",
    "StreamEventKind",
    "StreamEventProcessor",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "extract_assistant_text",
    "extract_thinking",
    "parse_langgraph_chunk",
]
