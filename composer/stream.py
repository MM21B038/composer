from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages.ai import AIMessageChunk

from .tools import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Typed stream events (like HumanMessage / AIMessage / ToolMessage)
# ---------------------------------------------------------------------------


@dataclass
class ThinkingEvent:
    text: str
    message: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> Literal["thinking"]:
        return "thinking"


@dataclass
class AssistantEvent:
    text: str
    message: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> Literal["assistant"]:
        return "assistant"


@dataclass
class ToolCallEvent:
    call: ToolCall
    message: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> Literal["tool_call"]:
        return "tool_call"

    def __repr__(self) -> str:
        return f"ToolCallEvent(call={self.call!r})"


@dataclass
class ToolResultEvent:
    result: ToolResult
    message: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> Literal["tool_result"]:
        return "tool_result"

    @property
    def text(self) -> str:
        return self.result.content

    def __repr__(self) -> str:
        preview = self.result.content[:60]
        if len(self.result.content) > 60:
            preview += "..."
        return f"ToolResultEvent(result={preview!r})"


StreamEvent = Union[ThinkingEvent, AssistantEvent, ToolCallEvent, ToolResultEvent]
StreamEventKind = Literal["thinking", "assistant", "tool_call", "tool_result"]

_KNOWN_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call"}
)
_RESPONSE_METADATA_STRING_FIELDS = frozenset(
    {"finish_reason", "model_name", "model", "id", "system_fingerprint"}
)


def dedupe_repeated_string(value: str) -> str:
    """Undo langchain merge_dicts concatenation when chunks repeat the same value."""
    if not value:
        return value
    length = len(value)
    for size in range(1, length // 2 + 1):
        if length % size != 0:
            continue
        base = value[:size]
        if base * (length // size) == value:
            return base
    return value


def normalize_finish_reason(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value in _KNOWN_FINISH_REASONS:
        return value
    deduped = dedupe_repeated_string(value)
    if deduped in _KNOWN_FINISH_REASONS:
        return deduped
    for reason in sorted(_KNOWN_FINISH_REASONS, key=len, reverse=True):
        if deduped.endswith(reason):
            return reason
    return deduped


def normalize_response_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fix response_metadata corrupted by AIMessageChunk accumulation."""
    if not metadata:
        return {}
    out = dict(metadata)
    for key in _RESPONSE_METADATA_STRING_FIELDS:
        val = out.get(key)
        if not isinstance(val, str):
            continue
        if key == "finish_reason":
            out[key] = normalize_finish_reason(val)
        else:
            out[key] = dedupe_repeated_string(val)
    return out


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class StreamEventProcessor:
    """Accumulates raw LLM chunks into typed stream events."""

    def __init__(self, *, input_message_count: int = 0) -> None:
        self._values_message_count = input_message_count
        self._ai_accum: Optional[AIMessageChunk] = None
        self._thinking_len = 0
        self._assistant_len = 0
        self._emitted_tool_call_ids: set[str] = set()

    def process_message_chunk(
        self,
        message: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[StreamEvent]:
        if getattr(message, "type", None) == "tool":
            return []

        meta = metadata or {}
        events: List[StreamEvent] = []

        if isinstance(message, AIMessageChunk):
            self._ai_accum = (
                message if self._ai_accum is None else self._ai_accum + message
            )
            message = self._ai_accum

        events.extend(self._thinking_delta_events(message, meta))
        events.extend(self._assistant_delta_events(message, meta))

        finish_reason = normalize_finish_reason(
            (getattr(message, "response_metadata", None) or {}).get("finish_reason")
        )
        if finish_reason == "tool_calls":
            events.extend(self._tool_call_events_from_message(message, meta))

        return events

    def process_values_update(self, data: Dict[str, Any]) -> List[StreamEvent]:
        messages = data.get("messages", [])
        if not messages:
            return []

        new_messages = messages[self._values_message_count :]
        self._values_message_count = len(messages)
        events: List[StreamEvent] = []

        for message in new_messages:
            msg_type = getattr(message, "type", None)

            if msg_type == "ai":
                events.extend(self._tool_call_events_from_message(message, {}))
                events.extend(self._thinking_from_complete_message(message))

            elif msg_type == "tool":
                events.append(
                    ToolResultEvent(
                        result=ToolResult.from_message(message),
                        message=message,
                    )
                )
                self._reset_turn()

        return events

    def _reset_turn(self) -> None:
        self._ai_accum = None
        self._thinking_len = 0
        self._assistant_len = 0

    def _thinking_delta_events(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        full = _extract_streaming_thinking(message)
        if len(full) <= self._thinking_len:
            return []
        delta = full[self._thinking_len :]
        self._thinking_len = len(full)
        return [ThinkingEvent(text=delta, message=message, metadata=metadata)]

    def _assistant_delta_events(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        full = extract_assistant_text(message)
        if len(full) <= self._assistant_len:
            return []
        delta = full[self._assistant_len :]
        self._assistant_len = len(full)
        return [AssistantEvent(text=delta, message=message, metadata=metadata)]

    def _thinking_from_complete_message(self, message: Any) -> List[StreamEvent]:
        return self._thinking_delta_events(message, {})

    def _tool_call_events_from_message(
        self,
        message: Any,
        metadata: Dict[str, Any],
    ) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            call = ToolCall.from_langchain(tool_call)
            if not call.id or call.id in self._emitted_tool_call_ids:
                continue
            self._emitted_tool_call_ids.add(call.id)
            events.append(
                ToolCallEvent(call=call, message=message, metadata=metadata)
            )
        return events


# ---------------------------------------------------------------------------
# Extractors (invoke + thread messages)
# ---------------------------------------------------------------------------


def extract_thinking(message: Any) -> str:
    return _extract_streaming_thinking(message)


def _extract_streaming_thinking(message: Any) -> str:
    """Single canonical thinking text for streaming deltas and invoke."""
    from_blocks = _extract_thinking_without_details(message)
    if from_blocks:
        return from_blocks
    return _extract_reasoning_details_text(message)


def _extract_reasoning_details_text(message: Any) -> str:
    parts = [
        part
        for detail in _iter_reasoning_details(message)
        if (part := _reasoning_detail_text(detail))
    ]
    return _join_thinking_parts(parts)


def extract_assistant_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and not _iter_content_blocks(message):
        return content

    parts: List[str] = []
    for block in _iter_content_blocks(message):
        if _block_type(block) == "text":
            part = _block_text(block, keys=("text", "content"))
            if part:
                parts.append(part)

    if parts:
        return "".join(parts)

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)

    return str(content) if content else ""


def parse_langgraph_chunk(chunk: Any) -> Optional[tuple[str, Any]]:
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]
    return None


def _join_thinking_parts(parts: List[str]) -> str:
    cleaned = [p for p in parts if p]
    if not cleaned:
        return ""
    if len(cleaned) > 1 and cleaned[-1].startswith(cleaned[0]):
        return cleaned[-1]
    return "".join(cleaned)


def _extract_thinking_without_details(message: Any) -> str:
    parts: List[str] = []

    reasoning_field = getattr(message, "reasoning", None)
    if reasoning_field:
        parts.append(
            reasoning_field
            if isinstance(reasoning_field, str)
            else str(reasoning_field)
        )

    block_parts: List[str] = []
    for block in _iter_content_blocks(message):
        if _block_type(block) == "reasoning":
            part = _block_text(block, keys=("reasoning", "text", "content"))
            if part:
                block_parts.append(part)
    if block_parts:
        parts.append(_join_thinking_parts(block_parts))

    kwargs = getattr(message, "additional_kwargs", None) or {}
    reasoning = (
        kwargs.get("reasoning_content")
        or kwargs.get("reasoning")
        or kwargs.get("thinking")
    )
    if reasoning:
        if isinstance(reasoning, str):
            parts.append(reasoning)
        elif isinstance(reasoning, dict):
            parts.append(reasoning.get("text") or reasoning.get("content") or "")

    return _join_thinking_parts(parts)


def _iter_content_blocks(message: Any) -> List[Any]:
    blocks = getattr(message, "content_blocks", None)
    if blocks:
        return list(blocks)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return content
    return []


def _block_type(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_text(block: Any, keys: tuple[str, ...]) -> str:
    if isinstance(block, dict):
        for key in keys:
            val = block.get(key)
            if val:
                return val if isinstance(val, str) else str(val)
        return ""
    for key in keys:
        val = getattr(block, key, None)
        if val:
            return val if isinstance(val, str) else str(val)
    return ""


def _iter_reasoning_details(message: Any) -> List[Any]:
    details: List[Any] = []
    message_details = getattr(message, "reasoning_details", None)
    if message_details:
        details.extend(message_details)

    kwargs = getattr(message, "additional_kwargs", None) or {}
    kw_details = kwargs.get("reasoning_details")
    if kw_details:
        details.extend(kw_details)

    response_meta = getattr(message, "response_metadata", None) or {}
    meta_details = response_meta.get("reasoning_details")
    if meta_details:
        details.extend(meta_details)

    return details


def _reasoning_detail_text(detail: Any) -> str:
    if not isinstance(detail, dict):
        return ""
    detail_type = detail.get("type") or ""
    if detail_type == "reasoning.text":
        return detail.get("text") or ""
    if detail_type == "reasoning.summary":
        return detail.get("summary") or ""
    return detail.get("text") or detail.get("summary") or ""
