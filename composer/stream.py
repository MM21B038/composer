from __future__ import annotations

import re
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
_CHUNK_METADATA_DROP_FIELDS = frozenset(
    {"created_at", "created", "timestamp"}
)
_RESTREAM_PREFIX_MIN = 20
_STUB_STREAM_TEXT_MAX_LEN = 2
_DEDUPE_MIN_RETAINED_RATIO = 0.8
_CUMULATIVE_MERGE_RE = re.compile(r"^(\S{2,})\1")


def is_stub_stream_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or len(stripped) <= _STUB_STREAM_TEXT_MAX_LEN


def _shared_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _same_stream_turn_content(previous: str, incoming: str) -> bool:
    """Detect reformatted cumulative restreams of the same thinking turn."""
    if not previous or not incoming:
        return False
    if previous == incoming:
        return True
    # Normal incremental streaming extends or retracts the prior snapshot.
    if previous.startswith(incoming) or incoming.startswith(previous):
        return False
    prev_norm = _collapse_ws(previous)
    inc_norm = _collapse_ws(incoming)
    if prev_norm == inc_norm:
        return True
    if len(prev_norm) > 30 and len(inc_norm) > 30 and (
        prev_norm in inc_norm or inc_norm in prev_norm
    ):
        return True
    tail = min(len(prev_norm), len(inc_norm), 40)
    if tail and prev_norm[-tail:] == inc_norm[-tail:]:
        return _shared_prefix_len(prev_norm, inc_norm) >= _RESTREAM_PREFIX_MIN
    return False


def merge_stream_text_delta(previous: str, incoming: str) -> tuple[str, str]:
    """Merge streamed cumulative/incremental text; return (snapshot, delta)."""
    if not incoming:
        return previous, ""
    if not previous:
        return incoming, incoming
    if incoming == previous:
        return previous, ""
    if incoming.startswith(previous):
        return incoming, incoming[len(previous) :]
    if previous.startswith(incoming):
        return previous, ""

    if len(incoming) >= _RESTREAM_PREFIX_MIN and incoming in previous:
        return previous, ""

    max_overlap = min(len(previous), len(incoming))
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == incoming[:size]:
            delta = incoming[size:]
            return previous + delta, delta

    shared = _shared_prefix_len(previous, incoming)
    if shared >= _RESTREAM_PREFIX_MIN and len(incoming) > shared:
        previous_tail = previous[shared:]
        incoming_tail = incoming[shared:]
        if incoming_tail.startswith(previous_tail):
            delta = incoming_tail[len(previous_tail) :]
            return incoming, delta
        if previous_tail.startswith(incoming_tail):
            return incoming, ""
        if incoming.startswith(previous):
            return incoming, incoming[len(previous) :]
        delta = incoming[shared:]
        if delta:
            return incoming, delta
        if incoming not in previous:
            return previous + incoming, incoming
        return previous, ""

    if shared >= _RESTREAM_PREFIX_MIN and len(incoming) <= len(previous):
        return previous, ""

    return previous + incoming, incoming


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
            if size == 1 and base.isdigit():
                return value
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


def _is_assistant_stream_complete(message: Any) -> bool:
    """True when a stream chunk marks the end of an assistant (non-tool) turn."""
    if getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None):
        return False
    metadata = getattr(message, "response_metadata", None) or {}
    finish_reason = normalize_finish_reason(metadata.get("finish_reason"))
    if finish_reason in ("stop", "length"):
        return True
    chunk_position = getattr(message, "chunk_position", None) or metadata.get(
        "chunk_position"
    )
    if chunk_position == "last":
        return True
    if metadata.get("status") == "completed":
        return True
    return False


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


def sanitize_chunk_for_merge(chunk: AIMessageChunk) -> AIMessageChunk:
    """Strip response fields that break LangChain AIMessageChunk merging."""
    metadata = dict(getattr(chunk, "response_metadata", None) or {})
    for key in list(metadata):
        val = metadata[key]
        if key in _CHUNK_METADATA_DROP_FIELDS or isinstance(val, float):
            metadata.pop(key)
    sanitized = normalize_response_metadata(metadata)
    if sanitized == getattr(chunk, "response_metadata", None):
        return chunk
    return chunk.model_copy(update={"response_metadata": sanitized})


def _json_safe_value(value: Any) -> Any:
    """Recursively convert SDK/Pydantic objects to JSON-safe plain values."""
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe_value(model_dump(mode="json", warnings=False))
    return value


def message_model_dump(message: Any, *, mode: str = "json") -> Dict[str, Any]:
    """Serialize a LangChain message without SDK content-block serializer warnings."""
    updates: Dict[str, Any] = {}
    content = getattr(message, "content", None)
    if content is not None:
        safe_content = _json_safe_value(content)
        if safe_content != content:
            updates["content"] = safe_content
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        safe_kwargs = _json_safe_value(additional_kwargs)
        if safe_kwargs != additional_kwargs:
            updates["additional_kwargs"] = safe_kwargs
    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        safe_metadata = _json_safe_value(response_metadata)
        if safe_metadata != response_metadata:
            updates["response_metadata"] = safe_metadata
    if updates and hasattr(message, "model_copy"):
        message = message.model_copy(update=updates)

    dump = message.model_dump(mode=mode, warnings=False)
    if dump.get("response_metadata"):
        dump["response_metadata"] = normalize_response_metadata(
            dump["response_metadata"]
        )
    return dump


_MODEL_CHUNK_KIND = "model_chunk"
_BULKY_KWARG_KEYS = frozenset({"reasoning", "thinking", "reasoning_details"})


def compact_ai_message_dump(dump: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky reasoning blobs after content/reasoning_content promotion."""
    kwargs = dict(dump.get("additional_kwargs") or {})
    for key in _BULKY_KWARG_KEYS:
        kwargs.pop(key, None)
    if kwargs:
        dump["additional_kwargs"] = kwargs
    elif "additional_kwargs" in dump:
        dump["additional_kwargs"] = {}

    response_metadata = dump.get("response_metadata")
    if isinstance(response_metadata, dict) and "reasoning_details" in response_metadata:
        metadata = dict(response_metadata)
        metadata.pop("reasoning_details", None)
        dump["response_metadata"] = metadata
    return dump


def _message_needs_invoke_fallback(message: Any) -> bool:
    if getattr(message, "tool_calls", None):
        return False
    if extract_assistant_text(message):
        return False
    content = getattr(message, "content", None)
    return not (isinstance(content, str) and content.strip())


def normalize_ai_message_dump(message: Any, dump: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Responses API block content to standard string AIMessage content."""
    content = dump.get("content")
    kwargs = dict(dump.get("additional_kwargs") or {})
    assistant_text = extract_assistant_text(message)
    thinking_text = extract_thinking(message)

    if isinstance(content, list):
        has_response_blocks = any(
            isinstance(block, dict)
            and block.get("type") in ("reasoning", "text", "refusal")
            for block in content
        )
        if has_response_blocks:
            if thinking_text:
                kwargs["reasoning_content"] = thinking_text
            dump["content"] = assistant_text
            if kwargs:
                dump["additional_kwargs"] = kwargs
            return compact_ai_message_dump(dump)

    if assistant_text and not (isinstance(content, str) and content):
        dump["content"] = assistant_text
    if thinking_text:
        kwargs["reasoning_content"] = thinking_text
    if kwargs:
        dump["additional_kwargs"] = kwargs
    return compact_ai_message_dump(dump)


def parse_custom_model_chunk(data: Any) -> Optional[Any]:
    if isinstance(data, dict) and data.get("kind") == _MODEL_CHUNK_KIND:
        return data.get("chunk")
    return None


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class StreamEventProcessor:
    """Accumulates raw LLM chunks into typed stream events."""

    def __init__(self, *, input_message_count: int = 0) -> None:
        self._values_message_count = input_message_count
        self._values_initialized = False
        self._ai_accum: Optional[AIMessageChunk] = None
        self._tracked_ai_message_id: Optional[str] = None
        self._thinking_len = 0
        self._assistant_len = 0
        self._reasoning_snapshot = ""
        self._assistant_snapshot = ""
        self._pending_assistant_text = ""
        self._emitted_assistant_text = ""
        self._assistant_content_chunk_count = 0
        self._turn_thinking_buffer = ""
        self._emitted_tool_call_ids: set[str] = set()
        self._last_chunk_fingerprint: Optional[tuple[Any, ...]] = None

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
            message = sanitize_chunk_for_merge(message)
            if self._is_duplicate_chunk(message):
                return []
            self._update_tracked_message_id(getattr(message, "id", None))
            self._ai_accum = (
                message if self._ai_accum is None else self._ai_accum + message
            )
            events.extend(self._thinking_delta_from_chunk(message, meta))
            self._record_thinking_progress()
            if not self._is_tool_turn_message(message) and not self._is_tool_turn_message(
                self._ai_accum
            ):
                events.extend(self._assistant_delta_from_chunk(message, meta))
            elif self._is_tool_turn_message(message):
                self._clear_assistant_state()
        else:
            events.extend(self._thinking_delta_events(message, meta))
            self._record_thinking_progress(message)
            events.extend(self._assistant_delta_events(message, meta))

        finish_reason = normalize_finish_reason(
            (
                getattr(self._ai_accum or message, "response_metadata", None) or {}
            ).get("finish_reason")
        )
        if finish_reason == "tool_calls":
            events.extend(self._flush_unemitted_thinking_gap(message, meta))
            self._ai_accum = None

        return events

    def process_values_update(self, data: Dict[str, Any]) -> List[StreamEvent]:
        messages = data.get("messages", [])
        if not messages:
            return []

        if not self._values_initialized:
            self._values_initialized = True
            self._values_message_count = len(messages)
            return []

        new_messages = messages[self._values_message_count :]
        self._values_message_count = len(messages)
        events: List[StreamEvent] = []

        for message in new_messages:
            msg_type = getattr(message, "type", None)

            if msg_type == "ai":
                self._sync_complete_ai_message_state(message)
                is_tool_turn = self._is_tool_turn_message(message)
                if is_tool_turn:
                    events.extend(self.flush_turn_thinking(message))
                    events.extend(self._thinking_from_complete_message(message))
                    self._clear_assistant_state()
                else:
                    events.extend(self._thinking_from_complete_message(message))
                    events.extend(self._assistant_from_complete_message(message))
                events.extend(self._tool_call_events_from_message(message, {}))
                if is_tool_turn:
                    self._ai_accum = None
                    self._turn_thinking_buffer = ""
                    self._reasoning_snapshot = ""
                    self._last_chunk_fingerprint = None

            elif msg_type == "tool":
                events.append(
                    ToolResultEvent(
                        result=ToolResult.from_message(message),
                        message=message,
                    )
                )
                self._reset_turn()

        return events

    def flush_turn_thinking(self, message: Any = None) -> List[StreamEvent]:
        """Emit buffered thinking before a tool-call turn ends."""
        buffer = self._reasoning_snapshot or self._turn_thinking_buffer
        if not buffer:
            return []
        message_thinking = (
            _extract_streaming_thinking(message) if message is not None else ""
        )
        if message_thinking and len(buffer) <= len(message_thinking):
            if message_thinking.startswith(buffer):
                buffer = message_thinking
            elif buffer.startswith(message_thinking):
                pass
            else:
                return []
        delta = buffer[self._thinking_len :]
        if not delta:
            return []
        self._thinking_len = len(buffer)
        self._reasoning_snapshot = buffer
        self._turn_thinking_buffer = buffer
        return [ThinkingEvent(text=delta, message=message, metadata={})]

    def _flush_unemitted_thinking_gap(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        """Emit thinking held in the snapshot but not yet streamed before tool_calls."""
        if self._ai_accum is not None:
            incoming = self._streaming_thinking_incoming(message)
            if incoming:
                self._extend_reasoning_snapshot(incoming)
        gap = self._reasoning_snapshot[self._thinking_len :]
        if not gap:
            return []
        self._thinking_len = len(self._reasoning_snapshot)
        return [ThinkingEvent(text=gap, message=message, metadata=metadata)]

    def flush(self, message: Any) -> List[StreamEvent]:
        """Emit any remaining thinking/assistant deltas from the final message."""
        self._sync_complete_ai_message_state(message)
        message_content = getattr(message, "content", None)
        has_explicit_content = (
            isinstance(message_content, str) and message_content.strip()
        )
        if self._ai_accum is not None and not has_explicit_content:
            accum_text = extract_assistant_text(self._ai_accum)
            if accum_text:
                self._pending_assistant_text, _ = self._extend_text_snapshot(
                    self._pending_assistant_text, accum_text
                )
        events: List[StreamEvent] = []
        events.extend(self._thinking_from_complete_message(message))
        events.extend(self._assistant_from_complete_message(message))
        return events

    def peek_turn_snapshots(self) -> tuple[str, str]:
        assistant = self._pending_assistant_text or self._assistant_snapshot
        if not assistant and self._ai_accum is not None:
            assistant = extract_assistant_text(self._ai_accum)
        return self._reasoning_snapshot, assistant

    def merge_turn_accumulation(self, message: Any) -> None:
        """Capture thinking from streamed chunks before a values update."""
        if self._reasoning_snapshot:
            self._turn_thinking_buffer = self._reasoning_snapshot
            return
        if message is None:
            return
        self._record_thinking_progress(message)

    def _reset_turn(self) -> None:
        self._ai_accum = None
        self._tracked_ai_message_id = None
        self._reset_stream_text_state()
        self._turn_thinking_buffer = ""
        self._last_chunk_fingerprint = None

    def _reset_stream_text_state(self) -> None:
        self._thinking_len = 0
        self._assistant_len = 0
        self._reasoning_snapshot = ""
        self._assistant_snapshot = ""
        self._pending_assistant_text = ""
        self._emitted_assistant_text = ""
        self._assistant_content_chunk_count = 0

    def _update_tracked_message_id(self, msg_id: Any) -> None:
        if msg_id is None or msg_id == self._tracked_ai_message_id:
            return
        self._tracked_ai_message_id = msg_id
        if not self._reasoning_snapshot and not self._assistant_snapshot:
            self._reset_stream_text_state()

    def _sync_complete_ai_message_state(self, message: Any) -> None:
        self._update_tracked_message_id(getattr(message, "id", None))
        self._reconcile_stream_lengths(message)

    @staticmethod
    def _is_tool_turn_message(message: Any) -> bool:
        if getattr(message, "tool_calls", None) or getattr(
            message, "tool_call_chunks", None
        ):
            return True
        finish_reason = normalize_finish_reason(
            (getattr(message, "response_metadata", None) or {}).get("finish_reason")
        )
        return finish_reason == "tool_calls"

    def _clear_assistant_state(self) -> None:
        self._assistant_len = 0
        self._assistant_snapshot = ""
        self._pending_assistant_text = ""
        self._emitted_assistant_text = ""
        self._assistant_content_chunk_count = 0

    def _note_assistant_content_chunk(self) -> None:
        self._assistant_content_chunk_count += 1

    def _assistant_delta_after_emitted(self, full: str) -> str:
        """Return assistant text not yet yielded to the consumer."""
        if not full:
            return ""
        emitted = self._emitted_assistant_text
        if not emitted:
            return full
        if full.startswith(emitted):
            return full[len(emitted) :]
        if emitted.startswith(full):
            return ""
        if _same_stream_turn_content(emitted, full):
            return ""
        shared = _shared_prefix_len(emitted, full)
        if shared >= _RESTREAM_PREFIX_MIN:
            tail = min(len(emitted), len(full), 60)
            if tail and emitted[-tail:] == full[-tail:]:
                if len(full) > len(emitted) and full.startswith(emitted[:shared]):
                    return full[len(emitted) :]
                return ""
        _, delta = merge_stream_text_delta(emitted, full)
        if delta and (full.startswith(emitted) or emitted.startswith(full)):
            return delta
        return ""

    def _record_emitted_assistant_delta(
        self, delta: str, *, snapshot: str
    ) -> None:
        if not delta:
            return
        self._emitted_assistant_text += delta
        self._assistant_snapshot = snapshot
        self._assistant_len = len(snapshot)

    def _buffer_assistant_text(self, chunk: AIMessageChunk) -> None:
        incoming = extract_streaming_assistant_text(chunk)
        if not incoming:
            return
        self._pending_assistant_text, _ = self._extend_text_snapshot(
            self._pending_assistant_text, incoming
        )

    def _release_pending_assistant(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        pending = self._pending_assistant_text
        if not pending:
            return []
        delta = self._assistant_delta_after_emitted(pending)
        self._assistant_snapshot = pending
        self._assistant_len = len(pending)
        self._pending_assistant_text = ""
        if not delta:
            return []
        self._record_emitted_assistant_delta(delta, snapshot=pending)
        return [AssistantEvent(text=delta, message=message, metadata=metadata)]

    def _chunk_fingerprint(self, chunk: AIMessageChunk) -> tuple[Any, ...]:
        kwargs = getattr(chunk, "additional_kwargs", None) or {}
        content = chunk.content
        if not isinstance(content, str):
            content = repr(content)
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
        return (
            getattr(chunk, "id", None),
            kwargs.get("reasoning_content"),
            content,
            _extract_streaming_response_text(chunk),
            extract_streaming_assistant_text(chunk),
            tuple(repr(item) for item in tool_call_chunks),
        )

    def _is_duplicate_chunk(self, chunk: AIMessageChunk) -> bool:
        fingerprint = self._chunk_fingerprint(chunk)
        if fingerprint == self._last_chunk_fingerprint:
            return True
        self._last_chunk_fingerprint = fingerprint
        return False

    def _extend_text_snapshot(self, snapshot: str, incoming: str) -> tuple[str, str]:
        return merge_stream_text_delta(snapshot, incoming)

    def _extend_reasoning_snapshot(self, incoming: str) -> str:
        if incoming:
            incoming = _maybe_dedupe_stream_incoming(
                incoming, self._reasoning_snapshot
            )
        if incoming and self._reasoning_snapshot and _same_stream_turn_content(
            self._reasoning_snapshot, incoming
        ):
            if len(incoming) > len(self._reasoning_snapshot):
                self._reasoning_snapshot = incoming
            return ""
        self._reasoning_snapshot, delta = self._extend_text_snapshot(
            self._reasoning_snapshot, incoming
        )
        return delta

    def _extend_assistant_snapshot(self, incoming: str) -> str:
        self._assistant_snapshot, delta = self._extend_text_snapshot(
            self._assistant_snapshot, incoming
        )
        return delta

    def _streaming_thinking_incoming(self, chunk: AIMessageChunk) -> str:
        chunk_text = _extract_streaming_thinking(chunk)
        if chunk_text and self._reasoning_snapshot and _same_stream_turn_content(
            self._reasoning_snapshot, chunk_text
        ):
            return chunk_text
        if self._ai_accum is None:
            return chunk_text
        accum_text = _extract_streaming_thinking(self._ai_accum)
        if not chunk_text:
            return accum_text
        if not accum_text:
            return chunk_text
        if chunk_text.startswith(self._reasoning_snapshot):
            return chunk_text
        if self._reasoning_snapshot and accum_text.startswith(self._reasoning_snapshot):
            if len(accum_text) > len(chunk_text):
                return accum_text
        return accum_text if len(accum_text) >= len(chunk_text) else chunk_text

    def _thinking_delta_from_chunk(
        self, chunk: AIMessageChunk, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        incoming = self._streaming_thinking_incoming(chunk)
        delta = self._extend_reasoning_snapshot(incoming)
        if not delta:
            return []
        self._thinking_len = len(self._reasoning_snapshot)
        return [ThinkingEvent(text=delta, message=chunk, metadata=metadata)]

    def _can_stream_assistant_incrementally(self, chunk: AIMessageChunk) -> bool:
        """Allow live assistant tokens during reasoning turns; defer plain preambles."""
        if self._is_tool_turn_message(chunk) or self._is_tool_turn_message(
            self._ai_accum
        ):
            return False
        if self._reasoning_snapshot:
            return True
        if self._emitted_assistant_text:
            return True
        if self._assistant_content_chunk_count >= 2:
            return True
        if _is_assistant_stream_complete(chunk):
            return True
        if getattr(self._ai_accum, "tool_call_chunks", None):
            return False
        return False

    def _assistant_delta_from_chunk(
        self, chunk: AIMessageChunk, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        if self._is_tool_turn_message(chunk) or self._is_tool_turn_message(
            self._ai_accum
        ):
            self._clear_assistant_state()
            return []
        live_text = _extract_live_assistant_text(chunk)
        if live_text:
            self._note_assistant_content_chunk()
            self._extend_assistant_snapshot(live_text)
        self._pending_assistant_text = self._assistant_snapshot
        if not self._can_stream_assistant_incrementally(chunk):
            return []
        if len(self._assistant_snapshot) <= self._assistant_len:
            return []
        delta = self._assistant_snapshot[self._assistant_len :]
        self._record_emitted_assistant_delta(
            delta, snapshot=self._assistant_snapshot
        )
        return [AssistantEvent(text=delta, message=chunk, metadata=metadata)]

    def _reconcile_stream_lengths(self, message: Any) -> None:
        full_thinking = _extract_streaming_thinking(message)
        if (
            full_thinking
            and self._thinking_len > len(full_thinking)
            and not self._reasoning_snapshot
        ):
            self._thinking_len = 0
            self._reasoning_snapshot = ""
        full_assistant = extract_assistant_text(message)
        self._reconcile_assistant_snapshot(full_assistant)

    def _reconcile_assistant_snapshot(self, full: str) -> None:
        if not full:
            return
        if self._assistant_len > len(full):
            self._assistant_len = 0
            self._assistant_snapshot = ""
            return
        if self._assistant_snapshot and not full.startswith(self._assistant_snapshot):
            self._assistant_len = 0
            self._assistant_snapshot = ""

    def _thinking_delta_events(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        self._reconcile_stream_lengths(message)
        full = _extract_streaming_thinking(message)
        if len(full) <= self._thinking_len:
            return []
        delta = full[self._thinking_len :]
        self._thinking_len = len(full)
        self._reasoning_snapshot = full
        return [ThinkingEvent(text=delta, message=message, metadata=metadata)]

    def _assistant_delta_events(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        full = extract_streaming_assistant_text(message)
        self._reconcile_assistant_snapshot(full)
        delta = self._assistant_delta_after_emitted(full)
        if not delta:
            self._assistant_snapshot = full
            self._assistant_len = len(full)
            return []
        self._record_emitted_assistant_delta(delta, snapshot=full)
        return [AssistantEvent(text=delta, message=message, metadata=metadata)]

    def _record_thinking_progress(self, message: Any | None = None) -> None:
        if self._reasoning_snapshot:
            self._turn_thinking_buffer = self._reasoning_snapshot
            return
        if message is None:
            return
        full = _extract_streaming_thinking(message)
        if len(full) > len(self._turn_thinking_buffer):
            self._turn_thinking_buffer = full

    def _thinking_from_complete_message(self, message: Any) -> List[StreamEvent]:
        message_thinking = _extract_streaming_thinking(message)
        if self._reasoning_snapshot:
            if not message_thinking:
                self._thinking_len = len(self._reasoning_snapshot)
                return []
            if message_thinking.startswith(self._reasoning_snapshot):
                if len(message_thinking) > len(self._reasoning_snapshot):
                    delta = message_thinking[len(self._reasoning_snapshot) :]
                    self._reasoning_snapshot = message_thinking
                    self._thinking_len = len(message_thinking)
                    return [
                        ThinkingEvent(text=delta, message=message, metadata={})
                    ]
                self._thinking_len = len(self._reasoning_snapshot)
                return []
            if self._reasoning_snapshot.startswith(message_thinking):
                self._thinking_len = len(self._reasoning_snapshot)
                return []
            if message_thinking in self._reasoning_snapshot:
                self._thinking_len = len(self._reasoning_snapshot)
                return []

        events = self._thinking_delta_events(message, {})
        if events:
            return events
        if not self._turn_thinking_buffer:
            return []
        if len(message_thinking) >= len(self._turn_thinking_buffer):
            return []
        delta = self._turn_thinking_buffer[self._thinking_len :]
        if not delta:
            return []
        self._thinking_len = len(self._turn_thinking_buffer)
        return [ThinkingEvent(text=delta, message=message, metadata={})]

    def _assistant_from_complete_message(self, message: Any) -> List[StreamEvent]:
        return self._assistant_complete_delta_events(message, {})

    def _assistant_complete_delta_events(
        self, message: Any, metadata: Dict[str, Any]
    ) -> List[StreamEvent]:
        full = extract_assistant_text(message)
        if full:
            self._pending_assistant_text, _ = self._extend_text_snapshot(
                self._pending_assistant_text, full
            )
        return self._release_pending_assistant(message, metadata)

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


def _extract_streaming_response_text(message: Any) -> str:
    """Assistant response text from reasoning.text blocks during live streaming."""
    reasoning_blocks = [
        block
        for block in _iter_content_blocks(message)
        if _block_type(block) == "reasoning"
    ]
    if reasoning_blocks:
        part = _reasoning_response_text(reasoning_blocks[-1])
        if part:
            return part

    kwargs = getattr(message, "additional_kwargs", None) or {}
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        part = _reasoning_response_text(reasoning)
        if part:
            return part

    return ""


def _extract_live_assistant_text(message: Any) -> str:
    """Best-effort assistant text from a live model chunk."""
    incoming = extract_streaming_assistant_text(message)
    response_text = _extract_streaming_response_text(message)
    if not incoming:
        return response_text
    if not response_text:
        return incoming
    if response_text.startswith(incoming):
        return response_text
    if incoming.startswith(response_text):
        return incoming
    return response_text if len(response_text) >= len(incoming) else incoming


def extract_streaming_assistant_text(message: Any) -> str:
    """Assistant text from live stream chunks (excludes in-progress reasoning.text)."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content and not _iter_content_blocks(message):
        return content

    parts: List[str] = []
    for block in _iter_content_blocks(message):
        block_type = _block_type(block)
        if block_type in ("text", "output_text"):
            part = _block_text(block, keys=("text", "content"))
            if part:
                parts.append(part)

    if parts:
        return "".join(parts)

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text") or "")
        if parts:
            return "".join(parts)

    return content if isinstance(content, str) else ""


def extract_assistant_text(message: Any) -> str:
    streaming_text = extract_streaming_assistant_text(message)
    if streaming_text:
        return streaming_text

    content = getattr(message, "content", None)
    reasoning_blocks = [
        block
        for block in _iter_content_blocks(message)
        if _block_type(block) == "reasoning"
    ]
    if reasoning_blocks:
        part = _reasoning_response_text(reasoning_blocks[-1])
        if part:
            return part

    kwargs = getattr(message, "additional_kwargs", None) or {}
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        part = _reasoning_response_text(reasoning)
        if part:
            return part

    return content if isinstance(content, str) else ""


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
    return _join_summary_fragments(cleaned)


def _join_summary_fragments(parts: List[str]) -> str:
    """Join incremental summary tokens while undoing chunk-merge duplication."""
    result = ""
    for part in parts:
        if not part:
            continue
        if result.endswith(part):
            continue
        if part.startswith(result):
            result = part
        else:
            result += part
    return result


def dedupe_cumulative_stream_text(value: str) -> str:
    """Recover the latest cumulative snapshot from LangChain string concatenation."""
    if not value:
        return value

    block_deduped = dedupe_repeated_string(value)
    if len(block_deduped) < len(value):
        return block_deduped

    if not _CUMULATIVE_MERGE_RE.match(value):
        return value

    pos = 0
    result = ""
    while pos < len(value):
        if not result:
            found = False
            for end in range(pos + 1, len(value) + 1):
                prefix = value[pos:end]
                rest = value[end:]
                if len(prefix) == 1:
                    if not prefix.isspace() or not (
                        rest.startswith(prefix) and len(rest) > 1
                    ):
                        continue
                elif not rest.startswith(prefix):
                    continue
                result = prefix
                pos = end
                found = True
                break
            if not found:
                return value
            continue

        found = False
        for end in range(pos + 1, len(value) + 1):
            segment = value[pos:end]
            rest = value[end:]
            if len(segment) <= len(result) or not segment.startswith(result):
                continue
            if len(segment) == len(result) and len(segment) == 1 and segment.isspace():
                continue
            if rest.startswith(segment) or not rest:
                result = segment
                pos = end
                found = True
                break
        if not found:
            return value

    if pos < len(value):
        return value
    if (
        len(result) < len(value)
        and " " not in value
        and len(result) / len(value) < _DEDUPE_MIN_RETAINED_RATIO
    ):
        return value
    return result


def _maybe_dedupe_stream_incoming(incoming: str, snapshot: str) -> str:
    """Dedupe LangChain merge artifacts without truncating numeric/code tokens."""
    block_deduped = dedupe_repeated_string(incoming)
    if len(block_deduped) < len(incoming):
        return block_deduped
    if snapshot and incoming.startswith(snapshot) and len(incoming) > len(snapshot):
        return dedupe_cumulative_stream_text(incoming)
    if not snapshot:
        deduped = dedupe_cumulative_stream_text(incoming)
        if len(deduped) < len(incoming):
            return deduped
    return incoming


def _reasoning_summary_text(block: Any) -> str:
    """Extract thinking/summary text from a reasoning block."""
    if not isinstance(block, dict):
        return ""
    summary = block.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    if isinstance(summary, list):
        parts = [
            item.get("text") or item.get("summary") or ""
            for item in summary
            if isinstance(item, dict)
        ]
        return _join_summary_fragments(parts)
    content = block.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts = [
            item.get("text") or item.get("summary") or ""
            for item in content
            if isinstance(item, dict)
        ]
        return "".join(part for part in parts if part)
    reasoning = block.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return ""


def _reasoning_response_text(block: Any) -> str:
    """Extract the final assistant response from a reasoning block."""
    if not isinstance(block, dict):
        return ""
    text = block.get("text")
    if isinstance(text, str) and text:
        return text
    return ""


def _extract_thinking_without_details(message: Any) -> str:
    parts: List[str] = []

    reasoning_field = getattr(message, "reasoning", None)
    if reasoning_field:
        parts.append(
            reasoning_field
            if isinstance(reasoning_field, str)
            else str(reasoning_field)
        )

    reasoning_blocks = [
        block
        for block in _iter_content_blocks(message)
        if _block_type(block) == "reasoning"
    ]
    block_thinking = (
        _reasoning_summary_text(reasoning_blocks[-1]) if reasoning_blocks else ""
    )
    if block_thinking:
        parts.append(block_thinking)

    kwargs = getattr(message, "additional_kwargs", None) or {}
    reasoning_content = kwargs.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        kw_thinking = reasoning_content
        if block_thinking and len(kw_thinking) <= 2:
            pass
        elif block_thinking and len(kw_thinking) < len(block_thinking):
            pass
        elif kw_thinking not in parts:
            parts.append(kw_thinking)
    elif not block_thinking:
        for key in ("reasoning", "thinking"):
            reasoning = kwargs.get(key)
            if isinstance(reasoning, dict):
                part = _reasoning_summary_text(reasoning)
                if part:
                    parts.append(part)
                    break
            elif isinstance(reasoning, str) and reasoning:
                parts.append(reasoning)
                break

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
