from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages import HumanMessage, SystemMessage

from composer import AIMessage as ComposerAIMessage
from composer.streaming_middleware import StreamingModelMiddleware
from composer.stream import (
    AssistantEvent,
    StreamEventProcessor,
    ThinkingEvent,
    ToolCallEvent,
)


class _StreamingModel:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks
        self.stream_calls = 0
        self.astream_calls = 0
        self.last_stream_config = None
        self.last_astream_config = None

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        del tools, tool_choice, kwargs
        return self

    def bind(self, **kwargs):
        del kwargs
        return self

    def stream(self, messages, config=None, **kwargs) -> Iterator[AIMessageChunk]:
        del messages, kwargs
        self.stream_calls += 1
        self.last_stream_config = config
        yield from self._chunks

    async def astream(self, messages, config=None, **kwargs) -> AsyncIterator[AIMessageChunk]:
        del messages, kwargs
        self.astream_calls += 1
        self.last_astream_config = config
        for chunk in self._chunks:
            yield chunk


def _request(model: _StreamingModel) -> ModelRequest[Any]:
    return ModelRequest(
        model=model,
        messages=[HumanMessage("hi")],
        tools=[],
    )


def test_streaming_middleware_uses_astream():
    model = _StreamingModel(
        [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "thinking"},
            ),
            AIMessageChunk(content="answer"),
        ]
    )
    middleware = StreamingModelMiddleware(max_retries=0)
    response = asyncio.run(
        middleware.awrap_model_call(_request(model), AsyncMock())
    )

    assert model.astream_calls == 1
    assert isinstance(response, ModelResponse)
    assert len(response.result) == 1
    assert isinstance(response.result[0], AIMessage)


def test_streaming_middleware_uses_stream_sync():
    model = _StreamingModel([AIMessageChunk(content="answer")])
    middleware = StreamingModelMiddleware(max_retries=0)
    response = middleware.wrap_model_call(_request(model), MagicMock())

    assert model.stream_calls == 1
    assert response.result[0].content == "answer"


def test_values_uses_turn_buffer_when_ai_message_has_no_reasoning():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    chunk = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "Need to snap"},
    )
    processor.merge_turn_accumulation(chunk)

    tool_ai = ComposerAIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )

    event_types = [type(event).__name__ for event in events]
    assert event_types.index("ThinkingEvent") < event_types.index("ToolCallEvent")
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "Need to snap"
    ]


def test_streaming_middleware_emits_model_chunks_via_stream_writer(monkeypatch):
    emitted: list[dict[str, Any]] = []

    def _writer(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    monkeypatch.setattr(
        "composer.streaming_middleware._get_stream_writer",
        lambda: _writer,
    )

    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "think"}),
        AIMessageChunk(content="answer"),
    ]
    model = _StreamingModel(chunks)
    middleware = StreamingModelMiddleware(max_retries=0)
    asyncio.run(middleware.awrap_model_call(_request(model), AsyncMock()))

    assert len(emitted) == 2
    assert all(item.get("kind") == "model_chunk" for item in emitted)
    assert emitted[0]["chunk"].additional_kwargs["reasoning_content"] == "think"


def test_streaming_middleware_preserves_tool_turn_reasoning_on_result():
    model = _StreamingModel(
        [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "Need to snap"},
            ),
            AIMessageChunk(
                content="",
                tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
            ),
        ]
    )
    middleware = StreamingModelMiddleware(max_retries=0)
    response = asyncio.run(
        middleware.awrap_model_call(_request(model), AsyncMock())
    )

    result = response.result[0]
    assert result.tool_calls
    assert result.additional_kwargs.get("reasoning_content") == "Need to snap"
    assert "reasoning" not in result.additional_kwargs
    assert result.content == ""


def test_streaming_middleware_populates_content_from_reasoning_text():
    model = _StreamingModel(
        [
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning": {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "thinking"}],
                        "text": "Hello there",
                    },
                    "reasoning_content": "thinking",
                },
            ),
        ]
    )
    middleware = StreamingModelMiddleware(max_retries=0)
    response = asyncio.run(
        middleware.awrap_model_call(_request(model), AsyncMock())
    )

    assert response.result[0].content == "Hello there"
    assert "reasoning" not in response.result[0].additional_kwargs


def test_streaming_middleware_falls_back_to_handler_when_stream_content_empty():
    model = _StreamingModel(
        [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "."},
                response_metadata={"chunk_position": "last"},
            ),
        ]
    )
    handler = AsyncMock(
        return_value=ModelResponse(result=[AIMessage(content="Hello from invoke")])
    )
    middleware = StreamingModelMiddleware(max_retries=0)
    response = asyncio.run(
        middleware.awrap_model_call(_request(model), handler)
    )

    handler.assert_awaited_once()
    assert response.result[0].content == "Hello from invoke"


def test_streaming_middleware_uses_streamed_assistant_text_over_empty_final_chunk():
    model = _StreamingModel(
        [
            AIMessageChunk(
                content=[{"type": "output_text", "text": "Streamed answer"}],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "."},
                response_metadata={"chunk_position": "last"},
            ),
        ]
    )
    handler = AsyncMock()
    middleware = StreamingModelMiddleware(max_retries=0)
    response = asyncio.run(
        middleware.awrap_model_call(_request(model), handler)
    )

    handler.assert_not_awaited()
    assert response.result[0].content == "Streamed answer"


def test_streaming_middleware_passes_runnable_config(monkeypatch):
    model = _StreamingModel([AIMessageChunk(content="answer")])
    sentinel_config = {"callbacks": ["cb"]}

    monkeypatch.setattr(
        "composer.streaming_middleware._get_runnable_config",
        lambda: sentinel_config,
    )

    middleware = StreamingModelMiddleware(max_retries=0)
    asyncio.run(middleware.awrap_model_call(_request(model), AsyncMock()))

    assert model.last_astream_config == sentinel_config


def test_values_baseline_skips_first_snapshot_with_system_message():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update(
        {"messages": [SystemMessage("sys"), HumanMessage("hi")]}
    )

    tool_ai = ComposerAIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning_content": "use tool",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "use tool"}],
            },
        },
    )
    events = processor.process_values_update(
        {"messages": [SystemMessage("sys"), HumanMessage("hi"), tool_ai]}
    )

    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "use tool"
    ]
    assert not any(isinstance(event, AssistantEvent) for event in events)
