from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .stream import (
    _extract_streaming_thinking,
    _message_needs_invoke_fallback,
    compact_ai_message_dump,
    extract_assistant_text,
    is_stub_stream_text,
    message_model_dump,
    normalize_ai_message_dump,
    sanitize_chunk_for_merge,
)
from .thread import AIMessage as ComposerAIMessage

_MODEL_CHUNK_KIND = "model_chunk"
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = ()


def _load_transient_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    try:
        from openai import APIConnectionError
    except ImportError:
        pass
    else:
        errors.append(APIConnectionError)
    try:
        import httpx
    except ImportError:
        pass
    else:
        errors.extend(
            exc
            for exc in (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout)
            if exc not in errors
        )
    return tuple(errors)


def _is_transient_error(exc: BaseException) -> bool:
    global _TRANSIENT_ERRORS
    if not _TRANSIENT_ERRORS:
        _TRANSIENT_ERRORS = _load_transient_errors()
    return isinstance(exc, _TRANSIENT_ERRORS)


def _should_fallback_to_handler_on_stream_error(exc: BaseException) -> bool:
    if isinstance(exc, ValueError):
        return "no generation chunks" in str(exc).lower()
    return False


def _get_runnable_config() -> RunnableConfig | None:
    try:
        from langgraph.config import get_config
    except ImportError:
        return None
    try:
        return get_config()
    except RuntimeError:
        return None


def _get_stream_writer() -> Callable[[Any], None] | None:
    try:
        from langgraph.config import get_stream_writer
    except ImportError:
        return None
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def _emit_model_chunk(chunk: AIMessageChunk | AIMessage) -> None:
    writer = _get_stream_writer()
    if writer is None:
        return
    writer({"kind": _MODEL_CHUNK_KIND, "chunk": chunk})


class StreamingModelMiddleware(AgentMiddleware):
    """Use model streaming so LangGraph can forward reasoning chunks each turn."""

    tools: tuple[BaseTool, ...] = ()

    def __init__(self, *, max_retries: int = 2, retry_backoff_s: float = 0.5) -> None:
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def _bind_model(self, request: ModelRequest[Any]) -> BaseChatModel:
        model = request.model
        tools = [tool for tool in (request.tools or []) if isinstance(tool, BaseTool)]
        settings = dict(request.model_settings or {})
        if tools:
            return model.bind_tools(
                tools,
                tool_choice=request.tool_choice,
                **settings,
            )
        if settings:
            return model.bind(**settings)
        return model

    @staticmethod
    def _prepare_messages(request: ModelRequest[Any]) -> list[Any]:
        messages = list(request.messages)
        if request.system_message is not None:
            messages.insert(0, request.system_message)
        return messages

    @staticmethod
    def _chunk_to_ai_message(chunk: AIMessageChunk | AIMessage | None) -> AIMessage:
        if chunk is None:
            return AIMessage(content="")
        if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk):
            return chunk
        dump = chunk.model_dump() if hasattr(chunk, "model_dump") else {}
        if dump.get("type") in ("AIMessageChunk",):
            dump["type"] = "ai"
        return AIMessage.model_validate(dump)

    @classmethod
    def _merge_stream_chunks(
        cls,
        chunks: Iterable[AIMessageChunk | AIMessage],
        *,
        turn_thinking: str = "",
    ) -> AIMessage:
        accum: AIMessageChunk | AIMessage | None = None
        latest_thinking = turn_thinking
        latest_assistant_text = ""
        for chunk in chunks:
            chunk = sanitize_chunk_for_merge(chunk)
            accum = chunk if accum is None else accum + chunk
            thinking = _extract_streaming_thinking(accum)
            if len(thinking) > len(latest_thinking):
                latest_thinking = thinking
            assistant_text = extract_assistant_text(accum)
            if len(assistant_text) > len(latest_assistant_text):
                latest_assistant_text = assistant_text
        message = cls._chunk_to_ai_message(accum)
        if latest_assistant_text and not extract_assistant_text(message):
            message = message.model_copy(update={"content": latest_assistant_text})
        return cls._finalize_model_message(message, latest_thinking)

    @staticmethod
    def _finalize_model_message(message: AIMessage, turn_thinking: str) -> AIMessage:
        message_thinking = _extract_streaming_thinking(message)
        if turn_thinking and (
            not message_thinking
            or is_stub_stream_text(message_thinking)
            or len(turn_thinking) > len(message_thinking)
        ):
            kwargs = dict(message.additional_kwargs or {})
            kwargs["reasoning_content"] = turn_thinking
            message = message.model_copy(update={"additional_kwargs": kwargs})
        dump = message_model_dump(message)
        if dump.get("type") in ("AIMessageChunk",):
            dump["type"] = "ai"
        dump = normalize_ai_message_dump(message, dump)
        dump = compact_ai_message_dump(dump)
        return ComposerAIMessage.model_validate(dump)

    def _stream_kwargs(self) -> dict[str, Any]:
        config = _get_runnable_config()
        return {"config": config} if config is not None else {}

    def _stream_model_sync(self, request: ModelRequest[Any]) -> AIMessage:
        model = self._bind_model(request)
        messages = self._prepare_messages(request)
        stream = getattr(model, "stream", None)
        if stream is None:
            raise RuntimeError("Model does not support streaming")

        stream_kwargs = self._stream_kwargs()
        accum: AIMessageChunk | AIMessage | None = None
        latest_thinking = ""
        latest_assistant_text = ""
        for chunk in stream(messages, **stream_kwargs):
            chunk = sanitize_chunk_for_merge(chunk)
            _emit_model_chunk(chunk)
            accum = chunk if accum is None else accum + chunk
            thinking = _extract_streaming_thinking(accum)
            if len(thinking) > len(latest_thinking):
                latest_thinking = thinking
            assistant_text = extract_assistant_text(accum)
            if len(assistant_text) > len(latest_assistant_text):
                latest_assistant_text = assistant_text
        message = self._chunk_to_ai_message(accum)
        if latest_assistant_text and not extract_assistant_text(message):
            message = message.model_copy(update={"content": latest_assistant_text})
        return self._finalize_model_message(
            message,
            latest_thinking,
        )

    async def _stream_model_async(self, request: ModelRequest[Any]) -> AIMessage:
        model = self._bind_model(request)
        messages = self._prepare_messages(request)
        astream = getattr(model, "astream", None)
        if astream is None:
            raise RuntimeError("Model does not support async streaming")

        stream_kwargs = self._stream_kwargs()
        accum: AIMessageChunk | AIMessage | None = None
        latest_thinking = ""
        latest_assistant_text = ""
        async for chunk in astream(messages, **stream_kwargs):
            chunk = sanitize_chunk_for_merge(chunk)
            _emit_model_chunk(chunk)
            accum = chunk if accum is None else accum + chunk
            thinking = _extract_streaming_thinking(accum)
            if len(thinking) > len(latest_thinking):
                latest_thinking = thinking
            assistant_text = extract_assistant_text(accum)
            if len(assistant_text) > len(latest_assistant_text):
                latest_assistant_text = assistant_text
        message = self._chunk_to_ai_message(accum)
        if latest_assistant_text and not extract_assistant_text(message):
            message = message.model_copy(update={"content": latest_assistant_text})
        return self._finalize_model_message(
            message,
            latest_thinking,
        )

    @staticmethod
    def _message_from_handler_response(
        response: ModelResponse[Any],
        *,
        turn_thinking: str = "",
    ) -> AIMessage:
        invoke_message = response.result[0]
        if isinstance(invoke_message, AIMessageChunk):
            invoke_message = StreamingModelMiddleware._chunk_to_ai_message(
                invoke_message
            )
        return StreamingModelMiddleware._finalize_model_message(
            invoke_message,
            turn_thinking,
        )

    def _resolve_streamed_message(
        self,
        request: ModelRequest[Any],
        streamed: AIMessage,
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> AIMessage:
        if not _message_needs_invoke_fallback(streamed):
            return streamed
        response = handler(request)
        turn_thinking = _extract_streaming_thinking(streamed)
        return self._message_from_handler_response(
            response,
            turn_thinking=turn_thinking,
        )

    async def _aresolve_streamed_message(
        self,
        request: ModelRequest[Any],
        streamed: AIMessage,
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> AIMessage:
        if not _message_needs_invoke_fallback(streamed):
            return streamed
        response = await handler(request)
        turn_thinking = _extract_streaming_thinking(streamed)
        return self._message_from_handler_response(
            response,
            turn_thinking=turn_thinking,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                streamed = self._stream_model_sync(request)
                message = self._resolve_streamed_message(request, streamed, handler)
                return ModelResponse(result=[message])
            except Exception as exc:
                if _should_fallback_to_handler_on_stream_error(exc):
                    response = handler(request)
                    message = self._message_from_handler_response(response)
                    return ModelResponse(result=[message])
                last_error = exc
                if attempt >= self.max_retries or not _is_transient_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        return ModelResponse(result=[AIMessage(content="")])

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                streamed = await self._stream_model_async(request)
                message = await self._aresolve_streamed_message(
                    request, streamed, handler
                )
                return ModelResponse(result=[message])
            except Exception as exc:
                if _should_fallback_to_handler_on_stream_error(exc):
                    response = await handler(request)
                    message = self._message_from_handler_response(response)
                    return ModelResponse(result=[message])
                last_error = exc
                if attempt >= self.max_retries or not _is_transient_error(exc):
                    raise
                await asyncio.sleep(self.retry_backoff_s * (attempt + 1))
        if last_error is not None:
            raise last_error
        return ModelResponse(result=[AIMessage(content="")])
