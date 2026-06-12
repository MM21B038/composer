import asyncio
import inspect
import queue
import threading
import warnings
from contextlib import contextmanager
from typing import Literal, List, Optional, Union, AsyncIterator, Iterator, Sequence, Any, Tuple, Dict, TypeVar

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from .mcp import MCPClient, MCPPromptInfo, MCPResourceInfo
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
    message_model_dump,
    normalize_ai_message_dump,
    normalize_response_metadata,
    parse_langgraph_chunk,
    parse_custom_model_chunk,
    compact_ai_message_dump,
    is_stub_stream_text,
    sanitize_chunk_for_merge,
)
from .streaming_middleware import StreamingModelMiddleware
from .thread import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    Thread,
)
from .tools import ToolCall, ToolResult, combine_tools, run_tool_call as _run_tool_call
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

PROVIDERS = Literal["custom"]
_T = TypeVar("_T")
_ASYNC_GEN_SENTINEL = object()


def _in_running_loop() -> bool:
    try:
        return asyncio.get_running_loop().is_running()
    except RuntimeError:
        return False


@contextmanager
def _suppress_openai_response_serializer_warnings():
    """LangChain's Responses API parser calls Response.model_dump() without warnings=False."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings:",
            category=UserWarning,
        )
        yield


class Agent:
    def __init__(
        self,
        provider: Optional[PROVIDERS] = "custom",
        model: Optional[Union[str, BaseChatModel]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
        *,
        auto_tool_call: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning = reasoning
        self.auto_tool_call = auto_tool_call

    def _prepare_messages(
        self,
        thread: Thread,
        *,
        system_prompt_override: Optional[str] = None,
    ):
        messages = thread.messages_for_model()
        active_prompt = (
            system_prompt_override
            if system_prompt_override is not None
            else self.system_prompt
        )

        if (
            active_prompt
            and messages
            and isinstance(messages[0], SystemMessage)
        ):
            messages = messages[1:]

        return messages

    def _get_model(self):
        if isinstance(self.model, BaseChatModel):
            return self.model

        if self.provider == "custom":
            extra_body: Dict[str, Any] = {}
            if self.reasoning is True:
                extra_body["reasoning"] = {"enabled": True}
            elif isinstance(self.reasoning, dict):
                extra_body["reasoning"] = self.reasoning

            return ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                extra_body=extra_body or None,
            )

        raise NotImplementedError(
            f"Provider {self.provider} is not supported yet."
        )

    def _resolve_tools(self, tools: Optional[List] = None) -> List[BaseTool]:
        resolved = self.tools if tools is None else tools
        return self._flatten_tools(resolved)

    async def _resolve_tools_async(self, tools: Optional[List] = None) -> List[BaseTool]:
        resolved = self.tools if tools is None else tools
        return await self._flatten_tools_async(resolved)

    def _flatten_tools(self, tools: Any) -> List[BaseTool]:
        if isinstance(tools, MCPClient):
            if not tools.loaded:
                raise RuntimeError(
                    "MCPClient tools not loaded. Call: await mcp.load_tools() first, "
                    "or use await agent.ainvoke(...) / await agent.astream_events(...) "
                    "which loads MCP tools automatically."
                )
            return list(tools.tools)
        if isinstance(tools, list):
            out: List[BaseTool] = []
            for item in tools:
                out.extend(self._flatten_tools(item))
            return out
        return [tools]

    async def _flatten_tools_async(self, tools: Any) -> List[BaseTool]:
        if isinstance(tools, MCPClient):
            return await tools.get_tools()
        if isinstance(tools, list):
            out: List[BaseTool] = []
            for item in tools:
                out.extend(await self._flatten_tools_async(item))
            return out
        return [tools]

    def _tool_requires_async(self, tool: BaseTool) -> bool:
        coroutine = getattr(tool, "coroutine", None)
        if coroutine is not None and inspect.iscoroutinefunction(coroutine):
            return True
        func = getattr(tool, "func", None)
        if func is not None and inspect.iscoroutinefunction(func):
            return True
        return False

    def _coerce_tool_call(self, call: Union[ToolCall, ToolCallEvent]) -> ToolCall:
        if isinstance(call, ToolCall):
            return call
        if isinstance(call, ToolCallEvent):
            return call.call
        raise TypeError(
            f"Expected ToolCall or ToolCallEvent, got {type(call).__name__}"
        )

    async def arun_tool_call(
        self,
        call: Union[ToolCall, ToolCallEvent],
        *,
        tools: Optional[List] = None,
        thread: Optional[Thread] = None,
    ) -> ToolResult:
        """Run one tool call and return its result (`.content` is the output text)."""
        tool_call = self._coerce_tool_call(call)
        langchain_tools = await self._resolve_tools_async(tools)
        result = await _run_tool_call(tool_call, langchain_tools)
        if thread is not None and thread.auto_append_tool_results:
            thread.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=result.tool_call_id or tool_call.id,
                    name=result.name or tool_call.name,
                )
            )
        return result

    def run_tool_call(
        self,
        call: Union[ToolCall, ToolCallEvent],
        *,
        tools: Optional[List] = None,
        thread: Optional[Thread] = None,
    ) -> ToolResult:
        """Sync wrapper for arun_tool_call."""
        return self._sync_run_async(
            self.arun_tool_call(call, tools=tools, thread=thread),
            tools=tools,
        )

    def _tools_need_async(self, tools: Optional[Any] = None) -> bool:
        raw = self.tools if tools is None else tools
        if isinstance(raw, MCPClient):
            return True
        if isinstance(raw, list):
            if any(isinstance(item, MCPClient) for item in raw):
                return True
            try:
                flat = self._flatten_tools(raw)
            except RuntimeError:
                return True
            return any(self._tool_requires_async(tool) for tool in flat)
        try:
            return self._tool_requires_async(self._flatten_tools(raw))
        except RuntimeError:
            return True

    def _collect_mcp_clients(self, tools: Optional[Any] = None) -> List[MCPClient]:
        raw = self.tools if tools is None else tools
        if isinstance(raw, MCPClient):
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, MCPClient)]
        return []

    def _reset_loop_bound_clients(self) -> None:
        model = self.model
        if not isinstance(model, BaseChatModel):
            return
        build_client = getattr(model, "_build_client", None)
        if build_client is not None:
            model.client = build_client()

    def _invalidate_mcp_tools(self, tools: Optional[Any] = None) -> None:
        for mcp in self._collect_mcp_clients(tools):
            mcp._tools = None

    async def _prepare_thread_loop_context(self, tools: Optional[Any] = None) -> None:
        self._reset_loop_bound_clients()
        for mcp in self._collect_mcp_clients(tools):
            mcp._tools = None
            await mcp.load_tools(reload=True)

    def _cleanup_thread_loop_context(self, tools: Optional[Any] = None) -> None:
        self._reset_loop_bound_clients()
        self._invalidate_mcp_tools(tools)

    def _sync_run_async(self, coro, *, tools: Optional[Any] = None):
        if not _in_running_loop():
            return asyncio.run(coro)

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        async def wrapped() -> Any:
            await self._prepare_thread_loop_context(tools)
            return await coro

        def runner() -> None:
            try:
                result_queue.put(("ok", asyncio.run(wrapped())))
            except Exception as exc:
                result_queue.put(("err", exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        kind, payload = result_queue.get()
        thread.join()
        self._cleanup_thread_loop_context(tools)
        if kind == "err":
            raise payload
        return payload

    def _sync_iterate_asyncgen(
        self, async_gen_factory: Any, *, tools: Optional[Any] = None
    ) -> Iterator[_T]:
        if _in_running_loop():
            item_queue: queue.Queue = queue.Queue()

            async def drain() -> None:
                await self._prepare_thread_loop_context(tools)
                try:
                    async for item in async_gen_factory():
                        item_queue.put(("ok", item))
                except Exception as exc:
                    item_queue.put(("err", exc))
                finally:
                    item_queue.put(("done", _ASYNC_GEN_SENTINEL))

            def runner() -> None:
                try:
                    asyncio.run(drain())
                except Exception as exc:
                    item_queue.put(("err", exc))
                    item_queue.put(("done", _ASYNC_GEN_SENTINEL))

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            while True:
                kind, payload = item_queue.get()
                if kind == "done":
                    break
                if kind == "err":
                    raise payload
                yield payload
            thread.join(timeout=1)
            self._cleanup_thread_loop_context(tools)
            return

        item_queue: queue.Queue = queue.Queue()

        async def drain() -> None:
            try:
                async for item in async_gen_factory():
                    item_queue.put(("ok", item))
            except Exception as exc:
                item_queue.put(("err", exc))
            finally:
                item_queue.put(("done", _ASYNC_GEN_SENTINEL))

        try:
            asyncio.run(drain())
        except Exception as exc:
            item_queue.put(("err", exc))
            item_queue.put(("done", _ASYNC_GEN_SENTINEL))

        while True:
            kind, payload = item_queue.get()
            if kind == "done":
                break
            if kind == "err":
                raise payload
            yield payload

    def _build_agent(
        self,
        langchain_tools: List[BaseTool],
        *,
        auto_tool_call: Optional[bool] = None,
        system_prompt: Optional[str] = None,
    ):
        run_tools = self.auto_tool_call if auto_tool_call is None else auto_tool_call
        return create_agent(
            model=self._get_model(),
            tools=langchain_tools,
            middleware=[StreamingModelMiddleware()],
            system_prompt=(
                system_prompt
                if system_prompt is not None
                else self.system_prompt
            ),
            interrupt_before=None if run_tools else ["tools"],
        )

    def _build_agent_for_invoke(
        self,
        tools: Optional[List] = None,
        *,
        auto_tool_call: Optional[bool] = None,
        system_prompt_override: Optional[str] = None,
    ):
        return self._build_agent(
            self._resolve_tools(tools),
            auto_tool_call=auto_tool_call,
            system_prompt=system_prompt_override,
        )

    async def _build_agent_for_invoke_async(
        self,
        tools: Optional[List] = None,
        *,
        auto_tool_call: Optional[bool] = None,
        system_prompt_override: Optional[str] = None,
    ):
        return self._build_agent(
            await self._resolve_tools_async(tools),
            auto_tool_call=auto_tool_call,
            system_prompt=system_prompt_override,
        )

    def _get_agent(
        self,
        tools: Optional[List] = None,
        *,
        auto_tool_call: Optional[bool] = None,
    ):
        return self._build_agent(
            self._resolve_tools(tools),
            auto_tool_call=auto_tool_call,
        )

    async def _aget_agent(
        self,
        tools: Optional[List] = None,
        *,
        auto_tool_call: Optional[bool] = None,
    ):
        return self._build_agent(
            await self._resolve_tools_async(tools),
            auto_tool_call=auto_tool_call,
        )


    def _to_ai_message(self, message) -> AIMessage:
        dump = message_model_dump(message)
        if dump.get("type") == "AIMessageChunk":
            dump["type"] = "ai"
        dump = normalize_ai_message_dump(message, dump)
        return AIMessage.model_validate(dump)

    def _enrich_ai_message_from_stream(
        self,
        message: Any,
        *,
        streamed_thinking: str = "",
        streamed_assistant: str = "",
    ) -> AIMessage:
        ai_message = self._to_ai_message(message)
        if not streamed_thinking and not streamed_assistant:
            return ai_message

        dump = message_model_dump(ai_message)
        kwargs = dict(dump.get("additional_kwargs") or {})
        existing_thinking = kwargs.get("reasoning_content", "")
        if streamed_thinking and (
            not existing_thinking or is_stub_stream_text(str(existing_thinking))
        ):
            kwargs["reasoning_content"] = streamed_thinking
        content = dump.get("content", "")
        if streamed_assistant and not (
            isinstance(content, str) and content.strip()
        ):
            dump["content"] = streamed_assistant
        dump["additional_kwargs"] = kwargs
        interim = AIMessage.model_validate({**dump, "type": "ai"})
        dump = normalize_ai_message_dump(interim, dump)
        dump = compact_ai_message_dump(dump)
        return AIMessage.model_validate(dump)

    def _append_stream_values_messages(
        self,
        thread: Thread,
        append_to: Optional[Thread],
        output_messages: Sequence[Any],
        start_index: int,
        *,
        streamed_thinking: str = "",
        streamed_assistant: str = "",
    ) -> None:
        target = append_to or thread
        if not target.appends_agent_messages():
            return
        pending_thinking = streamed_thinking
        pending_assistant = streamed_assistant
        for message in output_messages[start_index:]:
            if getattr(message, "type", None) == "ai" and (
                pending_thinking or pending_assistant
            ):
                message = self._enrich_ai_message_from_stream(
                    message,
                    streamed_thinking=pending_thinking,
                    streamed_assistant=pending_assistant,
                )
                pending_thinking = ""
                pending_assistant = ""
            if self._should_append_message(target, message):
                if getattr(message, "type", None) == "ai":
                    target.append(message)
                else:
                    target.append(self._to_thread_message(message))

    def _to_thread_message(self, message) -> Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]:
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            if isinstance(message, AIMessageChunk):
                return self._to_ai_message(message)
            if isinstance(message, AIMessage):
                return self._to_ai_message(message)
            return message

        msg_type = getattr(message, "type", None)
        if msg_type == "ai":
            return self._to_ai_message(message)

        dump = message_model_dump(message)
        if msg_type == "ai" and dump.get("type") in ("AIMessageChunk",):
            dump["type"] = "ai"

        type_map = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }
        cls = type_map.get(msg_type)
        if cls is None:
            raise ValueError(f"Unsupported message type: {msg_type!r}")
        if cls is AIMessage:
            return self._to_ai_message(message)
        return cls.model_validate(dump)

    def _should_append_message(self, thread: Thread, message: Any) -> bool:
        thread_message = self._to_thread_message(message)
        if isinstance(thread_message, ToolMessage):
            return thread.auto_append_tool_results
        if isinstance(thread_message, AIMessage):
            if thread_message.tool_calls:
                return thread.auto_append_tool_calls
            return thread.auto_append_ai_message
        return True

    def _new_messages_to_append(
        self,
        thread: Thread,
        input_messages: Sequence[Any],
        output_messages: Sequence[Any],
    ) -> List[Any]:
        existing_count = len(thread.get_messages())
        new_messages = list(output_messages[existing_count:])
        if not new_messages:
            new_messages = list(output_messages[len(input_messages) :])
        if not new_messages and output_messages:
            last = output_messages[-1]
            if getattr(last, "type", None) in ("ai", "AIMessage", "AIMessageChunk"):
                new_messages = [last]
        return new_messages

    def _append_new_messages(
        self,
        thread: Thread,
        input_messages: Sequence[Any],
        output_messages: Sequence[Any],
    ) -> None:
        if not thread.appends_agent_messages():
            return
        for message in self._new_messages_to_append(
            thread, input_messages, output_messages
        ):
            if self._should_append_message(thread, message):
                thread.append(self._to_thread_message(message))

    def _apply_stream_metadata(
        self,
        ai_message: AIMessage,
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
    ) -> AIMessage:
        metadata_source = last_finish_chunk or last_stream_chunk
        if metadata_source is None:
            return ai_message

        dump = message_model_dump(ai_message)
        if metadata_source.response_metadata:
            dump["response_metadata"] = normalize_response_metadata(
                {
                    **dump.get("response_metadata", {}),
                    **metadata_source.response_metadata,
                }
            )
        if metadata_source.usage_metadata:
            dump["usage_metadata"] = metadata_source.usage_metadata
        if metadata_source.id:
            dump["id"] = metadata_source.id
        merged = AIMessage.model_validate({**dump, "type": "ai"})
        dump = normalize_ai_message_dump(merged, dump)
        dump = compact_ai_message_dump(dump)
        return AIMessage.model_validate(dump)

    def _accumulate_ai_chunk(
        self,
        message: AIMessageChunk,
        accumulated: Optional[AIMessageChunk],
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
    ) -> Tuple[
        Optional[AIMessageChunk],
        Optional[AIMessageChunk],
        Optional[AIMessageChunk],
    ]:
        message = sanitize_chunk_for_merge(message)
        last_stream_chunk = message
        if message.response_metadata.get("finish_reason"):
            last_finish_chunk = message
        accumulated = message if accumulated is None else accumulated + message
        return accumulated, last_stream_chunk, last_finish_chunk

    def _process_stream_chunk(
        self,
        chunk: Any,
        stream_mode: Union[str, Sequence[str]],
        accumulated: Optional[AIMessageChunk],
        last_ai_message: Any,
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
    ) -> Tuple[
        Optional[AIMessageChunk],
        Any,
        Optional[AIMessageChunk],
        Optional[AIMessageChunk],
        Optional[List[Any]],
    ]:
        last_output_messages = None
        if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
            mode, data = chunk
        elif isinstance(stream_mode, str):
            mode, data = stream_mode, chunk
        else:
            return accumulated, last_ai_message, last_stream_chunk, last_finish_chunk, last_output_messages

        if mode == "messages":
            message = data[0] if isinstance(data, tuple) and len(data) == 2 else data
            if isinstance(message, AIMessageChunk):
                pass
            elif getattr(message, "type", None) in ("ai", "AIMessageChunk"):
                last_ai_message = message

        elif mode == "custom":
            message = parse_custom_model_chunk(data)
            if isinstance(message, AIMessageChunk):
                accumulated, last_stream_chunk, last_finish_chunk = (
                    self._accumulate_ai_chunk(
                        message,
                        accumulated,
                        last_stream_chunk,
                        last_finish_chunk,
                    )
                )

        elif mode == "values" and isinstance(data, dict):
            messages = data.get("messages", [])
            if messages:
                last_output_messages = messages
                if getattr(messages[-1], "type", None) in ("ai", "AIMessage"):
                    last_ai_message = messages[-1]

        return (
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
            last_output_messages,
        )

    def _stream_modes(
        self, stream_mode: Union[str, Sequence[str]]
    ) -> Tuple[Union[str, Sequence[str]], bool]:
        if stream_mode == "messages":
            return ["messages", "values"], True
        if stream_mode == "events":
            return ["messages", "values"], False
        return stream_mode, False

    def _events_stream_modes(self) -> List[str]:
        return ["messages", "values", "custom"]

    def _yield_stream_chunk(self, chunk: Any, filter_messages_only: bool) -> Optional[Any]:
        if not filter_messages_only:
            return chunk
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "messages":
            return chunk[1]
        if isinstance(chunk, tuple) and len(chunk) == 2 and not isinstance(chunk[0], str):
            return chunk
        return None

    def _finalize_stream_append(
        self,
        thread: Thread,
        append_to: Optional[Thread],
        input_messages: Sequence[Any],
        last_output_messages: Optional[List[Any]],
        accumulated: Optional[AIMessageChunk],
        last_ai_message: Any,
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
        *,
        output_messages_start_index: Optional[int] = None,
        streamed_thinking: str = "",
        streamed_assistant: str = "",
    ) -> None:
        target = append_to or thread
        if last_output_messages is not None:
            if output_messages_start_index is not None:
                pending_thinking = streamed_thinking
                pending_assistant = streamed_assistant
                for message in last_output_messages[output_messages_start_index:]:
                    if getattr(message, "type", None) == "ai" and (
                        pending_thinking or pending_assistant
                    ):
                        message = self._enrich_ai_message_from_stream(
                            message,
                            streamed_thinking=pending_thinking,
                            streamed_assistant=pending_assistant,
                        )
                        pending_thinking = ""
                        pending_assistant = ""
                    if self._should_append_message(target, message):
                        target.append(self._to_thread_message(message))
            else:
                self._append_new_messages(
                    target, input_messages, last_output_messages
                )
            return

        ai_message = self._resolve_stream_result(
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
        )
        if ai_message is not None and self._should_append_message(target, ai_message):
            target.append(ai_message)

    def _resolve_stream_result(
        self,
        accumulated: Optional[AIMessageChunk],
        last_ai_message: Any,
        last_stream_chunk: Optional[AIMessageChunk] = None,
        last_finish_chunk: Optional[AIMessageChunk] = None,
    ) -> Optional[AIMessage]:
        if last_ai_message is not None:
            ai_message = self._to_ai_message(last_ai_message)
        elif accumulated is not None:
            ai_message = self._to_ai_message(accumulated)
        else:
            return None
        return self._apply_stream_metadata(
            ai_message, last_stream_chunk, last_finish_chunk
        )

    def __call__(
        self,
        thread: Thread,
        *,
        append_to: Optional[Thread] = None,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
    ):
        return self.invoke(
            thread,
            append_to=append_to,
            tools=tools,
            auto_tool_call=auto_tool_call,
        )

    def invoke(
        self,
        thread: Thread,
        *,
        append_to: Optional[Thread] = None,
        record_output: bool = True,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
        system_prompt_override: Optional[str] = None,
    ):
        if self._tools_need_async(tools):
            return self._sync_run_async(
                self.ainvoke(
                    thread,
                    append_to=append_to,
                    record_output=record_output,
                    tools=tools,
                    auto_tool_call=auto_tool_call,
                    system_prompt_override=system_prompt_override,
                ),
                tools=tools,
            )

        agent = self._build_agent_for_invoke(
            tools,
            auto_tool_call=auto_tool_call,
            system_prompt_override=system_prompt_override,
        )
        input_messages = self._prepare_messages(
            thread,
            system_prompt_override=system_prompt_override,
        )

        with _suppress_openai_response_serializer_warnings():
            response = agent.invoke({"messages": input_messages})
        output_messages = response["messages"]
        if record_output:
            self._append_new_messages(
                append_to or thread, input_messages, output_messages
            )
        return self._to_ai_message(output_messages[-1])

    async def ainvoke(
        self,
        thread: Thread,
        *,
        append_to: Optional[Thread] = None,
        record_output: bool = True,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
        system_prompt_override: Optional[str] = None,
    ):
        agent = await self._build_agent_for_invoke_async(
            tools,
            auto_tool_call=auto_tool_call,
            system_prompt_override=system_prompt_override,
        )
        input_messages = self._prepare_messages(
            thread,
            system_prompt_override=system_prompt_override,
        )

        with _suppress_openai_response_serializer_warnings():
            response = await agent.ainvoke({"messages": input_messages})
        output_messages = response["messages"]
        if record_output:
            self._append_new_messages(
                append_to or thread, input_messages, output_messages
            )
        return self._to_ai_message(output_messages[-1])

    def _consume_stream_chunk(
        self,
        chunk: Any,
        internal_mode: Union[str, Sequence[str]],
        *,
        accumulated: Optional[AIMessageChunk],
        last_ai_message: Any,
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
        last_output_messages: Optional[List[Any]],
    ) -> Tuple[
        Optional[AIMessageChunk],
        Any,
        Optional[AIMessageChunk],
        Optional[AIMessageChunk],
        Optional[List[Any]],
    ]:
        (
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
            output_messages,
        ) = self._process_stream_chunk(
            chunk,
            internal_mode,
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
        )
        if output_messages is not None:
            last_output_messages = output_messages
        return (
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
            last_output_messages,
        )

    def _reset_turn_stream_state(
        self,
    ) -> tuple[None, None, None]:
        return None, None, None

    def _yield_events_from_chunk(
        self,
        chunk: Any,
        processor: StreamEventProcessor,
        *,
        accumulated: Optional[AIMessageChunk] = None,
        last_stream_chunk: Optional[AIMessageChunk] = None,
    ) -> Iterator[StreamEvent]:
        parsed = parse_langgraph_chunk(chunk)
        if parsed is None:
            return

        mode, data = parsed
        if mode == "messages":
            message = data[0] if isinstance(data, tuple) and len(data) == 2 else data
            metadata = data[1] if isinstance(data, tuple) and len(data) == 2 else {}
            if isinstance(message, AIMessageChunk):
                return
            yield from processor.process_message_chunk(message, metadata)
        elif mode == "custom":
            message = parse_custom_model_chunk(data)
            if message is not None:
                yield from processor.process_message_chunk(message, {})
        elif mode == "values" and isinstance(data, dict):
            turn_source = accumulated or last_stream_chunk
            if turn_source is not None:
                processor.merge_turn_accumulation(turn_source)
            yield from processor.process_values_update(data)

    def _yield_flush_events(
        self,
        processor: StreamEventProcessor,
        *,
        accumulated: Optional[AIMessageChunk],
        last_ai_message: Any,
        last_stream_chunk: Optional[AIMessageChunk],
        last_finish_chunk: Optional[AIMessageChunk],
    ) -> Iterator[StreamEvent]:
        final_message = self._resolve_stream_result(
            accumulated,
            last_ai_message,
            last_stream_chunk,
            last_finish_chunk,
        )
        if final_message is not None:
            yield from processor.flush(final_message)

    def stream(
        self,
        thread: Thread,
        stream_mode: Union[str, Sequence[str]] = "messages",
        *,
        append_to: Optional[Thread] = None,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
    ) -> Iterator:
        if stream_mode == "events":
            yield from self.stream_events(
                thread,
                append_to=append_to,
                tools=tools,
                auto_tool_call=auto_tool_call,
            )
            return

        if self._tools_need_async(tools):
            yield from self._sync_iterate_asyncgen(
                lambda: self.astream(
                    thread,
                    stream_mode=stream_mode,
                    append_to=append_to,
                    tools=tools,
                    auto_tool_call=auto_tool_call,
                ),
                tools=tools,
            )
            return

        agent = self._get_agent(tools, auto_tool_call=auto_tool_call)
        input_messages = self._prepare_messages(thread)
        internal_mode, filter_messages_only = self._stream_modes(stream_mode)
        accumulated: Optional[AIMessageChunk] = None
        last_ai_message = None
        last_stream_chunk: Optional[AIMessageChunk] = None
        last_finish_chunk: Optional[AIMessageChunk] = None
        last_output_messages: Optional[List[Any]] = None

        try:
            with _suppress_openai_response_serializer_warnings():
                for chunk in agent.stream(
                    {"messages": input_messages},
                    stream_mode=internal_mode,
                ):
                    (
                        accumulated,
                        last_ai_message,
                        last_stream_chunk,
                        last_finish_chunk,
                        last_output_messages,
                    ) = self._consume_stream_chunk(
                        chunk,
                        internal_mode,
                        accumulated=accumulated,
                        last_ai_message=last_ai_message,
                        last_stream_chunk=last_stream_chunk,
                        last_finish_chunk=last_finish_chunk,
                        last_output_messages=last_output_messages,
                    )
                    out = self._yield_stream_chunk(chunk, filter_messages_only)
                    if out is not None:
                        yield out
        finally:
            self._finalize_stream_append(
                thread,
                append_to,
                input_messages,
                last_output_messages,
                accumulated,
                last_ai_message,
                last_stream_chunk,
                last_finish_chunk,
            )

    def stream_events(
        self,
        thread: Thread,
        *,
        append_to: Optional[Thread] = None,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
    ) -> Iterator[StreamEvent]:
        """Stream typed events: thinking, assistant, tool_call, tool_result."""
        if self._tools_need_async(tools):
            yield from self._sync_iterate_asyncgen(
                lambda: self.astream_events(
                    thread,
                    append_to=append_to,
                    tools=tools,
                    auto_tool_call=auto_tool_call,
                ),
                tools=tools,
            )
            return

        agent = self._get_agent(tools, auto_tool_call=auto_tool_call)
        input_messages = self._prepare_messages(thread)
        internal_mode = self._events_stream_modes()
        accumulated: Optional[AIMessageChunk] = None
        last_ai_message = None
        last_stream_chunk: Optional[AIMessageChunk] = None
        last_finish_chunk: Optional[AIMessageChunk] = None
        last_output_messages: Optional[List[Any]] = None
        processor = StreamEventProcessor(input_message_count=len(input_messages))
        stream_output_count = len(input_messages)
        flush_accumulated: Optional[AIMessageChunk] = None
        flush_stream_chunk: Optional[AIMessageChunk] = None
        flush_finish_chunk: Optional[AIMessageChunk] = None
        last_stream_snapshots: tuple[str, str] = ("", "")

        try:
            with _suppress_openai_response_serializer_warnings():
                for chunk in agent.stream(
                    {"messages": input_messages},
                    stream_mode=internal_mode,
                ):
                    (
                        accumulated,
                        last_ai_message,
                        last_stream_chunk,
                        last_finish_chunk,
                        last_output_messages,
                    ) = self._consume_stream_chunk(
                        chunk,
                        internal_mode,
                        accumulated=accumulated,
                        last_ai_message=last_ai_message,
                        last_stream_chunk=last_stream_chunk,
                        last_finish_chunk=last_finish_chunk,
                        last_output_messages=last_output_messages,
                    )
                    parsed = parse_langgraph_chunk(chunk)
                    stream_snapshots = None
                    if (
                        parsed
                        and parsed[0] == "values"
                        and isinstance(parsed[1], dict)
                    ):
                        stream_snapshots = processor.peek_turn_snapshots()
                    yield from self._yield_events_from_chunk(
                        chunk,
                        processor,
                        accumulated=accumulated,
                        last_stream_chunk=last_stream_chunk,
                    )
                    if parsed and parsed[0] == "values":
                        values_messages = (
                            parsed[1].get("messages", [])
                            if isinstance(parsed[1], dict)
                            else []
                        )
                        if (
                            values_messages
                            and len(values_messages) > stream_output_count
                            and stream_snapshots is not None
                        ):
                            streamed_thinking, streamed_assistant = stream_snapshots
                            self._append_stream_values_messages(
                                thread,
                                append_to,
                                values_messages,
                                stream_output_count,
                                streamed_thinking=streamed_thinking,
                                streamed_assistant=streamed_assistant,
                            )
                            stream_output_count = len(values_messages)
                        flush_accumulated = accumulated
                        flush_stream_chunk = last_stream_chunk
                        flush_finish_chunk = last_finish_chunk
                        if stream_snapshots is not None:
                            last_stream_snapshots = stream_snapshots
                        accumulated, last_stream_chunk, last_finish_chunk = (
                            self._reset_turn_stream_state()
                        )
            yield from self._yield_flush_events(
                processor,
                accumulated=flush_accumulated,
                last_ai_message=last_ai_message,
                last_stream_chunk=flush_stream_chunk,
                last_finish_chunk=flush_finish_chunk,
            )
        finally:
            streamed_thinking, streamed_assistant = last_stream_snapshots
            self._finalize_stream_append(
                thread,
                append_to,
                input_messages,
                last_output_messages,
                flush_accumulated,
                last_ai_message,
                flush_stream_chunk,
                flush_finish_chunk,
                output_messages_start_index=stream_output_count,
                streamed_thinking=streamed_thinking,
                streamed_assistant=streamed_assistant,
            )

    async def astream(
        self,
        thread: Thread,
        stream_mode: Union[str, Sequence[str]] = "messages",
        *,
        append_to: Optional[Thread] = None,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
    ) -> AsyncIterator:
        if stream_mode == "events":
            async for event in self.astream_events(
                thread,
                append_to=append_to,
                tools=tools,
                auto_tool_call=auto_tool_call,
            ):
                yield event
            return

        agent = await self._aget_agent(tools, auto_tool_call=auto_tool_call)
        input_messages = self._prepare_messages(thread)
        internal_mode, filter_messages_only = self._stream_modes(stream_mode)
        accumulated: Optional[AIMessageChunk] = None
        last_ai_message = None
        last_stream_chunk: Optional[AIMessageChunk] = None
        last_finish_chunk: Optional[AIMessageChunk] = None
        last_output_messages: Optional[List[Any]] = None

        try:
            with _suppress_openai_response_serializer_warnings():
                async for chunk in agent.astream(
                    {"messages": input_messages},
                    stream_mode=internal_mode,
                ):
                    (
                        accumulated,
                        last_ai_message,
                        last_stream_chunk,
                        last_finish_chunk,
                        last_output_messages,
                    ) = self._consume_stream_chunk(
                        chunk,
                        internal_mode,
                        accumulated=accumulated,
                        last_ai_message=last_ai_message,
                        last_stream_chunk=last_stream_chunk,
                        last_finish_chunk=last_finish_chunk,
                        last_output_messages=last_output_messages,
                    )
                    out = self._yield_stream_chunk(chunk, filter_messages_only)
                    if out is not None:
                        yield out
        finally:
            self._finalize_stream_append(
                thread,
                append_to,
                input_messages,
                last_output_messages,
                accumulated,
                last_ai_message,
                last_stream_chunk,
                last_finish_chunk,
            )

    async def astream_events(
        self,
        thread: Thread,
        *,
        append_to: Optional[Thread] = None,
        tools: Optional[List] = None,
        auto_tool_call: Optional[bool] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async stream of typed events: thinking, assistant, tool_call, tool_result."""
        agent = await self._aget_agent(tools, auto_tool_call=auto_tool_call)
        input_messages = self._prepare_messages(thread)
        internal_mode = self._events_stream_modes()
        accumulated: Optional[AIMessageChunk] = None
        last_ai_message = None
        last_stream_chunk: Optional[AIMessageChunk] = None
        last_finish_chunk: Optional[AIMessageChunk] = None
        last_output_messages: Optional[List[Any]] = None
        processor = StreamEventProcessor(input_message_count=len(input_messages))
        stream_output_count = len(input_messages)
        flush_accumulated: Optional[AIMessageChunk] = None
        flush_stream_chunk: Optional[AIMessageChunk] = None
        flush_finish_chunk: Optional[AIMessageChunk] = None
        last_stream_snapshots: tuple[str, str] = ("", "")

        try:
            with _suppress_openai_response_serializer_warnings():
                async for chunk in agent.astream(
                    {"messages": input_messages},
                    stream_mode=internal_mode,
                ):
                    (
                        accumulated,
                        last_ai_message,
                        last_stream_chunk,
                        last_finish_chunk,
                        last_output_messages,
                    ) = self._consume_stream_chunk(
                        chunk,
                        internal_mode,
                        accumulated=accumulated,
                        last_ai_message=last_ai_message,
                        last_stream_chunk=last_stream_chunk,
                        last_finish_chunk=last_finish_chunk,
                        last_output_messages=last_output_messages,
                    )
                    parsed = parse_langgraph_chunk(chunk)
                    stream_snapshots = None
                    if (
                        parsed
                        and parsed[0] == "values"
                        and isinstance(parsed[1], dict)
                    ):
                        stream_snapshots = processor.peek_turn_snapshots()
                    for event in self._yield_events_from_chunk(
                        chunk,
                        processor,
                        accumulated=accumulated,
                        last_stream_chunk=last_stream_chunk,
                    ):
                        yield event
                    if parsed and parsed[0] == "values":
                        values_messages = (
                            parsed[1].get("messages", [])
                            if isinstance(parsed[1], dict)
                            else []
                        )
                        if (
                            values_messages
                            and len(values_messages) > stream_output_count
                            and stream_snapshots is not None
                        ):
                            streamed_thinking, streamed_assistant = stream_snapshots
                            self._append_stream_values_messages(
                                thread,
                                append_to,
                                values_messages,
                                stream_output_count,
                                streamed_thinking=streamed_thinking,
                                streamed_assistant=streamed_assistant,
                            )
                            stream_output_count = len(values_messages)
                        flush_accumulated = accumulated
                        flush_stream_chunk = last_stream_chunk
                        flush_finish_chunk = last_finish_chunk
                        if stream_snapshots is not None:
                            last_stream_snapshots = stream_snapshots
                        accumulated, last_stream_chunk, last_finish_chunk = (
                            self._reset_turn_stream_state()
                        )
            for event in self._yield_flush_events(
                processor,
                accumulated=flush_accumulated,
                last_ai_message=last_ai_message,
                last_stream_chunk=flush_stream_chunk,
                last_finish_chunk=flush_finish_chunk,
            ):
                yield event
        finally:
            streamed_thinking, streamed_assistant = last_stream_snapshots
            self._finalize_stream_append(
                thread,
                append_to,
                input_messages,
                last_output_messages,
                flush_accumulated,
                last_ai_message,
                flush_stream_chunk,
                flush_finish_chunk,
                output_messages_start_index=stream_output_count,
                streamed_thinking=streamed_thinking,
                streamed_assistant=streamed_assistant,
            )


__all__ = [
    "Agent",
    "MCPClient",
    "MCPPromptInfo",
    "MCPResourceInfo",
    "ToolCall",
    "ToolResult",
    "combine_tools",
    "StreamEvent",
    "StreamEventKind",
    "StreamEventProcessor",
    "ThinkingEvent",
    "AssistantEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "extract_thinking",
    "extract_assistant_text",
]