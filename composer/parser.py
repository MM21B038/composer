from __future__ import annotations

import asyncio
import inspect
import queue
import threading
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from langchain_core.messages import AIMessage as LangChainAIMessage
from langchain_core.messages import HumanMessage as LangChainHumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from composer.stream import (
    extract_assistant_text,
    message_model_dump,
    normalize_ai_message_dump,
)
from composer.thread import (
    AIMessage,
    HumanMessage,
    Message as ComposerMessage,
    SystemMessage,
    Thread,
)

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = Any  # type: ignore[misc,assignment]

T = TypeVar("T", bound=BaseModel)  # type: ignore[misc]


def _in_running_loop() -> bool:
    try:
        return asyncio.get_running_loop().is_running()
    except RuntimeError:
        return False


class Parser:
    def __init__(
        self,
        provider: Optional[str] = "custom",
        model: Optional[Union[str, BaseChatModel]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
        schema: Optional[Type[BaseModel]] = None,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.reasoning = reasoning
        self.schema = schema
        self.method = method

    def _get_model(self) -> BaseChatModel:
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

        raise NotImplementedError(f"Provider {self.provider} is not supported yet.")

    def _prepare_messages(
        self,
        input_obj: Union[Thread, List[ComposerMessage], str],
        *,
        system_prompt_override: Optional[str] = None,
    ) -> List[ComposerMessage]:
        active_prompt = (
            system_prompt_override if system_prompt_override is not None else self.system_prompt
        )

        if isinstance(input_obj, str):
            messages: List[ComposerMessage] = [HumanMessage(input_obj)]
        elif isinstance(input_obj, Thread):
            messages = input_obj.messages_for_model()
        else:
            messages = list(input_obj)

        if active_prompt:
            if messages and isinstance(messages[0], SystemMessage):
                messages = messages[1:]
            messages.insert(0, SystemMessage(active_prompt))

        return messages

    def _to_ai_message(self, message: BaseMessage) -> AIMessage:
        dump = message_model_dump(message)
        if dump.get("type") in ("AIMessageChunk",):
            dump["type"] = "ai"
        dump = normalize_ai_message_dump(message, dump)
        return AIMessage.model_validate(dump)

    def _build_structured(self, schema: Type[BaseModel]) -> Any:
        model = self._get_model()
        return model.with_structured_output(
            schema,
            method=self.method,
            include_raw=True,
        )

    def _sync_run_async(self, coro) -> Any:
        if not _in_running_loop():
            return asyncio.run(coro)

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result_queue.put(("ok", asyncio.run(coro)))
            except Exception as exc:
                result_queue.put(("err", exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        kind, payload = result_queue.get()
        thread.join()
        if kind == "err":
            raise payload
        return payload

    async def aparse(
        self,
        input_obj: Union[Thread, List[ComposerMessage], str],
        *,
        schema: Optional[Type[BaseModel]] = None,
        append_to: Optional[Thread] = None,
        record_output: bool = False,
        system_prompt_override: Optional[str] = None,
    ) -> Union[BaseModel, Dict[str, Any]]:
        resolved_schema = schema or self.schema
        if resolved_schema is None:
            raise ValueError(
                "Parser requires a schema. Pass schema=MyModel to constructor or to parse()."
            )

        messages = self._prepare_messages(input_obj, system_prompt_override=system_prompt_override)
        structured = self._build_structured(resolved_schema)

        response = await structured.ainvoke(messages)

        raw = response.get("raw")
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")

        if record_output and append_to is not None and raw is not None:
            ai_msg = self._to_ai_message(raw)
            append_to.append(ai_msg)

        if parsing_error is not None and parsed is None:
            output_text = _extract_output_text(raw)
            return {
                "error": True,
                "detail": str(parsing_error),
                "output": output_text,
            }

        if parsed is not None:
            return parsed

        output_text = _extract_output_text(raw)
        return {
            "error": True,
            "detail": "No parsed output returned by model",
            "output": output_text,
        }

    def parse(
        self,
        input_obj: Union[Thread, List[ComposerMessage], str],
        *,
        schema: Optional[Type[BaseModel]] = None,
        append_to: Optional[Thread] = None,
        record_output: bool = False,
        system_prompt_override: Optional[str] = None,
    ) -> Union[BaseModel, Dict[str, Any]]:
        return self._sync_run_async(
            self.aparse(
                input_obj,
                schema=schema,
                append_to=append_to,
                record_output=record_output,
                system_prompt_override=system_prompt_override,
            ),
        )


def _extract_output_text(raw: BaseMessage) -> str:
    import json

    content = getattr(raw, "content", "")
    if isinstance(content, list):
        return json.dumps(content, default=str)
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        return json.dumps(tool_calls, default=str)
    text = extract_assistant_text(raw)
    if text:
        return text
    if isinstance(content, str):
        return content
    return str(content)
