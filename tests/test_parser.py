from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from composer import Parser, Thread, HumanMessage, SystemMessage, AIMessage
from composer.thread import Message


class TestSchema(BaseModel):
    name: str
    value: int


class ErrorSchema(BaseModel):
    description: str
    severity: str


def _make_mock_structured(
    raw=None,
    parsed=None,
    parsing_error=None,
) -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "raw": raw,
            "parsed": parsed,
            "parsing_error": parsing_error,
        }
    )
    mock.invoke = MagicMock(
        return_value={
            "raw": raw,
            "parsed": parsed,
            "parsing_error": parsing_error,
        }
    )
    return mock


def _make_raw_message(
    content: str = "test output",
    tool_calls=None,
) -> MagicMock:
    raw = MagicMock()
    raw.content = content
    raw.tool_calls = tool_calls or []
    raw.type = "ai"
    raw.model_dump = MagicMock(return_value={"type": "ai", "content": content})
    raw.additional_kwargs = {}
    raw.response_metadata = {}
    return raw


class TestParserSuccess:
    def test_parse_returns_validated_model(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="hello", value=42),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("test input")

        assert isinstance(result, TestSchema)
        assert result.name == "hello"
        assert result.value == 42

    def test_parse_with_thread(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="threaded", value=99),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        thread = Thread()
        thread.append(SystemMessage("sys"))
        thread.append(HumanMessage("hi"))

        result = parser.parse(thread)

        assert isinstance(result, TestSchema)
        assert result.name == "threaded"
        assert result.value == 99

    def test_parse_with_list_of_messages(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="list", value=7),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        messages = [HumanMessage("hello")]
        result = parser.parse(messages)

        assert isinstance(result, TestSchema)
        assert result.name == "list"

    def test_aparse_returns_validated_model(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="async", value=123),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = asyncio.run(parser.aparse("async input"))

        assert isinstance(result, TestSchema)
        assert result.name == "async"

    def test_parse_schema_override_at_call(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            parsed=ErrorSchema(description="test error", severity="high"),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input", schema=ErrorSchema)

        assert isinstance(result, ErrorSchema)
        assert result.description == "test error"
        assert result.severity == "high"

    def test_parse_with_system_prompt(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
            system_prompt="You are an extractor.",
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="with_sys", value=1),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input")

        assert isinstance(result, TestSchema)
        assert result.name == "with_sys"

    def test_parse_system_prompt_override_at_call(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
            system_prompt="Default sys",
        )
        mock_structured = _make_mock_structured(
            parsed=TestSchema(name="override", value=2),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input", system_prompt_override="New sys")

        assert isinstance(result, TestSchema)
        assert result.name == "override"


class TestParserErrorHandling:
    def test_parse_returns_error_dict_on_validation_failure(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="raw text output"),
            parsed=None,
            parsing_error=Exception("Validation failed: missing required field"),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("bad input")

        assert isinstance(result, dict)
        assert result["error"] is True
        assert "missing required field" in result["detail"]
        assert result["output"] == "raw text output"

    def test_parse_error_dict_handles_empty_content_with_tool_calls(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        import json
        tool_calls_data = [{"name": "extract", "args": {"key": "val"}}]
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="", tool_calls=tool_calls_data),
            parsed=None,
            parsing_error=Exception("Schema mismatch"),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input")

        assert result["error"] is True
        assert json.loads(result["output"]) == tool_calls_data

    def test_parse_error_dict_handles_list_content(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        import json
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content=[{"type": "text", "text": "block"}]),
            parsed=None,
            parsing_error=Exception("Parse error"),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input")

        assert result["error"] is True
        parsed_content = json.loads(result["output"])
        assert isinstance(parsed_content, list)

    def test_parse_returns_error_dict_when_no_parsed_and_no_error(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="fallback output"),
            parsed=None,
            parsing_error=None,
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = parser.parse("input")

        assert result["error"] is True
        assert result["detail"] == "No parsed output returned by model"
        assert result["output"] == "fallback output"

    def test_parse_no_schema_raises_error(self):
        parser = Parser(model=MagicMock())

        with pytest.raises(ValueError, match="Parser requires a schema"):
            parser.parse("input")

    def test_aparse_returns_error_dict_on_validation_failure(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="raw"),
            parsed=None,
            parsing_error=Exception("Async validation error"),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        result = asyncio.run(parser.aparse("input"))

        assert result["error"] is True
        assert "Async validation error" in result["detail"]


class TestParserRecordOutput:
    def test_record_output_appends_ai_message_to_thread(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="extracted content"),
            parsed=TestSchema(name="recorded", value=10),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        thread = Thread()
        result = parser.parse(
            thread,
            record_output=True,
            append_to=thread,
        )

        assert isinstance(result, TestSchema)
        assert len(thread.thread) == 1
        assert thread.thread[0].content == "extracted content"

    def test_record_output_skipped_when_false(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        mock_structured = _make_mock_structured(
            raw=_make_raw_message(content="skip this"),
            parsed=TestSchema(name="skipped", value=20),
        )
        parser._build_structured = MagicMock(return_value=mock_structured)

        thread = Thread()
        result = parser.parse(
            thread,
            record_output=False,
            append_to=thread,
        )

        assert isinstance(result, TestSchema)
        assert len(thread.thread) == 0


class TestParserGetModel:
    def test_get_model_with_string_model(self):
        parser = Parser(
            provider="custom",
            model="gpt-4",
            base_url="https://api.example.com",
            api_key="test-key",
            reasoning={"effort": "medium"},
        )
        model = parser._get_model()

        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4"
        assert model.openai_api_base == "https://api.example.com"
        assert model.openai_api_key.get_secret_value() == "test-key"

    def test_get_model_with_chatopenai_instance(self):
        chat_model = ChatOpenAI(model="gpt-3.5", api_key="test-key")
        parser = Parser(model=chat_model, schema=TestSchema)

        model = parser._get_model()

        assert model is chat_model

    def test_get_model_with_reasoning_true(self):
        parser = Parser(
            provider="custom",
            model="test",
            api_key="test-key",
            reasoning=True,
        )
        model = parser._get_model()

        assert isinstance(model, ChatOpenAI)
        assert model.extra_body.get("reasoning") == {
            "enabled": True
        }

    def test_get_model_with_reasoning_dict(self):
        parser = Parser(
            provider="custom",
            model="test",
            api_key="test-key",
            reasoning={"effort": "low"},
        )
        model = parser._get_model()

        assert model.extra_body.get("reasoning") == {
            "effort": "low"
        }


class TestParserMessagesPreparation:
    def test_prepare_messages_str(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
            system_prompt="System",
        )
        messages = parser._prepare_messages("user text")

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == "System"
        assert messages[1].content == "user text"

    def test_prepare_messages_str_no_system_prompt(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        messages = parser._prepare_messages("user text")

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)

    def test_prepare_messages_thread_keeps_messages_for_model(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
            system_prompt="Override",
        )
        thread = Thread()
        thread.append(SystemMessage("existing sys"))
        thread.append(HumanMessage("msg1"))

        messages = parser._prepare_messages(thread)

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Override"

    def test_prepare_messages_thread_no_system_prompt(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
        )
        thread = Thread()
        thread.append(SystemMessage("keep sys"))
        thread.append(HumanMessage("msg"))

        messages = parser._prepare_messages(thread)

        assert len(messages) == 2
        assert messages[0].content == "keep sys"

    def test_prepare_messages_list(self):
        parser = Parser(
            model=MagicMock(),
            schema=TestSchema,
            system_prompt="sys",
        )
        messages = parser._prepare_messages(
            [HumanMessage("a"), HumanMessage("b")]
        )

        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
