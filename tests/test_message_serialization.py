import warnings

from langchain_core.messages.ai import AIMessageChunk

from composer import AIMessage, HumanMessage, SystemMessage, ToolMessage
from composer.stream import (
    AssistantEvent,
    StreamEventProcessor,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    extract_assistant_text,
    extract_thinking,
    message_model_dump,
    normalize_ai_message_dump,
    sanitize_chunk_for_merge,
)


class _FakeReasoningItem:
    def model_dump(self, mode="json", warnings=True):
        return {
            "id": "rs_test",
            "type": "reasoning",
            "content": [
                {
                    "text": "thinking",
                    "type": "output_text",
                    "annotations": [],
                }
            ],
            "status": "completed",
            "role": "assistant",
        }


def test_message_model_dump_json_safe_reasoning_blocks():
    message = AIMessage(content="placeholder")
    message.content = [
        _FakeReasoningItem(),
        {"type": "text", "text": "hello", "annotations": [], "id": "msg_1"},
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dump = message_model_dump(message)

    assert not caught
    assert dump["content"][0]["type"] == "reasoning"
    assert dump["content"][0]["content"][0]["type"] == "output_text"
    assert dump["content"][1]["text"] == "hello"


def test_composer_ai_message_populates_content_from_reasoning_text():
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "status": "in_progress",
                "summary": [{"type": "summary_text", "text": "Let me think"}],
                "text": "Hello! How can I help?",
            },
            "reasoning_content": "Let me think",
        },
    )

    assert message.content == "Hello! How can I help?"
    assert "reasoning" not in message.additional_kwargs
    assert message.additional_kwargs.get("reasoning_content") == "Let me think"


def test_normalize_ai_message_dump_uses_string_content():
    message = AIMessage(content="placeholder")
    message.content = [
        {
            "id": "rs_test",
            "type": "reasoning",
            "content": [{"text": "thinking", "type": "output_text", "annotations": []}],
            "status": "completed",
        },
        {"type": "text", "text": "hello", "annotations": [], "id": "msg_1"},
    ]
    dump = normalize_ai_message_dump(message, message_model_dump(message))

    assert dump["content"] == "hello"
    assert dump["additional_kwargs"]["reasoning_content"] == "thinking"
    assert "reasoning" not in dump["additional_kwargs"]


def test_normalize_ai_message_dump_uses_reasoning_text_when_content_empty():
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [
                    {"type": "summary_text", "text": "Let me think"},
                ],
                "text": "Here is the answer.",
            },
            "reasoning_content": "Let me think",
        },
    )
    dump = normalize_ai_message_dump(message, message_model_dump(message))

    assert dump["content"] == "Here is the answer."
    assert dump["additional_kwargs"]["reasoning_content"] == "Let me think"
    assert "reasoning" not in dump["additional_kwargs"]


def test_to_ai_message_strips_reasoning_blob():
    from composer.agent import Agent

    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "text": "answer",
            }
        },
    )
    agent = Agent.__new__(Agent)
    converted = agent._to_ai_message(message)

    assert converted.content == "answer"
    assert converted.additional_kwargs.get("reasoning_content") == "thinking"
    assert "reasoning" not in converted.additional_kwargs


def test_buffer_beats_stub_reasoning_on_values_ai():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})
    processor._turn_thinking_buffer = "Need to use the browser for this step"
    processor._thinking_len = 0

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning_content": "Need",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need"}],
            },
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )

    thinking = "".join(event.text for event in events if isinstance(event, ThinkingEvent))
    assert "browser" in thinking
    event_types = [type(event).__name__ for event in events]
    assert event_types.index("ThinkingEvent") < event_types.index("ToolCallEvent")


def test_custom_model_chunk_routes_to_thinking_events():
    from composer.stream import parse_custom_model_chunk

    processor = StreamEventProcessor()
    chunk = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "Need to snap"},
    )
    payload = {"kind": "model_chunk", "chunk": chunk}
    assert parse_custom_model_chunk(payload) is chunk

    events = processor.process_message_chunk(parse_custom_model_chunk(payload), {})
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "Need to snap"
    ]


def test_persistence_round_trip_compacts_ai_message():
    from composer.persistence.serializers import message_to_payload, payload_to_message

    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "text": "Hello",
            }
        },
    )
    payload = message_to_payload(message)
    restored = payload_to_message(payload)

    assert restored.content == "Hello"
    assert restored.additional_kwargs.get("reasoning_content") == "thinking"
    assert "reasoning" not in restored.additional_kwargs


def test_join_summary_fragments_handles_merge_duplication():
    from composer.stream import _join_summary_fragments

    assert _join_summary_fragments(["The", "The", " user"]) == "The user"
    assert _join_summary_fragments(["Let", " me", " think"]) == "Let me think"


def test_dedupe_cumulative_stream_text_handles_reasoning_content_merge():
    from composer.stream import dedupe_cumulative_stream_text

    assert dedupe_cumulative_stream_text("LetLet me") == "Let me"
    assert dedupe_cumulative_stream_text("LetLet meLet me think") == "Let me think"
    assert dedupe_cumulative_stream_text("\n\nHi there!") == "\n\nHi there!"


def test_reasoning_content_streams_incrementally():
    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let me"}),
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "Let me think"},
        ),
    ]
    thinking = "".join(
        event.text
        for chunk in chunks
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "Let me think"


def test_thinking_from_merged_accum_preserves_delta_tokens():
    processor = StreamEventProcessor()
    thinking = "".join(
        event.text
        for chunk in [
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let me"}),
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": " start"}),
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": " by"}),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "Let me start by"


def test_reasoning_snapshot_preserves_double_newlines():
    processor = StreamEventProcessor()
    thinking = "".join(
        event.text
        for chunk in [
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "\n\n"}),
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "\n\nPlan the login flow."},
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "\n\nPlan the login flow."


def test_assistant_response_preserves_markdown_and_newlines():
    processor = StreamEventProcessor()
    final = (
        "Here's the **complete summary** of everything done:\n\n"
        "---\n\n"
        "## ✅ Task Completed\n\n"
        "1. Go to `https://example.com/login`\n"
        "2. Fill out the form"
    )
    assistant = "".join(
        event.text
        for chunk in [
            AIMessageChunk(content="Here's the **complete"),
            AIMessageChunk(content="Here's the **complete summary** of"),
            AIMessageChunk(
                content=final,
                response_metadata={"finish_reason": "stop"},
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, AssistantEvent)
    )

    assert assistant == final
    assert "\n\n---\n\n" in assistant
    assert "**complete summary**" in assistant


def test_single_values_update_emits_thinking():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})
    final_ai = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "text": "answer",
            },
            "reasoning_content": "thinking",
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), final_ai]}
    )

    assert [e.text for e in events if isinstance(e, ThinkingEvent)] == ["thinking"]


def test_extract_assistant_text_ignores_reasoning_list_content():
    message = AIMessage(
        content=[
            {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            }
        ]
    )

    assert extract_assistant_text(message) == ""
    assert extract_thinking(message) == "thinking"


def test_stream_processor_emits_clean_reasoning_deltas():
    processor = StreamEventProcessor()
    first = AIMessageChunk(
        content=[
            {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "index": 0,
            }
        ]
    )
    second = AIMessageChunk(
        content=[
            {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "index": 0,
                "summary": [{"type": "summary_text", "text": "The"}],
            }
        ]
    )
    third = AIMessageChunk(
        content=[
            {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "index": 0,
                "summary": [
                    {"type": "summary_text", "text": "The"},
                    {"type": "summary_text", "text": " user"},
                ],
            }
        ]
    )

    first_events = processor.process_message_chunk(first)
    second_events = processor.process_message_chunk(second)
    third_events = processor.process_message_chunk(third)

    assert first_events == []
    thinking_text = "".join(
        event.text for event in second_events + third_events if isinstance(event, ThinkingEvent)
    )
    assistant_text = "".join(
        event.text
        for event in second_events + third_events
        if isinstance(event, AssistantEvent)
    )

    assert thinking_text == "The user"
    assert assistant_text == ""


def test_stream_processor_emits_response_after_reasoning_text():
    processor = StreamEventProcessor()
    thinking_chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [{"type": "summary_text", "text": "Let me think"}],
            }
        }
    )
    response_chunk = AIMessageChunk(
        content="Final answer.",
        response_metadata={"finish_reason": "stop"},
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "Let me think"}],
                "text": "Final answer.",
            },
            "reasoning_content": "Let me think",
        }
    )

    thinking_events = processor.process_message_chunk(thinking_chunk)
    response_events = processor.process_message_chunk(response_chunk)

    assert [event.text for event in thinking_events if isinstance(event, ThinkingEvent)] == [
        "Let me think"
    ]
    assert [
        event.text for event in response_events if isinstance(event, AssistantEvent)
    ] == ["Final answer."]


def test_values_update_emits_assistant_from_reasoning_text():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    final_ai = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "Let me think"}],
                "text": "Here is the answer.",
            },
            "reasoning_content": "Let me think",
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), final_ai]}
    )

    thinking = [e.text for e in events if isinstance(e, ThinkingEvent)]
    assistant = [e.text for e in events if isinstance(e, AssistantEvent)]

    assert thinking == ["Let me think"]
    assert assistant == ["Here is the answer."]


def test_values_baseline_ignores_prepended_system_message():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update(
        {"messages": [SystemMessage("sys"), HumanMessage("hi")]}
    )

    final_ai = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "text": "done",
            },
            "reasoning_content": "thinking",
        },
    )
    events = processor.process_values_update(
        {
            "messages": [
                SystemMessage("sys"),
                HumanMessage("hi"),
                final_ai,
            ]
        }
    )

    assert [e.text for e in events if isinstance(e, AssistantEvent)] == ["done"]


def test_values_stub_reasoning_does_not_reemit_streamed_thinking():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    full = (
        "The user wants me to solve a reCAPTCHA v2 challenge. "
        "Let me start by creating a session and going to the URL"
    )
    streamed = ""
    for event in processor.process_message_chunk(
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": full})
    ):
        if isinstance(event, ThinkingEvent):
            streamed += event.text

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={"reasoning_content": "The user wants"},
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    extra = "".join(event.text for event in events if isinstance(event, ThinkingEvent))

    assert extra == ""
    assert streamed.count("The user wants me to solve") == 1


def test_values_recovers_tool_turn_thinking_after_blocked_stream_length():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})
    processor._thinking_len = 50

    tool_ai = AIMessage(
        content="",
        id="ai_tool_1",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning": {
                "id": "rs_tool",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "use tool"}],
            },
            "reasoning_content": "use tool",
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )

    assert [e.text for e in events if isinstance(e, ThinkingEvent)] == ["use tool"]


def test_multi_turn_tool_and_final_thinking_both_emit():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    tool_ai = AIMessage(
        content="",
        id="ai_tool_1",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning_content": "use tool",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "use tool"}],
            },
        },
    )
    tool_events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    processor.process_values_update(
        {
            "messages": [
                HumanMessage("hi"),
                tool_ai,
                ToolMessage(content="ok", tool_call_id="call_1", name="snap"),
            ]
        }
    )

    final_ai = AIMessage(
        content="",
        id="ai_final_1",
        additional_kwargs={
            "reasoning_content": "summarize",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "summarize"}],
                "text": "Final response.",
            },
        },
    )
    final_events = processor.process_values_update(
        {
            "messages": [
                HumanMessage("hi"),
                tool_ai,
                ToolMessage(content="ok", tool_call_id="call_1", name="snap"),
                final_ai,
            ]
        }
    )

    assert [e.text for e in tool_events if isinstance(e, ThinkingEvent)] == ["use tool"]
    assert [e.text for e in final_events if isinstance(e, ThinkingEvent)] == ["summarize"]
    assert [e.text for e in final_events if isinstance(e, AssistantEvent)] == [
        "Final response."
    ]


def test_tool_call_deferred_until_values_emits_thinking_first():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    chunk = AIMessageChunk(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        response_metadata={"finish_reason": "tool_calls"},
    )
    chunk_events = processor.process_message_chunk(chunk)
    assert not any(isinstance(e, ToolCallEvent) for e in chunk_events)

    tool_ai = AIMessage(
        content="",
        id="ai_tool_1",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning_content": "Need to snap",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need to snap"}],
            },
        },
    )
    values_events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    event_types = [type(e).__name__ for e in values_events]
    assert event_types.index("ThinkingEvent") < event_types.index("ToolCallEvent")
    assert [e.text for e in values_events if isinstance(e, ThinkingEvent)] == [
        "Need to snap"
    ]
    assert [
        e.call.name for e in values_events if isinstance(e, ToolCallEvent)
    ] == ["snap"]


def test_incremental_thinking_then_values_emits_tool_call_only():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    thinking_chunk = AIMessageChunk(
        content="",
        id="ai_tool_1",
        additional_kwargs={"reasoning_content": "Need to snap"},
    )
    tool_chunk = AIMessageChunk(
        content="",
        id="ai_tool_1",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        response_metadata={"finish_reason": "tool_calls"},
    )
    chunk_events = []
    chunk_events.extend(processor.process_message_chunk(thinking_chunk))
    chunk_events.extend(processor.process_message_chunk(tool_chunk))

    assert [e.text for e in chunk_events if isinstance(e, ThinkingEvent)] == [
        "Need to snap"
    ]
    assert not any(isinstance(e, ToolCallEvent) for e in chunk_events)

    tool_ai = AIMessage(
        content="",
        id="ai_tool_1",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning_content": "Need to snap",
            "reasoning": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need to snap"}],
            },
        },
    )
    values_events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    assert not any(isinstance(e, ThinkingEvent) for e in values_events)
    assert [
        e.call.name for e in values_events if isinstance(e, ToolCallEvent)
    ] == ["snap"]


def test_multi_turn_values_emits_final_response_only():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning": {
                "id": "rs_tool",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "use tool"}],
            },
            "reasoning_content": "use tool",
        },
    )
    tool_events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    tool_result = ToolMessage(content="ok", tool_call_id="call_1", name="snap")
    processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai, tool_result]}
    )

    final_ai = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_final",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "summarize"}],
                "text": "Final response.",
            },
            "reasoning_content": "summarize",
        },
    )
    final_events = processor.process_values_update(
        {
            "messages": [
                HumanMessage("hi"),
                tool_ai,
                tool_result,
                final_ai,
            ]
        }
    )

    assert [e.text for e in tool_events if isinstance(e, ThinkingEvent)] == ["use tool"]
    assert [e.text for e in tool_events if isinstance(e, AssistantEvent)] == []
    event_types = [type(e).__name__ for e in tool_events]
    assert event_types[:1] == ["ThinkingEvent"]
    assert "ToolCallEvent" in event_types
    assert [type(e).__name__ for e in tool_events if isinstance(e, ToolResultEvent)] == []

    assert [e.text for e in final_events if isinstance(e, ThinkingEvent)] == ["summarize"]
    assert [e.text for e in final_events if isinstance(e, AssistantEvent)] == [
        "Final response."
    ]


def test_flush_emits_remaining_assistant_text():
    processor = StreamEventProcessor()
    partial = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [{"type": "summary_text", "text": "Let me"}],
            }
        }
    )
    processor.process_message_chunk(partial)

    final = AIMessage(
        content="",
        additional_kwargs={
            "reasoning": {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "Let me think"}],
                "text": "Complete answer.",
            },
            "reasoning_content": "Let me think",
        },
    )
    flushed = processor.flush(final)

    thinking = "".join(e.text for e in flushed if isinstance(e, ThinkingEvent))
    assistant = "".join(e.text for e in flushed if isinstance(e, AssistantEvent))

    assert thinking == " think"
    assert assistant == "Complete answer."


def test_sanitize_chunk_for_merge_allows_different_created_at():
    first = AIMessageChunk(
        content="",
        response_metadata={"created_at": 1.0, "model": "test"},
    )
    second = AIMessageChunk(
        content="hi",
        response_metadata={"created_at": 2.0, "model": "test"},
    )
    merged = sanitize_chunk_for_merge(first) + sanitize_chunk_for_merge(second)
    assert merged.content


def test_process_message_chunk_merges_created_at_metadata():
    processor = StreamEventProcessor(input_message_count=1)
    first = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "think"},
        response_metadata={"created_at": 1.0, "model": "test"},
    )
    second = AIMessageChunk(
        content="answer",
        response_metadata={"created_at": 2.0, "model": "test", "finish_reason": "stop"},
    )

    processor.process_message_chunk(first)
    events = processor.process_message_chunk(second)

    assert any(isinstance(event, AssistantEvent) for event in events)
    assert "".join(
        event.text for event in events if isinstance(event, AssistantEvent)
    ) == "answer"


def test_merge_turn_accumulation_ignores_doubled_ai_accum():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    plan = "The user wants me to solve a recaptcha."
    streamed = ""
    for part in ["The user", " wants me to solve a recaptcha."]:
        for event in processor.process_message_chunk(
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": part})
        ):
            if isinstance(event, ThinkingEvent):
                streamed += event.text

    doubled_accum = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning_content": streamed + plan,
        },
    )
    processor.merge_turn_accumulation(doubled_accum)

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={"reasoning_content": plan},
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    extra = "".join(event.text for event in events if isinstance(event, ThinkingEvent))

    assert extra == ""
    assert streamed.count("The user") == 1


def test_merge_stream_text_delta_cumulative_markdown():
    from composer.stream import merge_stream_text_delta

    snap, delta = merge_stream_text_delta(
        "Here's the **complete",
        "Here's the **complete summary of everything done",
    )
    assert snap == "Here's the **complete summary of everything done"
    assert delta == " summary of everything done"


def test_merge_stream_text_delta_numbered_list():
    from composer.stream import merge_stream_text_delta

    snap = ""
    parts = [
        "The user wants me to:\n1. Go ",
        "The user wants me to:\n1. Go to https://example.com/login\n2. Analyze",
    ]
    out = ""
    for part in parts:
        snap, delta = merge_stream_text_delta(snap, part)
        out += delta
    assert out == parts[-1]
    assert "\n2. Analyze" in out


def test_merge_stream_text_delta_overlap_incremental():
    from composer.stream import merge_stream_text_delta

    snap, delta = merge_stream_text_delta("**complete summary", " summary of done")
    assert snap == "**complete summary of done"
    assert delta == " of done"


def test_merge_stream_text_delta_suppresses_duplicate_block():
    from composer.stream import dedupe_cumulative_stream_text, merge_stream_text_delta

    plan = "The user wants me to solve a recaptcha. Let me start."
    snap, delta = merge_stream_text_delta(plan, plan)
    assert delta == ""
    deduped = plan + plan
    snap, delta = merge_stream_text_delta(
        plan, dedupe_cumulative_stream_text(deduped)
    )
    assert delta == ""


def test_assistant_streams_markdown_on_stop():
    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="Here's the **complete"),
        AIMessageChunk(content="Here's the **complete summary"),
        AIMessageChunk(
            content="Here's the **complete summary of everything done:\n\n---\n\n## ✅ Task",
            response_metadata={"finish_reason": "stop"},
        ),
    ]
    assistant = ""
    for chunk in chunks:
        for event in processor.process_message_chunk(chunk):
            if isinstance(event, AssistantEvent):
                assistant += event.text
    assert assistant == chunks[-1].content


def test_incremental_thinking_ignores_cumulative_restream():
    processor = StreamEventProcessor()
    out = ""
    parts = [
        "The user wants me to:",
        " Go to https://2captcha.com/demo/recaptcha-v2",
        " Analyze the webpage",
        "Let me start by creating a session and going to the URL",
    ]
    for part in parts:
        for event in processor.process_message_chunk(
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": part})
        ):
            if isinstance(event, ThinkingEvent):
                out += event.text

    full_restart = """The user wants me to:
1. Go to https://2captcha.com/demo/recaptcha-v2
2. Analyze the webpage
Let me start by creating a session and going to the URL"""
    for event in processor.process_message_chunk(
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": full_restart})
    ):
        if isinstance(event, ThinkingEvent):
            out += event.text

    assert out.count("The user wants me to:") == 1


def test_streamed_thinking_not_repeated_on_values_id_change():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    thinking = "The user wants me to solve a recaptcha."
    for part in ["The", " user", " wants me to solve a recaptcha."]:
        processor.process_message_chunk(
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": part},
            )
        )

    tool_ai = AIMessage(
        content="",
        id="resp_tool_turn",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={"reasoning_content": thinking},
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )

    assert [e.text for e in events if isinstance(e, ThinkingEvent)] == []
    assert [e.call.name for e in events if isinstance(e, ToolCallEvent)] == ["snap"]
    assert not any(isinstance(e, AssistantEvent) for e in events)


def test_tool_turn_suppresses_streamed_assistant_preamble():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    events = []
    events.extend(
        processor.process_message_chunk(
            AIMessageChunk(content="I'll help you solve this.")
        )
    )
    events.extend(
        processor.process_message_chunk(
            AIMessageChunk(
                content="",
                tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
                response_metadata={"finish_reason": "tool_calls"},
            )
        )
    )

    assert not any(isinstance(e, AssistantEvent) for e in events)


def test_agent_routes_model_chunks_only_from_custom_stream():
    from composer.agent import Agent

    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "The"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": " user"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": " wants"}),
    ]
    agent = Agent.__new__(Agent)
    thinking = ""
    for chunk in chunks:
        for event in agent._yield_events_from_chunk(
            ("messages", (chunk, {})),
            processor,
        ):
            if isinstance(event, ThinkingEvent):
                thinking += event.text
        for event in agent._yield_events_from_chunk(
            ("custom", {"kind": "model_chunk", "chunk": chunk}),
            processor,
        ):
            if isinstance(event, ThinkingEvent):
                thinking += event.text

    assert thinking == "The user wants"


def test_incremental_thinking_survives_double_chunk_processing():
    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "The"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": " user"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": " wants"}),
    ]
    thinking = ""
    for chunk in chunks:
        for _ in range(2):
            for event in processor.process_message_chunk(chunk):
                if isinstance(event, ThinkingEvent):
                    thinking += event.text

    assert thinking == "The user wants"


def test_cumulative_reasoning_content_streams_cleanly():
    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "The"}),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "The user"}),
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "The user wants"},
        ),
    ]
    thinking = "".join(
        event.text
        for chunk in chunks
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "The user wants"


def test_values_emits_assistant_after_streamed_thinking_only():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})
    for chunk in [
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "The user said hi."},
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={
                "reasoning_content": "The user said hi. I should greet them."
            },
        ),
    ]:
        processor.process_message_chunk(chunk)

    final = AIMessage(
        content="\n\nHi there! How can I help?",
        additional_kwargs={
            "reasoning_content": "The user said hi. I should greet them."
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), final]}
    )
    assistant = "".join(
        event.text for event in events if isinstance(event, AssistantEvent)
    )

    assert assistant == "\n\nHi there! How can I help?"


def test_cumulative_assistant_content_streams_cleanly():
    processor = StreamEventProcessor()
    chunks = [
        AIMessageChunk(content="\n\nHi there!"),
        AIMessageChunk(
            content="\n\nHi there! How can I help you today?",
            response_metadata={"finish_reason": "stop"},
        ),
    ]
    assistant = "".join(
        event.text
        for chunk in chunks
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, AssistantEvent)
    )

    assert assistant == "\n\nHi there! How can I help you today?"


def test_streaming_reasoning_text_does_not_block_final_assistant():
    processor = StreamEventProcessor()
    partial = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "status": "in_progress",
                "text": "Hi there partial",
                "summary": [{"type": "summary_text", "text": "think"}],
            }
        },
    )
    processor.process_message_chunk(partial)

    final = AIMessage(
        content="Hi there!",
        additional_kwargs={"reasoning_content": "think"},
    )
    flushed = processor.flush(final)
    assistant = "".join(
        event.text for event in flushed if isinstance(event, AssistantEvent)
    )

    assert assistant == "Hi there!"


def test_assistant_emits_on_chunk_position_last():
    processor = StreamEventProcessor()
    assistant = "".join(
        event.text
        for chunk in [
            AIMessageChunk(
                content=[{"type": "output_text", "text": "Streamed"}],
            ),
            AIMessageChunk(
                content=[{"type": "output_text", "text": "Streamed answer"}],
                response_metadata={"chunk_position": "last"},
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, AssistantEvent)
    )

    assert assistant == "Streamed answer"


def test_assistant_emits_reasoning_text_on_complete():
    processor = StreamEventProcessor()
    assistant = "".join(
        event.text
        for chunk in [
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning": {
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [{"type": "summary_text", "text": "think"}],
                        "text": "Hello",
                    }
                },
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning": {
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [{"type": "summary_text", "text": "think"}],
                        "text": "Hello world",
                    }
                },
                response_metadata={"status": "completed"},
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, AssistantEvent)
    )

    assert assistant == "Hello world"


def test_flush_merges_ai_accum_when_values_stub():
    processor = StreamEventProcessor()
    processor.process_message_chunk(
        AIMessageChunk(
            content=[{"type": "output_text", "text": "Final answer"}],
        )
    )
    stub = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "."},
    )
    flushed = processor.flush(stub)
    assistant = "".join(
        event.text for event in flushed if isinstance(event, AssistantEvent)
    )

    assert assistant == "Final answer"


def test_multi_turn_final_response_after_tools_responses_api():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
        additional_kwargs={
            "reasoning": {
                "id": "rs_tool",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "use tool"}],
            },
            "reasoning_content": "use tool",
        },
    )
    processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    processor.process_values_update(
        {
            "messages": [
                HumanMessage("hi"),
                tool_ai,
                ToolMessage(content="ok", tool_call_id="call_1", name="snap"),
            ]
        }
    )

    final_events = list(
        processor.process_message_chunk(
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning": {
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [{"type": "summary_text", "text": "summarize"}],
                        "text": "Task complete.",
                    }
                },
                response_metadata={"status": "completed"},
            )
        )
    )

    assistant = "".join(
        event.text for event in final_events if isinstance(event, AssistantEvent)
    )
    assert assistant == "Task complete."


def test_dedupe_preserves_ip_and_numeric_tokens():
    from composer.stream import dedupe_cumulative_stream_text

    for value in ("11", "117", "112.114", "10.10.112.114"):
        assert dedupe_cumulative_stream_text(value) == value


def test_dedupe_still_fixes_langchain_merge():
    from composer.stream import dedupe_cumulative_stream_text

    assert dedupe_cumulative_stream_text("LetLet me") == "Let me"
    assert dedupe_cumulative_stream_text("LetLet meLet me think") == "Let me think"


def test_thinking_streams_ip_address_incrementally():
    processor = StreamEventProcessor()
    thinking = "".join(
        event.text
        for chunk in [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "10.10.11"},
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "10.10.112.114"},
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "10.10.112.114"


def test_thinking_streams_117_not_17():
    processor = StreamEventProcessor()
    thinking = "".join(
        event.text
        for chunk in [
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "11"}),
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "117"}),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == "117"


def test_merge_stream_text_delta_no_tail_reemit():
    from composer.stream import merge_stream_text_delta

    shared = "x" * 20
    previous = f"{shared}10.10.112.1"
    incoming = f"{shared}10.10.117"
    snap, delta = merge_stream_text_delta(previous, incoming)

    assert snap == incoming
    assert delta == "7"
    assert "112.1" in previous
    assert "117" in snap


def test_cumulative_thinking_prefix_extension_streams_incrementally():
    processor = StreamEventProcessor()
    thinking = "".join(
        event.text
        for chunk in [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "We need to perform IDOR (Insecure"},
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "reasoning_content": (
                        "We need to perform IDOR (Insecure Direct Object Reference)"
                    )
                },
            ),
        ]
        for event in processor.process_message_chunk(chunk)
        if isinstance(event, ThinkingEvent)
    )

    assert thinking == (
        "We need to perform IDOR (Insecure Direct Object Reference)"
    )


def test_tool_calls_flushes_unemitted_thinking_gap():
    processor = StreamEventProcessor()
    processor.process_message_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "We need to perform IDOR (Insecure"},
        )
    )
    processor._reasoning_snapshot = (
        "We need to perform IDOR (Insecure Direct Object Reference)"
    )
    processor._thinking_len = len("We need to perform IDOR (Insecure")

    events = processor.process_message_chunk(
        AIMessageChunk(
            content="",
            tool_calls=[{"id": "c1", "name": "http", "args": {}}],
            response_metadata={"finish_reason": "tool_calls"},
        )
    )
    thinking = "".join(
        event.text for event in events if isinstance(event, ThinkingEvent)
    )

    assert thinking == " Direct Object Reference)"


def test_values_emits_remaining_tool_turn_thinking_after_partial_stream():
    processor = StreamEventProcessor(input_message_count=1)
    processor.process_values_update({"messages": [HumanMessage("hi")]})
    processor.process_message_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "We need to perform IDOR (Insecure"},
        )
    )

    tool_ai = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "http", "args": {}}],
        additional_kwargs={
            "reasoning_content": (
                "We need to perform IDOR (Insecure Direct Object Reference)"
            )
        },
    )
    events = processor.process_values_update(
        {"messages": [HumanMessage("hi"), tool_ai]}
    )
    thinking = "".join(
        event.text for event in events if isinstance(event, ThinkingEvent)
    )

    assert thinking == " Direct Object Reference)"


def test_assistant_streams_incrementally_after_reasoning():
    processor = StreamEventProcessor()
    processor.process_message_chunk(
        AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "Let me answer."},
        )
    )
    deltas = []
    for chunk in [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(
            content="Hello world",
            response_metadata={"finish_reason": "stop"},
        ),
    ]:
        for event in processor.process_message_chunk(chunk):
            if isinstance(event, AssistantEvent):
                deltas.append(event.text)

    assert deltas == ["Hello", " world"]
    assert "".join(deltas) == "Hello world"
