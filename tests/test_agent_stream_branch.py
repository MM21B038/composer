from unittest.mock import MagicMock

import pytest

from composer import AIMessage, Agent, HumanMessage, SystemMessage, Thread
from composer.stream import AssistantEvent
from composer.thread import AgentInvoke


def _compression_agent(summary: str = "compressed summary"):
    agent = MagicMock(spec=Agent)

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        if kwargs.get("record_output") is False:
            return AIMessage(content=summary)
        msg = AIMessage(content="streamed reply")
        (append_to or invoke_thread).append(msg)
        return msg

    agent.invoke.side_effect = _invoke
    return agent


def _stream_events_agent():
    agent = MagicMock(spec=Agent)

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        if kwargs.get("record_output") is False:
            return AIMessage(content="compressed summary")
        msg = AIMessage(content="streamed reply")
        (append_to or invoke_thread).append(msg)
        return msg

    def _stream_events(invoke_thread, *, append_to=None, record_output=True, **kwargs):
        if record_output:
            (append_to or invoke_thread).append(AIMessage(content="streamed reply"))
        yield AssistantEvent(text="streamed reply")

    agent.invoke.side_effect = _invoke
    agent.stream_events.side_effect = _stream_events
    return agent


def test_agent_invoke_stream_events_triggers_compression():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("x" * 200) | thread
    HumanMessage("tail") | thread

    agent = _stream_events_agent()
    before = len(thread.get_messages())

    events = list(
        AgentInvoke(
            agent,
            thread.branch.active_view().copy(),
            thread,
            thread.branch,
        ).stream_events()
    )

    assert len(events) == 1
    assert events[0].text == "streamed reply"
    assert agent.invoke.call_count == 1
    assert agent.invoke.call_args.kwargs["record_output"] is False
    assert len(thread.get_messages()) == before + 1
    assert thread.get_messages()[-1].content == "streamed reply"


def test_thread_stream_events_delegates_to_branch_prepare():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("x" * 200) | thread
    HumanMessage("tail") | thread

    agent = _stream_events_agent()
    events = list(thread.stream_events(agent))

    assert len(events) == 1
    assert agent.invoke.call_count == 1


def test_agent_stream_record_output_false_skips_append():
    thread = Thread()
    HumanMessage("hello") | thread
    agent = MagicMock(spec=Agent)

    def _stream_events(invoke_thread, *, append_to=None, record_output=True, **kwargs):
        yield AssistantEvent(text="partial")

    agent.stream_events.side_effect = _stream_events
    before = len(thread.get_messages())

    events = list(agent.stream_events(thread, record_output=False))
    assert len(events) == 1
    assert len(thread.get_messages()) == before
