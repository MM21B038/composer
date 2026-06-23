"""Integration tests: compression on invoke vs streaming paths."""

from unittest.mock import MagicMock

import pytest

from composer import AIMessage, Agent, CompressedMessage, HumanMessage, SystemMessage, Thread
from composer.stream import AssistantEvent
from composer.thread import AgentInvoke


SUMMARY = "SUMMARY: preserved older context"
REPLY = "main reply"
STREAM_REPLY = "stream reply"


def _over_limit_thread() -> Thread:
    thread = Thread(
        compression_prompt="Summarize older turns for continuation.",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("old-" + "x" * 200) | thread
    HumanMessage("middle-" + "y" * 100) | thread
    HumanMessage("tail-msg") | thread
    return thread


def _tracking_agent():
    """Agent that records compression vs main-call thread views."""
    agent = MagicMock(spec=Agent)
    state = {
        "compression_calls": 0,
        "main_invoke_views": [],
        "main_stream_views": [],
    }

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        if kwargs.get("record_output") is False:
            state["compression_calls"] += 1
            return AIMessage(content=SUMMARY)
        msg = AIMessage(content=REPLY)
        (append_to or invoke_thread).append(msg)
        state["main_invoke_views"].append(
            [type(m).__name__ + ":" + str(m.content)[:40] for m in invoke_thread.get_messages()]
        )
        return msg

    def _stream_events(invoke_thread, *, append_to=None, record_output=True, **kwargs):
        state["main_stream_views"].append(
            [type(m).__name__ + ":" + str(m.content)[:40] for m in invoke_thread.get_messages()]
        )
        if record_output:
            (append_to or invoke_thread).append(AIMessage(content=STREAM_REPLY))
        yield AssistantEvent(text=STREAM_REPLY)

    agent.invoke.side_effect = _invoke
    agent.stream_events.side_effect = _stream_events
    return agent, state


def _expected_collapsed_view_labels():
    return [
        "SystemMessage:sys",
        f"CompressedMessage:{SUMMARY}",
        "HumanMessage:tail-msg",
    ]


def _assert_compression_outcome(thread: Thread, *, reply_content: str) -> None:
    root = thread.get_messages()
    assert not any(isinstance(m, CompressedMessage) for m in root)
    assert root[-1].content == reply_content

    active = thread.branch.active
    assert active.compressed is not None
    assert active.compressed.content == SUMMARY
    assert active.compressed_through == 3

    # Pre-reply collapsed slice: system + summary + preserved tail only.
    pre_reply = root[active.compressed_through : active.compressed_through + 1]
    assert len(pre_reply) == 1
    assert pre_reply[0].content == "tail-msg"

    # After reply, active_view also includes new root messages beyond compression.
    view_contents = [m.content for m in thread.branch.active_view().get_messages()]
    assert view_contents[:3] == ["sys", SUMMARY, "tail-msg"]
    assert view_contents[-1] == reply_content
    assert len(root) == 5  # sys + 3 human + 1 ai reply


def test_invoke_path_compresses_before_agent():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()
    before = len(thread.get_messages())

    result = AgentInvoke(
        agent,
        thread.branch.active_view().copy(),
        thread,
        thread.branch,
    ).resolve()

    assert result.content == REPLY
    assert state["compression_calls"] == 1
    assert state["main_invoke_views"] == [_expected_collapsed_view_labels()]
    assert len(thread.get_messages()) == before + 1
    _assert_compression_outcome(thread, reply_content=REPLY)


def test_thread_or_agent_operator_compresses():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()

    result = thread | agent

    assert result.content == REPLY
    assert state["compression_calls"] == 1
    _assert_compression_outcome(thread, reply_content=REPLY)


def test_stream_events_path_compresses_before_agent():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()
    before = len(thread.get_messages())

    events = list(thread.stream_events(agent))

    assert len(events) == 1
    assert events[0].text == STREAM_REPLY
    assert state["compression_calls"] == 1
    assert state["main_stream_views"] == [_expected_collapsed_view_labels()]
    assert len(thread.get_messages()) == before + 1
    _assert_compression_outcome(thread, reply_content=STREAM_REPLY)


def test_agent_invoke_stream_events_compresses():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()

    events = list(
        AgentInvoke(
            agent,
            thread.branch.active_view().copy(),
            thread,
            thread.branch,
        ).stream_events()
    )

    assert len(events) == 1
    assert state["compression_calls"] == 1
    assert state["main_stream_views"] == [_expected_collapsed_view_labels()]
    _assert_compression_outcome(thread, reply_content=STREAM_REPLY)


def test_direct_agent_invoke_bypasses_compression():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()
    before = len(thread.get_messages())

    result = agent.invoke(thread)

    assert result.content == REPLY
    assert state["compression_calls"] == 0
    assert len(thread.get_messages()) == before + 1
    assert thread.branch.active.compressed is None
    assert len(thread.branch.active_view()) == before + 1


def test_direct_agent_stream_events_bypasses_compression():
    thread = _over_limit_thread()
    agent, state = _tracking_agent()
    before = len(thread.get_messages())

    events = list(agent.stream_events(thread))

    assert len(events) == 1
    assert state["compression_calls"] == 0
    assert thread.branch.active.compressed is None
    assert len(thread.get_messages()) == before + 1


def test_invoke_and_stream_leave_equivalent_branch_graph():
    def run_invoke():
        thread = _over_limit_thread()
        agent, _ = _tracking_agent()
        thread | agent
        return thread

    def run_stream():
        thread = _over_limit_thread()
        agent, _ = _tracking_agent()
        list(thread.stream_events(agent))
        return thread

    invoke_thread = run_invoke()
    stream_thread = run_stream()

    for thread in (invoke_thread, stream_thread):
        assert thread.branch.active.compressed.content == SUMMARY
        assert thread.branch.active.compressed_through == 3

    invoke_collapsed = [
        m.content
        for m in invoke_thread.branch.active_view().get_messages()[:3]
    ]
    stream_collapsed = [
        m.content
        for m in stream_thread.branch.active_view().get_messages()[:3]
    ]
    assert invoke_collapsed == stream_collapsed == ["sys", SUMMARY, "tail-msg"]


def test_no_compression_when_below_token_limit_invoke_and_stream():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=96_000,
    )
    HumanMessage("short") | thread
    agent, state = _tracking_agent()

    thread | agent
    assert state["compression_calls"] == 0

    thread2 = Thread(
        compression_prompt="summarize",
        compression_max_tokens=96_000,
    )
    HumanMessage("short") | thread2
    agent2, state2 = _tracking_agent()
    list(thread2.stream_events(agent2))
    assert state2["compression_calls"] == 0
