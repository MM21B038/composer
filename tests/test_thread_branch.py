from unittest.mock import MagicMock

import pytest

from composer import (
    AIMessage,
    Agent,
    CompressedMessage,
    HumanMessage,
    SystemMessage,
    Thread,
    ToolMessage,
    ToolResultHideRule,
    get_original_content,
    is_hidden_for_model,
)
from composer.thread import AgentInvoke


def _mock_agent(summary: str = "compressed summary"):
    agent = MagicMock()
    agent.invoke.return_value = AIMessage(content=summary)
    return agent


def test_active_view_root_only():
    thread = Thread(compression_prompt="summarize")
    SystemMessage("sys") | thread
    HumanMessage("hello") | thread
    AIMessage("hi") | thread

    view = thread.branch.active_view()
    assert len(view) == 3
    assert isinstance(view[0], SystemMessage)
    assert view[1].content == "hello"
    assert view[2].content == "hi"


def test_active_view_after_compression():
    thread = Thread(
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | thread
    HumanMessage("one") | thread
    HumanMessage("two") | thread
    HumanMessage("three") | thread

    agent = _mock_agent("summary text")
    child = thread.branch.compress(agent)

    assert child is not None
    view = thread.branch.active_view()
    assert len(view) == 3
    assert isinstance(view[0], SystemMessage)
    assert isinstance(view[1], CompressedMessage)
    assert view[1].content == "summary text"
    assert view[2].content == "three"

    assert len(thread.get_messages()) == 4
    assert not any(isinstance(m, CompressedMessage) for m in thread.get_messages())


def test_compression_preserves_tail_by_message_count():
    thread = Thread(
        compression_prompt="summarize",
        compression_tail_messages=2,
    )
    SystemMessage("sys") | thread
    for i in range(5):
        HumanMessage(f"msg-{i}") | thread

    agent = _mock_agent("packed")
    thread.branch.compress(agent)

    view = thread.branch.active_view()
    tail_contents = [m.content for m in view if isinstance(m, HumanMessage)]
    assert tail_contents == ["msg-3", "msg-4"]
    assert thread.branch.active.compressed_through == 4


def test_branch_parent_child_switch():
    thread = Thread(
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | thread
    HumanMessage("a") | thread
    HumanMessage("b") | thread

    agent = _mock_agent("c1")
    thread.branch.compress(agent)
    first_child = thread.branch.tail

    HumanMessage("c") | thread
    agent.invoke.return_value = AIMessage(content="c2")
    thread.branch.compress(agent)
    second_child = thread.branch.tail

    assert thread.branch.parent(second_child).id == first_child
    assert first_child in thread.branch.children(thread.branch.head)

    thread.branch.switch_to(first_child)
    assert thread.branch.tail == first_child
    view = thread.branch.active_view()
    assert view[-1].content == "b"

    thread.branch.switch_to_tail()
    assert thread.branch.tail == second_child


def test_maybe_compress_respects_token_limit():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("x" * 200) | thread
    HumanMessage("tail") | thread

    agent = _mock_agent("done")
    result = thread.branch.maybe_compress(agent)

    assert result is not None
    agent.invoke.assert_called_once()


def test_maybe_compress_skips_below_limit():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=96_000,
    )
    HumanMessage("short") | thread

    agent = _mock_agent()
    assert thread.branch.maybe_compress(agent) is None
    agent.invoke.assert_not_called()


def test_maybe_compress_retries_with_shrunk_tail():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=15,
        compression_tail_tokens=40,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("alpha " * 30) | thread
    HumanMessage("beta " * 30) | thread
    HumanMessage("gamma " * 30) | thread
    HumanMessage("tail-msg") | thread

    agent = MagicMock()
    agent.invoke.return_value = AIMessage(content="summary " * 80)
    thread.branch.maybe_compress(agent)
    assert agent.invoke.call_count >= 1


def test_prepare_for_agent_applies_emergency_trim_when_stuck():
    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("x" * 500) | thread
    HumanMessage("tail") | thread

    agent = _mock_agent("y" * 500)
    with pytest.warns(UserWarning, match="emergency token trim"):
        view = thread.branch.prepare_for_agent(agent)

    assert view.max_tokens_for_model == 10
    assert view.token_count_for_model() <= 10


def test_amaybe_compress_uses_ainvoke():
    import asyncio
    from unittest.mock import AsyncMock

    thread = Thread(
        compression_prompt="summarize",
        compression_max_tokens=10,
        compression_tail_messages=1,
        model_view_encoder="cl100k_base",
    )
    SystemMessage("sys") | thread
    HumanMessage("x" * 200) | thread
    HumanMessage("tail") | thread

    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value=AIMessage(content="done"))

    async def _run() -> None:
        result = await thread.branch.amaybe_compress(agent)
        assert result is not None
        agent.ainvoke.assert_called_once()

    asyncio.run(_run())


def test_agent_invoke_appends_to_root_not_branch_view():
    thread = Thread(compression_prompt="summarize")
    SystemMessage("sys") | thread
    HumanMessage("start") | thread

    agent = MagicMock(spec=Agent)

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        msg = AIMessage(content="reply")
        target = append_to or invoke_thread
        target.append(msg)
        return msg

    agent.invoke.side_effect = _invoke

    before = len(thread.get_messages())
    pending = AgentInvoke(
        agent,
        thread.branch.active_view().copy(),
        thread,
        thread.branch,
    )
    result = pending.resolve()

    assert result.content == "reply"
    assert len(thread.get_messages()) == before + 1
    assert thread.get_messages()[-1].content == "reply"
    assert not any(isinstance(m, CompressedMessage) for m in thread.get_messages())
    agent.invoke.assert_called_once()
    assert agent.invoke.call_args.kwargs["append_to"] is thread


def test_history_lists_root_to_active():
    thread = Thread(
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | thread
    HumanMessage("a") | thread
    HumanMessage("b") | thread

    agent = _mock_agent("s1")
    thread.branch.compress(agent)
    path = thread.branch.history()
    assert len(path) == 2
    assert path[0].parent_id is None
    assert path[1].compressed is not None


def test_no_duplicate_system_in_active_view():
    thread = Thread(
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | thread
    HumanMessage("a") | thread
    HumanMessage("b") | thread

    agent = _mock_agent("sum")
    thread.branch.compress(agent)

    view = thread.branch.active_view()
    system_msgs = [m for m in view if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1


def test_mixed_hide_modes():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="persist_tool",
                on_hide_message="[persisted]",
                hide_mode="persist",
            ),
            ToolResultHideRule(
                tool_name="invoke_tool",
                on_hide_message="[invoke]",
                hide_mode="invoke_only",
            ),
        ],
    )
    thread.append(ToolMessage(content="big", name="persist_tool", tool_call_id="1"))
    thread.append(ToolMessage(content="latest", name="persist_tool", tool_call_id="2"))
    thread.append(ToolMessage(content="first", name="invoke_tool", tool_call_id="3"))
    thread.append(ToolMessage(content="second", name="invoke_tool", tool_call_id="4"))

    stored = thread.get_messages()
    persist_hidden = [m for m in stored if getattr(m, "name", None) == "persist_tool"]
    assert persist_hidden[0].content == "[persisted]"
    assert get_original_content(persist_hidden[0]) == "big"
    assert is_hidden_for_model(persist_hidden[0])

    invoke_stored = [m for m in stored if getattr(m, "name", None) == "invoke_tool"]
    assert invoke_stored[0].content == "first"
    assert invoke_stored[1].content == "second"

    model_tool_contents = [
        m.content
        for m in thread.messages_for_model()
        if isinstance(m, ToolMessage)
    ]
    assert model_tool_contents == [
        "[persisted]",
        "latest",
        "[invoke]",
        "second",
    ]


def test_thread_persist_flag_still_applies_without_explicit_hide_mode():
    thread = Thread(
        persist_tool_hides=True,
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
            )
        ],
    )
    thread.append(ToolMessage(content="original", name="snap", tool_call_id="1"))
    thread.append(ToolMessage(content="latest", name="snap", tool_call_id="2"))

    stored = thread.get_messages()
    assert stored[0].content == "[hidden]"
    assert get_original_content(stored[0]) == "original"
