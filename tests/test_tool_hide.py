import pytest

from composer import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    Thread,
    ToolMessage,
    ToolResultHideRule,
    get_original_content,
    is_hidden_for_model,
    message_matches_rule,
)


def _tool(name: str, content: str, tool_call_id: str = "1") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)


def test_message_matches_rule_with_server_prefix():
    rule = ToolResultHideRule(server="butcher", tool_name="browser_snapshot")
    msg = _tool("butcher_browser_snapshot", "big payload")
    assert message_matches_rule(msg, rule)


def test_message_matches_wildcard():
    rule = ToolResultHideRule(tool_name="butcher_browser_*")
    assert message_matches_rule(_tool("butcher_browser_snapshot", "a"), rule)
    assert not message_matches_rule(_tool("task_manager_create", "a"), rule)


def test_message_matches_whole_server():
    rule = ToolResultHideRule(server="butcher", tool_name="*")
    assert message_matches_rule(_tool("butcher_browser_snapshot", "a"), rule)
    assert not message_matches_rule(_tool("task_manager_create", "a"), rule)


def test_keep_latest_one():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="butcher_browser_snapshot",
                on_hide_message="[hidden]",
            )
        ]
    )
    thread.append(_tool("butcher_browser_snapshot", "first", "1"))
    thread.append(_tool("butcher_browser_snapshot", "second", "2"))
    thread.append(_tool("butcher_browser_snapshot", "third", "3"))

    model_msgs = thread.messages_for_model()
    tool_contents = [m.content for m in model_msgs if isinstance(m, ToolMessage)]
    assert tool_contents == ["[hidden]", "[hidden]", "third"]

    storage = [m.content for m in thread.get_messages() if isinstance(m, ToolMessage)]
    assert storage == ["first", "second", "third"]


def test_keep_latest_two():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
                keep_latest=2,
            )
        ]
    )
    for i in range(4):
        thread.append(_tool("snap", f"payload-{i}", str(i)))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["[hidden]", "[hidden]", "payload-2", "payload-3"]


def test_min_tokens_to_hide():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
                min_tokens_to_hide=10,
            )
        ]
    )
    thread.append(_tool("snap", "tiny", "1"))
    thread.append(_tool("snap", "x" * 100, "2"))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["tiny", "x" * 100]


def test_scope_since_last_human():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
                scope="since_last_human",
            )
        ]
    )
    thread.append(_tool("snap", "turn1-a", "1"))
    thread.append(_tool("snap", "turn1-b", "2"))
    thread.append(HumanMessage("next turn"))
    thread.append(_tool("snap", "turn2-a", "3"))
    thread.append(_tool("snap", "turn2-b", "4"))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["turn1-a", "turn1-b", "[hidden]", "turn2-b"]


def test_summarize_strategy():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_strategy="summarize",
                summarize_fn=lambda msg: f"summary:{msg.content[:3]}",
            )
        ]
    )
    thread.append(_tool("snap", "abcdef", "1"))
    thread.append(_tool("snap", "ghijkl", "2"))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["summary:abc", "ghijkl"]


def test_drop_strategy():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_strategy="drop",
            )
        ]
    )
    thread.append(_tool("snap", "first", "1"))
    thread.append(_tool("snap", "second", "2"))

    model_msgs = thread.messages_for_model()
    assert len([m for m in model_msgs if isinstance(m, ToolMessage)]) == 1
    assert model_msgs[-1].content == "second"


def test_max_hidden_results_drops_oldest():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
                keep_latest=1,
                max_hidden_results=1,
            )
        ]
    )
    for i in range(4):
        thread.append(_tool("snap", f"p{i}", str(i)))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["[hidden]", "p3"]


def test_persist_tool_hides():
    thread = Thread(
        persist_tool_hides=True,
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
            )
        ],
    )
    thread.append(_tool("snap", "original", "1"))
    thread.append(_tool("snap", "latest", "2"))

    stored = thread.get_messages()
    hidden_msg = stored[0]
    assert hidden_msg.content == "[hidden]"
    assert is_hidden_for_model(hidden_msg)
    assert get_original_content(hidden_msg) == "original"
    assert stored[1].content == "latest"

    restored = thread.restore_hidden_tool_messages()
    assert restored == 1
    assert thread.get_messages()[0].content == "original"


def test_callable_on_hide_message():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message=lambda msg, rule: f"hidden:{msg.name}",
            )
        ]
    )
    thread.append(_tool("snap", "a", "1"))
    thread.append(_tool("snap", "b", "2"))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["hidden:snap", "b"]


def test_rolling_window_preserves_system():
    thread = Thread(max_messages_for_model=3)
    thread.append(SystemMessage("sys"))
    thread.append(HumanMessage("1"))
    thread.append(AIMessage("2"))
    thread.append(HumanMessage("3"))
    thread.append(AIMessage("4"))

    model_msgs = thread.messages_for_model()
    assert len(model_msgs) == 3
    assert isinstance(model_msgs[0], SystemMessage)
    assert model_msgs[-1].content == "4"


def test_token_budget_trim():
    thread = Thread(max_tokens_for_model=20)
    thread.append(HumanMessage("hello world"))
    thread.append(AIMessage("short"))
    thread.append(HumanMessage("x" * 500))

    model_msgs = thread.messages_for_model()
    assert len(model_msgs) >= 1
    # trim keeps one oversized tail message when nothing else fits the budget
    assert model_msgs[-1].content == "x" * 500


def test_scope_since_last_assistant():
    thread = Thread(
        tool_hide_rules=[
            ToolResultHideRule(
                tool_name="snap",
                on_hide_message="[hidden]",
                scope="since_last_assistant",
            )
        ]
    )
    thread.append(AIMessage(content="", tool_calls=[{"id": "1", "name": "snap", "args": {}}]))
    thread.append(_tool("snap", "step1-a", "1"))
    thread.append(_tool("snap", "step1-b", "2"))
    thread.append(AIMessage(content="done step 1"))
    thread.append(AIMessage(content="", tool_calls=[{"id": "2", "name": "snap", "args": {}}]))
    thread.append(_tool("snap", "step2-a", "2"))
    thread.append(_tool("snap", "step2-b", "3"))

    tool_contents = [
        m.content for m in thread.messages_for_model() if isinstance(m, ToolMessage)
    ]
    assert tool_contents == ["step1-a", "step1-b", "[hidden]", "step2-b"]


def test_add_and_remove_rule():
    thread = Thread()
    rule = ToolResultHideRule(tool_name="snap", on_hide_message="[hidden]")
    thread.add_tool_hide_rule(rule)
    assert len(thread.tool_hide_rules) == 1
    assert thread.remove_tool_hide_rule(tool_name="snap") is True
    assert thread.tool_hide_rules == []
