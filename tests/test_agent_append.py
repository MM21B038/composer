from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

from composer import Agent, AIMessage, HumanMessage, SystemMessage, Thread


def test_append_uses_target_thread_length_not_input_length():
    thread = Thread()
    thread.append(SystemMessage("sys"))
    thread.append(HumanMessage("hi"))

    agent = Agent(model=MagicMock())
    input_msgs = [HumanMessage("hi")]
    ai_msg = AIMessage(content="answer")
    mock_response = {"messages": [SystemMessage("sys"), HumanMessage("hi"), ai_msg]}

    with patch.object(agent, "_build_agent_for_invoke_async", new_callable=AsyncMock) as build:
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value=mock_response)
        build.return_value = mock_agent

        asyncio.run(
            agent.ainvoke(
                thread.branch.active_view().copy(),
                append_to=thread,
            )
        )

    assert len(thread.thread) == 3
    assert thread.thread[-1].content == "answer"


def test_enrich_ai_message_from_stream_replaces_stub_reasoning():
    agent = Agent.__new__(Agent)
    stub = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "."},
        tool_calls=[{"id": "call_1", "name": "snap", "args": {}}],
    )
    enriched = agent._enrich_ai_message_from_stream(
        stub,
        streamed_thinking="Need to take a snapshot",
        streamed_assistant="",
    )

    assert enriched.content == ""
    assert enriched.additional_kwargs["reasoning_content"] == "Need to take a snapshot"


def test_append_stream_values_messages_replaces_empty_final_content():
    agent = Agent.__new__(Agent)
    thread = Thread([HumanMessage("hi")])
    thinking = "The user asked about the password key."
    assistant = "The page uses RSA-OAEP encryption."
    stub_ai = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "."},
    )
    output_messages = [HumanMessage("hi"), stub_ai]

    agent._append_stream_values_messages(
        thread,
        None,
        output_messages,
        1,
        streamed_thinking=thinking,
        streamed_assistant=assistant,
    )

    assert thread[-1].content == assistant
    assert thread[-1].additional_kwargs["reasoning_content"] == thinking
