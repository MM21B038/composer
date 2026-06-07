from unittest.mock import MagicMock

import pytest

from chat.models import ChatSession as ChatSessionModel
from chat.models import Project as ProjectModel
from chat.models import StoredMessage
from composer import (
    AIMessage,
    Agent,
    ChatProject,
    ChatSession,
    HumanMessage,
    SystemMessage,
)
from composer.persistence.repository import SessionRepository

pytestmark = pytest.mark.django_db


@pytest.fixture
def project():
    return ChatProject.create("test-project")


def test_create_project_and_session(project):
    session = project.new_session(name="first-chat", compression_prompt="summarize")
    assert session.name == "first-chat"
    assert session.project.name == "test-project"
    assert len(project.sessions()) == 1


def test_append_save_reload(project):
    session = project.new_session(name="persist-me")
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("hello"))
    session.append(AIMessage("hi there"))

    loaded = ChatSession.load(session_id=session.id)
    contents = [m.content for m in loaded.thread.get_messages()]
    assert contents == ["sys", "hello", "hi there"]


def test_lookup_by_name(project):
    session = project.new_session(name="named")
    found = project.get_session(name="named")
    assert found.id == session.id


def test_list_projects():
    ChatProject.create("alpha")
    ChatProject.create("beta")
    names = [p.name for p in ChatProject.list_all()]
    assert names == ["alpha", "beta"]


def test_branch_graph_round_trip(project):
    session = project.new_session(
        name="branchy",
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("one"))
    session.append(HumanMessage("two"))
    session.append(HumanMessage("three"))

    agent = MagicMock()
    agent.invoke.return_value = AIMessage(content="summary")
    session.branch.compress(agent)
    session.save()

    loaded = ChatSession.load(session_id=session.id)
    graph = loaded.branch_graph()
    assert len(graph["nodes"]) == 2
    assert graph["tail"] != graph["head"]
    assert graph["nodes"][graph["tail"]]["has_compressed"] is True

    stored_types = list(
        StoredMessage.objects.filter(session_id=session.id).values_list(
            "message_type", flat=True
        )
    )
    assert "compressed" not in stored_types


def test_continue_from_tail_after_reload(project):
    session = project.new_session(
        name="tail-test",
        compression_prompt="summarize",
        compression_tail_messages=1,
    )
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("a"))
    session.append(HumanMessage("b"))

    agent = MagicMock()
    agent.invoke.return_value = AIMessage(content="c1")
    session.branch.compress(agent)
    first_tail = session.branch.tail

    session.append(HumanMessage("c"))
    agent.invoke.return_value = AIMessage(content="c2")
    session.branch.compress(agent)
    second_tail = session.branch.tail
    session.save()

    session.branch.switch_to(first_tail)
    session.save()

    loaded = ChatSession.load(session_id=session.id)
    assert loaded.branch.tail == second_tail


def test_invoke_appends_to_root(project):
    session = project.new_session(name="invoke")
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("question"))

    agent = MagicMock(spec=Agent)

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        msg = AIMessage(content="answer")
        (append_to or invoke_thread).append(msg)
        return msg

    agent.invoke.side_effect = _invoke
    result = session.invoke(agent)

    assert result.content == "answer"
    reloaded = ChatSession.load(session_id=session.id)
    assert reloaded.thread.get_messages()[-1].content == "answer"


def test_get_session_by_id(project):
    session = project.new_session(name="by-id")
    loaded = SessionRepository.get_session_by_id(session.id)
    assert loaded.name == "by-id"


def test_delete_session(project):
    session = project.new_session(name="to-delete")
    session_id = session.id
    session.delete()
    assert not ChatSessionModel.objects.filter(pk=session_id).exists()


def test_delete_session_by_id(project):
    session = project.new_session(name="by-id-delete")
    session_id = session.id
    ChatSession.delete_by_id(session_id)
    assert not ChatSessionModel.objects.filter(pk=session_id).exists()


def test_delete_project():
    project = ChatProject.create("doomed")
    project.new_session(name="child")
    project_id = project.id
    project.delete()
    assert not ProjectModel.objects.filter(pk=project_id).exists()
    assert ChatSessionModel.objects.filter(project_id=project_id).count() == 0


def test_delete_project_by_name():
    ChatProject.create("remove-me")
    ChatProject.delete_by(name="remove-me")
    assert not ProjectModel.objects.filter(name="remove-me").exists()


def test_incognito_session_not_persisted(project):
    session = project.new_incognito_session(name="secret")
    assert session.is_incognito
    assert session.id is None
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("hidden"))
    assert ChatSessionModel.objects.filter(name="secret").count() == 0


def test_incognito_invoke_not_persisted(project):
    session = ChatSession.incognito(name="tmp")
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("q"))

    agent = MagicMock(spec=Agent)

    def _invoke(invoke_thread, *, append_to=None, **kwargs):
        msg = AIMessage(content="a")
        (append_to or invoke_thread).append(msg)
        return msg

    agent.invoke.side_effect = _invoke
    session.invoke(agent)
    assert ChatSessionModel.objects.count() == 0
    assert len(session.thread.get_messages()) == 3


def test_promote_incognito_to_persistent(project):
    session = project.new_incognito_session(name="draft")
    SystemMessage("sys") | session.thread
    session.append(HumanMessage("keep me"))

    saved = session.promote_to(project, name="draft-saved")
    assert not saved.is_incognito
    assert saved.id is not None

    loaded = ChatSession.load(session_id=saved.id)
    assert [m.content for m in loaded.thread.get_messages()] == ["sys", "keep me"]


def test_list_sessions_lightweight(project):
    project.new_session(name="one")
    project.new_session(name="two")
    rows = project.list_sessions()
    names = {row["name"] for row in rows}
    assert names == {"one", "two"}
    assert "id" in rows[0]
    assert "updated_at" in rows[0]
