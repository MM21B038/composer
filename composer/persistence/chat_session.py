from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from chat.models import ChatSession as ChatSessionModel
from chat.models import Project as ProjectModel

from ..thread import AIMessage, Message, Thread
from ..thread_branch import ThreadBranchGraph
from .django_setup import ensure_django
from .repository import SessionRepository

if TYPE_CHECKING:
    from ..agent import Agent


class ChatProject:
    def __init__(self, project: ProjectModel) -> None:
        self._project = project

    @property
    def id(self) -> UUID:
        return self._project.id

    @property
    def name(self) -> str:
        return self._project.name

    @property
    def created_at(self) -> datetime:
        return self._project.created_at

    @property
    def updated_at(self) -> datetime:
        return self._project.updated_at

    @classmethod
    def create(cls, name: str) -> ChatProject:
        return cls(SessionRepository.create_project(name))

    @classmethod
    def get(
        cls,
        *,
        id: UUID | str | None = None,
        name: str | None = None,
    ) -> ChatProject:
        return cls(SessionRepository.get_project(id=id, name=name))

    @classmethod
    def list_all(cls) -> list[ChatProject]:
        ensure_django()
        return [cls(project) for project in SessionRepository.list_projects()]

    @classmethod
    def delete_by(
        cls,
        *,
        id: UUID | str | None = None,
        name: str | None = None,
    ) -> None:
        project = SessionRepository.get_project(id=id, name=name)
        SessionRepository.delete_project(project)

    def delete(self) -> None:
        SessionRepository.delete_project(self._project)

    def new_session(
        self,
        name: str = "",
        *,
        incognito: bool = False,
        **thread_kwargs: Any,
    ) -> ChatSession:
        if incognito:
            return ChatSession.incognito(
                project=self,
                name=name,
                **thread_kwargs,
            )
        session = SessionRepository.create_session(
            self._project,
            name=name,
            **thread_kwargs,
        )
        return ChatSession.from_model(session)

    def new_incognito_session(
        self,
        name: str = "",
        **thread_kwargs: Any,
    ) -> ChatSession:
        return self.new_session(name=name, incognito=True, **thread_kwargs)

    def sessions(self) -> list[ChatSession]:
        ensure_django()
        return [
            ChatSession.from_model(session)
            for session in SessionRepository.list_sessions(self._project)
        ]

    def list_sessions(self) -> list[dict[str, Any]]:
        """Lightweight session listing without loading full thread state."""
        ensure_django()
        return [
            {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
            for session in SessionRepository.list_sessions(self._project)
        ]

    def get_session(
        self,
        *,
        id: UUID | str | None = None,
        name: str | None = None,
    ) -> ChatSession:
        session = SessionRepository.get_session(self._project, id=id, name=name)
        return ChatSession.from_model(session)

    def __repr__(self) -> str:
        return f"ChatProject(name={self.name!r}, id={self.id})"


class ChatSession:
    def __init__(
        self,
        session: ChatSessionModel | None,
        thread: Thread,
        graph: ThreadBranchGraph,
        *,
        incognito: bool = False,
        project: ChatProject | None = None,
        name: str = "",
    ) -> None:
        self._session = session
        self._thread = thread
        self._graph = graph
        self._incognito = incognito
        self._project = project
        self._name = name

    @property
    def is_incognito(self) -> bool:
        return self._incognito

    @property
    def id(self) -> UUID | None:
        if self._incognito:
            return None
        assert self._session is not None
        return self._session.id

    @property
    def name(self) -> str:
        if self._session is not None:
            return self._session.name
        return self._name

    @property
    def project(self) -> ChatProject:
        if self._session is not None:
            return ChatProject(self._session.project)
        if self._project is None:
            raise ValueError("incognito session has no project")
        return self._project

    @property
    def thread(self) -> Thread:
        return self._thread

    @property
    def branch(self) -> ThreadBranchGraph:
        return self._graph

    @classmethod
    def incognito(
        cls,
        *,
        project: ChatProject | None = None,
        name: str = "",
        **thread_kwargs: Any,
    ) -> ChatSession:
        """Create an in-memory session that is not written to the database."""
        thread = Thread(**thread_kwargs)
        graph = thread.branch
        return cls(
            session=None,
            thread=thread,
            graph=graph,
            incognito=True,
            project=project,
            name=name,
        )

    @classmethod
    def from_model(cls, session: ChatSessionModel) -> ChatSession:
        loaded = cls.load(session_id=session.id)
        return loaded

    @classmethod
    def load(
        cls,
        *,
        session_id: UUID | str | None = None,
        project: ChatProject | None = None,
        name: str | None = None,
    ) -> ChatSession:
        ensure_django()
        if session_id is not None:
            session, thread, graph = SessionRepository.load_session_state(session_id)
        elif project is not None and name is not None:
            row = SessionRepository.get_session(project._project, name=name)
            session, thread, graph = SessionRepository.load_session_state(row.id)
        else:
            raise ValueError("load requires session_id or project+name")
        wrapper = cls(session, thread, graph, incognito=False)
        wrapper.continue_from_tail(save=False)
        return wrapper

    @classmethod
    def delete_by_id(cls, session_id: UUID | str) -> None:
        ensure_django()
        session = SessionRepository.get_session_by_id(session_id)
        SessionRepository.delete_session(session)

    def save(self) -> None:
        if self._incognito:
            return
        assert self._session is not None
        SessionRepository.save_session_state(self._session, self._thread, self._graph)

    def append(self, message: Message) -> None:
        self._thread.append(message)
        self.save()

    def continue_from_tail(self, *, save: bool = True) -> None:
        self._graph.switch_to_tail()
        if save:
            self.save()

    def promote_to(
        self,
        project: ChatProject,
        name: str | None = None,
    ) -> ChatSession:
        """Persist an incognito session to the database under a project."""
        if not self._incognito:
            raise ValueError("only incognito sessions can be promoted")
        session_name = self.name if name is None else name
        row = SessionRepository._create_session_row(
            project=project._project,
            name=session_name,
            thread=self._thread,
            graph=self._graph,
        )
        return ChatSession(row, self._thread, self._graph, incognito=False)

    def delete(self) -> None:
        if self._incognito:
            self._thread.clear_thread()
            self._graph = self._thread.branch
            return
        assert self._session is not None
        SessionRepository.delete_session(self._session)
        self._session = None

    def branch_graph(self) -> dict[str, Any]:
        nodes = {
            node_id: {
                "id": node.id,
                "parent_id": node.parent_id,
                "compressed_through": node.compressed_through,
                "visible_end": node.visible_end,
                "children": list(node.children),
                "has_compressed": node.compressed is not None,
            }
            for node_id, node in self._graph.nodes.items()
        }
        return {
            "session_id": str(self.id) if self.id is not None else None,
            "incognito": self._incognito,
            "head": self._graph.head,
            "tail": self._graph.tail,
            "active_id": self._graph.active_id,
            "history": [node.id for node in self._graph.history()],
            "nodes": nodes,
        }

    def invoke(self, agent: Agent) -> AIMessage:
        ensure_django()
        self.continue_from_tail(save=False)
        try:
            asyncio.get_running_loop()
            raise RuntimeError("use await session.ainvoke(agent) inside async context")
        except RuntimeError as exc:
            if "async context" in str(exc):
                raise

        from ..thread import AgentInvoke

        pending = AgentInvoke(
            agent,
            self._thread.branch.active_view().copy(),
            self._thread,
            self._graph,
        )
        result = pending.resolve()
        self.save()
        return result

    async def ainvoke(self, agent: Agent) -> AIMessage:
        ensure_django()
        self.continue_from_tail(save=False)
        from ..thread import AgentInvoke

        pending = AgentInvoke(
            agent,
            self._thread.branch.active_view().copy(),
            self._thread,
            self._graph,
        )
        result = await pending
        self.save()
        return result

    def __repr__(self) -> str:
        if self._incognito:
            return f"ChatSession(incognito, name={self.name!r})"
        return f"ChatSession(name={self.name!r}, id={self.id})"
