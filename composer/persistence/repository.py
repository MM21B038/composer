from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from django.db import transaction

from ..thread import Thread
from ..thread_branch import ThreadBranchGraph
from .django_setup import ensure_django, sync_db

ensure_django()
from chat.models import BranchNodeRecord, ChatSession, Project, StoredMessage
from .serializers import (
    attach_branch_graph,
    branch_graph_from_session,
    branch_graph_to_record_payloads,
    message_to_payload,
    message_type_name,
    payload_to_message,
    thread_config_from_json,
    thread_config_to_json,
)


class SessionRepository:
    @staticmethod
    @sync_db
    def create_project(name: str) -> Project:
        ensure_django()
        return Project.objects.create(name=name)

    @staticmethod
    @sync_db
    def get_project(*, id: UUID | str | None = None, name: str | None = None) -> Project:
        ensure_django()
        if id is not None:
            return Project.objects.get(pk=id)
        if name is not None:
            return Project.objects.get(name=name)
        raise ValueError("get_project requires id or name")

    @staticmethod
    @sync_db
    def list_projects():
        ensure_django()
        return list(Project.objects.all())

    @staticmethod
    @sync_db
    def create_session(
        project: Project,
        *,
        name: str = "",
        **thread_kwargs: Any,
    ) -> ChatSession:
        ensure_django()
        thread = Thread(**thread_kwargs)
        graph = thread.branch
        return SessionRepository._create_session_row(
            project=project,
            name=name,
            thread=thread,
            graph=graph,
        )

    @staticmethod
    @sync_db
    def _create_session_row(
        *,
        project: Project,
        name: str,
        thread: Thread,
        graph: ThreadBranchGraph,
    ) -> ChatSession:
        config = thread_config_to_json(thread._thread_kwargs())
        session = ChatSession.objects.create(
            project=project,
            name=name,
            config=config,
            active_branch_id=UUID(graph.active_id),
        )
        SessionRepository.save_session_state(session, thread, graph)
        return session

    @staticmethod
    @sync_db
    def list_sessions(project: Project):
        ensure_django()
        return list(project.sessions.all())

    @staticmethod
    @sync_db
    def get_session(
        project: Project,
        *,
        id: UUID | str | None = None,
        name: str | None = None,
    ) -> ChatSession:
        ensure_django()
        qs = project.sessions.all()
        if id is not None:
            return qs.get(pk=id)
        if name is not None:
            return qs.get(name=name)
        raise ValueError("get_session requires id or name")

    @staticmethod
    @sync_db
    def get_session_by_id(session_id: UUID | str) -> ChatSession:
        ensure_django()
        return ChatSession.objects.select_related("project").get(pk=session_id)

    @staticmethod
    @sync_db
    @transaction.atomic
    def save_session_state(
        session: ChatSession,
        thread: Thread,
        graph: ThreadBranchGraph,
    ) -> None:
        ensure_django()
        session.config = thread_config_to_json(thread._thread_kwargs())
        session.active_branch_id = UUID(graph.active_id)
        session.save(update_fields=["config", "active_branch_id", "updated_at"])

        StoredMessage.objects.filter(session=session).delete()
        BranchNodeRecord.objects.filter(session=session).delete()

        StoredMessage.objects.bulk_create(
            [
                StoredMessage(
                    session=session,
                    position=index,
                    message_type=message_type_name(message),
                    payload=message_to_payload(message),
                )
                for index, message in enumerate(thread.get_messages())
            ]
        )

        record_payloads = branch_graph_to_record_payloads(graph)
        id_map: dict[UUID, BranchNodeRecord] = {}
        pending = list(record_payloads)
        while pending:
            progress = False
            remaining: list[dict[str, Any]] = []
            for payload in pending:
                parent_id = payload["parent_id"]
                if parent_id is not None and parent_id not in id_map:
                    remaining.append(payload)
                    continue
                parent_record = id_map.get(parent_id) if parent_id else None
                record = BranchNodeRecord.objects.create(
                    id=payload["id"],
                    session=session,
                    parent=parent_record,
                    compressed_payload=payload["compressed_payload"],
                    compressed_through=payload["compressed_through"],
                    visible_end=payload["visible_end"],
                    child_order=[str(child_id) for child_id in payload["child_order"]],
                )
                id_map[payload["id"]] = record
                progress = True
            if not progress and remaining:
                raise RuntimeError("failed to persist branch graph: unresolved parent links")
            pending = remaining

    @staticmethod
    @sync_db
    def load_session_state(session_id: UUID | str) -> tuple[ChatSession, Thread, ThreadBranchGraph]:
        ensure_django()
        session = SessionRepository.get_session_by_id(session_id)
        config = thread_config_from_json(session.config or {})
        messages = [
            payload_to_message(row.payload)
            for row in session.messages.order_by("position")
        ]
        thread = Thread(messages, **config)
        records = list(session.branch_nodes.select_related("parent").all())
        graph = branch_graph_from_session(thread, session, records)
        return session, thread, graph

    @staticmethod
    @sync_db
    def delete_project(project: Project) -> None:
        ensure_django()
        project.delete()

    @staticmethod
    @sync_db
    def delete_session(session: ChatSession) -> None:
        ensure_django()
        session.delete()
