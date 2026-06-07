from __future__ import annotations

from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..thread import (
    AIMessage,
    CompressedMessage,
    HumanMessage,
    ImageMessage,
    Message,
    SystemMessage,
    Thread,
    ToolMessage,
)
from ..thread_branch import BranchNode, ThreadBranchGraph
from ..tool_hide import ToolResultHideRule

if TYPE_CHECKING:
    from chat.models import BranchNodeRecord, ChatSession

_MESSAGE_CLASSES: dict[str, type[Message]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
    "compressed": CompressedMessage,
}


def message_type_name(message: Message) -> str:
    if isinstance(message, ImageMessage):
        return "image"
    if isinstance(message, CompressedMessage):
        return "compressed"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    return "human"


def message_to_payload(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json")


def payload_to_message(data: dict[str, Any]) -> Message:
    msg_type = data.get("type") or data.get("message_type") or "human"
    if msg_type == "AIMessageChunk":
        msg_type = "ai"
    cls = _MESSAGE_CLASSES.get(msg_type, HumanMessage)
    dump = dict(data)
    dump["type"] = msg_type if msg_type in _MESSAGE_CLASSES else "human"
    return cls.model_validate(dump)


def _rule_to_dict(rule: ToolResultHideRule) -> dict[str, Any]:
    data = asdict(rule)
    if callable(data.get("on_hide_message")):
        data["on_hide_message"] = None
        data["_on_hide_message_callable"] = True
    if callable(data.get("summarize_fn")):
        data["summarize_fn"] = None
        data["_summarize_fn_callable"] = True
    return data


def _rule_from_dict(data: dict[str, Any]) -> ToolResultHideRule:
    allowed = {field.name for field in fields(ToolResultHideRule)}
    kwargs = {key: value for key, value in data.items() if key in allowed}
    if kwargs.get("on_hide_message") is None and not data.get("_on_hide_message_callable"):
        kwargs["on_hide_message"] = (
            "[Earlier {tool_name} result hidden — see latest result above]"
        )
    kwargs["summarize_fn"] = None
    return ToolResultHideRule(**kwargs)


def thread_config_to_json(kwargs: dict[str, Any]) -> dict[str, Any]:
    config = dict(kwargs)
    rules = config.get("tool_hide_rules") or []
    config["tool_hide_rules"] = [_rule_to_dict(rule) for rule in rules]
    return config


def thread_config_from_json(data: dict[str, Any]) -> dict[str, Any]:
    config = dict(data)
    rules = config.get("tool_hide_rules") or []
    config["tool_hide_rules"] = [_rule_from_dict(rule) for rule in rules]
    return config


def attach_branch_graph(thread: Thread, graph: ThreadBranchGraph) -> None:
    thread._branch_graph = graph
    graph.root = thread


def branch_nodes_from_records(
    records: list["BranchNodeRecord"],
) -> dict[str, BranchNode]:
    nodes: dict[str, BranchNode] = {}
    for record in records:
        compressed = None
        if record.compressed_payload:
            compressed = payload_to_message(record.compressed_payload)
            if not isinstance(compressed, CompressedMessage):
                compressed = CompressedMessage(content=compressed.content)
        parent_id = str(record.parent_id) if record.parent_id else None
        nodes[str(record.id)] = BranchNode(
            id=str(record.id),
            parent_id=parent_id,
            compressed=compressed,
            compressed_through=record.compressed_through,
            visible_end=record.visible_end,
            children=[str(child_id) for child_id in (record.child_order or [])],
        )
    return nodes


def branch_graph_from_session(
    thread: Thread,
    session: "ChatSession",
    records: list["BranchNodeRecord"],
) -> ThreadBranchGraph:
    nodes = branch_nodes_from_records(records)
    config = thread_config_from_json(session.config or {})
    graph = ThreadBranchGraph.from_state(
        thread,
        nodes,
        str(session.active_branch_id),
        compression_prompt=config.get("compression_prompt"),
        compression_max_tokens=config.get("compression_max_tokens", 96_000),
        compression_tail_tokens=config.get("compression_tail_tokens", 8_000),
        compression_tail_messages=config.get("compression_tail_messages"),
        encoder=config.get("model_view_encoder", "cl100k_base"),
    )
    attach_branch_graph(thread, graph)
    return graph


def branch_graph_to_record_payloads(
    graph: ThreadBranchGraph,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for node_id, node in graph.nodes.items():
        compressed_payload = None
        if node.compressed is not None:
            compressed_payload = message_to_payload(node.compressed)
        parent_uuid = UUID(node.parent_id) if node.parent_id else None
        payloads.append(
            {
                "id": UUID(node_id),
                "parent_id": parent_uuid,
                "compressed_payload": compressed_payload,
                "compressed_through": node.compressed_through,
                "visible_end": node.visible_end,
                "child_order": [UUID(child_id) for child_id in node.children],
            }
        )
    return payloads
