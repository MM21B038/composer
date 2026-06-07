from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .thread import (
    AIMessage,
    CompressedMessage,
    EncoderType,
    HumanMessage,
    Message,
    SystemMessage,
    TokenCalculator,
    ToolMessage,
    _message_to_text,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .thread import Thread


@dataclass
class BranchNode:
    id: str
    parent_id: str | None
    compressed: CompressedMessage | None
    compressed_through: int
    visible_end: int | None = None
    children: list[str] = field(default_factory=list)


class ThreadBranchGraph:
    """Internal branch graph for compressed context windows on a root thread."""

    def __init__(
        self,
        root: "Thread",
        *,
        compression_prompt: str | None = None,
        compression_max_tokens: int = 96_000,
        compression_tail_tokens: int | None = 8_000,
        compression_tail_messages: int | None = None,
        encoder: EncoderType = "cl100k_base",
    ) -> None:
        self.root = root
        self.compression_prompt = compression_prompt
        self.compression_max_tokens = compression_max_tokens
        self.compression_tail_tokens = compression_tail_tokens
        self.compression_tail_messages = compression_tail_messages
        self.encoder = encoder

        root_id = str(uuid.uuid4())
        content_start = 0
        if root.thread and isinstance(root.thread[0], SystemMessage):
            content_start = 1

        self.nodes: dict[str, BranchNode] = {
            root_id: BranchNode(
                id=root_id,
                parent_id=None,
                compressed=None,
                compressed_through=content_start,
            )
        }
        self.active_id = root_id

    @classmethod
    def from_state(
        cls,
        root: "Thread",
        nodes: dict[str, BranchNode],
        active_id: str,
        *,
        compression_prompt: str | None = None,
        compression_max_tokens: int = 96_000,
        compression_tail_tokens: int | None = 8_000,
        compression_tail_messages: int | None = None,
        encoder: EncoderType = "cl100k_base",
    ) -> "ThreadBranchGraph":
        graph = cls.__new__(cls)
        graph.root = root
        graph.compression_prompt = compression_prompt
        graph.compression_max_tokens = compression_max_tokens
        graph.compression_tail_tokens = compression_tail_tokens
        graph.compression_tail_messages = compression_tail_messages
        graph.encoder = encoder
        graph.nodes = dict(nodes)
        graph.active_id = active_id
        return graph

    @property
    def head(self) -> str:
        for node_id, node in self.nodes.items():
            if node.parent_id is None:
                return node_id
        raise RuntimeError("branch graph has no root node")

    @property
    def tail(self) -> str:
        return self.active_id

    @property
    def active(self) -> BranchNode:
        return self.nodes[self.active_id]

    def parent(self, node_id: str) -> BranchNode | None:
        node = self.nodes[node_id]
        if node.parent_id is None:
            return None
        return self.nodes[node.parent_id]

    def children(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].children)

    def history(self) -> list[BranchNode]:
        path: list[BranchNode] = []
        node_id: str | None = self.active_id
        while node_id is not None:
            path.append(self.nodes[node_id])
            node_id = self.nodes[node_id].parent_id
        path.reverse()
        return path

    def switch_to(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"unknown branch node: {node_id}")
        self.active_id = node_id

    def switch_to_tail(self) -> None:
        node_id = self.head
        while self.nodes[node_id].children:
            node_id = self.nodes[node_id].children[-1]
        self.active_id = node_id

    def active_view(self) -> "Thread":
        from .thread import Thread

        messages: list[Message] = []
        root_msgs = self.root.thread

        if root_msgs and isinstance(root_msgs[0], SystemMessage):
            messages.append(root_msgs[0])

        for node in self.history():
            if node.compressed is not None:
                messages.append(node.compressed)

        active = self.active
        visible_end = self._visible_end(active)
        messages.extend(root_msgs[active.compressed_through : visible_end])

        return Thread(messages, **self.root._thread_kwargs())

    def _visible_end(self, node: BranchNode) -> int:
        if node.visible_end is not None:
            return node.visible_end
        return len(self.root.thread)

    def maybe_compress(self, agent: "Agent") -> BranchNode | None:
        if not self.compression_prompt:
            return None
        view = self.active_view()
        if view.token_count(self.encoder) < self.compression_max_tokens:
            return None
        return self.compress(agent)

    def compress(self, agent: "Agent") -> BranchNode | None:
        if not self.compression_prompt:
            raise ValueError("compression_prompt is required to compress")

        active = self.active
        content_start = active.compressed_through
        root_msgs = self.root.thread

        if content_start >= len(root_msgs):
            return None

        candidates = root_msgs[content_start:]
        if not candidates:
            return None

        compressible, tail = self._split_tail(candidates)
        if not compressible:
            return None

        summary_text = self._run_compression(agent, compressible)
        compressed_msg = CompressedMessage(content=summary_text)

        new_through = content_start + len(compressible)
        child_id = str(uuid.uuid4())
        child = BranchNode(
            id=child_id,
            parent_id=self.active_id,
            compressed=compressed_msg,
            compressed_through=new_through,
        )
        active.visible_end = new_through
        self.nodes[child_id] = child
        active.children.append(child_id)
        self.active_id = child_id
        return child

    def _split_tail(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
        if not messages:
            return [], []

        if self.compression_tail_messages is not None:
            tail_count = min(self.compression_tail_messages, len(messages))
            if tail_count >= len(messages):
                return [], list(messages)
            return list(messages[:-tail_count]), list(messages[-tail_count:])

        if self.compression_tail_tokens is not None:
            calculator = TokenCalculator(self.encoder)
            counts = calculator.counts(messages)
            tail: list[Message] = []
            running = 0
            for idx in range(len(messages) - 1, -1, -1):
                cost = counts[idx]
                if running + cost <= self.compression_tail_tokens:
                    tail.insert(0, messages[idx])
                    running += cost
                elif not tail:
                    tail.insert(0, messages[idx])
                    break
                else:
                    break
            if len(tail) >= len(messages):
                return [], list(messages)
            return list(messages[: len(messages) - len(tail)]), tail

        return list(messages), []

    def _format_messages_for_compression(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            elif isinstance(msg, CompressedMessage):
                role = "compressed"
            else:
                role = "message"
            lines.append(f"[{role}]: {_message_to_text(msg)}")
        return "\n\n".join(lines)

    def _run_compression(self, agent: "Agent", messages: list[Message]) -> str:
        from .thread import Thread

        prompt = self.compression_prompt
        assert prompt is not None
        formatted = self._format_messages_for_compression(messages)
        compress_thread = Thread(
            [
                SystemMessage(prompt),
                HumanMessage(content=formatted),
            ]
        )
        result = agent.invoke(
            compress_thread,
            record_output=False,
            system_prompt_override=prompt,
        )
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, default=str))
            return "\n".join(parts)
        return str(content)
