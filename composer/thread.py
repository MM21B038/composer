from typing import List, Union, Optional, TYPE_CHECKING, get_args, Literal, ClassVar
import asyncio
import json

import tiktoken
from pydantic import model_validator
from tiktoken.core import Encoding

from langchain_core.messages import (
    HumanMessage as BaseHumanMessage,
    AIMessage as BaseAIMessage,
    SystemMessage as BaseSystemMessage,
    ToolMessage as BaseToolMessage,
)

from .image import (
    ImageAttach,
    ImageSource,
    build_image_block,
    merge_image_content,
    resolve_config,
)
from .tool_hide import (
    ToolResultHideRule,
    apply_rolling_message_window,
    apply_tool_hide_rules_to_messages,
    get_original_content,
    is_hidden_for_model,
    persist_tool_hide_rules,
    restore_hidden_tool_messages,
    split_hide_rules_by_mode,
    trim_messages_to_token_budget,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .thread_branch import ThreadBranchGraph


class _AttachDescriptor:
    """Supports HumanMessage.attach(...) and HumanMessage(...).attach(...)."""

    def __get__(self, obj: Optional["HumanMessage"], owner: type["HumanMessage"]):
        def bound(
            source: ImageSource,
            config: ImageAttach | None = None,
            **kwargs: object,
        ) -> "HumanMessage":
            cfg = resolve_config(config, **kwargs)
            image_block = build_image_block(source, cfg)
            if obj is None:
                return owner(content=[image_block])
            return owner(content=merge_image_content(obj.content, image_block))

        return bound


class HumanMessage(BaseHumanMessage):
    attach: ClassVar[_AttachDescriptor] = _AttachDescriptor()


class ImageMessage(HumanMessage):
    """Human message containing a single image block (no text)."""

    def __init__(
        self,
        source: ImageSource,
        config: ImageAttach | None = None,
        **kwargs: object,
    ) -> None:
        cfg = resolve_config(config, **kwargs)
        super().__init__(content=[build_image_block(source, cfg)])


class AIMessage(BaseAIMessage):
    @model_validator(mode="after")
    def _normalize_reasoning_content(self) -> "AIMessage":
        from .stream import extract_assistant_text, extract_thinking

        assistant_text = extract_assistant_text(self)
        thinking_text = extract_thinking(self)
        content = self.content

        if assistant_text and not (isinstance(content, str) and content):
            if not isinstance(content, list):
                object.__setattr__(self, "content", assistant_text)
            else:
                object.__setattr__(self, "content", assistant_text)

        kwargs = dict(self.additional_kwargs or {})
        if thinking_text:
            kwargs["reasoning_content"] = thinking_text
        for key in ("reasoning", "thinking", "reasoning_details"):
            kwargs.pop(key, None)
        object.__setattr__(self, "additional_kwargs", kwargs)
        return self


class SystemMessage(BaseSystemMessage):
    pass


class ToolMessage(BaseToolMessage):
    pass


class CompressedMessage(BaseHumanMessage):
    pass


Message = Union[
    HumanMessage,
    ImageMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    CompressedMessage,
]

EncoderType = Literal[
    "gpt2",
    "r50k_base",
    "p50k_base",
    "p50k_edit",
    "cl100k_base",
    "o200k_base",
    "o200k_harmony",
]

_encoders: dict[EncoderType, Encoding] = {}


def _get_encoder(encoder: EncoderType) -> Encoding:
    if encoder not in _encoders:
        _encoders[encoder] = tiktoken.get_encoding(encoder)
    return _encoders[encoder]


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _message_to_text(message: Message) -> str:
    parts = [_content_to_text(message.content)]
    if isinstance(message, AIMessage) and message.tool_calls:
        parts.append(json.dumps(message.tool_calls, default=str))
    if isinstance(message, ToolMessage) and message.tool_call_id:
        parts.append(message.tool_call_id)
    return "\n".join(parts)


class TokenCalculator:
    def __init__(self, encoder: EncoderType = "cl100k_base"):
        self.encoder = encoder
        self._enc = _get_encoder(encoder)

    def count_message(self, message: Message) -> int:
        return len(self._enc.encode(_message_to_text(message)))

    def count(self, messages: List[Message]) -> int:
        return sum(self.count_message(message) for message in messages)

    def counts(self, messages: List[Message]) -> List[int]:
        return [self.count_message(message) for message in messages]

    @classmethod
    def count_messages(
        cls,
        messages: Union["Thread", List[Message]],
        encoder: EncoderType = "cl100k_base",
    ) -> int:
        return cls(encoder).count(cls._resolve_messages(messages))

    @classmethod
    def message_counts(
        cls,
        messages: Union["Thread", List[Message]],
        encoder: EncoderType = "cl100k_base",
    ) -> List[int]:
        return cls(encoder).counts(cls._resolve_messages(messages))

    @staticmethod
    def _resolve_messages(messages: Union["Thread", List[Message]]) -> List[Message]:
        if isinstance(messages, Thread):
            return messages.thread
        return messages


class AgentInvoke:
    __slots__ = ("_agent", "_thread", "_append_to", "_branch", "_prepared", "_result")

    def __init__(
        self,
        agent: "Agent",
        thread: "Thread",
        append_to: "Thread",
        branch: Optional["ThreadBranchGraph"] = None,
    ):
        self._agent = agent
        self._thread = thread
        self._append_to = append_to
        self._branch = branch
        self._prepared = False
        self._result: Optional[AIMessage] = None

    def _prepare(self) -> None:
        if self._prepared:
            return
        if self._branch is not None:
            self._branch.maybe_compress(self._agent)
            self._thread = self._branch.active_view().copy()
        self._prepared = True

    def __await__(self):
        self._prepare()
        return self._agent.ainvoke(
            self._thread, append_to=self._append_to
        ).__await__()

    def resolve(self) -> AIMessage:
        self._prepare()
        if self._result is None:
            self._result = self._agent.invoke(
                self._thread, append_to=self._append_to
            )
        return self._result

    def __getattr__(self, name: str):
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        if self._result is not None:
            return repr(self._result)
        return "<AgentInvoke pending>"

    def _ipython_display_(self) -> None:
        """Resolve and display in Jupyter instead of showing '<AgentInvoke pending>'."""
        from IPython.display import display

        display(self.resolve())


class Thread:
    def __init__(
        self,
        thread: Optional[List[Message]] = None,
        *,
        auto_append_ai_message: bool = True,
        auto_append_tool_calls: bool = True,
        auto_append_tool_results: bool = True,
        tool_hide_rules: Optional[List[ToolResultHideRule]] = None,
        persist_tool_hides: bool = False,
        max_messages_for_model: Optional[int] = None,
        max_tokens_for_model: Optional[int] = None,
        model_view_encoder: EncoderType = "cl100k_base",
        compression_prompt: Optional[str] = None,
        compression_max_tokens: int = 96_000,
        compression_tail_tokens: Optional[int] = 8_000,
        compression_tail_messages: Optional[int] = None,
    ):
        self.thread = thread or []
        self.auto_append_ai_message = auto_append_ai_message
        self.auto_append_tool_calls = auto_append_tool_calls
        self.auto_append_tool_results = auto_append_tool_results
        self.tool_hide_rules: List[ToolResultHideRule] = list(tool_hide_rules or [])
        self.persist_tool_hides = persist_tool_hides
        self.max_messages_for_model = max_messages_for_model
        self.max_tokens_for_model = max_tokens_for_model
        self.model_view_encoder = model_view_encoder
        self.compression_prompt = compression_prompt
        self.compression_max_tokens = compression_max_tokens
        self.compression_tail_tokens = compression_tail_tokens
        self.compression_tail_messages = compression_tail_messages
        self._branch_graph: Optional["ThreadBranchGraph"] = None

    @property
    def branch(self) -> "ThreadBranchGraph":
        if self._branch_graph is None:
            from .thread_branch import ThreadBranchGraph

            self._branch_graph = ThreadBranchGraph(
                self,
                compression_prompt=self.compression_prompt,
                compression_max_tokens=self.compression_max_tokens,
                compression_tail_tokens=self.compression_tail_tokens,
                compression_tail_messages=self.compression_tail_messages,
                encoder=self.model_view_encoder,
            )
        return self._branch_graph

    def _thread_kwargs(self) -> dict:
        return {
            "auto_append_ai_message": self.auto_append_ai_message,
            "auto_append_tool_calls": self.auto_append_tool_calls,
            "auto_append_tool_results": self.auto_append_tool_results,
            "tool_hide_rules": list(self.tool_hide_rules),
            "persist_tool_hides": self.persist_tool_hides,
            "max_messages_for_model": self.max_messages_for_model,
            "max_tokens_for_model": self.max_tokens_for_model,
            "model_view_encoder": self.model_view_encoder,
            "compression_prompt": self.compression_prompt,
            "compression_max_tokens": self.compression_max_tokens,
            "compression_tail_tokens": self.compression_tail_tokens,
            "compression_tail_messages": self.compression_tail_messages,
        }

    def copy(self):
        return Thread(self.thread.copy(), **self._thread_kwargs())

    def _persist_hide_rules(self) -> List[ToolResultHideRule]:
        _, persist_rules = split_hide_rules_by_mode(
            self.tool_hide_rules,
            self.persist_tool_hides,
        )
        return persist_rules

    def _invoke_hide_rules(self) -> List[ToolResultHideRule]:
        invoke_rules, _ = split_hide_rules_by_mode(
            self.tool_hide_rules,
            self.persist_tool_hides,
        )
        return invoke_rules

    def add_tool_hide_rule(self, rule: ToolResultHideRule) -> None:
        self.tool_hide_rules.append(rule)
        if self._persist_hide_rules():
            self.apply_tool_hide_rules(retroactive=True)

    def remove_tool_hide_rule(
        self,
        *,
        tool_name: str,
        server: Optional[str] = None,
    ) -> bool:
        before = len(self.tool_hide_rules)
        self.tool_hide_rules = [
            rule
            for rule in self.tool_hide_rules
            if not (rule.tool_name == tool_name and rule.server == server)
        ]
        return len(self.tool_hide_rules) < before

    def apply_tool_hide_rules(self, *, retroactive: bool = True) -> None:
        if not retroactive or not self.tool_hide_rules:
            return
        persist_rules = self._persist_hide_rules()
        if persist_rules:
            persist_tool_hide_rules(self.thread, persist_rules)

    def messages_for_model(
        self,
        *,
        max_tokens: Optional[int] = None,
        max_messages: Optional[int] = None,
        encoder: Optional[EncoderType] = None,
    ) -> List[Message]:
        """Collapsed view of thread messages for LLM context."""
        enc = encoder or self.model_view_encoder
        invoke_rules = self._invoke_hide_rules()
        messages = apply_tool_hide_rules_to_messages(
            self.thread,
            invoke_rules,
            persist=False,
        )

        window = (
            max_messages
            if max_messages is not None
            else self.max_messages_for_model
        )
        if window is not None:
            messages = apply_rolling_message_window(messages, window)

        budget = max_tokens if max_tokens is not None else self.max_tokens_for_model
        if budget is not None:
            messages = trim_messages_to_token_budget(messages, budget, enc)

        return messages

    def get_original_tool_content(self, message: ToolMessage) -> str:
        return get_original_content(message)

    def is_tool_hidden_for_model(self, message: ToolMessage) -> bool:
        return is_hidden_for_model(message)

    def appends_agent_messages(self) -> bool:
        return (
            self.auto_append_ai_message
            or self.auto_append_tool_calls
            or self.auto_append_tool_results
        )

    def append(self, message: Message):
        if (
            isinstance(message, SystemMessage)
            and self.thread
            and isinstance(self.thread[0], SystemMessage)
        ):
            self.thread[0] = message

        elif (
            isinstance(message, SystemMessage)
            and self.thread
            and not isinstance(self.thread[0], SystemMessage)
        ):
            self.thread.insert(0, message)

        else:
            self.thread.append(message)
            if isinstance(message, ToolMessage) and self._persist_hide_rules():
                self.apply_tool_hide_rules(retroactive=True)

    def get_messages(self) -> List[Message]:
        return self.thread

    def clear_messages(self):
        if (
            self.thread
            and isinstance(self.thread[0], SystemMessage)
        ):
            self.thread = self.thread[:1]
        else:
            self.thread = []

    def clear_thread(self):
        self.thread = []

    def pop(self, index):
        return self.thread.pop(index)

    def restore_hidden_tool_messages(self) -> int:
        return restore_hidden_tool_messages(self.thread)

    def token_count(
        self,
        encoder: EncoderType = "cl100k_base",
    ) -> int:
        return TokenCalculator.count_messages(self, encoder)

    def token_count_for_model(
        self,
        encoder: Optional[EncoderType] = None,
    ) -> int:
        enc = encoder or self.model_view_encoder
        return TokenCalculator.count_messages(
            self.messages_for_model(encoder=enc),
            enc,
        )

    def token_counts(
        self,
        encoder: EncoderType = "cl100k_base",
    ) -> List[int]:
        return TokenCalculator.message_counts(self, encoder)

    # -------------------------
    # Operators
    # -------------------------

    def __add__(self, other: "Thread") -> "Thread":
        new_thread = self.copy()

        for message in other.get_messages():
            new_thread.append(message)

        return new_thread

    def __or__(
        self,
        other: Union[
            "Agent",
        ],
    ):
        from .agent import Agent

        if isinstance(other, Thread):
            raise TypeError(
                "Use + to combine threads (thread_a + thread_b), not |"
            )

        if isinstance(other, get_args(Message)):
            raise TypeError(
                "Use message | thread to append a message, not thread | message"
            )

        if isinstance(other, Agent):
            invoke_thread = self.branch.active_view().copy()
            pending = AgentInvoke(other, invoke_thread, self, self.branch)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return pending.resolve()
            return pending

        raise TypeError(f"Unsupported operand for |: Thread and {type(other).__name__}")

    def __ror__(self, other):
        from .agent import Agent

        if isinstance(other, Thread):
            raise TypeError(
                "Use + to combine threads (thread_a + thread_b), not |"
            )

        if isinstance(other, get_args(Message)):
            self.append(other)
            return self

        if isinstance(other, Agent):
            invoke_thread = self.branch.active_view().copy()
            pending = AgentInvoke(other, invoke_thread, self, self.branch)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return pending.resolve()
            return pending

        return NotImplemented

    def __rshift__(
        self,
        other: Optional[CompressedMessage] = None,
    ) -> "Thread":
        new_thread = self.copy()
        new_thread.clear_messages()

        return (other | new_thread) if other else new_thread

    # -------------------------
    # Collection behavior
    # -------------------------

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Thread(self.thread[index], **self._thread_kwargs())
        return self.thread[index]

    def __len__(self):
        return len(self.thread)

    def __iter__(self):
        return iter(self.thread)

    def __repr__(self):
        return f"Thread(messages={len(self.thread)})"