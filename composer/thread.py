from typing import List, Union, Optional, TYPE_CHECKING, get_args
import asyncio

from langchain_core.messages import (
    HumanMessage as BaseHumanMessage,
    AIMessage as BaseAIMessage,
    SystemMessage as BaseSystemMessage,
    ToolMessage as BaseToolMessage,
)

if TYPE_CHECKING:
    from .agent import Agent


class HumanMessage(BaseHumanMessage):
    pass


class AIMessage(BaseAIMessage):
    pass


class SystemMessage(BaseSystemMessage):
    pass


class ToolMessage(BaseToolMessage):
    pass

class CompressedMessage(BaseHumanMessage):
    pass

class UpdateMessage(BaseHumanMessage):
    pass


Message = Union[
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    CompressedMessage,
    UpdateMessage,
]


class AgentInvoke:
    __slots__ = ("_agent", "_thread", "_append_to", "_result")

    def __init__(self, agent: "Agent", thread: "Thread", append_to: "Thread"):
        self._agent = agent
        self._thread = thread
        self._append_to = append_to
        self._result: Optional[AIMessage] = None

    def __await__(self):
        return self._agent.ainvoke(
            self._thread, append_to=self._append_to
        ).__await__()

    def resolve(self) -> AIMessage:
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


class Thread:
    def __init__(
        self,
        thread: Optional[List[Message]] = None,
        auto_append_ai_message = True
    ):
        self.thread = thread or []
        self.auto_append_ai_message = auto_append_ai_message

    def copy(self):
        return Thread(
            self.thread.copy(),
            auto_append_ai_message=self.auto_append_ai_message,
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

        elif (
            isinstance(message, UpdateMessage)
        ):
            for i in range(len(self.thread) - 1, -1, -1):
                if isinstance(self.thread[i], UpdateMessage):
                    self.thread.pop(i)
                    break
            self.thread.append(message)

        else:
            self.thread.append(message)

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
            new_thread = self.copy()
            pending = AgentInvoke(other, new_thread, self)
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
            new_thread = self.copy()
            pending = AgentInvoke(other, new_thread, self)
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
            return Thread(
                self.thread[index],
                auto_append_ai_message=self.auto_append_ai_message,
            )
        return self.thread[index]

    def __len__(self):
        return len(self.thread)

    def __iter__(self):
        return iter(self.thread)

    def __repr__(self):
        return f"Thread(messages={len(self.thread)})"