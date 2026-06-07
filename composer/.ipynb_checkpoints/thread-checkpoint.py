from typing import List, Union, Optional, TYPE_CHECKING

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


Message = Union[
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    CompressedMessage,
]


class Thread:
    def __init__(
        self,
        thread: Optional[List[Message]] = None,
        auto_append_ai_message = True
    ):
        self.thread = thread or []
        self.auto_append_ai_message = auto_append_ai_message

    def copy(self):
        return Thread(self.thread.copy())

    def add_message(self, message: Message):
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

    # -------------------------
    # Operators
    # -------------------------

    def __add__(self, other: "Thread") -> "Thread":
        new_thread = self.copy()

        for message in other.get_messages():
            new_thread.add_message(message)

        return new_thread

    def __or__(
        self,
        other: Union[
            Message,
            "Thread",
            "Agent",
        ],
    ):
        from .agent import Agent

        new_thread = self.copy()

        if isinstance(other, Thread):
            return new_thread + other

        if isinstance(other, Agent):
            ai_message = other(new_thread)
            if self.auto_append_ai_message and isinstance(ai_message, AIMessage):
                self.thread.add_message(ai_message)
            return ai_message

        new_thread.add_message(other)

        return new_thread

    def __ror__(self, other):
        from .agent import Agent

        new_thread = self.copy()

        if isinstance(other, Thread):
            return other + new_thread

        if isinstance(other, Message.__args__):
            temp = Thread()
            temp.add_message(other)
            return temp + new_thread

        if isinstance(other, Agent):
            ai_message = other(new_thread)
            if self.auto_append_ai_message and isinstance(ai_message, AIMessage):
                self.thread.add_message(ai_message)
            return ai_message

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
        return self.thread[index]

    def __len__(self):
        return len(self.thread)

    def __iter__(self):
        return iter(self.thread)

    def __repr__(self):
        return f"Thread(messages={len(self.thread)})"