from typing import Literal, List, Optional, Union, AsyncIterator, Iterator
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from .thread import SystemMessage, AIMessage, Thread
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

PROVIDERS = Literal["custom"]


class Agent:
    def __init__(
        self,
        provider: Optional[PROVIDERS] = "custom",
        model: Optional[Union[str, BaseChatModel]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
    ):
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.base_url = base_url
        self.api_key = api_key

    def _prepare_messages(self, thread: Thread):
        messages = thread.get_messages()
        
        if (
            self.system_prompt
            and messages
            and isinstance(messages[0], SystemMessage)
        ):
            messages = messages[1:]

        return messages

    def _get_model(self):
        if isinstance(self.model, BaseChatModel):
            return self.model

        if self.provider == "custom":
            return ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
            )

        raise NotImplementedError(
            f"Provider {self.provider} is not supported yet."
        )

    def _get_agent(self):
        model = self._get_model()

        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )


    def __call__(self, thread: Thread):
        return self.invoke(thread)

    def invoke(self, thread: Thread):
        agent = self._get_agent()

        response = agent.invoke(
            {
                "messages": self._prepare_messages(thread)
            }
        )
        return AIMessage(response["messages"][-1])

    async def ainvoke(self, thread: Thread):
        agent = self._get_agent()

        response = await agent.ainvoke(
            {
                "messages": self._prepare_messages(thread)
            }
        )

        return AIMessage(response["messages"][-1])

    def stream(
        self,
        thread: Thread,
        stream_mode: str = "messages",
    ) -> Iterator:

        agent = self._get_agent()

        for chunk in agent.stream(
            {
                "messages": self._prepare_messages(thread)
            },
            stream_mode=stream_mode,
        ):
            yield chunk

    async def astream(
        self,
        thread: Thread,
        stream_mode: str = "messages",
    ) -> AsyncIterator:

        agent = self._get_agent()

        async for chunk in agent.astream(
            {
                "messages": self._prepare_messages(thread)
            },
            stream_mode=stream_mode,
        ):
            yield chunk