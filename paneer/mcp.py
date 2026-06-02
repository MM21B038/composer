from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPClient:
    """Paneer wrapper around MultiServerMCPClient — use server config dicts, not raw JSON blobs."""

    def __init__(
        self,
        servers: Optional[Dict[str, Any]] = None,
        *,
        tool_name_prefix: bool = False,
        callbacks: Any = None,
        tool_interceptors: Any = None,
    ) -> None:
        self._client = MultiServerMCPClient(
            servers or {},
            callbacks=callbacks,
            tool_interceptors=tool_interceptors,
            tool_name_prefix=tool_name_prefix,
        )
        self._tools: Optional[List[BaseTool]] = None

    @property
    def client(self) -> MultiServerMCPClient:
        return self._client

    @property
    def loaded(self) -> bool:
        return self._tools is not None

    @property
    def tools(self) -> List[BaseTool]:
        if self._tools is None:
            raise RuntimeError(
                "MCP tools not loaded. Call: await mcp.load()  (or await mcp.get_tools())"
            )
        return self._tools

    async def get_tools(self, *, reload: bool = False) -> List[BaseTool]:
        if self._tools is None or reload:
            self._tools = await self._client.get_tools()
        return self._tools

    async def load(self) -> List[BaseTool]:
        return await self.get_tools()

    def __repr__(self) -> str:
        count = len(self._tools) if self._tools else "?"
        return f"MCPClient(servers={list(self._client.connections.keys())}, tools={count})"
