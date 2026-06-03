from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

Message = Union[HumanMessage, AIMessage]


@dataclass(frozen=True)
class MCPPromptInfo:
    name: str
    server: str
    description: Optional[str] = None
    arguments: Optional[List[Any]] = None


@dataclass(frozen=True)
class MCPResourceInfo:
    uri: str
    server: str
    name: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = None


class MCPClient:
    """Composer wrapper around MultiServerMCPClient — use server config dicts, not raw JSON blobs."""

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
        self._prompts: Optional[List[MCPPromptInfo]] = None
        self._resources: Optional[List[MCPResourceInfo]] = None

    @property
    def client(self) -> MultiServerMCPClient:
        return self._client

    @property
    def loaded(self) -> bool:
        return self._tools is not None

    @property
    def prompts_loaded(self) -> bool:
        return self._prompts is not None

    @property
    def resources_loaded(self) -> bool:
        return self._resources is not None

    @property
    def tools(self) -> List[BaseTool]:
        if self._tools is None:
            raise RuntimeError(
                "MCP tools not loaded. Call: await mcp.load_tools()  (or await mcp.get_tools())"
            )
        return self._tools

    @property
    def prompts(self) -> List[MCPPromptInfo]:
        if self._prompts is None:
            raise RuntimeError(
                "MCP prompts not loaded. Call: await mcp.load_prompts()  (or await mcp.get_prompts())"
            )
        return self._prompts

    @property
    def resources(self) -> List[MCPResourceInfo]:
        if self._resources is None:
            raise RuntimeError(
                "MCP resources not loaded. Call: await mcp.load_resources()  "
                "(or await mcp.get_resources())"
            )
        return self._resources

    async def get_tools(self, *, reload: bool = False) -> List[BaseTool]:
        if self._tools is None or reload:
            self._tools = await self._client.get_tools()
        return self._tools

    async def load_tools(self, *, reload: bool = False) -> List[BaseTool]:
        return await self.get_tools(reload=reload)

    async def load(self) -> List[BaseTool]:
        return await self.load_tools()

    async def get_prompts(self, *, reload: bool = False) -> List[MCPPromptInfo]:
        if self._prompts is None or reload:
            self._prompts = await self._list_prompts()
        return self._prompts

    async def load_prompts(self, *, reload: bool = False) -> List[MCPPromptInfo]:
        return await self.get_prompts(reload=reload)

    async def get_resources(self, *, reload: bool = False) -> List[MCPResourceInfo]:
        if self._resources is None or reload:
            self._resources = await self._list_resources()
        return self._resources

    async def load_resources(self, *, reload: bool = False) -> List[MCPResourceInfo]:
        return await self.get_resources(reload=reload)

    async def get_prompt(
        self,
        name: str,
        *,
        server: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """Fetch a prompt by name. Resolves server from the loaded prompt catalog when omitted."""
        resolved_server = server or self._resolve_prompt_server(name)
        return await self._client.get_prompt(
            resolved_server,
            name,
            arguments=arguments,
        )

    async def get_resource(
        self,
        uri: str,
        *,
        server: Optional[str] = None,
    ) -> List[Blob]:
        """Fetch resource content by URI. Resolves server from the loaded resource catalog when omitted."""
        resolved_server = server or self._resolve_resource_server(uri)
        return await self._client.get_resources(
            server_name=resolved_server,
            uris=uri,
        )

    def _resolve_prompt_server(self, name: str) -> str:
        if self._prompts is None:
            raise RuntimeError(
                f"Cannot resolve prompt {name!r} without a server. "
                "Call await mcp.load_prompts() first, or pass server=..."
            )
        matches = [p for p in self._prompts if p.name == name]
        if not matches:
            known = [p.name for p in self._prompts]
            raise ValueError(
                f"Unknown MCP prompt {name!r}. Known prompts: {known or '(none)'}"
            )
        if len(matches) > 1:
            servers = [p.server for p in matches]
            raise ValueError(
                f"Prompt {name!r} exists on multiple servers: {servers}. Pass server=..."
            )
        return matches[0].server

    def _resolve_resource_server(self, uri: str) -> str:
        if self._resources is None:
            raise RuntimeError(
                f"Cannot resolve resource {uri!r} without a server. "
                "Call await mcp.load_resources() first, or pass server=..."
            )
        matches = [r for r in self._resources if r.uri == uri]
        if not matches:
            known = [r.uri for r in self._resources]
            raise ValueError(
                f"Unknown MCP resource {uri!r}. Known resources: {known or '(none)'}"
            )
        if len(matches) > 1:
            servers = [r.server for r in matches]
            raise ValueError(
                f"Resource {uri!r} exists on multiple servers: {servers}. Pass server=..."
            )
        return matches[0].server

    async def _list_prompts(self) -> List[MCPPromptInfo]:
        prompts: List[MCPPromptInfo] = []
        for server_name in self._client.connections:
            async with self._client.session(server_name) as session:
                result = await session.list_prompts()
                for prompt in result.prompts:
                    prompts.append(
                        MCPPromptInfo(
                            name=prompt.name,
                            server=server_name,
                            description=prompt.description,
                            arguments=list(prompt.arguments) if prompt.arguments else None,
                        )
                    )
        return prompts

    async def _list_resources(self) -> List[MCPResourceInfo]:
        resources: List[MCPResourceInfo] = []
        for server_name in self._client.connections:
            async with self._client.session(server_name) as session:
                result = await session.list_resources()
                for resource in result.resources:
                    resources.append(
                        MCPResourceInfo(
                            uri=str(resource.uri),
                            server=server_name,
                            name=resource.name,
                            description=resource.description,
                            mime_type=resource.mimeType,
                        )
                    )
        return resources

    def __repr__(self) -> str:
        tool_count = len(self._tools) if self._tools else "?"
        prompt_count = len(self._prompts) if self._prompts else "?"
        resource_count = len(self._resources) if self._resources else "?"
        return (
            f"MCPClient(servers={list(self._client.connections.keys())}, "
            f"tools={tool_count}, prompts={prompt_count}, resources={resource_count})"
        )
