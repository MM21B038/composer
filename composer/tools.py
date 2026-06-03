from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import BaseTool


@dataclass
class ToolCall:
    """Model-initiated tool invocation (composer type, not a JSON blob)."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    type: str = "tool_call"

    def __repr__(self) -> str:
        return f"ToolCall(name={self.name!r}, args={self.args!r}, id={self.id!r})"

    @classmethod
    def from_langchain(cls, tool_call: Any) -> "ToolCall":
        if isinstance(tool_call, cls):
            return tool_call
        if isinstance(tool_call, dict):
            data = tool_call
        elif hasattr(tool_call, "model_dump"):
            data = tool_call.model_dump()
        else:
            data = dict(tool_call)

        raw_args = data.get("args", {})
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            args = _parse_args_string(raw_args)
        else:
            args = {}

        return cls(
            id=data.get("id") or "",
            name=data.get("name") or "",
            args=args,
            type=data.get("type") or "tool_call",
        )


@dataclass
class ToolResult:
    """Output returned from running a tool."""

    content: str
    tool_call_id: str = ""
    name: Optional[str] = None

    @classmethod
    def from_message(cls, message: Any) -> "ToolResult":
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else str(content)
        return cls(
            content=text,
            tool_call_id=getattr(message, "tool_call_id", None) or "",
            name=getattr(message, "name", None),
        )


def _parse_args_string(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": raw}


def format_tool_output(result: Any) -> str:
    """Normalize LangChain tool return values to a string."""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)
    return str(result)


def tools_by_name(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


def lookup_tool(name: str, tools: List[BaseTool]) -> BaseTool:
    by_name = tools_by_name(tools)
    if name not in by_name:
        available = ", ".join(sorted(by_name))
        raise KeyError(f"Tool {name!r} not found. Available: {available}")
    return by_name[name]


async def invoke_tool(tool: BaseTool, args: Dict[str, Any]) -> str:
    """Run one LangChain tool and return string content."""
    try:
        if hasattr(tool, "ainvoke"):
            output = await tool.ainvoke(args)
        else:
            output = tool.invoke(args)
    except Exception as exc:
        return f"Tool execution failed: {exc}"
    return format_tool_output(output)


async def run_tool_call(
    call: ToolCall,
    tools: List[BaseTool],
) -> ToolResult:
    """Execute a ToolCall against a resolved tool list; returns ToolResult."""
    tool = lookup_tool(call.name, tools)
    content = await invoke_tool(tool, call.args)
    return ToolResult(
        content=content,
        tool_call_id=call.id,
        name=call.name,
    )


async def combine_tools(*sources: Any) -> List[BaseTool]:
    """Merge local tools and MCPClient-loaded tools into one LangChain tool list."""
    from .mcp import MCPClient

    combined: List[BaseTool] = []
    for source in sources:
        if isinstance(source, MCPClient):
            combined.extend(await source.get_tools())
        elif isinstance(source, list):
            combined.extend(source)
        else:
            combined.append(source)
    return combined
