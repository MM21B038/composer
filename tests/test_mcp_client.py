"""Tests for MCPClient persistent session management."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from composer.mcp import MCPClient


def _stdio_server(name: str = "demo") -> dict:
    return {
        name: {
            "transport": "stdio",
            "command": "python",
            "args": ["-c", "print('noop')"],
        }
    }


def test_connect_opens_one_session_per_server():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())
        mock_session = AsyncMock()

        with patch.object(mcp._client, "session", autospec=True) as session_cm:
            session_cm.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.return_value.__aexit__ = AsyncMock(return_value=None)

            await mcp.connect()

            assert mcp.connected
            assert mcp.sessions == {"demo": mock_session}
            session_cm.assert_called_once_with("demo")

            await mcp.disconnect()
            assert not mcp.connected

    asyncio.run(_run())


def test_get_tools_uses_persistent_sessions():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())
        mock_session = AsyncMock()
        fake_tools = [MagicMock(name="tool_a")]

        with patch.object(
            mcp._client, "session", autospec=True
        ) as session_cm, patch(
            "composer.mcp.load_mcp_tools", new_callable=AsyncMock
        ) as load_tools:
            session_cm.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.return_value.__aexit__ = AsyncMock(return_value=None)
            load_tools.return_value = fake_tools

            tools = await mcp.get_tools()

            assert tools == fake_tools
            load_tools.assert_awaited_once()
            args, kwargs = load_tools.await_args
            assert args[0] is mock_session
            assert kwargs["server_name"] == "demo"

            await mcp.disconnect()

    asyncio.run(_run())


def test_get_tools_reuses_cached_tools_without_reload():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())
        mcp._tools = [MagicMock(name="cached")]

        with patch.object(mcp, "connect", new_callable=AsyncMock) as connect:
            tools = await mcp.get_tools()

        connect.assert_awaited_once()
        assert tools is mcp._tools

    asyncio.run(_run())


def test_async_context_manager_connects_and_disconnects():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())

        with patch.object(mcp, "connect", new_callable=AsyncMock) as connect, patch.object(
            mcp, "disconnect", new_callable=AsyncMock
        ) as disconnect:
            async with mcp as client:
                assert client is mcp
            connect.assert_awaited_once()
            disconnect.assert_awaited_once()

    asyncio.run(_run())


def test_ensure_connected_reconnects_on_loop_change():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())
        mcp._sessions = {"demo": AsyncMock()}
        mcp._loop = MagicMock(name="old_loop")
        exit_stack = AsyncMock()
        exit_stack.aclose = AsyncMock()
        mcp._exit_stack = exit_stack

        with patch.object(mcp, "connect", new_callable=AsyncMock) as connect:
            await mcp.ensure_connected()

        exit_stack.aclose.assert_awaited_once()
        connect.assert_awaited_once()

    asyncio.run(_run())


def test_disconnect_clears_catalog_cache():
    async def _run() -> None:
        mcp = MCPClient(_stdio_server())
        mcp._tools = []
        mcp._prompts = []
        mcp._resources = []
        mcp._sessions = {"demo": AsyncMock()}
        mcp._exit_stack = AsyncMock()
        mcp._exit_stack.aclose = AsyncMock()

        await mcp.disconnect()

        assert mcp._tools is None
        assert mcp._prompts is None
        assert mcp._resources is None

    asyncio.run(_run())
