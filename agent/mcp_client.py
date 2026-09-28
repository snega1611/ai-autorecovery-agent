"""
Thin async MCP client wrapper. Launches mcp_server/server.py as a stdio
subprocess and exposes list_tools()/call_tool() to the LangGraph agent, and
a helper to convert MCP tool schemas into the format Ollama's /api/chat
`tools` field expects (they're both JSON-schema based, but the wrapping
differs slightly).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).resolve().parent.parent / "mcp_server" / "server.py")


@asynccontextmanager
async def mcp_session():
    """Async context manager yielding a live, initialized MCP ClientSession
    connected to the tools server subprocess."""
    params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_ollama_tools(session: ClientSession) -> List[Dict[str, Any]]:
    """Fetches MCP tool schemas and reshapes them into Ollama's expected
    `tools` format: [{type: "function", function: {name, description,
    parameters}}, ...]."""
    resp = await session.list_tools()
    ollama_tools = []
    for t in resp.tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        })
    return ollama_tools


async def call_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> Any:
    """Calls an MCP tool and returns the parsed JSON result (our tools all
    return a single TextContent block containing a JSON string)."""
    result = await session.call_tool(name, arguments=arguments)
    if not result.content:
        return {"error": "empty tool result"}
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
