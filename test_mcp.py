import asyncio
from agent.mcp_client import mcp_session


async def main():
    async with mcp_session() as session:
        result = await session.list_tools()
        print("MCP CONNECTED")
        print("Tools:", [tool.name for tool in result.tools])


asyncio.run(main())