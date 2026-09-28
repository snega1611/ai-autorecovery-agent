import asyncio
from agent.mcp_client import mcp_session, list_ollama_tools

async def main():
    async with mcp_session() as session:
        tools = await list_ollama_tools(session)

        for tool in tools:
            if tool["function"]["name"] == "propose_recovery":
                print(tool)

asyncio.run(main())