from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_server import tools as tool_impl

server = Server("recall-tools")


TOOL_SCHEMAS = [
    Tool(
        name="get_record",
        description="Fetch a customer request record by its record_id.",
        inputSchema={
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    ),
    Tool(
        name="get_transcript",
        description="Fetch the full call transcript for a customer_id.",
        inputSchema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    ),
    Tool(
        name="search_transcript",
        description=(
            "Semantically search a customer's call transcript for evidence "
            "relevant to a query (e.g. 'policy number the customer stated'). "
            "Returns matching turns, catching paraphrases plain keyword "
            "search would miss."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "query": {"type": "string"},
                "record_id": {"type": "string", "description": "for audit logging"},
            },
            "required": ["customer_id", "query"],
        },
    ),
    Tool(
        name="get_policy",
        description="Retrieve the company policy text relevant to a topic (e.g. 'cancellation requirements', 'same day date change').",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "record_id": {"type": "string", "description": "for audit logging"},
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="lookup_customer_by_phone",
        description="Look up which customer_id(s) a phone number matches, for identity verification.",
        inputSchema={
            "type": "object",
            "properties": {
                "phone_number": {"type": "string"},
                "record_id": {"type": "string", "description": "for audit logging"},
            },
            "required": ["phone_number"],
        },
    ),
    Tool(
        name="validate_field",
        description=(
            "Deterministically validate ONE field at a time. "
            "IMPORTANT: field_name MUST be exactly ONE string: "
            "'policy_number' OR 'effective_date'. "
            "candidate_values must contain only the values for that ONE field. "
            "NEVER put multiple field names in field_name. "
            "If both policy_number and effective_date need validation, "
            "make TWO separate validate_field calls. "
            "ALWAYS call this before proposing a recovery -- do not decide "
            "validity yourself."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "enum": ["policy_number", "effective_date"]
                },
                "candidate_values": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "request_type": {"type": "string"},
                "record_id": {
                    "type": "string",
                    "description": "for audit logging"
                },
            },
            "required": ["field_name", "candidate_values"],
        },
    ),
    Tool(
        name="propose_recovery",
        description=(
            "Propose the recovered field values for a record, along with the "
            "validate_field results for each. The system (not you) makes the "
            "final auto-apply-vs-escalate decision from those validations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "fields": {"type": "object"},
                "field_validations": {"type": "object"},
                "evidence_summary": {"type": "string"},
            },
            "required": ["record_id", "fields", "field_validations", "evidence_summary"],
        },
    ),
    Tool(
        name="escalate_to_human",
        description="Escalate this record for human review without proposing a recovery (e.g. no usable evidence found).",
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "reason": {"type": "string"},
                "evidence_summary": {"type": "string"},
            },
            "required": ["record_id", "reason", "evidence_summary"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOL_SCHEMAS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in tool_impl.TOOL_REGISTRY:
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool {name}"}))]
    try:
        result = tool_impl.TOOL_REGISTRY[name](**arguments)
    except Exception as e:
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, default=str))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
