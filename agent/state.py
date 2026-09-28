from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    record_id: str
    record: Dict[str, Any]                 # the FAILED record under investigation
    messages: List[Dict[str, Any]]         # full chat history sent to the LLM
    evidence: List[Dict[str, Any]]         # every tool result gathered so far
    tool_call_count: int
    max_tool_calls: int
    done: bool
    final_outcome: Optional[str]           # "auto_recovered" | "escalated" | "escalated_max_