"""
Append-only JSONL audit log. Every tool call, agent decision, and human
action gets written here so any record's full history can be reconstructed.
Intentionally boring/simple: this is the part that's cheap to build but that
interviewers specifically ask about ("how do you know what the agent did?").
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterator, Dict, Any

from common.schema import AuditEntry

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit" / "audit_log.jsonl"


def log(record_id: str, event: str, detail: Dict[str, Any]) -> AuditEntry:
    entry = AuditEntry(record_id=record_id, event=event, detail=detail)
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")
    return entry


def read_all() -> Iterator[Dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_for_record(record_id: str):
    return [e for e in read_all() if e.get("record_id") == record_id]


if __name__ == "__main__":
    # quick smoke test
    log("demo_record_1", "tool_call", {"tool": "get_transcript", "result_len": 42})
    log("demo_record_1", "decision", {"decision": "escalate", "reason": "conflicting values"})
    for e in read_for_record("demo_record_1"):
        print(e)
