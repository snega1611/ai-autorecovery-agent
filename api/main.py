from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from common.audit_log import (
    log as audit_log,
    read_for_record,
)
from common.schema import RecordStatus


# =========================================================
# Paths
# =========================================================

DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data"
)

RECORDS_PATH = DATA_DIR / "records.json"

PENDING_PATH = (
    DATA_DIR / "pending_approvals.json"
)

TRANSCRIPTS_PATH = (
    DATA_DIR / "transcripts.json"
)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Recall - Human Approval Queue"
)


# =========================================================
# JSON helpers
# =========================================================

def _load(path: Path, default):
    """
    Load JSON from disk.

    If the file does not exist or is empty,
    return the supplied default value.
    """

    if not path.exists():
        return default

    text = path.read_text().strip()

    if not text:
        return default

    return json.loads(text)


def _save(path: Path, data):
    """
    Save JSON data to disk.
    """

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        )
    )


def _get_transcript(transcript_id: str):
    """
    Find a transcript using the transcript_id stored
    on the corresponding customer record.
    """

    if not transcript_id:
        return None

    transcripts = _load(
        TRANSCRIPTS_PATH,
        [],
    )

    for transcript in transcripts:

        if transcript.get("transcript_id") == transcript_id:
            return transcript

    return None


# =========================================================
# Request models
# =========================================================

class ApprovalDecision(BaseModel):
    """
    Human decision submitted from the review console.

    approved_fields allows the reviewer to confirm or
    correct recovery values before approving the request.
    """

    approved_fields: Dict[str, str] = {}

    note: str = ""


# =========================================================
# Pending recovery queue
# =========================================================

@app.get("/pending")
def list_pending() -> List[Dict[str, Any]]:
    """
    Return cases escalated by the AI agent.

    Each pending case is enriched with:

    - customer_id
    - original customer transcript
    - AI investigation / audit trail

    This gives the human reviewer the context required
    to make an approval decision without needing to
    manually search the underlying JSON files.
    """

    pending = _load(
        PENDING_PATH,
        [],
    )

    records = _load(
        RECORDS_PATH,
        [],
    )

    enriched = []

    for case in pending:

        record_id = case["record_id"]

        # -------------------------------------------------
        # Find the original record
        # -------------------------------------------------

        record = next(
            (
                r
                for r in records
                if r["record_id"] == record_id
            ),
            None,
        )

        # -------------------------------------------------
        # Resolve the original conversation
        # -------------------------------------------------

        transcript = None

        if record:

            transcript = _get_transcript(
                record.get("transcript_id")
            )

        # -------------------------------------------------
        # Resolve audit trail
        # -------------------------------------------------

        trail = read_for_record(
            record_id
        )

        # -------------------------------------------------
        # Build enriched pending case
        # -------------------------------------------------

        enriched.append(
            {
                **case,

                "customer_id": (
                    record.get("customer_id")
                    if record
                    else None
                ),

                "transcript": transcript,

                "audit_trail": trail,
            }
        )

    return enriched


# =========================================================
# Record lookup
# =========================================================

@app.get("/records/{record_id}")
def get_record(
    record_id: str,
) -> Dict[str, Any]:

    records = _load(
        RECORDS_PATH,
        [],
    )

    for record in records:

        if record["record_id"] == record_id:
            return record

    raise HTTPException(
        404,
        f"record {record_id} not found",
    )


# =========================================================
# Audit trail
# =========================================================

@app.get("/records/{record_id}/audit")
def get_audit_trail(
    record_id: str,
) -> List[Dict[str, Any]]:

    return read_for_record(
        record_id
    )


# =========================================================
# Human approval
# =========================================================

@app.post("/approve/{record_id}")
def approve(
    record_id: str,
    decision: ApprovalDecision,
):

    records = _load(
        RECORDS_PATH,
        [],
    )

    found = False

    for record in records:

        if record["record_id"] == record_id:

            # Human reviewer can confirm or edit
            # recovery values before approving.
            if decision.approved_fields:

                record.update(
                    decision.approved_fields
                )

            record["status"] = (
                RecordStatus.APPROVED.value
            )

            found = True

            break

    if not found:

        raise HTTPException(
            404,
            f"record {record_id} not found",
        )

    _save(
        RECORDS_PATH,
        records,
    )

    # Remove the case from the pending queue.
    pending = _load(
        PENDING_PATH,
        [],
    )

    pending = [
        case
        for case in pending
        if case["record_id"] != record_id
    ]

    _save(
        PENDING_PATH,
        pending,
    )

    # Record the human action.
    audit_log(
        record_id,
        "human_action",
        {
            "action": "approved",
            "fields": decision.approved_fields,
            "note": decision.note,
        },
    )

    return {
        "record_id": record_id,
        "status": "APPROVED",
    }


# =========================================================
# Human rejection
# =========================================================

@app.post("/reject/{record_id}")
def reject(
    record_id: str,
    decision: ApprovalDecision,
):

    records = _load(
        RECORDS_PATH,
        [],
    )

    found = False

    for record in records:

        if record["record_id"] == record_id:

            record["status"] = (
                RecordStatus.REJECTED.value
            )

            found = True

            break

    if not found:

        raise HTTPException(
            404,
            f"record {record_id} not found",
        )

    _save(
        RECORDS_PATH,
        records,
    )

    # Remove the case from the pending queue.
    pending = _load(
        PENDING_PATH,
        [],
    )

    pending = [
        case
        for case in pending
        if case["record_id"] != record_id
    ]

    _save(
        PENDING_PATH,
        pending,
    )

    # Record the human action.
    audit_log(
        record_id,
        "human_action",
        {
            "action": "rejected",
            "note": decision.note,
        },
    )

    return {
        "record_id": record_id,
        "status": "REJECTED",
    }


# =========================================================
# Operational statistics
# =========================================================

@app.get("/stats")
def stats() -> Dict[str, Any]:

    records = _load(
        RECORDS_PATH,
        [],
    )

    from collections import Counter

    counts = Counter(
        record["status"]
        for record in records
    )

    return {
        "total_records": len(records),
        "by_status": dict(counts),
    }
