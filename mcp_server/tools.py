"""
The actual tool implementations exposed via MCP (see server.py for the MCP
wiring). Kept separate from server.py so they can be unit/integration
tested directly without going through the MCP protocol layer.

Tools:
  get_record(record_id)
  get_transcript(customer_id)
  search_transcript(customer_id, query)     -> RAG (semantic) evidence search
  get_policy(topic)                          -> RAG policy retrieval
  lookup_customer_by_phone(phone_number)     -> identity resolution
  validate_field(field_name, candidate_values, request_type)  -> deterministic
  propose_recovery(record_id, fields, evidence_summary)
  escalate_to_human(record_id, reason, evidence_summary)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.audit_log import log as audit_log
from common.schema import RecordStatus
from mcp_server import validators

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUEUE_DIR = DATA_DIR / "queue"
PENDING_APPROVALS_PATH = DATA_DIR / "pending_approvals.json"
RECORDS_PATH = DATA_DIR / "records.json"

try:
    from rag.retriever import retrieve_policy, search_transcript as rag_search_transcript
except ImportError:
    # allows validators/data-layer to be exercised even before chromadb /
    # sentence-transformers are installed locally
    def retrieve_policy(query, n_results=3):
        raise RuntimeError("RAG index not available -- run rag/build_index.py after `pip install -r requirements.txt`")

    def rag_search_transcript(customer_id, query, n_results=5):
        raise RuntimeError("RAG index not available -- run rag/build_index.py after `pip install -r requirements.txt`")


def _load_json(path: Path, default):
    if not path.exists():
        return default

    text = path.read_text().strip()

    if not text:
        return default

    return json.loads(text)


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def get_record(record_id: str) -> Dict[str, Any]:
    records = _load_json(RECORDS_PATH, [])
    for r in records:
        if r["record_id"] == record_id:
            audit_log(record_id, "tool_call", {"tool": "get_record"})
            return r
    return {"error": f"record {record_id} not found"}


def get_transcript(
    transcript_id: str = "",
    customer_id: str = "",
) -> Dict[str, Any]:
    transcripts = _load_json(
        DATA_DIR / "transcripts.json",
        [],
    )

    # transcripts.json is expected to contain a list of
    # transcript objects.
    if not isinstance(transcripts, list):
        return {
            "error": (
                "transcripts.json has an unexpected format; "
                "expected a list of transcript records"
            )
        }

    # Preferred path: retrieve the exact transcript
    # linked to the record being investigated.
    if transcript_id:
        for t in transcripts:
            if not isinstance(t, dict):
                continue

            if t.get("transcript_id") == transcript_id:
                audit_log(
                    "",
                    "tool_call",
                    {
                        "tool": "get_transcript",
                        "transcript_id": transcript_id,
                    },
                )
                return t

        return {
            "error": (
                f"transcript {transcript_id} not found"
            )
        }

    # Fallback for cases where an exact transcript ID
    # is genuinely unavailable.
    if customer_id:
        for t in reversed(transcripts):
            if not isinstance(t, dict):
                continue

            if t.get("customer_id") == customer_id:
                audit_log(
                    "",
                    "tool_call",
                    {
                        "tool": "get_transcript",
                        "customer_id": customer_id,
                        "fallback": True,
                    },
                )
                return t

        return {
            "error": (
                f"no transcript for customer "
                f"{customer_id}"
            )
        }

    return {
        "error": (
            "transcript_id or customer_id is required"
        )
    }

def search_transcript(customer_id: str, query: str, record_id: str = "") -> List[Dict[str, Any]]:
    """Semantic search over the customer's transcript. This is the RAG call
    used mid-investigation, distinct from get_policy's RAG call."""
    results = rag_search_transcript(customer_id, query)
    audit_log(record_id, "tool_call", {
        "tool": "search_transcript", "customer_id": customer_id, "query": query,
        "n_results": len(results),
    })
    return results


def get_policy(topic: str, record_id: str = "") -> List[Dict[str, Any]]:
    """RAG retrieval over the company policy documents."""
    results = retrieve_policy(topic)
    audit_log(record_id, "tool_call", {
        "tool": "get_policy", "topic": topic, "n_results": len(results),
    })
    return results


def lookup_customer_by_phone(phone_number: str, record_id: str = "") -> Dict[str, Any]:
    """Simulated customer DB lookup. In this portfolio project the
    synthetic dataset doesn't populate phone numbers for most customers on
    purpose -- it's meant to exercise the 'identity cannot be resolved'
    escalation path, which is a real and common failure mode."""
    records = _load_json(RECORDS_PATH, [])
    matches = [r["customer_id"] for r in records if r.get("phone_number") == phone_number]
    result = {"phone_number": phone_number, "matching_customer_ids": list(set(matches))}
    audit_log(record_id, "tool_call", {"tool": "lookup_customer_by_phone", "result": result})
    return result


# ---------------------------------------------------------------------------
# Validation tool (deterministic, non-LLM)
# ---------------------------------------------------------------------------

def validate_field(
    field_name: str,
    candidate_values: List[str],
    request_type: str = "cancellation",
    record_id: str = "",
) -> Dict[str, Any]:
    normalized_values = candidate_values

    if field_name == "effective_date":
        normalized_values = []

        for value in candidate_values:
            try:
                normalized_values.append(
                    validators.normalize_date_candidate(value)
                )
            except ValueError:
                normalized_values.append(value)

        result = validators.validate_effective_date(
            normalized_values,
            request_type,
        )

    elif field_name == "policy_number":
        result = validators.validate_policy_number(
            candidate_values
        )

    else:
        result = validators.ValidationResult(
            False,
            f"unknown_field:{field_name}",
            0.0,
        )

    payload = {
        "field": field_name,
        "valid": result.valid,
        "reason": result.reason,
        "confidence": result.confidence,
    }

    if field_name == "effective_date":
        payload["candidate_values"] = candidate_values
        payload["normalized_values"] = normalized_values

    audit_log(
        record_id,
        "tool_call",
        {
            "tool": "validate_field",
            **payload,
        },
    )

    return payload


def _build_escalation_reason(field_validations: Dict[str, Dict[str, Any]]) -> str:
    reasons = []

    for field, result in field_validations.items():
        if not result.get("valid", False):
            reason = result.get("reason", "validation_failed")
            reasons.append(f"{field}: {reason}")

    if reasons:
        return "; ".join(reasons)

    return "recovery_not_safe"

# ---------------------------------------------------------------------------
# Write / decision tools
# ---------------------------------------------------------------------------

def propose_recovery(
    record_id: str,
    fields: Dict[str, str],
    field_validations: Dict[str, Dict[str, Any]],
    evidence_summary: str,
) -> Dict[str, Any]:
    """Submit a recovery candidate to the deterministic safety layer.

    The agent may propose a recovery, but the deterministic layer verifies
    that every required field is present and has been explicitly validated.
    """

    from mcp_server.validators import (
        ValidationResult,
        is_safe_to_auto_recover,
    )

    records = _load_json(RECORDS_PATH, [])

    record = None

    for r in records:
        if r["record_id"] == record_id:
            record = r
            break

    if record is None:
        return {
            "outcome": "rejected",
            "safety_reason": "record_not_found",
            "escalation_reason": "record_not_found",
        }

    # ---------------------------------------------------------
    # Determine required fields for this request type
    # ---------------------------------------------------------

    required_fields = {
        "cancellation": [
            "policy_number",
            "effective_date",
        ],
        "date_change": [
            "policy_number",
            "effective_date",
        ],
        "address_change": [
            "policy_number",
        ],
    }

    request_type = record.get("request_type")

    required = required_fields.get(
        request_type,
        [],
    )

    # ---------------------------------------------------------
    # Deterministic completeness gate
    # ---------------------------------------------------------

    missing_fields = [
        field
        for field in required
        if not fields.get(field)
    ]

    unvalidated_fields = [
        field
        for field in required
        if field not in field_validations
    ]

    if missing_fields:
        return {
            "outcome": "rejected",
            "safety_reason": "required_fields_missing",
            "escalation_reason": (
                "Recovery proposal is missing required fields: "
                + ", ".join(missing_fields)
            ),
        }

    if unvalidated_fields:
        return {
            "outcome": "rejected",
            "safety_reason": "required_fields_not_validated",
            "escalation_reason": (
                "Recovery proposal contains fields that were not "
                "validated: "
                + ", ".join(unvalidated_fields)
            ),
        }

    # ---------------------------------------------------------
    # Reconstruct deterministic validation results
    # ---------------------------------------------------------

    reconstructed = {
        f: ValidationResult(
            v["valid"],
            v["reason"],
            v["confidence"],
        )
        for f, v in field_validations.items()
    }

    safety = is_safe_to_auto_recover(
        reconstructed
    )

    # ---------------------------------------------------------
    # Apply final deterministic decision
    # ---------------------------------------------------------

    if safety.valid:
        record.update(fields)
        record["status"] = RecordStatus.RECOVERED.value
    else:
        record["status"] = RecordStatus.ESCALATED.value

    _save_json(
        RECORDS_PATH,
        records,
    )

    escalation_reason = _build_escalation_reason(
        field_validations
    )

    outcome = (
        "auto_recovered"
        if safety.valid
        else "escalated_failed_validation"
    )

    audit_log(
        record_id,
        "decision",
        {
            "decision": outcome,
            "fields": fields,
            "safety_reason": safety.reason,
            "evidence_summary": evidence_summary,
        },
    )

    if not safety.valid:
        _queue_for_human(
            record_id,
            reason=escalation_reason,
            evidence_summary=evidence_summary,
            proposed_fields=fields,
        )

    return {
        "outcome": outcome,
        "safety_reason": safety.reason,
        "escalation_reason": (
            escalation_reason
            if not safety.valid
            else ""
        ),
    }


def escalate_to_human(record_id: str, reason: str, evidence_summary: str) -> Dict[str, Any]:
    """Direct escalation path the agent takes when it decides on its own
    that it should not even attempt a recovery proposal (e.g. no evidence
    found at all, or distress language detected)."""
    records = _load_json(RECORDS_PATH, [])
    for r in records:
        if r["record_id"] == record_id:
            r["status"] = RecordStatus.ESCALATED.value
            break
    _save_json(RECORDS_PATH, records)

    audit_log(record_id, "decision", {
        "decision": "escalated_by_agent", "reason": reason, "evidence_summary": evidence_summary,
    })
    _queue_for_human(record_id, reason=reason, evidence_summary=evidence_summary, proposed_fields={})
    return {"outcome": "escalated", "reason": reason}


def _queue_for_human(record_id: str, reason: str, evidence_summary: str, proposed_fields: Dict[str, str]):
    pending = _load_json(PENDING_APPROVALS_PATH, [])
    pending = [p for p in pending if p["record_id"] != record_id]  # replace if re-queued
    pending.append({
        "record_id": record_id,
        "reason": reason,
        "evidence_summary": evidence_summary,
        "proposed_fields": proposed_fields,
    })
    _save_json(PENDING_APPROVALS_PATH, pending)


TOOL_REGISTRY = {
    "get_record": get_record,
    "get_transcript": get_transcript,
    "search_transcript": search_transcript,
    "get_policy": get_policy,
    "lookup_customer_by_phone": lookup_customer_by_phone,
    "validate_field": validate_field,
    "propose_recovery": propose_recovery,
    "escalate_to_human": escalate_to_human,
}
