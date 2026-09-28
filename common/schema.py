from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid


class RequestType(str, Enum):
    CANCELLATION = "cancellation"
    DATE_CHANGE = "date_change"
    ADDRESS_CHANGE = "address_change"


class RecordStatus(str, Enum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CONFLICTING = "CONFLICTING"
    RECOVERED = "RECOVERED"          # auto-recovered by the agent
    ESCALATED = "ESCALATED"          # waiting on a human
    APPROVED = "APPROVED"            # human approved the recovery
    REJECTED = "REJECTED"            # human rejected the recovery


@dataclass
class CustomerRecord:
    """Structured customer request produced from the voice conversation."""

    record_id: str
    customer_id: str
    request_type: str

    policy_number: Optional[str] = None
    effective_date: Optional[str] = None   # ISO date, e.g. "2026-09-28"
    phone_number: Optional[str] = None

    status: str = RecordStatus.FAILED.value

    # Why the record could not safely be completed.
    failure_reason: Optional[str] = None

    # Specific validation problems found during intake.
    validation_errors: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    transcript_id: Optional[str] = None

    def missing_fields(self) -> List[str]:
        """
        Return required fields that have no value at all.

        Note:
        A field being present does NOT automatically mean the record
        is valid. The validator will separately check whether the value
        is complete, valid, and unambiguous.
        """
        required = REQUIRED_FIELDS.get(self.request_type, [])

        return [
            field_name
            for field_name in required
            if not getattr(self, field_name, None)
        ]

    def has_missing_fields(self) -> bool:
        """Return True if one or more required fields are completely missing."""
        return bool(self.missing_fields())

    def has_validation_errors(self) -> bool:
        """Return True if validation found an invalid or ambiguous value."""
        return bool(self.validation_errors)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Fields that must exist before a request can be considered actionable.
#
# IMPORTANT:
# This does NOT mean that merely having a value makes the record COMPLETE.
# The validation layer will also check whether those values are valid,
# complete, and unambiguous.
REQUIRED_FIELDS = {
    RequestType.CANCELLATION.value: [
        "policy_number",
        "effective_date",
    ],
    RequestType.DATE_CHANGE.value: [
        "policy_number",
        "effective_date",
    ],
    RequestType.ADDRESS_CHANGE.value: [
        "policy_number",
    ],
}


@dataclass
class TranscriptTurn:
    speaker: str   # "customer" or "agent"
    text: str


@dataclass
class Transcript:
    transcript_id: str
    customer_id: str
    turns: List[TranscriptTurn] = field(default_factory=list)

    def full_text(self) -> str:
        return "\n".join(
            f"{t.speaker}: {t.text}"
            for t in self.turns
        )

    def customer_text(self) -> str:
        return "\n".join(
            t.text
            for t in self.turns
            if t.speaker == "customer"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "customer_id": self.customer_id,
            "turns": [asdict(t) for t in self.turns],
        }


@dataclass
class Evidence:
    """One piece of evidence the agent gathered, with provenance."""

    source: str
    content: str
    field_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditEntry:
    entry_id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)
    event: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

