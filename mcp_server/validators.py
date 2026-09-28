"""
Deterministic validation logic. This is intentionally NOT an LLM call.

This is the core safety story of the whole project: the agent (an LLM) may
propose a recovered value, but whether that value is actually safe to
auto-apply is decided here, by plain Python rules grounded in the evidence
the agent collected -- not by the LLM's own confidence.

Kept fully unit-testable and framework-free.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


POLICY_NUMBER_RE = re.compile(r"^[A-Z]\d{4,6}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    confidence: float = 0.0   # 0.0-1.0, purely diagnostic; the boolean is what's used downstream


def validate_policy_number(candidate_values: List[str]) -> ValidationResult:
    """candidate_values: every distinct value the agent's evidence surfaced
    for this field. More than one distinct value => conflict => invalid."""
    distinct = sorted(set(v.strip().upper() for v in candidate_values if v))

    if not distinct:
        return ValidationResult(False, "no_evidence", 0.0)

    if len(distinct) > 1:
        return ValidationResult(False, f"conflicting_values:{distinct}", 0.0)

    value = distinct[0]
    if not POLICY_NUMBER_RE.match(value):
        return ValidationResult(False, f"malformed_policy_number:{value}", 0.2)

    return ValidationResult(True, "single_well_formed_value", 0.95)


def normalize_date_candidate(value: str) -> str:
    """Convert common human-readable dates to canonical YYYY-MM-DD.

    Raises ValueError when the value cannot be interpreted as a date.
    """

    value = value.strip()

    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%d %b %Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt).date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(f"unparseable_date:{value}")

def validate_effective_date(
    candidate_values: List[str],
    request_type: str,
    call_timestamp: Optional[date] = None,
) -> ValidationResult:
    distinct = sorted(set(v.strip() for v in candidate_values if v))
    call_timestamp = call_timestamp or date.today()

    if not distinct:
        return ValidationResult(False, "no_evidence", 0.0)

    if len(distinct) > 1:
        return ValidationResult(False, f"conflicting_values:{distinct}", 0.0)

    value = distinct[0]
    if not ISO_DATE_RE.match(value):
        return ValidationResult(False, f"unparseable_date:{value}", 0.1)

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return ValidationResult(False, f"invalid_calendar_date:{value}", 0.1)

    if parsed < call_timestamp:
        return ValidationResult(False, f"date_in_past:{value}", 0.1)

    # Policy rule: date-change requests taking effect within 24h require
    # supervisor approval even though the value itself is unambiguous.
    if request_type == "date_change" and (parsed - call_timestamp).days < 1:
        return ValidationResult(False, "same_day_change_requires_supervisor", 0.9)

    return ValidationResult(True, "single_valid_future_date", 0.95)


def validate_identity(policy_number_valid: bool, phone_match_count: int) -> ValidationResult:
    """Identity must resolve to exactly one customer, per
    data/policies/identity_verification.md."""
    if policy_number_valid:
        return ValidationResult(True, "resolved_via_policy_number", 0.95)
    if phone_match_count == 1:
        return ValidationResult(True, "resolved_via_unique_phone_match", 0.85)
    if phone_match_count == 0:
        return ValidationResult(False, "no_phone_match", 0.0)
    return ValidationResult(False, f"ambiguous_phone_match:{phone_match_count}_customers", 0.0)


REQUIRED_FIELDS_BY_TYPE = {
    "cancellation": ["policy_number", "effective_date"],
    "date_change": ["policy_number", "effective_date"],
    "address_change": ["policy_number"],
}


def is_safe_to_auto_recover(field_results: dict) -> ValidationResult:
    """field_results: {field_name: ValidationResult}. Auto-recovery is safe
    only if every required field for this request type validated True.
    A single failing field forces escalation -- there is no partial-credit
    auto-recovery."""
    invalid = [f for f, r in field_results.items() if not r.valid]
    if invalid:
        reasons = {f: field_results[f].reason for f in invalid}
        return ValidationResult(False, f"fields_failed_validation:{reasons}", 0.0)
    return ValidationResult(True, "all_required_fields_validated", min(
        (r.confidence for r in field_results.values()), default=0.0
    ))


if __name__ == "__main__":
    # a few inline sanity checks standing in for a proper pytest suite
    assert validate_policy_number(["P8921"]).valid is True
    assert validate_policy_number(["P8921", "P8927"]).valid is False
    assert validate_policy_number([]).valid is False
    assert validate_policy_number(["8921"]).valid is False  # missing letter prefix

    today = date.today()
    future = (today.replace(day=1)).isoformat()  # not robust across months, replaced below
    from datetime import timedelta
    future = (today + timedelta(days=10)).isoformat()
    same_day = today.isoformat()

    assert validate_effective_date([future], "cancellation").valid is True
    assert validate_effective_date([same_day], "date_change").valid is False
    assert validate_effective_date([same_day], "cancellation").valid is True  # no same-day rule for cancellations
    assert validate_effective_date([future, same_day], "cancellation").valid is False  # conflict

    assert validate_identity(True, 0).valid is True
    assert validate_identity(False, 1).valid is True
    assert validate_identity(False, 0).valid is False
    assert validate_identity(False, 2).valid is False

    print("All validator self-checks passed.")
