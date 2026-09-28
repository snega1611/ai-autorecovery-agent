from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

from common.schema import CustomerRecord, RecordStatus


@dataclass
class ValidationResult:
    """
    Result of validating a structured customer request.

    valid:
        True only when the record is safe to treat as COMPLETE.

    status:
        COMPLETE, FAILED, or CONFLICTING.

    errors:
        Specific reasons why the record is not safely actionable.
    """

    valid: bool
    status: str
    errors: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


def validate_date(value: Optional[str]) -> List[str]:
    """
    Validate that effective_date is a complete ISO date.

    Expected format:
        YYYY-MM-DD

    Examples:
        2026-09-28  -> valid
        28 September -> invalid
        2026-09     -> invalid
        2020-09-28  -> syntactically valid
                       (business-rule validation can be added later)
    """

    if not value:
        return ["effective_date is missing"]

    value = value.strip()

    try:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()

    except ValueError:
        return [
            "effective_date is not a complete valid date "
            "in YYYY-MM-DD format"
        ]

    # Basic sanity check:
    # Do not accept dates that are implausibly far in the past.
    #
    # We are deliberately keeping this rule conservative for now.
    # Business-specific date rules can be added later.
    if parsed.year < 2025:
        return [
            f"effective_date {value} is outside the supported "
            "business date range"
        ]

    return []


def validate_policy_number(
    policy_number: Optional[str],
) -> List[str]:
    """Validate that a policy number is present and structurally usable."""

    if not policy_number:
        return ["policy_number is missing"]

    value = policy_number.strip().upper()

    if not value:
        return ["policy_number is empty"]

    if not value.startswith("P"):
        return [
            "policy_number must start with 'P'"
        ]

    if len(value) != 6:
        return [
            "policy_number must contain 6 characters"
        ]

    if not value[1:].isdigit():
        return [
            "policy_number must have numeric digits after 'P'"
        ]

    return []


def validate_record(
    record: CustomerRecord,
) -> ValidationResult:
    """
    Determine whether a structured customer record is safe to process.

    IMPORTANT:
    A non-empty field is NOT automatically considered valid.

    The record must satisfy:
        1. required fields are present
        2. field values are structurally valid
        3. field values are complete enough to act on

    Transcript-vs-record conflict detection will be added separately
    once the extraction/voice flow is wired into this validator.
    """

    errors: List[str] = []

    request_type = record.request_type

    # ---------------------------------------------------------
    # Required-field validation
    # ---------------------------------------------------------

    missing = record.missing_fields()

    for field_name in missing:
        errors.append(
            f"{field_name} is missing"
        )

    # ---------------------------------------------------------
    # Policy validation
    # ---------------------------------------------------------

    if request_type in {
        "cancellation",
        "date_change",
        "address_change",
    }:
        errors.extend(
            validate_policy_number(
                record.policy_number
            )
        )

    # ---------------------------------------------------------
    # Date validation
    # ---------------------------------------------------------

    if request_type in {
        "cancellation",
        "date_change",
    }:
        errors.extend(
            validate_date(
                record.effective_date
            )
        )

    # Remove duplicate errors while preserving order.
    errors = list(dict.fromkeys(errors))

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------

    if errors:
        return ValidationResult(
            valid=False,
            status=RecordStatus.FAILED.value,
            errors=errors,
            failure_reason="VALIDATION_FAILED",
        )

    return ValidationResult(
        valid=True,
        status=RecordStatus.COMPLETE.value,
        errors=[],
        failure_reason=None,
    )


def apply_validation(
    record: CustomerRecord,
) -> CustomerRecord:
    """
    Validate a CustomerRecord and update its status/error information.

    This function does NOT perform recovery.

    It only answers:

        "Can this record safely be treated as COMPLETE?"
    """

    result = validate_record(record)

    record.status = result.status
    record.validation_errors = result.errors
    record.failure_reason = result.failure_reason

    return record
