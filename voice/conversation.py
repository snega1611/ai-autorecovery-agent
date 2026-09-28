from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent import llm_client
from agent.mcp_client import mcp_session, call_tool

from common.schema import (
    CustomerRecord,
    Transcript,
    TranscriptTurn,
    RecordStatus,
    new_id,
)

from common.validator import apply_validation

from common.customer_lookup import (
    lookup_customer_by_phone,
    lookup_customer_by_policy,
)

from mcp_server.validators import normalize_date_candidate


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MAX_TURNS = 6


# ============================================================================
# REGEX
# ============================================================================

# Accept:
#   P10001
#   P-10001
#   P 10001
POLICY_NUMBER_RE = re.compile(
    r"\bP(?:[\s-]*\d){4,6}\b",
    re.IGNORECASE,
)

CANNOT_IDENTIFY_POLICY_RE = re.compile(
    r"\b("
    r"i\s*(don'?t|do not)\s*(know|remember)"
    r"|don'?t\s*know"
    r"|don'?t\s*remember"
    r"|do\s*not\s*remember"
    r"|not\s*sure"
    r"|no\s*idea"
    r"|i\s*can'?t\s*(tell|identify|remember)"
    r"|cannot\s*(tell|identify|remember)"
    r"|unsure"
    r"|i\s*forgot"
    r"|forgot\s*(my\s*)?(policy\s*)?(number)?"
    r")\b",
    re.IGNORECASE,
)


# Month names accepted by the deterministic date extractor.
MONTH_PATTERN = (
    r"(?:"
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r")"
)


# ============================================================================
# LLM SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are a customer service voice assistant for an insurance company.

Supported requests:
- cancellation
- effective date change
- address change

Required information:
- policy_number
- effective_date for cancellation and date_change
- new address for address_change


CUSTOMER IDENTIFICATION RULES

1. Ask for the policy number first.

2. If the customer provides a policy number, the application
   will verify it against the trusted customer database.

3. Once the application has verified a policy number, NEVER ask
   for the phone number merely to identify the customer.

4. Ask for the phone number ONLY when the customer explicitly
   says they do not know or remember their policy number.

5. If the customer does not know their policy number and provides
   a phone number, the application may identify the customer,
   but the unresolved request must be escalated to a human.

6. Never invent policy numbers, customer IDs, phone numbers,
   dates, or addresses.

7. Never ask the customer to repeat information that has already
   been verified by the application.

8. Ask ONE short natural question at a time.

9. Once the application provides a VERIFIED POLICY, treat that
   policy number as trusted application state.


DATE RULES

10. For effective dates:

    - Extract a date only when the customer actually provides it.
    - Return complete dates as YYYY-MM-DD.
    - Never invent a missing year.
    - If the date is incomplete or missing, return null.
    - If the customer gives multiple conflicting dates in the
      same utterance, do not choose one.


11. When all required information is available, return ONLY JSON:

{
  "done": true,
  "request_type": "date_change",
  "policy_number": "P10001",
  "effective_date": "2026-09-28",
  "phone_number": null
}


12. When more information is required, return ONLY JSON:

{
  "done": false,
  "reply": "What complete date would you like?"
}

Never invent missing information.
"""


# ============================================================================
# POLICY HELPERS
# ============================================================================

def _normalize_policy_number(
    value: Optional[str],
) -> Optional[str]:

    if not value:
        return None

    value = value.strip().upper()

    # P-10001 -> P10001
    # P 10001 -> P10001
    value = re.sub(r"[\s-]+", "", value)

    return value


# ============================================================================
# DATE EXTRACTION
# ============================================================================

def _normalize_date(
    value: str,
) -> Optional[str]:

    try:
        return normalize_date_candidate(
            value.strip()
        )
    except Exception:
        return None


def _extract_complete_dates(
    text: str,
) -> List[str]:

    """
    Extract dates that already contain a year.

    Examples:

        September 29, 2026
        29 September 2026
        29/09/2026
        2026-09-29

    Returns normalized YYYY-MM-DD values.
    """

    candidates: List[str] = []

    patterns = [
        # September 29, 2026
        rf"\b{MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b",

        # September 29 2026
        rf"\b{MONTH_PATTERN}\s+\d{{1,2}}\s+\d{{4}}\b",

        # 29 September 2026
        rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b",

        # 29/09/2026
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",

        # 29-09-2026
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",

        # 2026-09-29
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):
            raw = match.group(0)

            normalized = _normalize_date(
                raw
            )

            if normalized:
                candidates.append(
                    normalized
                )

    return list(
        dict.fromkeys(candidates)
    )


def _extract_partial_month_day_dates(
    text: str,
) -> List[str]:

    """
    Extract month/day values WITHOUT a year.

    Examples:

        September 28
        Sep 28
        28 September

    These are returned as raw month/day strings.

    The year is deliberately not invented here.
    The caller can safely supply a year only when another
    complete date in the same utterance provides an explicit year.
    """

    candidates: List[str] = []

    patterns = [
        # September 28
        rf"\b{MONTH_PATTERN}\s+\d{{1,2}}\b",

        # 28 September
        rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\b",
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):
            raw = match.group(0)

            candidates.append(
                raw.strip()
            )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(candidates)
    )


def _extract_complete_dates_with_inference(
    text: str,
) -> List[str]:

    """
    Deterministically extract all date candidates from one customer
    utterance.

    Important behavior:

        "September 28. No, September 29, 2026"

    becomes:

        2026-09-28
        2026-09-29

    We are NOT guessing an arbitrary year.

    The year is borrowed only because the same utterance explicitly
    contains a complete date with that year.

    If there is no explicit year anywhere in the utterance,
    an incomplete date remains incomplete.
    """

    complete_dates = _extract_complete_dates(
        text
    )

    partial_dates = (
        _extract_partial_month_day_dates(
            text
        )
    )

    # If there is exactly one explicit year in the
    # complete dates, it can safely contextualize
    # the partial date candidate.
    explicit_years = {
        value[:4]
        for value in complete_dates
        if re.match(
            r"^\d{4}-\d{2}-\d{2}$",
            value,
        )
    }

    inferred_dates: List[str] = list(
        complete_dates
    )

    if len(explicit_years) == 1:

        year = next(
            iter(explicit_years)
        )

        for partial in partial_dates:

            # Avoid treating the month/day prefix of an already
            # complete date as a separate candidate.
            normalized_partial = (
                _normalize_partial_date(
                    partial,
                    year,
                )
            )

            if normalized_partial:
                inferred_dates.append(
                    normalized_partial
                )

    return list(
        dict.fromkeys(
            inferred_dates
        )
    )


def _normalize_partial_date(
    value: str,
    year: str,
) -> Optional[str]:

    """
    Convert a month/day expression into YYYY-MM-DD
    using an explicitly supplied year.
    """

    value = value.strip()

    patterns = [
        "%B %d",
        "%b %d",
        "%d %B",
        "%d %b",
    ]

    for pattern in patterns:

        try:
            parsed = __import__(
                "datetime"
            ).datetime.strptime(
                value,
                pattern,
            )

            candidate = (
                f"{year}-"
                f"{parsed.month:02d}-"
                f"{parsed.day:02d}"
            )

            normalized = _normalize_date(
                candidate
            )

            if normalized:
                return normalized

        except ValueError:
            continue

    return None


# ============================================================================
# RUN CONVERSATION
# ============================================================================

async def run_conversation(
    use_voice: bool = True,
) -> Dict[str, Any]:

    if use_voice:

        from voice.stt import (
            listen_and_transcribe,
        )

        from voice.tts import speak

    else:

        speak = (
            lambda text:
            print(f"AI: {text}")
        )

        listen_and_transcribe = (
            lambda seconds=6.0:
            input("You: ")
        )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    turns: List[TranscriptTurn] = []

    # ------------------------------------------------------------------
    # TRUSTED APPLICATION STATE
    # ------------------------------------------------------------------

    resolved_customer: Optional[
        Dict[str, Any]
    ] = None

    resolved_policy: Optional[str] = None

    provided_phone: Optional[str] = None

    phone_lookup_pending = False

    # ------------------------------------------------------------------
    # OPENING
    # ------------------------------------------------------------------

    opening = (
        "How can I help you today?"
    )

    speak(opening)

    turns.append(
        TranscriptTurn(
            "agent",
            opening,
        )
    )

    # ------------------------------------------------------------------
    # CONVERSATION LOOP
    # ------------------------------------------------------------------

    for _ in range(MAX_TURNS):

        customer_text = (
            listen_and_transcribe()
            if use_voice
            else listen_and_transcribe(0)
        )

        customer_text = (
            customer_text.strip()
        )

        if not customer_text:
            continue

        print(
            f"[customer]: "
            f"{customer_text}"
        )

        turns.append(
            TranscriptTurn(
                "customer",
                customer_text,
            )
        )

        # ==============================================================
        # PATH A
        # CUSTOMER DOES NOT KNOW POLICY NUMBER
        # ==============================================================

        if (
            resolved_policy is None
            and CANNOT_IDENTIFY_POLICY_RE.search(
                customer_text
            )
        ):

            phone_lookup_pending = True

            reply = (
                "No problem. What is the phone number "
                "on your account?"
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            continue

        # ==============================================================
        # PATH B
        # PHONE NUMBER AFTER CUSTOMER DOES NOT KNOW POLICY
        # ==============================================================

        if phone_lookup_pending:

            if (
                customer_text.isdigit()
                and len(customer_text) == 10
            ):

                provided_phone = (
                    customer_text
                )

                lookup_result = (
                    lookup_customer_by_phone(
                        customer_text
                    )
                )

                if lookup_result["status"] in {
                    "resolved",
                    "ambiguous",
                }:

                    resolved_customer = (
                        lookup_result
                    )

                    verified_phone = (
                        lookup_result.get(
                            "phone_number",
                            customer_text,
                        )
                    )

                    return (
                        await _create_and_escalate_record(
                            turns=turns,
                            parsed={
                                "request_type": (
                                    _infer_request_type(
                                        turns
                                    )
                                ),
                                "policy_number": None,
                                "effective_date": None,
                                "phone_number": verified_phone,
                            },
                            resolved_customer=(
                                resolved_customer
                            ),
                            resolved_policy=None,
                            provided_phone=(
                                verified_phone
                            ),
                            reason=(
                                "CUSTOMER_DOES_NOT_KNOW_POLICY_NUMBER"
                            ),
                            evidence_summary=(
                                "Customer requested a policy "
                                "change but could not provide "
                                "the policy number. The customer "
                                "was identified using the verified "
                                "phone number, but the applicable "
                                "policy could not be safely "
                                "identified during voice intake. "
                                "Human review is required."
                            ),
                        )
                    )

                reply = (
                    "I couldn't find an account with that "
                    "phone number. Please provide the phone "
                    "number again."
                )

                speak(reply)

                turns.append(
                    TranscriptTurn(
                        "agent",
                        reply,
                    )
                )

                continue

            reply = (
                "Please provide the 10-digit phone number "
                "on your account."
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            continue

        # ==============================================================
        # PATH C
        # CUSTOMER PROVIDES POLICY NUMBER
        # ==============================================================

        policy_match = (
            POLICY_NUMBER_RE.search(
                customer_text
            )
        )

        if policy_match:

            candidate_policy = (
                _normalize_policy_number(
                    policy_match.group(0)
                )
            )

            lookup_result = (
                lookup_customer_by_policy(
                    candidate_policy
                )
            )

            # ----------------------------------------------------------
            # POLICY VERIFIED
            # ----------------------------------------------------------

            if lookup_result["status"] == "resolved":

                resolved_customer = (
                    lookup_result
                )

                resolved_policy = (
                    _normalize_policy_number(
                        lookup_result["policy"][
                            "policy_number"
                        ]
                    )
                )

                phone_lookup_pending = False

                request_type = (
                    _infer_request_type(
                        turns
                    )
                )

                messages.append({
                    "role": "system",
                    "content": (
                        "APPLICATION STATE UPDATE:\n"
                        f"verified_policy_number: "
                        f"{resolved_policy}\n"
                        f"customer_id: "
                        f"{resolved_customer['customer_id']}\n"
                        f"request_type: "
                        f"{request_type}\n\n"
                        "The policy number has been verified "
                        "by the application. Do not ask for it "
                        "again. Do not ask for a phone number "
                        "for identification.\n"
                        "Continue collecting the remaining "
                        "required information."
                    ),
                })

                if request_type in {
                    "date_change",
                    "cancellation",
                }:

                    if request_type == "date_change":

                        reply = (
                            "Thanks, I've verified your policy. "
                            "What date would you like the "
                            "effective date changed to?"
                        )

                    else:

                        reply = (
                            "Thanks, I've verified your policy. "
                            "What effective date would you like "
                            "for the cancellation?"
                        )

                    speak(reply)

                    turns.append(
                        TranscriptTurn(
                            "agent",
                            reply,
                        )
                    )

                    continue

                if request_type == "address_change":

                    reply = (
                        "Thanks, I've verified your policy. "
                        "What is the new address?"
                    )

                    speak(reply)

                    turns.append(
                        TranscriptTurn(
                            "agent",
                            reply,
                        )
                    )

                    continue

            # ----------------------------------------------------------
            # POLICY NOT FOUND
            # ----------------------------------------------------------

            reply = (
                f"I couldn't find policy "
                f"{candidate_policy}. "
                "Please provide the correct policy number, "
                "or say that you don't know it."
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            continue

        # ==============================================================
        # PHONE PROVIDED BEFORE POLICY
        # ==============================================================

        if (
            customer_text.isdigit()
            and len(customer_text) == 10
        ):

            reply = (
                "I need your policy number first. "
                "If you don't know it, you can say so."
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            continue

        # ==============================================================
        # NORMAL LLM PROCESSING
        # ==============================================================
        #
        # ONE QWEN CALL PER TURN.
        #
        # Qwen:
        #   - understands natural language
        #   - extracts structured information
        #   - decides what information is still missing
        #
        # Application:
        #   - verifies policy
        #   - owns trusted state
        #   - detects deterministic conflicts
        #   - validates final values
        #
        # There is NO second Qwen call for date validation.
        # ==============================================================

        messages.append({
            "role": "user",
            "content": customer_text,
        })

        request_type = (
            _infer_request_type(
                turns
            )
        )

        if resolved_policy is not None:

            verified_customer_id = (
                resolved_customer["customer_id"]
                if resolved_customer
                else "unknown"
            )

            messages.append({
                "role": "system",
                "content": (
                    "CURRENT VERIFIED APPLICATION STATE:\n"
                    f"request_type: {request_type}\n"
                    f"verified_policy_number: "
                    f"{resolved_policy}\n"
                    f"customer_id: "
                    f"{verified_customer_id}\n\n"
                    "The policy number is already verified "
                    "by the application.\n"
                    "Do not ask for the policy number again.\n"
                    "Do not ask for a phone number for "
                    "identification.\n\n"
                    "Process the customer's latest message.\n"
                    "If the customer provides a complete "
                    "effective date, extract it as YYYY-MM-DD.\n"
                    "If the date is incomplete or missing, "
                    "return effective_date as null.\n"
                    "Never invent missing information."
                ),
            })

        # --------------------------------------------------------------
        # SINGLE QWEN CALL
        # --------------------------------------------------------------

        message = llm_client.chat(
            messages
        )

        content = message.get(
            "content",
            "",
        )

        parsed = (
            llm_client.extract_json_object(
                content
            )
        )

        # --------------------------------------------------------------
        # MODEL DID NOT RETURN JSON
        # --------------------------------------------------------------

        if not parsed:

            speak(content)

            turns.append(
                TranscriptTurn(
                    "agent",
                    content,
                )
            )

            messages.append({
                "role": "assistant",
                "content": content,
            })

            continue

        # ==============================================================
        # LLM SAYS IT NEEDS MORE INFORMATION
        # ==============================================================

        if not parsed.get("done"):

            reply = parsed.get(
                "reply",
                "Could you provide that information again?",
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            messages.append({
                "role": "assistant",
                "content": json.dumps(
                    parsed
                ),
            })

            continue

        # ==============================================================
        # LLM SAYS REQUEST IS COMPLETE
        # ==============================================================

        extracted_policy_number = (
            parsed.get(
                "policy_number"
            )
        )

        # --------------------------------------------------------------
        # APPLICATION MUST OWN POLICY IDENTITY
        # --------------------------------------------------------------

        if resolved_policy is None:

            reply = (
                "Before I can continue, I need "
                "your policy number. If you don't "
                "know it, you can say so."
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            messages.append({
                "role": "assistant",
                "content": json.dumps(
                    parsed
                ),
            })

            continue

        # --------------------------------------------------------------
        # POLICY EXTRACTION CONFLICT
        #
        # Normalize both values first.
        #
        # P-10001 and P10001 are the SAME policy.
        # --------------------------------------------------------------

        normalized_extracted_policy = (
            _normalize_policy_number(
                extracted_policy_number
            )
        )

        normalized_verified_policy = (
            _normalize_policy_number(
                resolved_policy
            )
        )

        if (
            normalized_extracted_policy
            and normalized_extracted_policy
            != normalized_verified_policy
        ):

            parsed[
                "extraction_conflict"
            ] = True

            parsed[
                "extracted_policy_number"
            ] = normalized_extracted_policy

        # Never allow Qwen to replace the verified
        # application policy.

        parsed["policy_number"] = (
            normalized_verified_policy
        )

        # Application determines request type.

        parsed["request_type"] = (
            request_type
        )

        # ==============================================================
        # IMPORTANT:
        # DETECT MULTIPLE DATE CANDIDATES BEFORE ACCEPTING COMPLETE
        # ==============================================================

        if request_type in {
            "date_change",
            "cancellation",
        }:

            date_candidates = (
                _extract_complete_dates_with_inference(
                    customer_text
                )
            )

            if len(date_candidates) > 1:

                parsed[
                    "effective_date_conflict"
                ] = True

                parsed[
                    "conflicting_effective_dates"
                ] = date_candidates

                # Preserve the first candidate only as a
                # representative value. _finalize() will
                # override the status to CONFLICTING.
                parsed[
                    "effective_date"
                ] = date_candidates[0]

            # ----------------------------------------------------------
            # If exactly one deterministic date exists in the latest
            # utterance, prefer that date over an older model value.
            #
            # This prevents Qwen from accidentally retaining an
            # earlier date when the latest utterance contains the
            # customer's actual complete date.
            # ----------------------------------------------------------

            elif len(date_candidates) == 1:

                parsed[
                    "effective_date"
                ] = date_candidates[0]

        # --------------------------------------------------------------
        # DATE STILL INCOMPLETE
        # --------------------------------------------------------------

        if (
            request_type in {
                "date_change",
                "cancellation",
            }
            and not parsed.get(
                "effective_date"
            )
        ):

            reply = (
                "Please provide the complete effective "
                "date, including the year."
            )

            speak(reply)

            turns.append(
                TranscriptTurn(
                    "agent",
                    reply,
                )
            )

            messages.append({
                "role": "assistant",
                "content": json.dumps(
                    parsed
                ),
            })

            continue

        # --------------------------------------------------------------
        # REQUIRED INFORMATION IS PRESENT
        #
        # Do NOT decide COMPLETE here.
        #
        # The deterministic validator gets the final say.
        # --------------------------------------------------------------

        closing = (
            "Thanks, I have what I need "
            "to process your request."
        )

        speak(closing)

        turns.append(
            TranscriptTurn(
                "agent",
                closing,
            )
        )

        return _finalize(
            turns=turns,
            parsed=parsed,
            resolved_customer=resolved_customer,
            resolved_policy=resolved_policy,
            provided_phone=provided_phone,
        )

    # ==============================================================
    # MAX TURNS
    # ==============================================================

    return _finalize(
        turns=turns,
        parsed={
            "request_type": (
                _infer_request_type(
                    turns
                )
            ),
            "policy_number": (
                resolved_policy
            ),
            "effective_date": None,
            "phone_number": (
                provided_phone
            ),
        },
        resolved_customer=(
            resolved_customer
        ),
        resolved_policy=(
            resolved_policy
        ),
        provided_phone=(
            provided_phone
        ),
    )


# ============================================================================
# FINALIZE NORMAL REQUEST
# ============================================================================

def _finalize(
    turns: List[TranscriptTurn],
    parsed: Dict[str, Any],
    resolved_customer: Optional[
        Dict[str, Any]
    ] = None,
    resolved_policy: Optional[str] = None,
    provided_phone: Optional[str] = None,
) -> Dict[str, Any]:

    transcript_id = new_id(
        "txn"
    )

    if resolved_customer:

        customer_id = (
            resolved_customer[
                "customer_id"
            ]
        )

    else:

        customer_id = new_id(
            "C"
        )

    policy_number = (
        resolved_policy
        or parsed.get(
            "policy_number"
        )
    )

    phone_number = (
        provided_phone
        or parsed.get(
            "phone_number"
        )
    )

    transcript = Transcript(
        transcript_id=transcript_id,
        customer_id=customer_id,
        turns=turns,
    )

    record = CustomerRecord(
        record_id=new_id("rec"),
        customer_id=customer_id,
        request_type=parsed.get(
            "request_type",
            "cancellation",
        ),
        policy_number=policy_number,
        effective_date=parsed.get(
            "effective_date"
        ),
        phone_number=phone_number,
        transcript_id=transcript_id,
    )

    # ------------------------------------------------------------------
    # DETERMINISTIC VALIDATION
    # ------------------------------------------------------------------

    record = apply_validation(
        record
    )

    # ------------------------------------------------------------------
    # DATE CONFLICT HAS PRIORITY
    #
    # This must happen before extraction_conflict.
    # ------------------------------------------------------------------

    if parsed.get(
        "effective_date_conflict"
    ):

        record.status = (
            RecordStatus.CONFLICTING.value
        )

        record.failure_reason = (
            "CONFLICTING_EFFECTIVE_DATES"
        )

        conflicting_dates = (
            parsed.get(
                "conflicting_effective_dates",
                [],
            )
        )

        record.validation_errors.append(
            "Customer provided multiple conflicting "
            f"effective dates: {conflicting_dates}"
        )

        record.validation_errors = list(
            dict.fromkeys(
                record.validation_errors
            )
        )

    # ------------------------------------------------------------------
    # POLICY EXTRACTION CONFLICT
    #
    # Only used when policy values genuinely differ
    # after normalization.
    # ------------------------------------------------------------------

    elif parsed.get(
        "extraction_conflict"
    ):

        record.status = (
            RecordStatus.CONFLICTING.value
        )

        record.failure_reason = (
            "EXTRACTION_CONFLICT"
        )

        record.validation_errors.append(
            "LLM extraction conflicts with "
            "verified application policy"
        )

        record.validation_errors = list(
            dict.fromkeys(
                record.validation_errors
            )
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    _save_record_and_transcript(
        transcript=transcript,
        record=record,
    )

    print(
        f"\nFinal record ({record.status}): "
        f"{record.to_dict()}"
    )

    # ------------------------------------------------------------------
    # DEBUG CONFLICT INFORMATION
    # ------------------------------------------------------------------

    if parsed.get(
        "effective_date_conflict"
    ):

        print(
            "\n[CONFLICT] Customer provided "
            "multiple effective dates:"
        )

        for value in parsed.get(
            "conflicting_effective_dates",
            [],
        ):

            print(
                f"  - {value}"
            )

    if parsed.get(
        "extraction_conflict"
    ):

        print(
            "\n[CONFLICT] Verified policy:",
            resolved_policy,
        )

        print(
            "[CONFLICT] Qwen extracted:",
            parsed.get(
                "extracted_policy_number"
            ),
        )

    return record.to_dict()


# ============================================================================
# CREATE FAILED RECORD + MCP ESCALATION
# ============================================================================

async def _create_and_escalate_record(
    turns: List[TranscriptTurn],
    parsed: Dict[str, Any],
    resolved_customer: Optional[
        Dict[str, Any]
    ],
    resolved_policy: Optional[str],
    provided_phone: Optional[str],
    reason: str,
    evidence_summary: str,
) -> Dict[str, Any]:

    transcript_id = new_id(
        "txn"
    )

    if resolved_customer:

        customer_id = (
            resolved_customer[
                "customer_id"
            ]
        )

    else:

        customer_id = new_id(
            "C"
        )

    policy_number = (
        resolved_policy
        or parsed.get(
            "policy_number"
        )
    )

    phone_number = (
        provided_phone
        or parsed.get(
            "phone_number"
        )
    )

    transcript = Transcript(
        transcript_id=transcript_id,
        customer_id=customer_id,
        turns=turns,
    )

    record = CustomerRecord(
        record_id=new_id("rec"),
        customer_id=customer_id,
        request_type=parsed.get(
            "request_type",
            "cancellation",
        ),
        policy_number=policy_number,
        effective_date=parsed.get(
            "effective_date"
        ),
        phone_number=phone_number,
        transcript_id=transcript_id,
    )

    record.status = (
        RecordStatus.FAILED.value
    )

    record.failure_reason = reason

    record.validation_errors = [
        reason
    ]

    _save_record_and_transcript(
        transcript=transcript,
        record=record,
    )

    print(
        "\nCreated failed record for escalation: "
        f"{record.to_dict()}"
    )

    escalation_result = (
        await _escalate_via_mcp(
            record_id=record.record_id,
            reason=reason,
            evidence_summary=evidence_summary,
        )
    )

    if escalation_result.get(
        "error"
    ):

        print(
            "\n[MCP] Escalation failed:",
            escalation_result,
        )

        return record.to_dict()

    record.status = (
        RecordStatus.ESCALATED.value
    )

    # Persist the final status as well.
    _update_record_status(
        record_id=record.record_id,
        status=(
            RecordStatus.ESCALATED.value
        ),
    )

    print(
        "\nEscalated record: "
        f"{record.to_dict()}"
    )

    return record.to_dict()


# ============================================================================
# MCP ESCALATION
# ============================================================================

async def _escalate_via_mcp(
    record_id: str,
    reason: str,
    evidence_summary: str,
) -> Dict[str, Any]:

    async with mcp_session() as session:

        return await call_tool(
            session,
            "escalate_to_human",
            {
                "record_id": record_id,
                "reason": reason,
                "evidence_summary": (
                    evidence_summary
                ),
            },
        )


# ============================================================================
# SAVE RECORD + TRANSCRIPT
# ============================================================================

def _save_record_and_transcript(
    transcript: Transcript,
    record: CustomerRecord,
):

    _append_json(
        DATA_DIR / "transcripts.json",
        transcript.to_dict(),
    )

    _append_json(
        DATA_DIR / "records.json",
        record.to_dict(),
    )


def _append_json(
    path: Path,
    item: Dict[str, Any],
):

    data = (
        json.loads(
            path.read_text()
        )
        if path.exists()
        else []
    )

    data.append(
        item
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        )
    )


def _update_record_status(
    record_id: str,
    status: str,
):

    path = (
        DATA_DIR /
        "records.json"
    )

    if not path.exists():
        return

    data = json.loads(
        path.read_text()
    )

    changed = False

    for item in data:

        if (
            item.get("record_id")
            == record_id
        ):

            item["status"] = status
            changed = True
            break

    if changed:

        path.write_text(
            json.dumps(
                data,
                indent=2,
                default=str,
            )
        )


# ============================================================================
# REQUEST TYPE DETECTION
# ============================================================================

def _infer_request_type(
    turns: List[TranscriptTurn],
) -> str:

    text = " ".join(
        turn.text
        for turn in turns
        if turn.speaker == "customer"
    ).lower()

    if any(
        phrase in text
        for phrase in [
            "change my date",
            "change the date",
            "change date",
            "modify my date",
            "modify the date",
            "change my policy date",
            "change policy date",
            "policy date",
            "new date",
            "reschedule",
            "reschedule my",
            "change effective date",
            "modify effective date",
        ]
    ):

        return "date_change"

    if any(
        phrase in text
        for phrase in [
            "change my address",
            "change address",
            "update my address",
            "modify my address",
            "update address",
        ]
    ):

        return "address_change"

    if any(
        phrase in text
        for phrase in [
            "cancel my policy",
            "cancel policy",
            "cancel my insurance",
            "cancel insurance",
            "cancellation",
        ]
    ):

        return "cancellation"

    return "cancellation"


# ============================================================================
# DIRECT EXECUTION
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        action="store_true",
        help=(
            "type instead of speaking "
            "(no mic/TTS needed)"
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        run_conversation(
            use_voice=not args.text
        )
    )