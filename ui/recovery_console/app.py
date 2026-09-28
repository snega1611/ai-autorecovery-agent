import requests
import streamlit as st


BACKEND_URL = "http://127.0.0.1:8000"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Recall — Recovery Operations",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# HELPERS
# =========================================================

def api_get(path: str):
    response = requests.get(
        f"{BACKEND_URL}{path}",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(
        f"{BACKEND_URL}{path}",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def format_event(event):
    return (
        event.get("event", "unknown"),
        event.get("detail", {}),
    )


# =========================================================
# HEADER
# =========================================================

st.title("Recall — Recovery Operations Console")

st.caption(
    "Internal operations console for reviewing "
    "AI-escalated customer recovery requests."
)


# =========================================================
# BACKEND CONNECTION
# =========================================================

try:

    stats = api_get("/stats")
    pending_cases = api_get("/pending")

except requests.RequestException as exc:

    st.error(
        "Unable to connect to the Recall backend. "
        "Make sure FastAPI is running on port 8000."
    )

    st.code(str(exc))

    st.stop()


# =========================================================
# STATISTICS
# =========================================================

counts = stats.get(
    "by_status",
    {},
)


total_records = stats.get(
    "total_records",
    0,
)


failed = counts.get(
    "FAILED",
    0,
)


conflicting = counts.get(
    "CONFLICTING",
    0,
)


escalated = counts.get(
    "ESCALATED",
    0,
)


recovered = counts.get(
    "RECOVERED",
    0,
)


# =========================================================
# STATISTICS DISPLAY
# =========================================================

col1, col2, col3, col4, col5, col6 = st.columns(6)


with col1:

    st.metric(
        "Total Records",
        total_records,
    )


with col2:

    st.metric(
        "Failed",
        failed,
    )


with col3:

    st.metric(
        "Conflicting",
        conflicting,
    )


with col4:

    st.metric(
        "Escalated",
        escalated,
    )


with col5:

    st.metric(
        "Pending Review",
        len(pending_cases),
    )


with col6:

    st.metric(
        "Recovered",
        recovered,
    )


# =========================================================
# RECOVERY LIFECYCLE
# =========================================================

st.divider()

st.subheader("Recovery Lifecycle")

st.caption(
    "A FAILED or CONFLICTING record becomes recovery-eligible. "
    "Spark prioritizes eligible records, then the Recovery Agent "
    "investigates them. Human review is required when the agent "
    "cannot safely recover the request."
)


lifecycle_cols = st.columns(5)


with lifecycle_cols[0]:

    st.markdown("### 1️⃣")

    st.markdown(
        "**Voice Intake**"
    )

    st.caption(
        "Customer request is captured and validated."
    )


with lifecycle_cols[1]:

    st.markdown("### 2️⃣")

    st.markdown(
        "**Failed / Conflicting**"
    )

    st.caption(
        f"{failed + conflicting} record(s) currently require recovery triage."
    )


with lifecycle_cols[2]:

    st.markdown("### 3️⃣")

    st.markdown(
        "**Spark Batch**"
    )

    st.caption(
        "Eligible records are prioritized for investigation."
    )


with lifecycle_cols[3]:

    st.markdown("### 4️⃣")

    st.markdown(
        "**Recovery Agent**"
    )

    st.caption(
        "Agent investigates evidence and policy."
    )


with lifecycle_cols[4]:

    st.markdown("### 5️⃣")

    st.markdown(
        "**Human Review**"
    )

    st.caption(
        f"{len(pending_cases)} case(s) currently awaiting approval."
    )


# =========================================================
# CURRENT RECOVERY STATUS
# =========================================================

if conflicting > 0:

    st.warning(
        f"⚠️ {conflicting} conflicting record(s) are currently "
        "eligible for Spark recovery triage."
    )


if failed > 0:

    st.info(
        f"ℹ️ {failed} failed record(s) are currently "
        "eligible for Spark recovery triage."
    )


if conflicting == 0 and failed == 0:

    st.success(
        "No FAILED or CONFLICTING records are currently "
        "waiting for recovery triage."
    )


# =========================================================
# RECOVERY QUEUE
# =========================================================

st.divider()

st.subheader("Human Recovery Queue")


if not pending_cases:

    st.success(
        "No recovery cases are currently waiting for human review."
    )

else:

    st.caption(
        f"{len(pending_cases)} case(s) currently require human review."
    )


# =========================================================
# CASES
# =========================================================

for case in pending_cases:

    record_id = case.get(
        "record_id",
        "Unknown record",
    )

    customer_id = case.get(
        "customer_id",
        "Unknown customer",
    )

    reason = case.get(
        "reason",
        "No reason provided.",
    )

    evidence_summary = case.get(
        "evidence_summary",
        "No evidence summary available.",
    )

    transcript = case.get(
        "transcript"
    )

    audit_trail = case.get(
        "audit_trail",
        [],
    )

    proposed_fields = case.get(
        "proposed_fields",
        {},
    )


    # =====================================================
    # CASE CONTAINER
    # =====================================================

    with st.container(border=True):


        # -------------------------------------------------
        # CASE HEADER
        # -------------------------------------------------

        header_left, header_right = st.columns(
            [4, 1]
        )


        with header_left:

            st.markdown(
                f"## {record_id}"
            )

            st.caption(
                f"Customer ID: {customer_id}"
            )


        with header_right:

            st.warning(
                "PENDING REVIEW"
            )


        # -------------------------------------------------
        # ESCALATION REASON
        # -------------------------------------------------

        st.markdown(
            "### Escalation Reason"
        )

        st.error(
            reason
        )


        # -------------------------------------------------
        # EVIDENCE SUMMARY
        # -------------------------------------------------

        st.markdown(
            "### Evidence Summary"
        )

        st.info(
            evidence_summary
        )


        # -------------------------------------------------
        # CUSTOMER CONVERSATION
        # -------------------------------------------------

        st.markdown(
            "### Customer Conversation"
        )


        if not transcript:

            st.warning(
                "No transcript was returned for this recovery case."
            )

        else:

            turns = transcript.get(
                "turns",
                [],
            )


            if not turns:

                st.warning(
                    "Transcript exists but contains no turns."
                )

            else:

                for turn in turns:

                    speaker = turn.get(
                        "speaker",
                        "unknown",
                    ).lower()

                    text = turn.get(
                        "text",
                        "",
                    )


                    if speaker == "customer":

                        with st.chat_message("user"):

                            st.markdown(
                                f"**Customer**\n\n{text}"
                            )


                    elif speaker == "agent":

                        with st.chat_message("assistant"):

                            st.markdown(
                                f"**Agent**\n\n{text}"
                            )


                    else:

                        with st.chat_message("assistant"):

                            st.markdown(
                                f"**{speaker.title()}**\n\n{text}"
                            )


        # -------------------------------------------------
        # PROPOSED RECOVERY
        # -------------------------------------------------

        st.markdown(
            "### Proposed Recovery Fields"
        )


        if proposed_fields:

            for field, value in proposed_fields.items():

                st.write(
                    f"**{field}:** `{value}`"
                )

        else:

            st.caption(
                "The agent did not submit recovery fields "
                "for this escalation."
            )


        # -------------------------------------------------
        # AI INVESTIGATION
        # -------------------------------------------------

        with st.expander(
            "View AI Investigation & Audit Trail",
            expanded=False,
        ):

            if not audit_trail:

                st.info(
                    "No audit events available."
                )

            else:

                for index, event in enumerate(
                    audit_trail,
                    start=1,
                ):

                    event_type, detail = format_event(
                        event
                    )


                    if event_type == "decision":

                        st.markdown(
                            "#### Decision"
                        )

                        decision = detail.get(
                            "decision",
                            "Unknown",
                        )

                        decision_reason = detail.get(
                            "reason",
                            "No reason provided.",
                        )

                        decision_evidence = detail.get(
                            "evidence_summary",
                            "No evidence summary.",
                        )

                        st.write(
                            f"**Decision:** `{decision}`"
                        )

                        st.write(
                            f"**Reason:** {decision_reason}"
                        )

                        st.write(
                            f"**Evidence:** {decision_evidence}"
                        )


                    else:

                        st.markdown(
                            f"#### Step {index}: `{event_type}`"
                        )

                        tool_name = detail.get(
                            "tool",
                            "unknown",
                        )

                        arguments = detail.get(
                            "arguments",
                            {},
                        )

                        st.write(
                            f"**Tool:** `{tool_name}`"
                        )

                        st.write(
                            "**Arguments:**"
                        )

                        st.json(
                            arguments
                        )


                    st.divider()


        # =================================================
        # HUMAN-IN-THE-LOOP
        # =================================================

        st.markdown(
            "### Human Decision"
        )

        st.caption(
            "Review the AI evidence above before making "
            "the final recovery decision."
        )


        # -------------------------------------------------
        # OPTIONAL CORRECTION FIELDS
        # -------------------------------------------------

        if proposed_fields:

            st.markdown(
                "**Recovery values to approve**"
            )

            approved_fields = {}


            for field, value in proposed_fields.items():

                approved_value = st.text_input(
                    field.replace("_", " ").title(),
                    value=str(value),
                    key=f"{record_id}_{field}",
                )

                approved_fields[field] = approved_value

        else:

            approved_fields = {}


        # -------------------------------------------------
        # REVIEWER NOTE
        # -------------------------------------------------

        reviewer_note = st.text_area(
            "Reviewer note",
            placeholder=(
                "Add a reason for your approval or rejection..."
            ),
            key=f"{record_id}_note",
        )


        # -------------------------------------------------
        # DECISION BUTTONS
        # -------------------------------------------------

        approve_col, reject_col = st.columns(2)


        with approve_col:

            approve_clicked = st.button(
                "Approve Recovery",
                type="primary",
                use_container_width=True,
                key=f"{record_id}_approve",
            )


        with reject_col:

            reject_clicked = st.button(
                "Reject Recovery",
                use_container_width=True,
                key=f"{record_id}_reject",
            )


        # -------------------------------------------------
        # APPROVE
        # -------------------------------------------------

        if approve_clicked:

            try:

                result = api_post(
                    f"/approve/{record_id}",
                    {
                        "approved_fields": approved_fields,
                        "note": reviewer_note,
                    },
                )

                st.success(
                    f"Recovery approved for {record_id}."
                )

                st.json(
                    result
                )

                st.rerun()

            except requests.RequestException as exc:

                st.error(
                    "Unable to approve this recovery."
                )

                st.code(
                    str(exc)
                )


        # -------------------------------------------------
        # REJECT
        # -------------------------------------------------

        if reject_clicked:

            try:

                result = api_post(
                    f"/reject/{record_id}",
                    {
                        "approved_fields": {},
                        "note": reviewer_note,
                    },
                )

                st.warning(
                    f"Recovery rejected for {record_id}."
                )

                st.json(
                    result
                )

                st.rerun()

            except requests.RequestException as exc:

                st.error(
                    "Unable to reject this recovery."
                )

                st.code(
                    str(exc)
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Recall Recovery Operations Console • Internal use"
)