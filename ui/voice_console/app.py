from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Recall — Voice Operations",
    page_icon="🎙️",
    layout="wide",
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "call_active" not in st.session_state:
    st.session_state.call_active = False

if "last_record" not in st.session_state:
    st.session_state.last_record = None

if "error" not in st.session_state:
    st.session_state.error = None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🎙️ Recall Voice Operations")

st.caption(
    "Customer voice intake and request recovery pipeline"
)


# ---------------------------------------------------------
# ARCHITECTURE
# ---------------------------------------------------------

st.info(
    """
    **Voice Intake → STT → AI Conversation → Validation → Record Storage**

    Complete requests stop here. Failed or conflicting requests become
    candidates for the downstream Spark recovery batch.
    """
)


# ---------------------------------------------------------
# CUSTOMER CALL
# ---------------------------------------------------------

st.subheader("Customer Call")

st.write(
    """
    **How this demo works**

    Click **Start Customer Call**. Recall will listen to the customer,
    process the conversation using the existing voice pipeline, speak the
    AI response, and create the customer request record.
    """
)


# ---------------------------------------------------------
# START CALL
# ---------------------------------------------------------

start_call = st.button(
    "🎙️ Start Customer Call",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.call_active,
)


if start_call:

    st.session_state.call_active = True
    st.session_state.error = None

    try:

        from voice.conversation import run_conversation

        st.info(
            "🎙️ **Call started.** Speak when Recall is listening."
        )

        with st.spinner(
            "Recall is processing the conversation..."
        ):

            record = asyncio.run(
                run_conversation(
                    use_voice=True
                )
            )

        st.session_state.last_record = record

        st.success(
            "Customer call completed."
        )

    except Exception as exc:

        st.session_state.error = str(exc)

        st.error(
            "The voice conversation could not be completed."
        )

    finally:

        st.session_state.call_active = False


# ---------------------------------------------------------
# ERROR
# ---------------------------------------------------------

if st.session_state.error:

    st.error(
        f"Voice pipeline error: {st.session_state.error}"
    )


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

record = st.session_state.last_record

if record:

    st.divider()

    st.subheader("Call Result")

    record_id = record.get(
        "record_id",
        "unknown",
    )

    status = record.get(
        "status",
        "UNKNOWN",
    )

    request_type = record.get(
        "request_type",
        "unknown",
    )

    transcript_id = record.get(
        "transcript_id"
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Record",
            record_id,
        )

    with col2:

        st.metric(
            "Status",
            status,
        )

    with col3:

        st.metric(
            "Request Type",
            request_type,
        )


    # -----------------------------------------------------
    # STATUS MESSAGE
    # -----------------------------------------------------

    if status == "COMPLETE":

        st.success(
            "✅ Customer request is complete. "
            "No recovery is required."
        )


    elif status in {
        "FAILED",
        "CONFLICTING",
    }:

        st.warning(
            f"""
            ⚠️ This request is **{status}**.

            The record has been persisted and is eligible for
            the downstream recovery batch.
            """
        )

        st.info(
            """
            **Next stage**

            `records.json → Spark batch → prioritized_failed.json → Recovery Agent`
            """
        )


    elif status == "ESCALATED":

        st.error(
            """
            🧑‍💼 This request was escalated during voice intake.

            No automated recovery will be attempted.
            """
        )


    else:

        st.info(
            f"Record finished with status: **{status}**"
        )


    # -----------------------------------------------------
    # RECORD DETAILS
    # -----------------------------------------------------

    with st.expander(
        "View Record Details"
    ):

        st.json(record)


    # -----------------------------------------------------
    # TRANSCRIPT
    # -----------------------------------------------------

    if transcript_id:

        with st.expander(
            "Transcript Information"
        ):

            st.write(
                f"Transcript ID: `{transcript_id}`"
            )

            st.caption(
                "The transcript is persisted by the existing "
                "voice conversation pipeline."
            )


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

st.divider()

st.subheader("Recall Pipeline")


cols = st.columns(4)


with cols[0]:

    st.markdown("### 🎙️ 1")

    st.markdown(
        "**Voice Intake**"
    )

    st.caption(
        "Customer speaks with Recall."
    )


with cols[1]:

    st.markdown("### 🧠 2")

    st.markdown(
        "**AI + Validation**"
    )

    st.caption(
        "Speech is converted into a structured request."
    )


with cols[2]:

    st.markdown("### ⚡ 3")

    st.markdown(
        "**Spark Batch**"
    )

    st.caption(
        "FAILED / CONFLICTING records enter recovery triage."
    )


with cols[3]:

    st.markdown("### 🤖 4")

    st.markdown(
        "**Recovery Agent**"
    )

    st.caption(
        "Agent investigates and recovers or escalates."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Recall — Autonomous Voice Request Recovery Agent"
)