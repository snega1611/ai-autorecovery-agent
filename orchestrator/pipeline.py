from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT_DIR),
)

QUEUE_PATH = (
    ROOT_DIR
    / "data"
    / "queue"
    / "prioritized_failed.json"
)


RECOVERY_ELIGIBLE_STATUSES = {
    "CONFLICTING",
    "FAILED",
}


def run_spark_triage():
    """
    Run the full Spark batch.

    Spark reads all records from records.json and rebuilds the
    derived recovery queue.
    """

    print(
        "\n[Pipeline] Running Spark batch triage..."
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT_DIR
                / "spark_job"
                / "filter_failed_records.py"
            ),
        ],
        cwd=ROOT_DIR,
        check=True,
    )

    return result.returncode


def get_prioritized_queue():
    """
    Read the queue produced by the Spark batch.
    """

    if not QUEUE_PATH.exists():

        print(
            "[Pipeline] Spark recovery queue was not found."
        )

        return []

    queue = json.loads(
        QUEUE_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        queue,
        list,
    ):

        print(
            "[Pipeline] Spark queue is not a list."
        )

        return []

    return queue


async def process_recovery_queue(
    queue_path: Path,
):
    """
    Send the Spark-generated recovery queue to the Recovery Agent.

    The Recovery Agent owns investigation and terminal decisions.
    """

    from agent.graph import run_queue

    print(
        "\n[Pipeline] Spark batch completed."
    )

    print(
        "[Pipeline] Starting Recovery Agent "
        "on the generated queue..."
    )

    await run_queue(
        queue_path
    )


async def run_pipeline():
    """
    End-to-end Recall pipeline.

        Voice AI
            ↓
        records.json
            ↓
        FAILED / CONFLICTING
            ↓
        Spark batch
            ↓
        prioritized_failed.json
            ↓
        Recovery Agent
            ↓
        Qwen chooses MCP tools
            ↓
        RECOVERED / ESCALATED
    """

    from voice.conversation import (
        run_conversation,
    )

    print(
        "\n========== "
        "RECALL VOICE INTAKE "
        "=========="
    )

    # --------------------------------------------------
    # VOICE INTAKE
    # --------------------------------------------------

    record = await run_conversation(
        use_voice=True
    )

    record_id = record[
        "record_id"
    ]

    status = record[
        "status"
    ]

    print(
        f"\n[Pipeline] Created record: "
        f"{record_id} ({status})"
    )

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    if status == "COMPLETE":

        print(
            "\n[Pipeline] Record is COMPLETE."
        )

        print(
            "[Pipeline] No recovery is needed."
        )

        return record

    # --------------------------------------------------
    # ALREADY ESCALATED
    # --------------------------------------------------

    if status == "ESCALATED":

        print(
            "\n[Pipeline] Record was already "
            "escalated during voice intake."
        )

        print(
            "[Pipeline] Skipping Spark and "
            "Recovery Agent."
        )

        return record

    # --------------------------------------------------
    # RECOVERY ELIGIBILITY
    # --------------------------------------------------

    if status not in RECOVERY_ELIGIBLE_STATUSES:

        print(
            f"\n[Pipeline] Unexpected "
            f"record status: {status}"
        )

        return record

    print(
        f"\n[Pipeline] Record is {status}."
    )

    print(
        "[Pipeline] Starting batch recovery pipeline..."
    )

    # --------------------------------------------------
    # SPARK BATCH
    # --------------------------------------------------

    run_spark_triage()

    # --------------------------------------------------
    # READ SPARK QUEUE
    # --------------------------------------------------

    queue = get_prioritized_queue()

    print(
        f"\n[Pipeline] Spark generated "
        f"{len(queue)} recovery-eligible record(s)."
    )

    # --------------------------------------------------
    # VERIFY CURRENT RECORD IS IN QUEUE
    # --------------------------------------------------

    current_record = next(
        (
            item
            for item in queue
            if item.get(
                "record_id"
            )
            == record_id
        ),
        None,
    )

    if current_record is None:

        print(
            f"[Pipeline] Current record "
            f"{record_id} is not in the Spark queue."
        )

        print(
            "[Pipeline] It may no longer be "
            "recovery-eligible."
        )

        return record

    print(
        f"[Pipeline] Current record "
        f"{record_id} is present in Spark queue."
    )

    print(
        "[Pipeline] Priority score: "
        f"{current_record.get('priority_score')}"
    )

    # --------------------------------------------------
    # RECOVERY AGENT
    # --------------------------------------------------

    await process_recovery_queue(
        QUEUE_PATH
    )

    print(
        "\n========== "
        "RECOVERY BATCH COMPLETE "
        "=========="
    )

    print(
        "[Pipeline] Spark completed the batch and "
        "the Recovery Agent processed the generated queue."
    )

    return record


if __name__ == "__main__":

    asyncio.run(
        run_pipeline()
    )