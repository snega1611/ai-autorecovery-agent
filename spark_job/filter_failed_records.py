"""
Recall Spark batch job.

Purpose:
    Scan accumulated customer request records and build the
    recovery work queue for FAILED and CONFLICTING requests.

Normal operation:
    python spark_job/filter_failed_records.py

The job:
    1. Reads all records from data/records.json.
    2. Selects FAILED and CONFLICTING records.
    3. Calculates recovery priority.
    4. Refreshes the derived recovery queue.
    5. Removes records that are no longer recovery-eligible.

The queue is a derived batch output, not the source of truth.

Source of truth:
    data/records.json

Derived recovery queue:
    data/queue/prioritized_failed.json

Optional debugging:
    --record-id <id>

The --record-id option is intended only for development/testing.
The normal Recall workflow runs the full batch without it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
)

QUEUE_DIR = (
    DATA_DIR
    / "queue"
)

INPUT_PATH = (
    DATA_DIR
    / "records.json"
)

OUTPUT_PATH = (
    QUEUE_DIR
    / "prioritized_failed.json"
)


# ------------------------------------------------------------
# RECOVERY RULES
# ------------------------------------------------------------

RECOVERY_ELIGIBLE_STATUSES = [
    "FAILED",
    "CONFLICTING",
]


REQUIRED_FIELDS_BY_TYPE = {
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


# ------------------------------------------------------------
# SPARK SCHEMA
# ------------------------------------------------------------

RECORD_SCHEMA = StructType([
    StructField(
        "record_id",
        StringType(),
        False,
    ),
    StructField(
        "customer_id",
        StringType(),
        False,
    ),
    StructField(
        "request_type",
        StringType(),
        False,
    ),
    StructField(
        "policy_number",
        StringType(),
        True,
    ),
    StructField(
        "effective_date",
        StringType(),
        True,
    ),
    StructField(
        "phone_number",
        StringType(),
        True,
    ),
    StructField(
        "status",
        StringType(),
        False,
    ),
    StructField(
        "failure_reason",
        StringType(),
        True,
    ),
    StructField(
        "created_at",
        DoubleType(),
        False,
    ),
    StructField(
        "transcript_id",
        StringType(),
        True,
    ),
])


# ------------------------------------------------------------
# OUTPUT FIELDS
# ------------------------------------------------------------

QUEUE_FIELDS = [
    "record_id",
    "customer_id",
    "request_type",
    "policy_number",
    "effective_date",
    "phone_number",
    "status",
    "failure_reason",
    "created_at",
    "transcript_id",
    "missing_field_count",
    "age_days",
    "conflict_priority",
    "priority_score",
]


# ------------------------------------------------------------
# SPARK
# ------------------------------------------------------------

def build_spark() -> SparkSession:

    return (
        SparkSession.builder
        .appName(
            "recall-recovery-batch"
        )
        .master("local[*]")
        .config(
            "spark.sql.shuffle.partitions",
            "4",
        )
        .config(
            "spark.ui.showConsoleProgress",
            "false",
        )
        .getOrCreate()
    )


# ------------------------------------------------------------
# MISSING FIELD CALCULATION
# ------------------------------------------------------------

def missing_field_count_udf():

    from pyspark.sql.functions import udf

    def _count_missing(
        request_type,
        policy_number,
        effective_date,
    ):

        required = (
            REQUIRED_FIELDS_BY_TYPE.get(
                request_type,
                [],
            )
        )

        values = {
            "policy_number": policy_number,
            "effective_date": effective_date,
        }

        return sum(
            1
            for field_name in required
            if not values.get(field_name)
        )

    return udf(
        _count_missing,
        IntegerType(),
    )


# ------------------------------------------------------------
# QUEUE HELPERS
# ------------------------------------------------------------

def _read_existing_queue(
    output_path: Path,
) -> list[dict]:

    if not output_path.exists():
        return []

    try:

        content = output_path.read_text(
            encoding="utf-8"
        )

        if not content.strip():
            return []

        data = json.loads(content)

        if not isinstance(data, list):

            print(
                "[Spark] Existing queue is not a list."
            )

            return []

        return data

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        print(
            "[Spark] Could not read existing queue: "
            f"{exc}"
        )

        return []


def _sort_queue(
    rows: list[dict],
) -> list[dict]:

    def priority_value(row):

        try:

            return float(
                row.get(
                    "priority_score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    return sorted(
        rows,
        key=priority_value,
        reverse=True,
    )


def _write_queue(
    output_path: Path,
    rows: list[dict],
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            rows,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# BATCH JOB
# ------------------------------------------------------------

def run(
    input_path: Path = None,
    output_path: Path = None,
    record_id: str = None,
):

    input_path = (
        input_path
        or INPUT_PATH
    )

    output_path = (
        output_path
        or OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = build_spark()

    try:

        # ----------------------------------------------------
        # READ ALL ACCUMULATED RECORDS
        # ----------------------------------------------------

        print(
            "[Spark] Starting Recall recovery batch."
        )

        print(
            f"[Spark] Input: {input_path}"
        )

        df = (
            spark.read
            .schema(RECORD_SCHEMA)
            .option(
                "multiLine",
                True,
            )
            .json(
                str(input_path)
            )
        )

        total_count = df.count()

        print(
            f"[Spark] Scanned {total_count} record(s)."
        )

        # ----------------------------------------------------
        # SELECT RECOVERY-ELIGIBLE RECORDS
        # ----------------------------------------------------

        eligible_df = (
            df.filter(
                F.col("status").isin(
                    RECOVERY_ELIGIBLE_STATUSES
                )
            )
        )

        # ----------------------------------------------------
        # OPTIONAL DEBUG FILTER
        #
        # This is NOT used by normal batch operation.
        # ----------------------------------------------------

        if record_id:

            print(
                "[Spark] DEBUG MODE: "
                f"processing only {record_id}"
            )

            eligible_df = (
                eligible_df.filter(
                    F.col("record_id")
                    == record_id
                )
            )

        eligible_count = (
            eligible_df.count()
        )

        print(
            "[Spark] Recovery-eligible records: "
            f"{eligible_count}"
        )

        # ----------------------------------------------------
        # CALCULATE PRIORITY
        # ----------------------------------------------------

        count_missing = (
            missing_field_count_udf()
        )

        now = time.time()

        prioritized = (
            eligible_df

            .withColumn(
                "missing_field_count",
                count_missing(
                    F.col(
                        "request_type"
                    ),
                    F.col(
                        "policy_number"
                    ),
                    F.col(
                        "effective_date"
                    ),
                ),
            )

            .withColumn(
                "age_days",
                (
                    F.lit(now)
                    - F.col("created_at")
                ) / 86400.0,
            )

            .withColumn(
                "conflict_priority",
                F.when(
                    F.col("status")
                    == "CONFLICTING",
                    F.lit(20.0),
                ).otherwise(
                    F.lit(0.0)
                ),
            )

            .withColumn(
                "priority_score",
                (
                    F.col(
                        "missing_field_count"
                    )
                    * F.lit(10.0)

                    + F.col(
                        "conflict_priority"
                    )

                    + F.col(
                        "age_days"
                    )
                ),
            )

            .select(
                *QUEUE_FIELDS
            )
        )

        # ----------------------------------------------------
        # COLLECT BATCH RESULT
        # ----------------------------------------------------

        new_rows = [
            row.asDict()
            for row in prioritized.collect()
        ]

        # ----------------------------------------------------
        # READ PREVIOUS QUEUE
        #
        # This is only used for reporting/debugging here.
        # The batch output is rebuilt from records.json.
        # ----------------------------------------------------

        existing_rows = (
            _read_existing_queue(
                output_path
            )
        )

        print(
            "[Spark] Previous queue contained "
            f"{len(existing_rows)} record(s)."
        )

        # ----------------------------------------------------
        # TRUE BATCH REFRESH
        #
        # records.json is the source of truth.
        #
        # We DO NOT preserve old queue entries blindly.
        #
        # Therefore:
        #
        #   FAILED       → included
        #   CONFLICTING  → included
        #   COMPLETE     → removed
        #   RECOVERED    → removed
        #   ESCALATED    → removed
        #   APPROVED     → removed
        #   REJECTED     → removed
        #
        # ----------------------------------------------------

        refreshed_rows = _sort_queue(
            new_rows
        )

        # ----------------------------------------------------
        # WRITE DERIVED RECOVERY QUEUE
        # ----------------------------------------------------

        _write_queue(
            output_path,
            refreshed_rows,
        )

        print(
            "[Spark] Batch complete."
        )

        print(
            "[Spark] Recovery queue contains "
            f"{len(refreshed_rows)} record(s)."
        )

        print(
            f"[Spark] Output: {output_path}"
        )

        # ----------------------------------------------------
        # SHOW QUEUE SUMMARY
        # ----------------------------------------------------

        if refreshed_rows:

            print(
                "[Spark] Queue priority order:"
            )

            for index, row in enumerate(
                refreshed_rows,
                start=1,
            ):

                print(
                    f"  {index}. "
                    f"{row.get('record_id')} | "
                    f"{row.get('status')} | "
                    f"priority={row.get('priority_score')}"
                )

        else:

            print(
                "[Spark] No recovery-eligible "
                "records found."
            )

        return refreshed_rows

    finally:

        spark.stop()


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the Recall recovery "
            "triage batch job."
        )
    )

    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help=(
            "Optional development/debug filter. "
            "Normal batch operation scans all records."
        ),
    )

    args = parser.parse_args()

    run(
        record_id=args.record_id
    )