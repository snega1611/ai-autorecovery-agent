from pathlib import Path
import sqlite3
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "recall.db"

def lookup_customer_by_policy(policy_number: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT
            c.customer_id,
            c.name,
            c.phone_number,
            p.policy_number,
            p.effective_date,
            p.status
        FROM policies p
        JOIN customers c
            ON p.customer_id = c.customer_id
        WHERE p.policy_number = ?
        """,
        (policy_number.strip().upper(),),
    ).fetchone()

    conn.close()

    if not row:
        return {
            "status": "not_found",
            "message": "No policy was found for this policy number.",
        }

    return {
        "status": "resolved",
        "customer_id": row["customer_id"],
        "name": row["name"],
        "phone_number": row["phone_number"],
        "policy": {
            "policy_number": row["policy_number"],
            "effective_date": row["effective_date"],
            "status": row["status"],
        },
    }

def lookup_customer_by_phone(phone_number: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT
            c.customer_id,
            c.name,
            c.phone_number,
            p.policy_number,
            p.effective_date,
            p.status
        FROM customers c
        LEFT JOIN policies p
            ON c.customer_id = p.customer_id
        WHERE c.phone_number = ?
        ORDER BY p.policy_number
        """,
        (phone_number.strip(),),
    ).fetchall()

    conn.close()

    if not rows:
        return {
            "status": "not_found",
            "message": "No customer was found for this phone number.",
        }

    policies = [
        {
            "policy_number": row["policy_number"],
            "effective_date": row["effective_date"],
            "status": row["status"],
        }
        for row in rows
        if row["policy_number"] is not None
    ]

    customer = {
        "customer_id": rows[0]["customer_id"],
        "name": rows[0]["name"],
        "phone_number": rows[0]["phone_number"],
    }

    if len(policies) == 1:
        return {
            "status": "resolved",
            **customer,
            "policy": policies[0],
        }

    return {
        "status": "ambiguous",
        **customer,
        "policies": policies,
        "message": (
            "Multiple policies are associated with this phone number. "
            "The correct policy cannot be selected from the phone number alone."
        ),
    }

