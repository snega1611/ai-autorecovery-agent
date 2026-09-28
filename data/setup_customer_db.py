from pathlib import Path
import sqlite3

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "recall.db"


def create_database():
    conn = sqlite3.connect(DB_PATH)

    conn.executescript(
        """
        DROP TABLE IF EXISTS policies;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone_number TEXT NOT NULL
        );

        CREATE TABLE policies (
            policy_number TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        INSERT INTO customers
            (customer_id, name, phone_number)
        VALUES
            ('C10001', 'Riya Sharma', '9389384938'),
            ('C10002', 'Arun Kumar', '9876543210');

        INSERT INTO policies
            (policy_number, customer_id, effective_date, status)
        VALUES
            ('P10001', 'C10001', '2026-09-20', 'ACTIVE'),
            ('P10005', 'C10001', '2026-08-15', 'ACTIVE'),
            ('P10002', 'C10002', '2026-09-15', 'ACTIVE');
        """
    )

    conn.commit()
    conn.close()

    print(f"Customer database created: {DB_PATH}")


if __name__ == "__main__":
    create_database()