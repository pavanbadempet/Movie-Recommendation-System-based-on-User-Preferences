import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_db_events():
    url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    if not url:
        print("DATABASE_URL not set.")
        return

    print("=" * 60)
    print("QUERYING EXACT RECORDS FROM NEON POSTGRESQL")
    print("=" * 60)

    conn = psycopg2.connect(url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    for table_name in ["fact_user_event", "nova_content_events", "experiment_results_snapshot"]:
        print(f"\nQuerying table: '{table_name}'...")
        try:
            cur.execute(f"SELECT * FROM {table_name} ORDER BY 1 DESC LIMIT 10;")
            rows = cur.fetchall()
            print(f"Total rows retrieved: {len(rows)}")
            for idx, r in enumerate(rows):
                print(f"[{idx + 1}] {dict(r)}")
        except Exception as err:
            print(f"Query error on {table_name}: {err}")
            conn.rollback()

    cur.close()
    conn.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_db_events()
