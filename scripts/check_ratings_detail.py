import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_ratings():
    url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    conn = psycopg2.connect(url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("=" * 60)
    print("SEARCHING FOR RECENT USER RATINGS IN NEON DB")
    print("=" * 60)

    cur.execute("""
        SELECT id, event_id, event_type, movie_id, rating, metadata, event_ts, created_at
        FROM nova_content_events
        WHERE event_type = 'rating'
        ORDER BY id DESC
        LIMIT 10;
    """)
    rows = cur.fetchall()
    print(f"Total Rating Events Found: {len(rows)}")
    for r in rows:
        print(
            f"ID: {r['id']} | Type: {r['event_type']} | Movie: {r['movie_id']} | Rating: {r['rating']} | Timestamp: {r['created_at']}"
        )
        print(f"Metadata: {r['metadata']}\n")

    print("\n--- Latest 5 Global Events Across Entire Platform ---")
    cur.execute("""
        SELECT id, event_type, movie_id, rating, created_at, metadata
        FROM nova_content_events
        ORDER BY id DESC
        LIMIT 5;
    """)
    for r in cur.fetchall():
        print(
            f"Event #{r['id']} | Type: {r['event_type']} | Movie ID: {r['movie_id']} | Rating: {r['rating']} | At: {r['created_at']}"
        )

    cur.close()
    conn.close()


if __name__ == "__main__":
    check_ratings()
