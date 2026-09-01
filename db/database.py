"""SQLite connection + init helpers for the CF analytics tool."""

import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "cf_data.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the database file and tables if they don't already exist."""
    schema_sql = SCHEMA_PATH.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH):
    """Context-managed connection with foreign keys enabled and Row access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- cache freshness helpers -------------------------------------------------

def get_last_refresh(conn: sqlite3.Connection, key: str) -> int | None:
    row = conn.execute(
        "SELECT last_refreshed_at FROM cache_meta WHERE key = ?", (key,)
    ).fetchone()
    return row["last_refreshed_at"] if row else None


def mark_refreshed(conn: sqlite3.Connection, key: str, when: int | None = None) -> None:
    when = when if when is not None else int(time.time())
    conn.execute(
        """
        INSERT INTO cache_meta (key, last_refreshed_at) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET last_refreshed_at = excluded.last_refreshed_at
        """,
        (key, when),
    )


def is_stale(conn: sqlite3.Connection, key: str, ttl_seconds: int) -> bool:
    """True if `key` has never been refreshed or is older than ttl_seconds."""
    last = get_last_refresh(conn, key)
    if last is None:
        return True
    return (time.time() - last) > ttl_seconds
