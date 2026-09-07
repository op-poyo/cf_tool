"""Local-only practice session tracking: sessions, per-problem attempts,
timers, notes. None of this comes from the Codeforces API -- CF never
tells you when you *started* a problem, only when you submitted -- so
everything here is entirely user-entered, stored only in the local DB.

TIMER MODEL: each attempt stores active_seconds (accumulated elapsed
time, NOT counting paused spans) plus last_resumed_at (set only while
running). Current elapsed time for a running attempt is always computed
as active_seconds + (now - last_resumed_at) via elapsed_seconds() below,
rather than kept ticking in the background. Streamlit reruns the whole
script on every widget interaction and there's no persistent process to
tick a clock in, so the displayed number updates whenever a rerun
happens to occur (any button click, page reload) rather than counting
up smoothly on its own. The stored number is always correct at the
moment it's read; it just doesn't animate between reads.
"""

import sqlite3
import time

import pandas as pd

STATUS_IN_PROGRESS = "in_progress"
STATUS_SOLVED = "solved"
STATUS_GAVE_UP = "gave_up"

TIMER_RUNNING = "running"
TIMER_PAUSED = "paused"
TIMER_STOPPED = "stopped"


# --- sessions ----------------------------------------------------------

def start_session(conn: sqlite3.Connection, handle: str) -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO sessions (handle, started_at) VALUES (?, ?)", (handle, now)
    )
    return cur.lastrowid


def end_session(conn: sqlite3.Connection, session_id: int, notes: str | None = None) -> None:
    now = int(time.time())
    conn.execute(
        "UPDATE sessions SET ended_at = ?, notes = COALESCE(?, notes) WHERE id = ?",
        (now, notes, session_id),
    )


def get_active_session(conn: sqlite3.Connection, handle: str) -> sqlite3.Row | None:
    """Most recent session for this handle that hasn't been ended yet.
    Assumes at most one active session per handle -- the UI only ever
    offers to start a new one when this returns None."""
    return conn.execute(
        """SELECT * FROM sessions
           WHERE handle = ? AND ended_at IS NULL
           ORDER BY started_at DESC LIMIT 1""",
        (handle,),
    ).fetchone()


def get_session_history(conn: sqlite3.Connection, handle: str, limit: int = 50) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT * FROM sessions
           WHERE handle = ? AND ended_at IS NOT NULL
           ORDER BY started_at DESC LIMIT ?""",
        conn, params=(handle, limit),
    )


def get_all_attempts(conn: sqlite3.Connection, handle: str) -> pd.DataFrame:
    """Every attempt ever logged across all of this handle's sessions --
    active and past alike. Backs the flat, sortable Logged Questions
    view, as opposed to get_session_attempts (one session at a time)."""
    return pd.read_sql_query(
        """SELECT sa.*, s.started_at AS session_started_at
           FROM session_attempts sa
           JOIN sessions s ON sa.session_id = s.id
           WHERE s.handle = ?
           ORDER BY sa.created_at DESC""",
        conn, params=(handle,),
    )


# --- attempts ------------------------------------------------------------

def add_attempt(conn: sqlite3.Connection, session_id: int, contest_id: int, problem_index: str) -> int:
    now = int(time.time())
    cur = conn.execute(
        """INSERT INTO session_attempts (session_id, contest_id, problem_index, created_at)
           VALUES (?, ?, ?, ?)""",
        (session_id, contest_id, problem_index, now),
    )
    return cur.lastrowid


def get_session_attempts(conn: sqlite3.Connection, session_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM session_attempts WHERE session_id = ? ORDER BY created_at",
        conn, params=(session_id,),
    )


def start_timer(conn: sqlite3.Connection, attempt_id: int) -> None:
    """No-op if already running -- guards against a double-click
    clobbering last_resumed_at and losing accumulated time."""
    now = int(time.time())
    conn.execute(
        """UPDATE session_attempts
           SET timer_state = ?, last_resumed_at = ?
           WHERE id = ? AND timer_state != ?""",
        (TIMER_RUNNING, now, attempt_id, TIMER_RUNNING),
    )


def pause_timer(conn: sqlite3.Connection, attempt_id: int) -> None:
    row = conn.execute(
        "SELECT active_seconds, last_resumed_at, timer_state FROM session_attempts WHERE id = ?",
        (attempt_id,),
    ).fetchone()
    if row is None or row["timer_state"] != TIMER_RUNNING:
        return
    now = int(time.time())
    elapsed = now - row["last_resumed_at"]
    conn.execute(
        """UPDATE session_attempts
           SET active_seconds = active_seconds + ?, timer_state = ?, last_resumed_at = NULL
           WHERE id = ?""",
        (elapsed, TIMER_PAUSED, attempt_id),
    )


def _finalize_timer(conn: sqlite3.Connection, attempt_id: int) -> None:
    """Fold any currently-running time into active_seconds and stop the
    timer. Internal -- called by mark_solved/mark_gave_up so neither has
    to duplicate this."""
    row = conn.execute(
        "SELECT active_seconds, last_resumed_at, timer_state FROM session_attempts WHERE id = ?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        return
    if row["timer_state"] == TIMER_RUNNING:
        now = int(time.time())
        elapsed = now - row["last_resumed_at"]
        conn.execute(
            "UPDATE session_attempts SET active_seconds = active_seconds + ? WHERE id = ?",
            (elapsed, attempt_id),
        )
    conn.execute(
        "UPDATE session_attempts SET timer_state = ?, last_resumed_at = NULL WHERE id = ?",
        (TIMER_STOPPED, attempt_id),
    )


def mark_solved(
    conn: sqlite3.Connection, attempt_id: int, felt_difficulty: str | None, notes: str | None
) -> None:
    _finalize_timer(conn, attempt_id)
    now = int(time.time())
    conn.execute(
        """UPDATE session_attempts
           SET status = ?, solved_at = ?, felt_difficulty = ?, notes = ?
           WHERE id = ?""",
        (STATUS_SOLVED, now, felt_difficulty, notes, attempt_id),
    )


def mark_gave_up(
    conn: sqlite3.Connection, attempt_id: int, felt_difficulty: str | None, notes: str | None
) -> None:
    _finalize_timer(conn, attempt_id)
    conn.execute(
        """UPDATE session_attempts
           SET status = ?, felt_difficulty = ?, notes = ?
           WHERE id = ?""",
        (STATUS_GAVE_UP, felt_difficulty, notes, attempt_id),
    )


def elapsed_seconds(attempt_row) -> int:
    """Correct current elapsed time for an attempt row (sqlite3.Row or
    dict with active_seconds, last_resumed_at, timer_state), whether
    running, paused, or stopped. See module docstring for the timer
    model this implements."""
    active = attempt_row["active_seconds"] or 0
    if attempt_row["timer_state"] == TIMER_RUNNING and attempt_row["last_resumed_at"]:
        active += int(time.time()) - attempt_row["last_resumed_at"]
    return active


def format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
