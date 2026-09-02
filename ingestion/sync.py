"""Glue between the CF API client and SQLite storage. This is the piece
that actually populates the schema: pulls from the API only when the
cache is stale, upserts into the tables, and handles the two failure
paths we designed -- invalid handle (caller should reprompt) and
transient API failure (fall back to existing cache if there is one).
"""

import time
import logging
import sqlite3

from db.database import get_last_refresh, mark_refreshed, is_stale
from .client import CFClient, InvalidHandleError
from .rate_limiter import IngestionError

logger = logging.getLogger(__name__)

GLOBAL_TTL_SECONDS = 86400   # contest list / problemset: refresh daily
USER_TTL_SECONDS = 3600      # per-user data: refresh hourly, or force-refresh on demand

# problemset.problems only refreshes on GLOBAL_TTL_SECONDS, so a contest
# that happened within the last few days can easily predate the last
# global sync -- until that catches up, _extract_embedded_problems() is
# the ONLY source for that contest's problems, and it only knows about
# problems the user actually submitted to. Left alone, an untouched
# problem in a recent contest has no row anywhere: num_problems undercounts
# (looks like "only the problems I attempted"), and every tag/weakness
# function is silently blind to it too. Bounded backfill window so this
# doesn't turn into one contest.standings call per contest in a veteran
# user's entire history on every sync -- older contests are already
# covered by the routine global problemset refresh by the time this runs.
RECENT_CONTEST_BACKFILL_SECONDS = 7 * 86400


# -- upserts ------------------------------------------------------------

def upsert_contests(conn: sqlite3.Connection, contests: list[dict]) -> None:
    now = int(time.time())
    rows = [
        {
            "id": c["id"],
            "name": c.get("name"),
            "phase": c.get("phase"),
            "type": c.get("type"),
            "start_time": c.get("startTimeSeconds"),
            "duration": c.get("durationSeconds"),
            "fetched_at": now,
        }
        for c in contests
    ]
    conn.executemany(
        """INSERT INTO contests (id, name, phase, type, start_time, duration, fetched_at)
           VALUES (:id, :name, :phase, :type, :start_time, :duration, :fetched_at)
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name, phase=excluded.phase, type=excluded.type,
             start_time=excluded.start_time, duration=excluded.duration,
             fetched_at=excluded.fetched_at""",
        rows,
    )


def upsert_problems_and_tags(conn: sqlite3.Connection, problems: list[dict]) -> None:
    now = int(time.time())
    problem_rows, tag_rows = [], []
    for p in problems:
        contest_id, index = p.get("contestId"), p.get("index")
        if contest_id is None or index is None:
            continue  # problems with no contestId (pure problemset entries) aren't usable for per-contest analytics
        problem_rows.append(
            {
                "contest_id": contest_id,
                "problem_index": index,
                "name": p.get("name"),
                "rating": p.get("rating"),
                "fetched_at": now,
            }
        )
        for tag in p.get("tags", []):
            tag_rows.append({"contest_id": contest_id, "problem_index": index, "tag": tag})

    conn.executemany(
        """INSERT INTO problems (contest_id, problem_index, name, rating, fetched_at)
           VALUES (:contest_id, :problem_index, :name, :rating, :fetched_at)
           ON CONFLICT(contest_id, problem_index) DO UPDATE SET
             name=excluded.name, rating=excluded.rating, fetched_at=excluded.fetched_at""",
        problem_rows,
    )
    # Simplest correct way to handle a problem's tag set changing: clear then reinsert.
    conn.executemany(
        "DELETE FROM problem_tags WHERE contest_id = :contest_id AND problem_index = :problem_index",
        [{"contest_id": r["contest_id"], "problem_index": r["problem_index"]} for r in problem_rows],
    )
    if tag_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO problem_tags (contest_id, problem_index, tag) "
            "VALUES (:contest_id, :problem_index, :tag)",
            tag_rows,
        )


def _extract_embedded_problems(submissions: list[dict]) -> list[dict]:
    """Every submission embeds a full Problem object (including rating and
    tags when known). problemset.problems only covers the main problemset,
    not gym problems -- so without this, a gym attempt would have no
    rating anywhere in our data and would be silently invisible to the
    tag-weighting functions. Submissions are the one place gym problem
    metadata is actually available to us."""
    seen: dict[tuple, dict] = {}
    for s in submissions:
        p = s.get("problem") or {}
        contest_id, index = p.get("contestId"), p.get("index")
        if contest_id is None or index is None:
            continue
        seen[(contest_id, index)] = p
    return list(seen.values())


def _backfill_recent_contest_problems(
    conn: sqlite3.Connection, client: CFClient, submissions: list[dict]
) -> None:
    """For any contest the user just submitted to that started within
    RECENT_CONTEST_BACKFILL_SECONDS, fetch its full problem list from
    contest.standings and upsert it -- overwriting whatever partial set
    _extract_embedded_problems left behind with the real, complete one.
    A no-op for contests the global problemset sync already fully covers
    (the upsert just rewrites the same rows)."""
    now = int(time.time())
    contest_ids = {
        s.get("problem", {}).get("contestId")
        for s in submissions
        if s.get("problem", {}).get("contestId") is not None
    }
    if not contest_ids:
        return

    placeholders = ",".join("?" * len(contest_ids))
    rows = conn.execute(
        f"SELECT id, start_time FROM contests WHERE id IN ({placeholders})",
        list(contest_ids),
    ).fetchall()
    recent_ids = [
        r["id"] for r in rows
        if r["start_time"] is not None
        and now - r["start_time"] <= RECENT_CONTEST_BACKFILL_SECONDS
    ]

    for contest_id in recent_ids:
        try:
            problems = client.get_contest_problems(contest_id)
            upsert_problems_and_tags(conn, problems)
        except IngestionError as exc:
            logger.warning(
                "Couldn't backfill full problem list for contest %s: %s", contest_id, exc
            )


def upsert_user_info(conn: sqlite3.Connection, handle: str, rating: int | None) -> None:
    now = int(time.time())
    conn.execute(
        """INSERT INTO users (handle, current_rating, last_synced_at) VALUES (?, ?, ?)
           ON CONFLICT(handle) DO UPDATE SET
             current_rating=excluded.current_rating, last_synced_at=excluded.last_synced_at""",
        (handle, rating, now),
    )


def upsert_rating_history(conn: sqlite3.Connection, handle: str, rating_changes: list[dict]) -> None:
    rows = [
        {
            "handle": handle,
            "contest_id": rc["contestId"],
            "rating_before": rc.get("oldRating"),
            "rating_after": rc.get("newRating"),
            "rank": rc.get("rank"),
            "rating_update_time": rc.get("ratingUpdateTimeSeconds"),
        }
        for rc in rating_changes
    ]
    conn.executemany(
        """INSERT INTO user_rating_history
             (handle, contest_id, rating_before, rating_after, rank, rating_update_time)
           VALUES (:handle, :contest_id, :rating_before, :rating_after, :rank, :rating_update_time)
           ON CONFLICT(handle, contest_id) DO UPDATE SET
             rating_before=excluded.rating_before, rating_after=excluded.rating_after,
             rank=excluded.rank, rating_update_time=excluded.rating_update_time""",
        rows,
    )


def upsert_submissions(conn: sqlite3.Connection, handle: str, submissions: list[dict]) -> None:
    rows = []
    for s in submissions:
        problem = s.get("problem", {})
        rows.append(
            {
                "submission_id": s["id"],
                "handle": handle,
                "contest_id": problem.get("contestId"),
                "problem_index": problem.get("index"),
                "verdict": s.get("verdict"),
                "creation_time": s["creationTimeSeconds"],
            }
        )
    conn.executemany(
        """INSERT INTO submissions
             (submission_id, handle, contest_id, problem_index, verdict, creation_time)
           VALUES (:submission_id, :handle, :contest_id, :problem_index, :verdict, :creation_time)
           ON CONFLICT(submission_id) DO UPDATE SET verdict=excluded.verdict""",
        rows,
    )


# -- orchestration --------------------------------------------------------

def sync_global_data(conn: sqlite3.Connection, client: CFClient, force: bool = False) -> None:
    """Refreshes contest.list and problemset.problems if stale. On a
    transient failure, logs and keeps whatever is already cached rather
    than raising -- global data changing slowly means stale-by-a-day data
    is a fine degraded state."""
    if force or is_stale(conn, "global:contest_list", GLOBAL_TTL_SECONDS):
        try:
            contests = client.get_contest_list()
            upsert_contests(conn, contests)
            mark_refreshed(conn, "global:contest_list")
        except IngestionError as exc:
            logger.warning("Couldn't refresh contest list, keeping cached data: %s", exc)

    if force or is_stale(conn, "global:problemset", GLOBAL_TTL_SECONDS):
        try:
            result = client.get_problemset_problems()
            upsert_problems_and_tags(conn, result["problems"])
            mark_refreshed(conn, "global:problemset")
        except IngestionError as exc:
            logger.warning("Couldn't refresh problemset, keeping cached data: %s", exc)


def sync_user_data(conn: sqlite3.Connection, client: CFClient, handle: str, force: bool = False) -> None:
    """Refreshes a user's data if stale.

    Raises InvalidHandleError if the handle doesn't exist -- the caller
    (CLI layer) is expected to catch this and reprompt for a username.

    On a transient API failure: falls back silently to existing cached
    data if there is any; if there's no cache at all (first-ever sync for
    this handle), re-raises so the caller can show a "couldn't reach
    Codeforces" message rather than an empty dashboard.
    """
    cache_key = f"user:{handle}"
    if not force and not is_stale(conn, cache_key, USER_TTL_SECONDS):
        return

    info = client.get_user_info(handle)  # InvalidHandleError propagates uncaught -- that's intentional

    try:
        rating_changes = client.get_user_rating(handle)
        submissions = client.get_user_status(handle)
    except IngestionError as exc:
        if get_last_refresh(conn, cache_key) is not None:
            logger.warning("Couldn't refresh data for %s, using cached data: %s", handle, exc)
            return
        raise

    upsert_user_info(conn, handle, info.get("rating"))
    upsert_rating_history(conn, handle, rating_changes)

    embedded_problems = _extract_embedded_problems(submissions)
    if embedded_problems:
        upsert_problems_and_tags(conn, embedded_problems)
    _backfill_recent_contest_problems(conn, client, submissions)
    upsert_submissions(conn, handle, submissions)
    mark_refreshed(conn, cache_key)


def sync_all(conn: sqlite3.Connection, client: CFClient, handle: str, force: bool = False) -> None:
    """Convenience entry point: refresh global data, then user data."""
    sync_global_data(conn, client, force=force)
    sync_user_data(conn, client, handle, force=force)
