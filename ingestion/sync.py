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


def _backfill_missing_contest_problems(
    conn: sqlite3.Connection,
    client: CFClient,
    submissions: list[dict],
    progress_callback=None,
) -> None:
    """For every contest the user has ever submitted to, ensure the local
    problems table has that contest's REAL, full problem list -- not just
    whatever _extract_embedded_problems could infer from the user's own
    attempts.

    Earlier version of this only backfilled contests started in the last
    7 days, on the assumption that anything older would already be
    covered by the routine global problemset.problems sync -- that
    assumption turned out to be wrong in practice: problemset.problems
    can apparently omit a contest for longer than a week (possibly
    indefinitely, for some contest types), so time-since-contest isn't a
    reliable signal for "already complete." This checks completeness
    directly instead: has this specific contest_id ever been backfilled
    from an authoritative source (contest.standings, right here)? If
    not, fetch it now and record that permanently via cache_meta, so it
    is NEVER re-fetched for this contest again, no matter how many times
    the user submits to it or how many future syncs happen -- a one-time
    cost per contest, ever, rather than a recurring one. This also no
    longer depends on the contests table already having this contest's
    start_time, which the time-window version silently required.

    CF only accepts contest.standings for a non-gym contest, as a
    non-admin, with contestId and NO other parameters -- so each call
    here downloads the full standings payload (every participant row),
    not just the problems we want. Combined with the rate limiter's
    ~1.75s floor between calls, backfilling a long submission history in
    one go can take real minutes on a first sync -- deliberately
    accepted here rather than capped: an earlier version capped this at
    a fixed number of contests per sync to keep any single sync fast,
    but that meant a large backlog only cleared a little at a time,
    across many manual refreshes. Runs to completion in one pass instead
    now. progress_callback(done, total), if given, lets the caller show
    real progress during that -- essential now, not optional, given how
    long this can run. Still ordered most-recent-contest-first (even
    though nothing is held back anymore) so if a sync is interrupted
    partway, whatever DID get backfilled is what the app's own weakness/
    tag analysis actually looks at first (bounded to the most recent
    MAX_RECENT_CONTESTS), not an arbitrary slice. Regardless, this is
    only ever slow once per contest: everything it touches is
    permanently marked via cache_meta, so it's never re-fetched again --
    after this first catch-up, every future sync is back to the small,
    fast case of just whatever's newly appeared since last time."""
    contest_ids = {
        s.get("problem", {}).get("contestId")
        for s in submissions
        if s.get("problem", {}).get("contestId") is not None
    }
    needs_backfill = [
        cid for cid in contest_ids
        if get_last_refresh(conn, f"contest_problems:{cid}") is None
    ]

    # Most recent submission per contest, as a recency proxy -- cheaper
    # than joining against the contests table, and correct for the same
    # reason: a contest we just submitted to is a contest we care about
    # getting right first.
    last_submitted_at: dict[int, int] = {}
    for s in submissions:
        cid = s.get("problem", {}).get("contestId")
        t = s.get("creationTimeSeconds")
        if cid is None or t is None:
            continue
        if cid not in last_submitted_at or t > last_submitted_at[cid]:
            last_submitted_at[cid] = t

    to_backfill = sorted(needs_backfill, key=lambda cid: last_submitted_at.get(cid, 0), reverse=True)

    total = len(to_backfill)
    for i, contest_id in enumerate(to_backfill, start=1):
        cache_key = f"contest_problems:{contest_id}"
        try:
            problems = client.get_contest_problems(contest_id)
            upsert_problems_and_tags(conn, problems)
            mark_refreshed(conn, cache_key)
        except IngestionError as exc:
            logger.warning(
                "Couldn't backfill full problem list for contest %s: %s", contest_id, exc
            )
        if progress_callback is not None:
            progress_callback(i, total)


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


def sync_user_data(
    conn: sqlite3.Connection,
    client: CFClient,
    handle: str,
    force: bool = False,
    backfill_progress_callback=None,
) -> None:
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
    _backfill_missing_contest_problems(
        conn, client, submissions, progress_callback=backfill_progress_callback
    )
    upsert_submissions(conn, handle, submissions)
    mark_refreshed(conn, cache_key)


def sync_all(
    conn: sqlite3.Connection,
    client: CFClient,
    handle: str,
    force: bool = False,
    backfill_progress_callback=None,
) -> None:
    """Convenience entry point: refresh global data, then user data."""
    sync_global_data(conn, client, force=force)
    sync_user_data(conn, client, handle, force=force, backfill_progress_callback=backfill_progress_callback)
