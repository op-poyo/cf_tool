"""Minimal driver for a full pipeline test run: loads the synthetic
fixture data into a fresh SQLite DB, builds the DataFrames each function
expects, runs all four functions, and prints the results.

This is a sanity-check tool, not the real CLI/dashboard -- it exists so
we can validate the pipeline's output shape before building presentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from db.database import init_db, get_connection
from fixtures.generate_fixtures import load as load_fixtures, HANDLE, CURRENT_RATING
from processing.derivations import participated_contest_ids
from processing.function1 import flag_weak_problems, weak_topics_summary
from processing.function2 import suggest_virtual_contests
from processing.function3a import solved_count_by_tag
from processing.function3b import tag_elo_breakdown
from processing.function4 import strong_weak_tag_ranking, strong_weak_tag_counts
from processing.recommendations import recommended_problems, problemset_browse_url
from processing.contest_history import contest_history

DB_PATH = Path(__file__).parent.parent / "test_run.db"


def load_dataframes(conn):
    contests_df = pd.read_sql_query("SELECT * FROM contests", conn)
    problems_df = pd.read_sql_query("SELECT * FROM problems", conn)
    tags_df = pd.read_sql_query("SELECT * FROM problem_tags", conn)
    submissions_df = pd.read_sql_query("SELECT * FROM submissions", conn)
    rating_hist_df = pd.read_sql_query(
        "SELECT * FROM user_rating_history WHERE handle = ?", conn, params=(HANDLE,)
    )
    return contests_df, problems_df, tags_df, submissions_df, rating_hist_df


def to_raw_rating_changes(rating_hist_df: pd.DataFrame) -> list[dict]:
    """Reconstruct the raw user.rating-shaped dicts our processing
    functions expect, from the DB's column names."""
    return [
        {
            "contestId": row.contest_id,
            "oldRating": row.rating_before,
            "newRating": row.rating_after,
            "ratingUpdateTimeSeconds": row.rating_update_time,
        }
        for row in rating_hist_df.itertuples()
    ]


def main():
    DB_PATH.unlink(missing_ok=True)
    init_db(DB_PATH)

    with get_connection(DB_PATH) as conn:
        load_fixtures(conn, HANDLE)

    with get_connection(DB_PATH) as conn:
        contests_df, problems_df, tags_df, submissions_df, rating_hist_df = load_dataframes(conn)

    rating_changes = to_raw_rating_changes(rating_hist_df)
    handle_subs = submissions_df[submissions_df.handle == HANDLE]
    finished_contests = contests_df[contests_df.phase == "FINISHED"]
    contest_start_times = dict(zip(contests_df.id, contests_df.start_time))

    # Uses the FIXED participation definition (time-window based, excludes
    # later upsolves) instead of "any submission ever for this contest_id".
    participated_ids = participated_contest_ids(submissions_df, HANDLE, contests_df)
    participated_sorted = (
        finished_contests[finished_contests.id.isin(participated_ids)]
        .sort_values("start_time", ascending=False)
        .id.tolist()
    )
    total_participated = len(participated_sorted)
    MAX_RECENT_CONTESTS = 20
    recent_participated_ids = participated_sorted[:MAX_RECENT_CONTESTS]

    print("=" * 60)
    print(f"TOTAL LIFETIME CONTESTS PARTICIPATED IN: {total_participated}")
    print("=" * 60)
    print(f"Recent (weakness-scan window, up to {MAX_RECENT_CONTESTS}): {recent_participated_ids}")

    print()
    print("=" * 60)
    print("FUNCTION 1 -- weakness flagging (recent PARTICIPATED contests)")
    print("=" * 60)
    flagged = flag_weak_problems(
        recent_participated_ids, problems_df, tags_df, handle_subs,
        rating_changes, contest_start_times,
    )
    print(flagged[["contest_id", "problem_index", "problem_rating", "tag"]] if not flagged.empty else "(none flagged)")
    print("\nWeak topics summary:")
    print(weak_topics_summary(flagged))

    print()
    print("=" * 60)
    print("FUNCTION 2 -- virtual contest suggestions (current-rating-based)")
    print("=" * 60)
    suggestions = suggest_virtual_contests(contests_df, problems_df, submissions_df, HANDLE, CURRENT_RATING)
    print(suggestions if not suggestions.empty else "(none suggested)")

    print()
    print("=" * 60)
    print("CONTEST HISTORY -- ALL-TIME participated contests (not capped)")
    print("=" * 60)
    print(contest_history(contests_df, problems_df, submissions_df, HANDLE, participated_sorted))

    print()
    print("=" * 60)
    print("FUNCTION 3A -- overall solved count per tag")
    print("=" * 60)
    print(solved_count_by_tag(submissions_df, tags_df, HANDLE))

    print()
    print("=" * 60)
    print("FUNCTION 3B -- elo-bucketed breakdown, per-tag bars for ['dp', 'implementation', 'greedy']")
    print("=" * 60)
    print(tag_elo_breakdown(
        submissions_df, problems_df, tags_df, HANDLE, ["dp", "implementation", "greedy"],
        rating_changes, recent_participated_ids, contest_start_times,
    ))

    print()
    print("=" * 60)
    print(f"FUNCTION 4 -- strong/weak tag ranking (current rating={CURRENT_RATING})")
    print("=" * 60)
    print(strong_weak_tag_ranking(
        problems_df, tags_df, handle_subs, rating_changes,
        recent_participated_ids, contest_start_times, CURRENT_RATING,
    ))

    print()
    print("=" * 60)
    print("FUNCTION 4 (raw) -- strong/weak tag counts, unweighted")
    print("=" * 60)
    print(strong_weak_tag_counts(
        problems_df, tags_df, handle_subs, rating_changes,
        recent_participated_ids, contest_start_times,
    ))

    print()
    print("=" * 60)
    print("RECOMMENDATIONS -- everything (no tag filter)")
    print("=" * 60)
    print(recommended_problems(
        problems_df, tags_df, handle_subs, rating_changes,
        recent_participated_ids, contest_start_times, tags=[],
    ))

    print()
    print("=" * 60)
    print("RECOMMENDATIONS -- filtered to tag='greedy' (should surface a should've-done-in-contest row)")
    print("=" * 60)
    print(recommended_problems(
        problems_df, tags_df, handle_subs, rating_changes,
        recent_participated_ids, contest_start_times, tags=["greedy"],
    ))
    print(problemset_browse_url(["greedy"], CURRENT_RATING, CURRENT_RATING + 250))
    print(problemset_browse_url([], CURRENT_RATING, CURRENT_RATING + 250))


if __name__ == "__main__":
    main()
