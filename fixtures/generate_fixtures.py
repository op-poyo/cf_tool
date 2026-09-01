"""Generates a small but deliberately varied synthetic dataset and loads
it through the *real* upsert functions from ingestion.sync -- so this
exercises the actual DB-writing code path, just with hand-built data
instead of live API responses. No network access needed.

Covers, on purpose:
  - a Div 3 contest with very high problem ratings the user never
    touched (must still be suggested by Function 2, division override)
  - a contest with an unrated problem mixed in (must be fully excluded
    from Function 2)
  - a contest whose first 3 problems are too far above the user's
    rating and isn't Div 3/4 (must be excluded from Function 2)
  - a normal eligible never-touched contest (must be suggested)
  - a BEFORE-phase (upcoming) contest (must be excluded everywhere)
  - a mix of first-attempt solves, later-attempt solves, attempted-
    but-unsolved, and never-attempted-but-should've-solved problems
    across multiple tags and elo buckets, for Functions 1/3/4
"""

import sqlite3

from ingestion.sync import (
    upsert_contests,
    upsert_problems_and_tags,
    upsert_user_info,
    upsert_rating_history,
    upsert_submissions,
)

HANDLE = "demo_user"

CONTESTS = [
    {"id": 1, "name": "Codeforces Round 1 (Div. 2)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_000_000, "durationSeconds": 7200},
    {"id": 2, "name": "Codeforces Round 2 (Div. 3)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_100_000, "durationSeconds": 7200},
    {"id": 3, "name": "Codeforces Round 3 (Div. 2)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_200_000, "durationSeconds": 7200},
    {"id": 4, "name": "Codeforces Round 4 (Div. 2)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_300_000, "durationSeconds": 7200},
    {"id": 5, "name": "Codeforces Round 5 (Div. 2)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_400_000, "durationSeconds": 7200},
    {"id": 6, "name": "Codeforces Round 6 (Div. 2)", "phase": "FINISHED",
     "type": "CF", "startTimeSeconds": 1_500_000, "durationSeconds": 7200},
    {"id": 7, "name": "Codeforces Round 7 (Div. 2)", "phase": "BEFORE",
     "type": "CF", "startTimeSeconds": 1_600_000, "durationSeconds": 7200},
]

PROBLEMS = [
    # Contest 1 -- user attempts A (first-try) and B (later-try); C never touched but out of range
    {"contestId": 1, "index": "A", "name": "Sum It", "rating": 800, "tags": ["math", "implementation"]},
    {"contestId": 1, "index": "B", "name": "Balanced", "rating": 1200, "tags": ["dp", "greedy"]},
    {"contestId": 1, "index": "C", "name": "Far Graph", "rating": 1600, "tags": ["math"]},

    # Contest 2 -- Div 3, absurdly high ratings, never touched -> must still be suggested
    {"contestId": 2, "index": "A", "name": "Trivial?", "rating": 3000, "tags": ["graphs"]},
    {"contestId": 2, "index": "B", "name": "Less Trivial", "rating": 3200, "tags": ["graphs"]},
    {"contestId": 2, "index": "C", "name": "Not Trivial", "rating": 3400, "tags": ["strings"]},

    # Contest 3 -- has one unrated problem -> whole contest excluded from Function 2
    {"contestId": 3, "index": "A", "name": "Warmup", "rating": 1000, "tags": ["implementation"]},
    {"contestId": 3, "index": "B", "name": "Subsequence", "rating": 1400, "tags": ["dp"]},
    {"contestId": 3, "index": "C", "name": "Big Jump", "rating": 2600, "tags": ["math"]},
    {"contestId": 3, "index": "D", "name": "Unrated Bonus", "rating": None, "tags": ["constructive"]},

    # Contest 4 -- fully rated, never touched, but first 3 way above rating+600, not Div3/4 -> excluded
    {"contestId": 4, "index": "A", "name": "Hard Graph 1", "rating": 2500, "tags": ["graphs"]},
    {"contestId": 4, "index": "B", "name": "Hard Graph 2", "rating": 2600, "tags": ["graphs"]},
    {"contestId": 4, "index": "C", "name": "Hard Strings", "rating": 2700, "tags": ["strings"]},

    # Contest 5 -- mix: never-attempted-should've-solved (A), attempted-unsolved (B), solved (C)
    {"contestId": 5, "index": "A", "name": "Easy Greedy", "rating": 900, "tags": ["greedy"]},
    {"contestId": 5, "index": "B", "name": "Mid Impl", "rating": 1300, "tags": ["implementation"]},
    {"contestId": 5, "index": "C", "name": "Mid DP", "rating": 1500, "tags": ["dp"]},

    # Contest 6 -- fully rated, never touched, within range, not Div3/4 -> normal suggestion
    {"contestId": 6, "index": "A", "name": "Fresh Graph", "rating": 1000, "tags": ["graphs"]},
    {"contestId": 6, "index": "B", "name": "Fresh String", "rating": 1300, "tags": ["strings"]},
    {"contestId": 6, "index": "C", "name": "Fresh DP", "rating": 1500, "tags": ["dp"]},

    # Contest 7 -- upcoming (BEFORE phase) -- must never appear anywhere
    {"contestId": 7, "index": "A", "name": "Not Yet", "rating": 1000, "tags": ["dp"]},
]

RATING_CHANGES = [
    {"contestId": 1, "oldRating": 1000, "newRating": 1100, "rank": 2000, "ratingUpdateTimeSeconds": 1_000_500},
    {"contestId": 3, "oldRating": 1100, "newRating": 1200, "rank": 1500, "ratingUpdateTimeSeconds": 1_200_500},
    {"contestId": 5, "oldRating": 1200, "newRating": 1300, "rank": 1000, "ratingUpdateTimeSeconds": 1_400_500},
]

SUBMISSIONS = [
    # Contest 1: A solved first try, B solved on 2nd try, C never touched
    {"id": 101, "verdict": "OK", "creationTimeSeconds": 1_000_100, "problem": {"contestId": 1, "index": "A"}},
    {"id": 102, "verdict": "WRONG_ANSWER", "creationTimeSeconds": 1_000_100, "problem": {"contestId": 1, "index": "B"}},
    {"id": 103, "verdict": "OK", "creationTimeSeconds": 1_000_200, "problem": {"contestId": 1, "index": "B"}},

    # Contest 3: A solved first try, B solved on 2nd try, C/D never touched
    {"id": 301, "verdict": "OK", "creationTimeSeconds": 1_200_100, "problem": {"contestId": 3, "index": "A"}},
    {"id": 302, "verdict": "WRONG_ANSWER", "creationTimeSeconds": 1_200_100, "problem": {"contestId": 3, "index": "B"}},
    {"id": 303, "verdict": "OK", "creationTimeSeconds": 1_200_300, "problem": {"contestId": 3, "index": "B"}},

    # Contest 5: A never touched, B attempted-unsolved, C solved first try
    {"id": 501, "verdict": "WRONG_ANSWER", "creationTimeSeconds": 1_400_100, "problem": {"contestId": 5, "index": "B"}},
    {"id": 502, "verdict": "OK", "creationTimeSeconds": 1_400_100, "problem": {"contestId": 5, "index": "C"}},
]

CURRENT_RATING = 1300  # matches the last rating_after in RATING_CHANGES


def load(conn: sqlite3.Connection, handle: str = HANDLE) -> None:
    upsert_contests(conn, CONTESTS)
    upsert_problems_and_tags(conn, PROBLEMS)
    upsert_user_info(conn, handle, CURRENT_RATING)
    upsert_rating_history(conn, handle, RATING_CHANGES)
    upsert_submissions(conn, handle, SUBMISSIONS)
