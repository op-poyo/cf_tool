"""Derivations shared across all four analytics functions.

Keeping these here (rating resolution, attempt numbering, the weighting
curve) means every function uses the exact same definitions instead of
each reimplementing its own notion of "attempt" or "rating at the time".
"""

import math
import pandas as pd

DEFAULT_RATING = 1000       # fallback for users with no rating history yet
SIGMOID_W_MIN = 0.2
SIGMOID_W_MAX = 2.0
SIGMOID_K = 150              # controls how fast weight ramps with rating diff


def resolve_rating_at_contest(
    rating_changes: list[dict],
    contest_id: int,
    contest_start_time: int,
    default: int = DEFAULT_RATING,
) -> int:
    """Rating the user had going *into* `contest_id`, which started at
    `contest_start_time`.

    `rating_changes` is the raw list from user.rating: dicts with
    'contestId', 'oldRating', 'newRating', 'ratingUpdateTimeSeconds'.
    Note ratingUpdateTimeSeconds is when the update was *applied* (usually
    shortly after the contest ends), not the contest's start time -- so it
    is only used for ordering, never for matching a specific contest.

    Logic:
      - If `contest_id` itself is in the user's rated history, use its
        oldRating (their rating before that contest's update was applied).
      - Otherwise, use the newRating from the most recent rating change
        that happened before this contest started (their rating carries
        forward through unrated contests).
      - If neither exists (this is before their first rated contest, or
        they have no rating history at all), fall back to `default`.
    """
    exact = [rc for rc in rating_changes if rc.get("contestId") == contest_id]
    if exact:
        return exact[0]["oldRating"]

    prior = [rc for rc in rating_changes if rc["ratingUpdateTimeSeconds"] < contest_start_time]
    if prior:
        most_recent = max(prior, key=lambda rc: rc["ratingUpdateTimeSeconds"])
        return most_recent["newRating"]

    return default


def compute_attempt_numbers(submissions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 1-indexed 'attempt_number' column per (handle, contest_id,
    problem_index), ordered by submission time. Every submission counts
    as an attempt -- no filtering by verdict."""
    df = submissions_df.sort_values("creation_time").copy()
    df["attempt_number"] = (
        df.groupby(["handle", "contest_id", "problem_index"]).cumcount() + 1
    )
    return df


def participated_contest_ids(
    submissions_df: pd.DataFrame, handle: str, contests_df: pd.DataFrame
) -> set:
    """Contests the user actually PARTICIPATED in -- has a submission
    whose creation_time falls within the contest's live window
    [start_time, start_time + duration].

    Deliberately NOT "any submission ever with this contest_id" -- that
    would wrongly count a much-later upsolve (practicing an old problem
    long after the contest ended) as participation, which could displace
    genuine recent participations out of a "last N contests" window.
    Uses only data already ingested (contest timing + submission
    timestamps) -- no extra API calls needed."""
    handle_subs = submissions_df[submissions_df.handle == handle]
    if handle_subs.empty:
        return set()

    merged = handle_subs.merge(
        contests_df[["id", "start_time", "duration"]].rename(columns={"id": "contest_id"}),
        on="contest_id",
        how="inner",
    )
    merged = merged[merged["start_time"].notna() & merged["duration"].notna()]
    within_window = merged[
        (merged["creation_time"] >= merged["start_time"])
        & (merged["creation_time"] <= merged["start_time"] + merged["duration"])
    ]
    return set(within_window["contest_id"].unique())


def solve_weight(
    problem_rating: int,
    user_rating: int,
    w_min: float = SIGMOID_W_MIN,
    w_max: float = SIGMOID_W_MAX,
    k: float = SIGMOID_K,
) -> float:
    """Sigmoid weight for a SOLVED problem (Function 4's green side).
    Problems well below the user's rating approach w_min; problems well
    above approach w_max; the user's own rating sits at the midpoint.
    Solving something above your level is more impressive than solving
    something well below it."""
    diff = problem_rating - user_rating
    sigmoid = 1 / (1 + math.exp(-diff / k))
    return w_min + (w_max - w_min) * sigmoid


def fail_weight(
    problem_rating: int,
    user_rating: int,
    w_min: float = SIGMOID_W_MIN,
    w_max: float = SIGMOID_W_MAX,
    k: float = SIGMOID_K,
) -> float:
    """Sigmoid weight for a FAILED problem (Function 4's red side) --
    the mirror image of solve_weight. Failing something well BELOW the
    user's rating approaches w_max (a strong weak-spot signal); failing
    something well above approaches w_min (unsurprising, barely counts).
    Same curve shape and parameters as solve_weight, just flipped."""
    diff = problem_rating - user_rating
    sigmoid = 1 / (1 + math.exp(diff / k))  # sign flipped vs solve_weight
    return w_min + (w_max - w_min) * sigmoid
