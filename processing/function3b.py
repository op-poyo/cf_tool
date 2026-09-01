"""Function 3 Chart B: for a toggled set of tags, bucket the user's
problems by elo (100pt buckets, open-ended at both ends), with a
SEPARATE bar per (bucket, tag) pair -- e.g. bucket 800 shows one bar
for 'greedy' and a distinct one for 'constructive' side by side, rather
than merging every selected tag into a single bar per bucket.

Each bar is internally stacked by FOUR categories:
  - first_attempt      : solved, and the very first submission was the solve
  - later_attempt       : solved, but took 2+ submissions (3rd, 4th... all
                           roll into this same bucket, per spec)
  - unsolved             : attempted at least once, never solved
  - should_have_solved   : NEVER attempted, but within Function 1's +400
                            fairness range in one of the last N contests
                            the user actually participated in

The 4th category exists to close a real gap: without it, a genuine
weakness (a problem you should've solved but never even tried) is
invisible in this attempted-only chart -- it only shows up in Function 1
and Function 4. Reuses flag_weak_problems (Function 1's own logic) so
the definition can never drift between the two.

`tags` is a union filter: a problem tagged with 2+ of the selected tags
appears in EACH of those tags' breakdowns (once per tag), not merged.
"""

import pandas as pd

from .derivations import compute_attempt_numbers
from .function1 import flag_weak_problems

BUCKET_WIDTH = 100


def compute_bucket(rating: float, width: int = BUCKET_WIDTH) -> int:
    """Floor of `rating` to the nearest bucket boundary, e.g. 1240 -> 1200."""
    return int(rating // width) * width


def tag_elo_breakdown(
    submissions_df: pd.DataFrame,    # handle, contest_id, problem_index, verdict, creation_time
    problems_df: pd.DataFrame,       # contest_id, problem_index, rating
    problem_tags_df: pd.DataFrame,   # contest_id, problem_index, tag
    handle: str,
    tags: list[str],
    rating_changes: list[dict],
    recent_participated_contest_ids: list[int],
    contest_start_times: dict,
    bucket_width: int = BUCKET_WIDTH,
) -> pd.DataFrame:
    """Returns one row per (bucket, tag, category) with a count column.

    IMPORTANT ordering note (attempted categories): attempt numbers are
    computed on the user's real (deduped) submissions BEFORE joining
    against tag-matching problems. A problem with 2+ selected tags
    produces multiple rows once joined against `problem_tags_df` (one
    per matching tag) -- if attempt numbers were computed *after* that
    join, the same real submission would appear multiple times in a
    groupby and corrupt the attempt count. Computing them first and
    letting the duplication happen afterward keeps each duplicate's
    attempt_number correct, since it's just the same already-correct
    value carried along.
    """
    user_subs = submissions_df[submissions_df.handle == handle]
    attempted_rows = []

    if not user_subs.empty:
        user_subs_numbered = compute_attempt_numbers(user_subs)

        matching_tags = problem_tags_df[problem_tags_df.tag.isin(tags)][
            ["contest_id", "problem_index", "tag"]
        ].drop_duplicates()

        rated_matching = matching_tags.merge(
            problems_df[["contest_id", "problem_index", "rating"]],
            on=["contest_id", "problem_index"],
            how="inner",
        )
        rated_matching = rated_matching[rated_matching["rating"].notna()]

        attempted = user_subs_numbered.merge(rated_matching, on=["contest_id", "problem_index"], how="inner")

        for (contest_id, problem_index, tag), group in attempted.groupby(
            ["contest_id", "problem_index", "tag"]
        ):
            rating = group["rating"].iloc[0]
            bucket = compute_bucket(rating, bucket_width)

            ok_attempts = group.loc[group.verdict == "OK", "attempt_number"]
            if ok_attempts.empty:
                category = "unsolved"
            elif ok_attempts.min() == 1:
                category = "first_attempt"
            else:
                category = "later_attempt"

            attempted_rows.append({"bucket": bucket, "tag": tag, "category": category})

    # -- should_have_solved: never attempted, but Function 1 flags it -----
    should_have_rows = []
    flagged = flag_weak_problems(
        recent_participated_contest_ids,
        problems_df,
        problem_tags_df,
        user_subs,
        rating_changes,
        contest_start_times,
    )
    if not flagged.empty:
        flagged = flagged[flagged.tag.isin(tags)]
    if not flagged.empty:
        ever_attempted_keys = set(zip(user_subs.contest_id, user_subs.problem_index))
        flagged_keys_series = pd.Series(
            list(zip(flagged.contest_id, flagged.problem_index)), index=flagged.index
        )
        flagged = flagged[~flagged_keys_series.isin(ever_attempted_keys)]
        flagged = flagged.drop_duplicates(subset=["contest_id", "problem_index", "tag"])
        for row in flagged.itertuples():
            bucket = compute_bucket(row.problem_rating, bucket_width)
            should_have_rows.append({"bucket": bucket, "tag": row.tag, "category": "should_have_solved"})

    all_rows = attempted_rows + should_have_rows
    if not all_rows:
        return pd.DataFrame(columns=["bucket", "tag", "category", "count"])

    breakdown = pd.DataFrame(all_rows)
    counts = (
        breakdown.groupby(["bucket", "tag", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(["bucket", "tag"])
        .reset_index(drop=True)
    )
    return counts
