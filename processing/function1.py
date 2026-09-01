"""Function 1: for a user's recent contests, find problems they *should*
plausibly have solved (rating <= their rating at that contest + 400) but
didn't -- and were never solved later either (upsolves don't count as a
weakness). Aggregate tags on those problems to surface weak topics.
"""

import pandas as pd

from .derivations import resolve_rating_at_contest, DEFAULT_RATING

ELIGIBILITY_MARGIN = 400  # problems above rating+this are not "should've solved"


def flag_weak_problems(
    recent_contest_ids: list[int],
    problems_df: pd.DataFrame,       # contest_id, problem_index, rating
    problem_tags_df: pd.DataFrame,   # contest_id, problem_index, tag
    submissions_df: pd.DataFrame,    # handle, contest_id, problem_index, verdict, creation_time
    rating_changes: list[dict],
    contest_start_times: dict[int, int],  # contest_id -> start_time
) -> pd.DataFrame:
    """Returns a DataFrame of flagged (contest_id, problem_index) rows the
    user should have solved but never did (ever, not just during the
    contest) -- with their tags attached for aggregation.
    """
    solved_ever = set(
        submissions_df.loc[submissions_df.verdict == "OK", ["contest_id", "problem_index"]]
        .itertuples(index=False, name=None)
    )

    flagged_frames = []
    for contest_id in recent_contest_ids:
        start_time = contest_start_times.get(contest_id)
        if start_time is None:
            continue  # can't resolve rating without contest timing info
        user_rating = resolve_rating_at_contest(rating_changes, contest_id, start_time, DEFAULT_RATING)
        threshold = user_rating + ELIGIBILITY_MARGIN

        contest_problems = problems_df[
            (problems_df.contest_id == contest_id)
            & problems_df.rating.notna()
            & (problems_df.rating <= threshold)
        ]
        if contest_problems.empty:
            continue

        keys = pd.Series(
            list(zip(contest_problems.contest_id, contest_problems.problem_index)),
            index=contest_problems.index,
        )
        unsolved = contest_problems[~keys.isin(solved_ever)]
        if unsolved.empty:
            continue

        unsolved = unsolved.assign(user_rating_at_contest=user_rating).rename(
            columns={"rating": "problem_rating"}
        )
        flagged_frames.append(
            unsolved[["contest_id", "problem_index", "problem_rating", "user_rating_at_contest"]]
        )

    if not flagged_frames:
        return pd.DataFrame(
            columns=["contest_id", "problem_index", "problem_rating", "user_rating_at_contest", "tag"]
        )

    flagged_df = pd.concat(flagged_frames, ignore_index=True)
    return flagged_df.merge(problem_tags_df, on=["contest_id", "problem_index"], how="left")


def weak_topics_summary(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates flagged problems by tag, most-frequent first."""
    if flagged_df.empty:
        return pd.DataFrame(columns=["tag", "count"])
    return (
        flagged_df.groupby("tag")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
