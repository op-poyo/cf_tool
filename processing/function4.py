"""Function 4: one stacked bar per tag.

  green = every solved problem with that tag (all-time, any contest),
          weighted by solve_weight -- solving ABOVE your level counts
          for more than solving well below it.

  red   = two components, both weighted the same way:
    (a) attempted at least once (anywhere -- including gym/practice),
        never solved. No recency or contest-participation restriction --
        an upsolve attempt you never finished still counts.
    (b) not solved, in one of the last N contests the user actually
        participated in, and within Function 1's +400 fairness range.
        This reuses flag_weak_problems but scoped to participated
        contests only -- NOT every finished contest CF has ever run,
        which would sweep in problems from contests the user never
        even entered. This scoping must match whatever the caller uses
        for Function 1's own "recent contests" window.

Red uses fail_weight, the MIRROR of solve_weight: failing something well
BELOW your level counts for more than failing something well above it --
missing an easy problem is a stronger weak-spot signal than missing a
hard one, which is somewhat expected. Green and red are intentionally
weighted in opposite directions.

Green has no upper eligibility gate (any solve counts). Part (a) of red
has no eligibility gate either -- the user already engaged with the
problem, so no fairness judgment is needed. Part (b) keeps Function 1's
+400 gate, since that's specifically about "should you have been
expected to solve this."

Weighting uses the user's CURRENT rating (a present-day strength
snapshot), whereas Function 1's eligibility gate uses the user's rating
AT THE TIME of each contest (fairness about what they could have been
expected to solve then). Two different rating references for two
different purposes.

Two public functions:
  strong_weak_tag_ranking -- weighted (solve_weight / fail_weight)
  strong_weak_tag_counts  -- raw, unweighted problem counts per tag
Both share the same problem-identification logic via _identify_components
so "what counts as solved/failed" can never drift between the two views.
"""

import pandas as pd

from .derivations import solve_weight, fail_weight
from .function1 import flag_weak_problems


def _identify_components(
    problems_df: pd.DataFrame,
    problem_tags_df: pd.DataFrame,
    handle_submissions_df: pd.DataFrame,
    rating_changes: list[dict],
    recent_participated_contest_ids: list[int],
    contest_start_times: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (solved_tagged, red_tagged) -- both with contest_id,
    problem_index, tag, rating columns, no weighting or aggregation
    applied yet. red_tagged already combines and dedupes parts (a) and
    (b), so callers never have to worry about double counting."""
    rated_problems = problems_df[problems_df["rating"].notna()]

    solved = (
        handle_submissions_df[handle_submissions_df.verdict == "OK"]
        [["contest_id", "problem_index"]]
        .drop_duplicates()
        .merge(rated_problems, on=["contest_id", "problem_index"], how="inner")
    )
    solved_tagged = solved.merge(problem_tags_df, on=["contest_id", "problem_index"], how="left")

    # -- red, part (a): attempted anywhere (gym included), never solved ---
    attempted = (
        handle_submissions_df[["contest_id", "problem_index"]]
        .drop_duplicates()
        .merge(rated_problems, on=["contest_id", "problem_index"], how="inner")
    )
    solved_keys = set(zip(solved.contest_id, solved.problem_index))
    attempted_keys_series = pd.Series(
        list(zip(attempted.contest_id, attempted.problem_index)), index=attempted.index
    )
    attempted_unsolved = attempted[~attempted_keys_series.isin(solved_keys)]
    au_tagged = attempted_unsolved.merge(problem_tags_df, on=["contest_id", "problem_index"], how="left")

    # -- red, part (b): not solved in a recent PARTICIPATED contest, ------
    #    within Function 1's +400 fairness range
    flagged = flag_weak_problems(
        recent_participated_contest_ids,
        problems_df,
        problem_tags_df,
        handle_submissions_df,
        rating_changes,
        contest_start_times,
    )
    if not flagged.empty:
        attempted_keys = set(zip(attempted.contest_id, attempted.problem_index))
        flagged_keys_series = pd.Series(
            list(zip(flagged.contest_id, flagged.problem_index)), index=flagged.index
        )
        flagged = flagged[~flagged_keys_series.isin(attempted_keys)]
        flagged = flagged.rename(columns={"problem_rating": "rating"}).drop_duplicates(
            subset=["contest_id", "problem_index", "tag"]
        )

    red_tagged = pd.concat(
        [
            au_tagged[["contest_id", "problem_index", "rating", "tag"]],
            flagged[["contest_id", "problem_index", "rating", "tag"]] if not flagged.empty else pd.DataFrame(
                columns=["contest_id", "problem_index", "rating", "tag"]
            ),
        ],
        ignore_index=True,
    )

    return solved_tagged, red_tagged


def strong_weak_tag_ranking(
    problems_df: pd.DataFrame,      # contest_id, problem_index, rating
    problem_tags_df: pd.DataFrame,  # contest_id, problem_index, tag
    handle_submissions_df: pd.DataFrame,  # ALREADY filtered to the target handle
    rating_changes: list[dict],
    recent_participated_contest_ids: list[int],
    contest_start_times: dict,
    current_rating: int,
) -> pd.DataFrame:
    """Weighted view: one row per tag, green_weight / red_weight.

    `recent_participated_contest_ids` should be the SAME list the caller
    passes to Function 1 -- the last N contests the user actually
    participated in.
    """
    solved_tagged, red_tagged = _identify_components(
        problems_df, problem_tags_df, handle_submissions_df, rating_changes,
        recent_participated_contest_ids, contest_start_times,
    )

    solved_tagged = solved_tagged.copy()
    solved_tagged["weight"] = solved_tagged["rating"].apply(lambda r: solve_weight(r, current_rating))
    green_by_tag = solved_tagged.groupby("tag")["weight"].sum()

    red_tagged = red_tagged.copy()
    red_tagged["weight"] = red_tagged["rating"].apply(lambda r: fail_weight(r, current_rating))
    red_by_tag = red_tagged.groupby("tag")["weight"].sum()

    all_tags = sorted(problem_tags_df["tag"].dropna().unique())
    return pd.DataFrame(
        {
            "tag": all_tags,
            "green_weight": [round(green_by_tag.get(t, 0.0), 3) for t in all_tags],
            "red_weight": [round(red_by_tag.get(t, 0.0), 3) for t in all_tags],
        }
    )


def strong_weak_tag_counts(
    problems_df: pd.DataFrame,
    problem_tags_df: pd.DataFrame,
    handle_submissions_df: pd.DataFrame,
    rating_changes: list[dict],
    recent_participated_contest_ids: list[int],
    contest_start_times: dict,
) -> pd.DataFrame:
    """Raw, unweighted view: one row per tag, green_count / red_count --
    plain problem counts, same solved/failed definitions as the weighted
    version above, just without the elo-based weighting applied."""
    solved_tagged, red_tagged = _identify_components(
        problems_df, problem_tags_df, handle_submissions_df, rating_changes,
        recent_participated_contest_ids, contest_start_times,
    )

    green_by_tag = solved_tagged.groupby("tag").size()
    red_by_tag = red_tagged.groupby("tag").size()

    all_tags = sorted(problem_tags_df["tag"].dropna().unique())
    return pd.DataFrame(
        {
            "tag": all_tags,
            "green_count": [int(green_by_tag.get(t, 0)) for t in all_tags],
            "red_count": [int(red_by_tag.get(t, 0)) for t in all_tags],
        }
    )
