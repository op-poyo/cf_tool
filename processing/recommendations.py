"""Tag-based recommendations, split into two independent pieces:

  1. recommended_problems -- combines two categories of "failed" problems
     into one list, each tagged with which category it fell into via a
     `status` column:
       - "attempted + failed": attempted at least once anywhere
         (including gym/practice), never solved. No recency restriction.
       - "should've done in contest": NEVER attempted, but within
         Function 1's +400 fairness range in one of the last N contests
         the user actually participated in. Reuses flag_weak_problems so
         the definition can never drift from Function 1's own.
     Sorted lowest-to-highest rating. Accepts a LIST of tags: empty/None
     means no tag filter (show everything); one or more tags means
     UNION -- a problem shows up if it has ANY of the selected tags.

     NOTE: this union behavior is deliberately the opposite of how
     Codeforces' own problemset filter works (see problemset_browse_url
     below), which is an INTERSECTION -- a problem there only matches if
     it has ALL the selected tags. Two different tools, two different
     semantics -- worth remembering when comparing what each one shows.

  2. problemset_browse_url -- builds a deep link straight to Codeforces'
     own problemset page, pre-filtered by the selected tag(s) and a
     rating range the user picks in our UI. We don't try to recommend
     specific "similar to your elo" problems ourselves -- CF's own
     filter already does this well, has full up-to-date coverage
     (including problems we haven't ingested), and avoids every open
     question (gym coverage, sort order, how many to show) that came up
     when we considered building it locally.
"""

from urllib.parse import quote

import pandas as pd

from .function1 import flag_weak_problems

STATUS_ATTEMPTED_FAILED = "attempted + failed"
STATUS_SHOULD_HAVE_DONE = "not done in contest"


def recommended_problems(
    problems_df: pd.DataFrame,       # contest_id, problem_index, name, rating
    problem_tags_df: pd.DataFrame,   # contest_id, problem_index, tag
    handle_submissions_df: pd.DataFrame,  # ALREADY filtered to the target handle
    rating_changes: list[dict],
    recent_participated_contest_ids: list[int],
    contest_start_times: dict,
    tags: list[str] | None = None,
) -> pd.DataFrame:
    """Every "failed" problem for the user -- attempted-and-unsolved
    (anywhere) plus never-attempted-but-should've-solved (recent
    participated contests, +400 range) -- optionally filtered to
    problems having ANY of `tags` (union). Sorted lowest rating first.
    Includes a readable 'tags' column, a 'status' column distinguishing
    the two categories, a CF-style 'problem_id' (e.g. '1987A'), and a
    direct problem URL."""
    rated_problems = problems_df[problems_df["rating"].notna()]

    attempted = handle_submissions_df[["contest_id", "problem_index"]].drop_duplicates()
    attempted_keys = set(zip(attempted.contest_id, attempted.problem_index))
    solved = handle_submissions_df.loc[
        handle_submissions_df.verdict == "OK", ["contest_id", "problem_index"]
    ].drop_duplicates()
    solved_keys = set(zip(solved.contest_id, solved.problem_index))

    # -- attempted + failed ------------------------------------------------
    attempted_rated = attempted.merge(rated_problems, on=["contest_id", "problem_index"], how="inner")
    attempted_keys_series = pd.Series(
        list(zip(attempted_rated.contest_id, attempted_rated.problem_index)), index=attempted_rated.index
    )
    attempted_failed = attempted_rated[~attempted_keys_series.isin(solved_keys)].copy()
    attempted_failed["status"] = STATUS_ATTEMPTED_FAILED

    # -- should've done in contest ------------------------------------------
    flagged = flag_weak_problems(
        recent_participated_contest_ids,
        problems_df,
        problem_tags_df,
        handle_submissions_df,
        rating_changes,
        contest_start_times,
    )
    if not flagged.empty:
        flagged_keys_series = pd.Series(
            list(zip(flagged.contest_id, flagged.problem_index)), index=flagged.index
        )
        flagged = flagged[~flagged_keys_series.isin(attempted_keys)]
        flagged = flagged.drop_duplicates(subset=["contest_id", "problem_index"]).rename(
            columns={"problem_rating": "rating"}
        )
        flagged = flagged.copy()
        flagged["status"] = STATUS_SHOULD_HAVE_DONE

    combined = pd.concat(
        [
            attempted_failed[["contest_id", "problem_index", "rating", "status"]],
            flagged[["contest_id", "problem_index", "rating", "status"]] if not flagged.empty else pd.DataFrame(
                columns=["contest_id", "problem_index", "rating", "status"]
            ),
        ],
        ignore_index=True,
    )

    if tags:
        matching = problem_tags_df[problem_tags_df.tag.isin(tags)]
        matching_keys = set(zip(matching.contest_id, matching.problem_index))
        combined_keys_series = pd.Series(
            list(zip(combined.contest_id, combined.problem_index)), index=combined.index
        )
        combined = combined[combined_keys_series.isin(matching_keys)]

    if combined.empty:
        return pd.DataFrame(
            columns=["problem_id", "contest_id", "problem_index", "name", "rating", "tags", "status", "url"]
        )

    combined = combined.merge(
        problems_df[["contest_id", "problem_index", "name"]], on=["contest_id", "problem_index"], how="left"
    )
    tag_lists = (
        problem_tags_df.groupby(["contest_id", "problem_index"])["tag"]
        .apply(lambda s: ", ".join(sorted(s)))
        .reset_index(name="tags")
    )
    combined = combined.merge(tag_lists, on=["contest_id", "problem_index"], how="left")
    combined["problem_id"] = combined["contest_id"].astype(int).astype(str) + combined["problem_index"]
    combined["url"] = (
        "https://codeforces.com/problemset/problem/"
        + combined["contest_id"].astype(int).astype(str)
        + "/"
        + combined["problem_index"]
    )

    return (
        combined[["problem_id", "contest_id", "problem_index", "name", "rating", "tags", "status", "url"]]
        .sort_values("rating")
        .reset_index(drop=True)
    )


def problemset_browse_url(tags: list[str], rating_min: int, rating_max: int) -> str:
    """Deep link to CF's own problemset page, filtered by tags + a rating
    range. CF encodes the rating range as one more comma-separated entry
    in the same `tags` param, e.g. tags=dp,greedy,1200-1600. An empty
    `tags` list is valid -- the link filters by rating only. NOTE: CF's
    own filter is an INTERSECTION across tags (a problem must have ALL
    of them), unlike recommended_problems above which is a union."""
    parts = [quote(t, safe="") for t in tags]
    parts.append(f"{int(rating_min)}-{int(rating_max)}")
    return "https://codeforces.com/problemset?tags=" + ",".join(parts)
