"""Function 3: tag analytics.

Chart A here: overall solved-problem count per tag. Chart B (elo-bucketed
stacked bars) is a separate, more involved piece -- see function3b.py.
"""

import pandas as pd


def solved_count_by_tag(
    submissions_df: pd.DataFrame,   # handle, contest_id, problem_index, verdict
    problem_tags_df: pd.DataFrame,  # contest_id, problem_index, tag
    handle: str,
) -> pd.DataFrame:
    """One row per tag: how many distinct problems with that tag the user
    has solved. A problem with multiple tags contributes to each of them."""
    solved = (
        submissions_df[(submissions_df.handle == handle) & (submissions_df.verdict == "OK")]
        [["contest_id", "problem_index"]]
        .drop_duplicates()
    )

    if solved.empty:
        return pd.DataFrame(columns=["tag", "count"])

    tagged = solved.merge(problem_tags_df, on=["contest_id", "problem_index"], how="left")

    return (
        tagged.groupby("tag")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
