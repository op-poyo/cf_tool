"""Function 2: suggest past contests worth doing as a virtual contest --
fully rated, never attempted, and either Div 3/4 (always suggested) or
with their first three problems (A-C) within +600 of the user's CURRENT
rating. Deliberately uses current rating, not their rating at the time
of the contest -- this is about whether the contest is a good fit for
the user to attempt RIGHT NOW, not a historical fairness judgment (that
distinction is what Function 1 is for).
"""

import re
import pandas as pd

RATING_MARGIN = 600
FIRST_N_PROBLEMS = 3

_DIV_3_4_PATTERN = re.compile(r"div(?:ision)?\.?\s*[34]\b", re.IGNORECASE)


def is_div_3_or_4(contest_name: str) -> bool:
    """CF's API doesn't expose a division field, so this is a name match --
    the naming convention ("Div. 3", "Division 4", etc.) is stable enough
    to rely on for this purpose."""
    return bool(_DIV_3_4_PATTERN.search(contest_name or ""))


def _is_fully_rated(contest_problems: pd.DataFrame) -> bool:
    return len(contest_problems) > 0 and contest_problems["rating"].notna().all()


def _meets_rating_check(contest_problems: pd.DataFrame, user_rating: int) -> bool:
    first_n = contest_problems.sort_values("problem_index").head(FIRST_N_PROBLEMS)
    if len(first_n) < FIRST_N_PROBLEMS:
        return False  # contest doesn't even have 3 problems -- skip rather than guess
    return bool((first_n["rating"] <= user_rating + RATING_MARGIN).all())


def suggest_virtual_contests(
    contests_df: pd.DataFrame,     # id, name, start_time, phase
    problems_df: pd.DataFrame,     # contest_id, problem_index, rating
    submissions_df: pd.DataFrame,  # handle, contest_id, problem_index (all submissions, any user)
    handle: str,
    current_rating: int,
) -> pd.DataFrame:
    finished = contests_df[contests_df.phase == "FINISHED"]

    attempted_contest_ids = set(
        submissions_df.loc[submissions_df.handle == handle, "contest_id"].dropna().unique()
    )

    suggestions = []
    for _, contest in finished.iterrows():
        contest_id = contest["id"]
        if contest_id in attempted_contest_ids:
            continue

        contest_problems = problems_df[problems_df.contest_id == contest_id]
        if not _is_fully_rated(contest_problems):
            continue

        if is_div_3_or_4(contest["name"]):
            eligible = True
        else:
            eligible = _meets_rating_check(contest_problems, current_rating)

        if eligible:
            suggestions.append(
                {
                    "contest_id": contest_id,
                    "name": contest["name"],
                    "start_time": contest["start_time"],
                    "num_problems": len(contest_problems),
                }
            )

    result = pd.DataFrame(suggestions)
    if result.empty:
        return result
    return result.sort_values("start_time", ascending=False).reset_index(drop=True)
