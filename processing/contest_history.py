"""Per-contest history: for a list of contests (typically the same
"recent participated" list used elsewhere), summarize how many problems
existed, how many the user solved DURING the contest's live window, and
how many they've solved for that contest OVERALL (including upsolves
made long after the contest ended).
"""

import pandas as pd


def contest_history(
    contests_df: pd.DataFrame,     # id, name, start_time, duration, phase
    problems_df: pd.DataFrame,     # contest_id, problem_index, rating
    submissions_df: pd.DataFrame,  # handle, contest_id, problem_index, verdict, creation_time
    handle: str,
    contest_ids: list[int],
) -> pd.DataFrame:
    """One row per contest in `contest_ids`, in the order given (typically
    most-recent-first). Columns: contest_id, name, date, num_problems,
    solved_during_contest, solved_overall, url."""
    handle_subs = submissions_df[submissions_df.handle == handle]
    contests_by_id = contests_df.set_index("id")

    rows = []
    for contest_id in contest_ids:
        if contest_id not in contests_by_id.index:
            continue
        contest = contests_by_id.loc[contest_id]
        start_time = contest["start_time"]
        duration = contest["duration"]

        num_problems = len(problems_df[problems_df.contest_id == contest_id])

        contest_subs = handle_subs[handle_subs.contest_id == contest_id]
        solved_subs = contest_subs[contest_subs.verdict == "OK"]
        solved_overall = solved_subs["problem_index"].nunique()

        if pd.notna(start_time) and pd.notna(duration):
            during = solved_subs[
                (solved_subs.creation_time >= start_time)
                & (solved_subs.creation_time <= start_time + duration)
            ]
            solved_during = during["problem_index"].nunique()
        else:
            solved_during = 0

        rows.append(
            {
                "contest_id": contest_id,
                "name": contest["name"],
                "date": (
                    pd.to_datetime(start_time, unit="s").strftime("%d %b %Y")
                    if pd.notna(start_time)
                    else "Unknown"
                ),
                "num_problems": num_problems,
                "solved_during_contest": solved_during,
                "solved_overall": solved_overall,
                "url": f"https://codeforces.com/contest/{contest_id}",
            }
        )

    return pd.DataFrame(rows)
