"""Streamlit dashboard. Launched via `streamlit run dashboard/app.py`,
normally through the CLI wrapper in main.py which passes the handle
along as a trailing arg. Can also be run directly and the handle typed
into the text box -- both paths converge on the same load/refresh logic.

CACHING NOTE: Streamlit reruns this entire script on every widget
interaction. Without caching, that means re-querying SQLite and
recomputing every function from scratch on every slider nudge or
checkbox click -- even ones that have nothing to do with what changed.
The pattern used throughout: `_load_dataframes_cached` is keyed on
(handle, sync_marker) where sync_marker is the DB's own
`last_synced_at` for that handle -- so it only actually re-hits SQLite
when a real sync happens, never on a UI interaction. Every other cached
function below re-derives its inputs from that same cached loader
(cheap, itself a cache hit) rather than taking large DataFrames
directly as cache-key parameters -- this keeps each cache key to small
hashable scalars/tuples instead of forcing Streamlit to hash the full
problemset on every call.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db.database import init_db, get_connection
from ingestion.client import CFClient, InvalidHandleError
from ingestion.rate_limiter import IngestionError
from ingestion.sync import sync_all
from processing.derivations import SIGMOID_W_MIN, SIGMOID_W_MAX, SIGMOID_K, participated_contest_ids
from processing.function1 import flag_weak_problems, weak_topics_summary
from processing.function2 import suggest_virtual_contests
from processing.function3a import solved_count_by_tag
from processing.function3b import tag_elo_breakdown
from processing.function4 import strong_weak_tag_ranking, strong_weak_tag_counts
from processing.recommendations import recommended_problems, problemset_browse_url
from processing.contest_history import contest_history

st.set_page_config(page_title="CF Analytics", layout="wide")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handle", default=None)
    parser.add_argument("--refresh", action="store_true")
    args, _ = parser.parse_known_args()  # Streamlit only forwards args after '--'
    return args


def load_dataframes(handle: str):
    with get_connection() as conn:
        contests_df = pd.read_sql_query("SELECT * FROM contests", conn)
        problems_df = pd.read_sql_query("SELECT * FROM problems", conn)
        tags_df = pd.read_sql_query("SELECT * FROM problem_tags", conn)
        submissions_df = pd.read_sql_query("SELECT * FROM submissions", conn)
        rating_hist_df = pd.read_sql_query(
            "SELECT * FROM user_rating_history WHERE handle = ?", conn, params=(handle,)
        )
        user_row = pd.read_sql_query(
            "SELECT * FROM users WHERE handle = ?", conn, params=(handle,)
        )
    return contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row


def to_raw_rating_changes(rating_hist_df: pd.DataFrame) -> list[dict]:
    return [
        {
            "contestId": r.contest_id,
            "oldRating": r.rating_before,
            "newRating": r.rating_after,
            "ratingUpdateTimeSeconds": r.rating_update_time,
        }
        for r in rating_hist_df.itertuples()
    ]


def get_last_synced(handle: str) -> int:
    """Cheap, uncached lookup -- used as the cache-busting key for
    everything else, so it must always reflect the real current value."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT last_synced_at FROM users WHERE handle = ?", (handle,)
        ).fetchone()
    return int(row["last_synced_at"]) if row and row["last_synced_at"] is not None else 0


@st.cache_data(show_spinner=False)
def _load_dataframes_cached(handle: str, sync_marker: int):
    return load_dataframes(handle)


@st.cache_data(show_spinner=False)
def _compute_participation(handle: str, sync_marker: int):
    """Contests the user participated in (see participated_contest_ids --
    excludes later upsolves), sorted most-recent-first. Also returns
    contest_start_times and rating_changes since callers usually need
    those alongside it."""
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    rating_changes = to_raw_rating_changes(rating_hist_df)
    contest_start_times = dict(zip(contests_df.id, contests_df.start_time))
    finished_contests = contests_df[contests_df.phase == "FINISHED"]
    participated_ids = participated_contest_ids(submissions_df, handle, contests_df)
    participated_sorted = (
        finished_contests[finished_contests.id.isin(participated_ids)]
        .sort_values("start_time", ascending=False)
        .id.tolist()
    )
    return participated_sorted, contest_start_times, rating_changes


@st.cache_data(show_spinner=False)
def _compute_function1(handle: str, sync_marker: int, recent_ids: tuple):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    handle_subs = submissions_df[submissions_df.handle == handle]
    _, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    flagged = flag_weak_problems(
        list(recent_ids), problems_df, tags_df, handle_subs, rating_changes, contest_start_times
    )
    return weak_topics_summary(flagged)


@st.cache_data(show_spinner=False)
def _compute_function2(handle: str, sync_marker: int, current_rating: int):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    return suggest_virtual_contests(contests_df, problems_df, submissions_df, handle, current_rating)


@st.cache_data(show_spinner=False)
def _compute_function3a(handle: str, sync_marker: int):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    return solved_count_by_tag(submissions_df, tags_df, handle)


@st.cache_data(show_spinner=False)
def _compute_function3b(handle: str, sync_marker: int, selected_tags: tuple, recent_ids: tuple):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    _, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    return tag_elo_breakdown(
        submissions_df, problems_df, tags_df, handle, list(selected_tags),
        rating_changes, list(recent_ids), contest_start_times,
    )


@st.cache_data(show_spinner=False)
def _compute_function4(handle: str, sync_marker: int, recent_ids: tuple, current_rating: int):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    handle_subs = submissions_df[submissions_df.handle == handle]
    _, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    ranking = strong_weak_tag_ranking(
        problems_df, tags_df, handle_subs, rating_changes,
        list(recent_ids), contest_start_times, current_rating,
    )
    raw_counts = strong_weak_tag_counts(
        problems_df, tags_df, handle_subs, rating_changes,
        list(recent_ids), contest_start_times,
    )
    return ranking, raw_counts


@st.cache_data(show_spinner=False)
def _compute_recommendations(handle: str, sync_marker: int, recent_ids: tuple, tags: tuple):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    handle_subs = submissions_df[submissions_df.handle == handle]
    _, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    return recommended_problems(
        problems_df, tags_df, handle_subs, rating_changes,
        list(recent_ids), contest_start_times, list(tags),
    )


@st.cache_data(show_spinner=False)
def _compute_contest_history(handle: str, sync_marker: int, contest_ids: tuple):
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    return contest_history(contests_df, problems_df, submissions_df, handle, list(contest_ids))


def _display_recommendations_table(df: pd.DataFrame):
    """Shared rendering for a recommended_problems() result -- problem_id
    leftmost, no numeric index, status + link columns."""
    if df.empty:
        st.write("No matching problems.")
        return
    st.dataframe(
        df[["problem_id", "name", "rating", "tags", "status", "url"]].rename(
            columns={
                "problem_id": "ID",
                "name": "Problem",
                "rating": "Rating",
                "tags": "Tags",
                "status": "Status",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"url": st.column_config.LinkColumn("Link", display_text="Open")},
    )


def main():
    cli_args = parse_cli_args()

    if "handle" not in st.session_state:
        st.session_state.handle = cli_args.handle
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    st.title("Codeforces Analytics")

    handle_input = st.text_input("Codeforces handle", value=st.session_state.handle or "")
    go_clicked = st.button("Load / Refresh")
    auto_load = bool(cli_args.handle) and not st.session_state.data_loaded

    if go_clicked or auto_load:
        handle = handle_input.strip()
        if not handle:
            st.warning("Enter a Codeforces handle.")
            st.stop()

        init_db()
        client = CFClient()
        try:
            with get_connection() as conn:
                sync_all(conn, client, handle, force=(go_clicked or cli_args.refresh))
            st.session_state.handle = handle
            st.session_state.data_loaded = True
        except InvalidHandleError:
            st.error(f"'{handle}' doesn't look like a valid Codeforces handle -- check the spelling and try again.")
            st.session_state.data_loaded = False
            st.stop()
        except IngestionError:
            st.error("Couldn't reach Codeforces, and there's no cached data for this handle yet. Try again shortly.")
            st.session_state.data_loaded = False
            st.stop()

    if not st.session_state.data_loaded:
        st.info("Enter a handle above and click Load / Refresh to get started.")
        st.stop()

    handle = st.session_state.handle
    sync_marker = get_last_synced(handle)
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )

    current_rating = (
        int(user_row.iloc[0]["current_rating"])
        if not user_row.empty and pd.notna(user_row.iloc[0]["current_rating"])
        else 1000
    )
    all_tags = sorted(tags_df.tag.dropna().unique())

    st.header(f"{handle} — current rating {current_rating}")

    # -- Function 1: weakness flagging -----------------------------------
    st.subheader("Weakness flagging (recent contests)")
    participated_sorted, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    total_participated = len(participated_sorted)
    st.caption(f"You've participated in {total_participated} contest{'s' if total_participated != 1 else ''} total.")

    if total_participated == 0:
        st.write("No contest participation found yet -- nothing to analyze.")
        recent_ids: list[int] = []
    else:
        n_recent = st.slider(
            "Number of recent contests you participated in",
            1, total_participated, min(10, total_participated),
        )
        recent_ids = participated_sorted[:n_recent]

        weak_summary = _compute_function1(handle, sync_marker, tuple(recent_ids))
        col1, col2 = st.columns([2, 1])
        with col1:
            if not weak_summary.empty:
                st.plotly_chart(
                    px.bar(weak_summary, x="tag", y="count", title="Weak topics"),
                    use_container_width=True,
                )
            else:
                st.write("No weaknesses flagged in the selected recent contests.")
        with col2:
            st.dataframe(weak_summary, hide_index=True, use_container_width=True)

        st.markdown("**Contest history**")
        history = _compute_contest_history(handle, sync_marker, tuple(recent_ids))
        if not history.empty:
            st.dataframe(
                history.rename(
                    columns={
                        "contest_id": "Contest ID",
                        "name": "Name",
                        "date": "Date",
                        "num_problems": "# Problems",
                        "solved_during_contest": "Solved During",
                        "solved_overall": "Solved Overall",
                        "url": "Link",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
            )
        else:
            st.write("No contest history to show.")

    st.divider()

    # -- Function 2: virtual contest suggestions --------------------------
    st.subheader("Virtual contest suggestions")
    suggestions = _compute_function2(handle, sync_marker, current_rating)
    if not suggestions.empty:
        display_df = suggestions.copy()
        display_df["Date"] = pd.to_datetime(display_df["start_time"], unit="s").dt.strftime("%d %b %Y")
        display_df["Link"] = display_df["contest_id"].apply(
            lambda cid: f"https://codeforces.com/contest/{cid}"
        )
        display_df = display_df.rename(columns={"contest_id": "Contest ID", "name": "Contest"})[
            ["Contest ID", "Contest", "Date", "Link"]
        ]
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )
    else:
        st.write("No eligible virtual contests found.")

    st.divider()

    # -- Function 3A: overall solved-per-tag -------------------------------
    st.subheader("Overall solved count per tag")
    tag_counts = _compute_function3a(handle, sync_marker)
    col3, col4 = st.columns([2, 1])
    with col3:
        if not tag_counts.empty:
            st.plotly_chart(
                px.bar(tag_counts, x="tag", y="count", title="Solved problems per tag"),
                use_container_width=True,
            )
        else:
            st.write("No solved problems yet.")
    with col4:
        st.dataframe(tag_counts, hide_index=True, use_container_width=True)

    st.divider()

    # -- Function 3B: elo-bucketed breakdown, one bar PER TAG per bucket --
    st.subheader("Elo breakdown by tag")
    selected_tags = st.multiselect("Tags", all_tags, default=all_tags[:1] if all_tags else [])
    if selected_tags:
        breakdown = _compute_function3b(handle, sync_marker, tuple(selected_tags), tuple(recent_ids))
        if not breakdown.empty:
            # Full grid of every (bucket, tag) combo actually present, sorted so buckets
            # increase left-to-right and tags are grouped together within each bucket.
            combos = breakdown[["bucket", "tag"]].drop_duplicates().sort_values(["bucket", "tag"])
            pivot = breakdown.pivot_table(
                index=["bucket", "tag"], columns="category", values="count", fill_value=0
            ).reindex(pd.MultiIndex.from_frame(combos))

            bucket_labels = [str(b) for b in combos["bucket"]]
            tag_labels = list(combos["tag"])

            fig = go.Figure()
            colors = {
                "first_attempt": "#1b7a1b",
                "later_attempt": "#8fd18f",
                "unsolved": "#d94f4f",
                "should_have_solved": "#7a0d0d",  # dark red -- distinct from the lighter 'unsolved' red
            }
            labels = {
                "first_attempt": "Solved (1st attempt)",
                "later_attempt": "Solved (2nd+ attempt)",
                "unsolved": "Unsolved (attempted)",
                "should_have_solved": "Not done in contest",
            }
            for category in ["first_attempt", "later_attempt", "unsolved", "should_have_solved"]:
                y_vals = pivot[category].values if category in pivot.columns else [0] * len(combos)
                fig.add_bar(
                    name=labels[category],
                    x=[bucket_labels, tag_labels],  # multicategory axis: bucket groups, tag sub-labels
                    y=y_vals,
                    marker_color=colors[category],
                )
            fig.update_layout(
                barmode="stack",
                title="Attempts by elo bucket, one bar per tag",
                xaxis_title="Elo bucket / tag",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Recommended problems for the selected tag(s)**")
            elo_recs = _compute_recommendations(handle, sync_marker, tuple(recent_ids), tuple(selected_tags))
            _display_recommendations_table(elo_recs)
        else:
            st.write("Nothing to show for the selected tag(s).")
    else:
        st.write("Select at least one tag to see the breakdown.")

    st.divider()

    # -- Recommendations: independent tag filters, failed list + CF browse link
    st.subheader("Recommended problems")

    st.markdown("**Attempted + failed, or should've done in contest — easiest first**")
    st.caption("Starts with everything. Check tags to narrow it down — matches ANY selected tag.")
    failed_rec_tags = st.multiselect("Filter by tag(s)", all_tags, default=[], key="failed_rec_tags")
    failed = _compute_recommendations(handle, sync_marker, tuple(recent_ids), tuple(failed_rec_tags))
    _display_recommendations_table(failed)

    st.markdown("**Browse more on Codeforces**")
    st.caption(
        "CF's own filter uses ALL selected tags (a problem must have every one), "
        "unlike the list above which matches ANY selected tag — different tools, different logic."
    )
    browse_rec_tags = st.multiselect("Filter by tag(s)", all_tags, default=[], key="browse_rec_tags")
    rating_bounds = st.slider(
        "Rating range",
        min_value=800,
        max_value=3500,
        value=(current_rating, min(current_rating + 250, 3500)),
        step=100,
    )
    browse_url = problemset_browse_url(browse_rec_tags, rating_bounds[0], rating_bounds[1])
    st.link_button("Problemset - Codeforces", browse_url)

    st.divider()

    ranking, raw_counts = _compute_function4(handle, sync_marker, tuple(recent_ids), current_rating)

    st.subheader("Strong / weak tags")
    sort_mode = st.radio(
        "Sort tags", ["Alphabetical", "Highest to lowest (total)"], horizontal=True
    )

    def _sorted(df: pd.DataFrame, green_col: str, red_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        if sort_mode == "Alphabetical":
            return df.sort_values("tag").reset_index(drop=True)
        return df.assign(_total=df[green_col] + df[red_col]).sort_values(
            "_total", ascending=False
        ).drop(columns="_total").reset_index(drop=True)

    raw_counts = _sorted(raw_counts, "green_count", "red_count")
    ranking = _sorted(ranking, "green_weight", "red_weight")

    # -- Function 4 (raw): same solved/failed definitions, no weighting ---
    st.markdown("**Raw counts**")
    st.caption(
        "\"Failed\" combines: attempted anywhere (including gym) and never solved, "
        f"plus never-attempted problems in your last {len(recent_ids)} participated contests "
        "that were within a fair range of your rating at the time. Plain counts, no weighting."
    )
    if not raw_counts.empty:
        fig4_raw = go.Figure()
        fig4_raw.add_bar(name="Solved", x=raw_counts.tag, y=raw_counts.green_count, marker_color="#2ca02c")
        fig4_raw.add_bar(name="Failed", x=raw_counts.tag, y=raw_counts.red_count, marker_color="#d62728")
        fig4_raw.update_layout(barmode="stack", title="Strong / weak tags (raw problem counts)")
        st.plotly_chart(fig4_raw, use_container_width=True)
        st.dataframe(raw_counts, hide_index=True, use_container_width=True)
    else:
        st.write("Not enough data yet.")

    st.divider()

    # -- Function 4 (weighted): same solved/failed definitions, elo-weighted
    st.markdown("**Weighted**")
    st.caption("Same solved/failed problems as above, weighted by how far each problem's rating is from yours.")
    with st.expander("How is the weighting calculated?"):
        st.latex(
            r"\text{weight} = w_{min} + (w_{max} - w_{min}) \cdot \frac{1}{1 + e^{\mp\, \text{diff}/k}}"
        )
        st.markdown(
            f"""
where `diff = problem_rating - your_current_rating`, and
`w_min = {SIGMOID_W_MIN}`, `w_max = {SIGMOID_W_MAX}`, `k = {SIGMOID_K}`.

- **Solved** problems use `-diff` in the exponent: solving something *above* your
  rating counts for more than solving something well below it.
- **Failed** problems use `+diff` (the mirrored curve): failing something
  *below* your rating counts for more than failing something well above it,
  since that's the stronger weak-spot signal.

At `diff = 0` (a problem exactly at your rating), both curves give the
same middle weight of `{round(SIGMOID_W_MIN + (SIGMOID_W_MAX - SIGMOID_W_MIN) * 0.5, 2)}`.
"""
        )
    if not ranking.empty:
        fig4 = go.Figure()
        fig4.add_bar(name="Solved", x=ranking.tag, y=ranking.green_weight, marker_color="#2ca02c")
        fig4.add_bar(name="Failed", x=ranking.tag, y=ranking.red_weight, marker_color="#d62728")
        fig4.update_layout(barmode="stack", title="Strong / weak tags (weighted, not normalized)")
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(ranking, hide_index=True, use_container_width=True)
    else:
        st.write("Not enough data yet to rank tags.")


if __name__ == "__main__":
    main()
