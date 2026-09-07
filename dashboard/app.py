"""Streamlit dashboard. Launched via `streamlit run dashboard/app.py`,
normally through the CLI wrapper in main.py which passes the handle
along as a trailing arg. Can also be run directly and the handle typed
into the text box -- both paths converge on the same load/refresh logic.

CACHING NOTE: Streamlit reruns this entire script on every widget
interaction. Without caching, that means re-querying SQLite and
recomputing every function from scratch on every slider nudge or
checkbox click -- even ones that have nothing to do with what changed.
The pattern used throughout: `_load_dataframes_cached` is keyed on
(handle, sync_marker) where sync_marker combines the user's own
last_synced_at with the two global data freshness markers (contest
list, problemset) -- see get_cache_marker() for why the global markers
matter separately, not just the user's own timestamp. So it only
actually re-hits SQLite when a real sync happens, never on a UI
interaction. Every other cached function below re-derives its inputs from that same cached loader
(cheap, itself a cache hit) rather than taking large DataFrames
directly as cache-key parameters -- this keeps each cache key to small
hashable scalars/tuples instead of forcing Streamlit to hash the full
problemset on every call.

STRUCTURE: eight tabs -- Summary, Contests, Weaknesses, Tag Overview,
Deep Dive, Practice Session, Previous Sessions, Logged Questions --
plus the handle input/header which stays outside any tab since it
drives the sync for everything below it.
"""

import sys
import re
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db.database import init_db, get_connection, get_last_refresh
from db import sessions as sessions_db
from ingestion.client import CFClient, InvalidHandleError
from ingestion.rate_limiter import IngestionError
from ingestion.sync import sync_global_data, sync_user_data
from processing.derivations import SIGMOID_W_MIN, SIGMOID_W_MAX, SIGMOID_K, participated_contest_ids
from processing.function2 import suggest_virtual_contests
from processing.function3a import solved_count_by_tag
from processing.function3b import tag_elo_breakdown
from processing.function4 import strong_weak_tag_ranking, strong_weak_tag_counts
from processing.recommendations import recommended_problems, problemset_browse_url
from processing.contest_history import contest_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CF Analytics", layout="wide")

# Fixed window for all "recent contests" weakness-scanning logic (Function 4's
# not-done-in-contest component, Function 3B's should_have_solved category,
# that same component in recommendations). NOT applied to the contest history
# table, which is deliberately all-time.
MAX_RECENT_CONTESTS = 20

# Codeforces handles: letters, digits, underscore, hyphen, dot; 3-24 chars.
# Client-side check only -- catches obvious typos before a network round
# trip, doesn't replace the real InvalidHandleError path below (CF may
# still reject a handle that matches this shape but doesn't exist).
HANDLE_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,24}$")

# Shared semantic color palette -- every chart in the app pulls from this
# instead of mixing Plotly Express defaults with ad-hoc hex per chart, so
# "solved" and "failed" always mean the same color no matter which tab
# you're looking at.
COLORS = {
    "solved": "#2ca02c",
    "failed": "#d62728",
    "first_attempt": "#1b7a1b",
    "later_attempt": "#8fd18f",
    "unsolved": "#d94f4f",
    "should_have_solved": "#7a0d0d",  # dark red -- distinct from the lighter 'unsolved' red
    "rating": "#1f77b4",
}


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


def get_cache_marker(handle: str) -> tuple:
    """Cache-busting key for everything below -- combines the user's own
    last_synced_at with the two GLOBAL freshness markers (contest list,
    problemset). These matter separately: sync_user_data can return early
    without bumping the user's own timestamp (its hourly TTL not yet
    expired) even in a run where sync_global_data just wrote fresh
    contest/problem data on ITS OWN daily TTL. Keying the cache on the
    user timestamp alone meant that fresh global data could silently sit
    unused until the user's own cache also happened to expire. Always
    uncached -- these are cheap single-row lookups, and their whole job
    is to catch changes, so caching them would defeat the purpose."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT last_synced_at FROM users WHERE handle = ?", (handle,)
        ).fetchone()
        user_marker = int(row["last_synced_at"]) if row and row["last_synced_at"] is not None else 0
        contest_marker = get_last_refresh(conn, "global:contest_list") or 0
        problemset_marker = get_last_refresh(conn, "global:problemset") or 0
    return (user_marker, int(contest_marker), int(problemset_marker))


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
def _compute_total_solved(handle: str, sync_marker: int) -> int:
    """Distinct (contest_id, problem_index) pairs the handle has an OK
    verdict on -- a problem solved via multiple submissions, or one that
    appears under both its main contest and a mirrored gym contest,
    still counts once."""
    contests_df, problems_df, tags_df, submissions_df, rating_hist_df, user_row = (
        _load_dataframes_cached(handle, sync_marker)
    )
    solved = submissions_df[
        (submissions_df.handle == handle) & (submissions_df.verdict == "OK")
    ]
    return int(solved[["contest_id", "problem_index"]].drop_duplicates().shape[0])


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
    leftmost, no numeric index, status + link columns. Kept as a table
    rather than cards (unlike Logged Questions/Practice Session) since
    this can return dozens of rows on an unfiltered Weaknesses tab --
    cards would trade density for a consistency that isn't worth it at
    that length. Sorted by rating so individual rows are easier to scan."""
    if df.empty:
        st.write("No matching problems. Try a different tag, or check back after your next contest.")
        return
    display_df = df.sort_values("rating", na_position="last")
    st.dataframe(
        display_df[["problem_id", "name", "rating", "tags", "status", "url"]].rename(
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

    # Consumed here, before any tab (specifically Deep Dive's tag
    # multiselect) has rendered this run -- Streamlit raises if you set
    # session_state for a widget's key after that widget has already been
    # instantiated in the same script execution, and every tab's body
    # runs every rerun regardless of which is visually active. The
    # "View in Deep Dive" jump button (Logged Questions tab, which
    # renders AFTER Deep Dive) can't safely set the "deep_dive_tags" key
    # directly for this reason -- it stashes the value here instead and
    # reruns; this block then moves it into place before Deep Dive reads it.
    if "pending_deep_dive_tags" in st.session_state:
        st.session_state["deep_dive_tags"] = st.session_state.pop("pending_deep_dive_tags")

    st.title("Codeforces Analytics")
    st.caption("Best viewed on a larger screen.")

    handle_input = st.text_input("Codeforces handle", value=st.session_state.handle or "")
    go_clicked = st.button("Load / Refresh")
    auto_load = bool(cli_args.handle) and not st.session_state.data_loaded

    if go_clicked or auto_load:
        handle = handle_input.strip()
        if not handle:
            st.warning("Enter a Codeforces handle.")
            st.stop()
        if not HANDLE_RE.match(handle):
            st.warning(
                "That doesn't look like a valid Codeforces handle -- "
                "check the spelling and try again."
            )
            st.stop()

        init_db()
        client = CFClient()
        try:
            with get_connection() as conn:
                with st.spinner("Fetching contest list and problemset..."):
                    sync_global_data(conn, client, force=(go_clicked or cli_args.refresh))
                with st.spinner(f"Fetching submissions for {handle}..."):
                    sync_user_data(conn, client, handle, force=(go_clicked or cli_args.refresh))
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
        except Exception:
            # Catch-all for anything unexpected (DB errors, bugs, etc.) --
            # log the real exception server-side with a full traceback, but
            # never render it to the page: an unhandled traceback here would
            # leak file paths and internal details to whoever's using this.
            logger.exception("Unexpected error syncing data for handle=%r", handle)
            st.error("Something went wrong syncing data -- try again in a moment.")
            st.session_state.data_loaded = False
            st.stop()

    if not st.session_state.data_loaded:
        st.info("Enter a handle above and click Load / Refresh to get started.")
        st.stop()

    handle = st.session_state.handle
    sync_marker = get_cache_marker(handle)
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

    participated_sorted, contest_start_times, rating_changes = _compute_participation(handle, sync_marker)
    total_participated = len(participated_sorted)
    st.caption(f"You've participated in {total_participated} contest{'s' if total_participated != 1 else ''} total.")

    # Fixed window for weakness-scanning logic -- NOT used for the contest
    # history table, which shows the user's full all-time history.
    recent_ids = participated_sorted[:MAX_RECENT_CONTESTS]

    ranking, raw_counts = _compute_function4(handle, sync_marker, tuple(recent_ids), current_rating)

    with get_connection() as conn:
        _active_session_banner = sessions_db.get_active_session(conn, handle)
        if _active_session_banner is not None:
            _banner_attempts = sessions_db.get_session_attempts(conn, int(_active_session_banner["id"]))
    if _active_session_banner is not None:
        _started = pd.to_datetime(_active_session_banner["started_at"], unit="s")
        _solved_n = int((_banner_attempts["status"] == sessions_db.STATUS_SOLVED).sum()) if not _banner_attempts.empty else 0
        _total_n = len(_banner_attempts)
        _running = not _banner_attempts.empty and (_banner_attempts["timer_state"] == sessions_db.TIMER_RUNNING).any()
        st.info(
            f"Practice session running since {_started.strftime('%H:%M')} -- "
            f"{_solved_n}/{_total_n} solved{' -- timer running' if _running else ''}. "
            "See the Practice Session tab."
        )

    tab_summary, tab_contests, tab_weaknesses, tab_tag_overview, tab_deep_dive, tab_practice, tab_prev_sessions, tab_logged = st.tabs(
        ["Summary", "Contests", "Weaknesses", "Tag Overview", "Deep Dive",
         "Practice Session", "Previous Sessions", "Logged Questions"]
    )

    # ======================================================================
    # TAB 0: Summary
    # ======================================================================
    with tab_summary:
        if not ranking.empty and ranking.red_weight.sum() > 0:
            top_weak = ranking.sort_values("red_weight", ascending=False).iloc[0]
            top_weak_failed = int(
                raw_counts.loc[raw_counts.tag == top_weak["tag"], "red_count"].sum()
            )
            st.markdown(f"### Your biggest weak spot: **{top_weak['tag']}**")
            st.caption(
                f"Weighted score {top_weak['red_weight']:.1f} across "
                f"{top_weak_failed} problem{'s' if top_weak_failed != 1 else ''}. "
                "See the Tag Overview tab to dig in."
            )
        else:
            st.markdown("### No clear weak spot yet")
            st.caption("Solve or attempt a few more problems and this will fill in.")

        st.divider()

        total_solved = _compute_total_solved(handle, sync_marker)
        weak_tag_count = int((raw_counts.red_count > 0).sum()) if not raw_counts.empty else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current rating", current_rating)
        col2.metric("Contests participated", total_participated)
        col3.metric("Problems solved", total_solved)
        col4.metric("Tags needing work", weak_tag_count)

        st.divider()

        st.markdown("**Top weak tags**")
        if not raw_counts.empty:
            top3 = (
                raw_counts[raw_counts.red_count > 0]
                .sort_values("red_count", ascending=False)
                .head(3)
            )
            if not top3.empty:
                st.dataframe(
                    top3[["tag", "red_count"]].rename(
                        columns={"tag": "Tag", "red_count": "Failed / missed"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.markdown("**Problems to try**")
                top3_tags = tuple(top3["tag"].tolist())
                top_weak_recs = _compute_recommendations(
                    handle, sync_marker, tuple(recent_ids), top3_tags
                )
                _display_recommendations_table(top_weak_recs)
            else:
                st.write("No weak tags right now -- nice work.")
        else:
            st.write("Not enough data yet -- participate in a few contests and check back.")

        st.divider()

        st.markdown("**Next virtual contest to try**")
        summary_suggestions = _compute_function2(handle, sync_marker, current_rating)
        if not summary_suggestions.empty:
            top_contest = summary_suggestions.iloc[0]
            st.write(f"Contest {top_contest['contest_id']}: {top_contest['name']} — not yet attempted.")
            st.link_button(
                "Open on Codeforces",
                f"https://codeforces.com/contest/{top_contest['contest_id']}",
            )
        else:
            st.write("No eligible virtual contests found. Check back after your next rated contest.")

        st.divider()

        st.markdown("**Rating over time**")
        if not rating_hist_df.empty:
            trend_df = rating_hist_df.sort_values("rating_update_time").copy()
            trend_df["Date"] = pd.to_datetime(trend_df["rating_update_time"], unit="s")
            fig_trend = px.line(
                trend_df, x="Date", y="rating_after", markers=True,
                color_discrete_sequence=[COLORS["rating"]],
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Rating", height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.write("No rating history yet -- compete in a rated contest to start building one.")

    # ======================================================================
    # TAB 1: Contests
    # ======================================================================
    with tab_contests:
        st.subheader("Contest history")
        if total_participated == 0:
            st.write("No contest participation found yet. Once you compete in a rated contest, it'll show up here.")
        else:
            history = _compute_contest_history(handle, sync_marker, tuple(participated_sorted))
            if not history.empty:
                display_history = history.copy()
                display_history["completion_pct"] = (
                    display_history["solved_overall"] / display_history["num_problems"] * 100
                ).clip(0, 100)
                st.dataframe(
                    display_history.rename(
                        columns={
                            "contest_id": "Contest ID",
                            "name": "Name",
                            "date": "Date",
                            "num_problems": "# Problems",
                            "solved_during_contest": "Solved During",
                            "solved_overall": "Solved Overall",
                            "completion_pct": "Completion",
                            "url": "Link",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                        "Completion": st.column_config.ProgressColumn(
                            "Completion", min_value=0, max_value=100, format="%.0f%%"
                        ),
                    },
                )
            else:
                st.write("No contest history to show. Try clicking Load / Refresh to pull the latest data.")

        st.divider()

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
            st.write("No eligible virtual contests found. Check back after your next rated contest.")

    # ======================================================================
    # TAB 2: Weaknesses
    # ======================================================================
    with tab_weaknesses:
        st.subheader("Weak tags")
        st.caption(
            "Combines: attempted anywhere (including gym) and never solved, "
            f"plus never-attempted problems in your last {len(recent_ids)} participated contests "
            "that were within a fair range of your rating at the time."
        )
        weak_tags_df = (
            raw_counts[raw_counts.red_count > 0][["tag", "red_count"]]
            .sort_values("red_count", ascending=False)
            .reset_index(drop=True)
        )
        colA, colB = st.columns([2, 1])
        with colA:
            if not weak_tags_df.empty:
                fig_weak = px.bar(
                    weak_tags_df, x="tag", y="red_count", title="Failed / missed problems per tag",
                    color_discrete_sequence=[COLORS["failed"]],
                )
                fig_weak.update_layout(xaxis_title="Tag", yaxis_title="Count")
                st.plotly_chart(fig_weak, use_container_width=True)
            else:
                st.write("No weaknesses found -- nice work. Keep competing to keep this fresh.")
        with colB:
            st.dataframe(
                weak_tags_df.rename(columns={"tag": "Tag", "red_count": "Failed"}),
                hide_index=True, use_container_width=True,
            )

        st.markdown("**Recommended problems**")
        st.caption("Starts with everything. Check tags to narrow it down — matches ANY selected tag.")
        failed_rec_tags = st.multiselect("Filter by tag(s)", all_tags, default=[], key="failed_rec_tags")
        failed = _compute_recommendations(handle, sync_marker, tuple(recent_ids), tuple(failed_rec_tags))
        _display_recommendations_table(failed)

    # ======================================================================
    # TAB 3: Tag Overview
    # ======================================================================
    with tab_tag_overview:
        st.subheader("Overall solved count per tag")
        tag_counts = _compute_function3a(handle, sync_marker)
        col3, col4 = st.columns([2, 1])
        with col3:
            if not tag_counts.empty:
                fig3a = px.bar(
                    tag_counts, x="tag", y="count", title="Solved problems per tag",
                    color_discrete_sequence=[COLORS["solved"]],
                )
                fig3a.update_layout(xaxis_title="Tag", yaxis_title="Count")
                st.plotly_chart(fig3a, use_container_width=True)
            else:
                st.write("No solved problems yet. Solve a few on Codeforces, then click Load / Refresh.")
        with col4:
            st.dataframe(
                tag_counts.rename(columns={"tag": "Tag", "count": "Count"}),
                hide_index=True, use_container_width=True,
            )

        st.divider()

        st.subheader("Overall failed / missed count per tag")
        st.caption(
            "\"Failed\" combines: attempted anywhere (including gym) and never solved, "
            f"plus never-attempted problems in your last {len(recent_ids)} participated contests "
            "that were within a fair range of your rating at the time. Every tag shown, including 0."
        )
        failed_counts_df = (
            raw_counts[["tag", "red_count"]]
            .sort_values("red_count", ascending=False)
            .reset_index(drop=True)
        )
        col5, col6 = st.columns([2, 1])
        with col5:
            if not failed_counts_df.empty and failed_counts_df.red_count.sum() > 0:
                fig_failed = px.bar(
                    failed_counts_df, x="tag", y="red_count", title="Failed / missed problems per tag",
                    color_discrete_sequence=[COLORS["failed"]],
                )
                fig_failed.update_layout(xaxis_title="Tag", yaxis_title="Count")
                st.plotly_chart(fig_failed, use_container_width=True)
            else:
                st.write("No failed/missed problems -- nice work.")
        with col6:
            st.dataframe(
                failed_counts_df.rename(columns={"tag": "Tag", "red_count": "Failed"}),
                hide_index=True, use_container_width=True,
            )

        st.divider()

        st.subheader("Strong / weak tags")
        sort_mode = st.radio(
            "Sort tags", ["Alphabetical", "Highest to lowest"], horizontal=True
        )
        st.caption(
            "\"Highest to lowest\" sorts the raw counts view by total (solved+failed), "
            "and the weighted view by net (solved−failed), matching its diverging chart below."
        )

        def _sorted(df: pd.DataFrame, green_col: str, red_col: str, use_net: bool = False) -> pd.DataFrame:
            if df.empty:
                return df
            if sort_mode == "Alphabetical":
                return df.sort_values("tag").reset_index(drop=True)
            key = (df[green_col] - df[red_col]) if use_net else (df[green_col] + df[red_col])
            return df.assign(_key=key).sort_values(
                "_key", ascending=False
            ).drop(columns="_key").reset_index(drop=True)

        raw_counts_sorted = _sorted(raw_counts, "green_count", "red_count")
        ranking_sorted = _sorted(ranking, "green_weight", "red_weight", use_net=True)

        st.markdown("**Raw counts**")
        st.caption(
            "\"Failed\" combines: attempted anywhere (including gym) and never solved, "
            f"plus never-attempted problems in your last {len(recent_ids)} participated contests "
            "that were within a fair range of your rating at the time. Plain counts, no weighting."
        )
        if not raw_counts_sorted.empty:
            fig4_raw = go.Figure()
            fig4_raw.add_bar(name="Solved", x=raw_counts_sorted.tag, y=raw_counts_sorted.green_count, marker_color=COLORS["solved"])
            fig4_raw.add_bar(name="Failed", x=raw_counts_sorted.tag, y=raw_counts_sorted.red_count, marker_color=COLORS["failed"])
            fig4_raw.update_layout(
                barmode="stack", title="Strong / weak tags (raw problem counts)",
                xaxis_title="Tag", yaxis_title="Count",
            )
            st.plotly_chart(fig4_raw, use_container_width=True)
            st.dataframe(
                raw_counts_sorted.rename(
                    columns={"tag": "Tag", "green_count": "Solved", "red_count": "Failed"}
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("Not enough data yet -- participate in a few contests and check back.")

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
        if not ranking_sorted.empty:
            fig4 = go.Figure()
            fig4.add_bar(name="Solved", x=ranking_sorted.tag, y=ranking_sorted.green_weight, marker_color=COLORS["solved"])
            fig4.add_bar(name="Failed", x=ranking_sorted.tag, y=-ranking_sorted.red_weight, marker_color=COLORS["failed"])
            fig4.update_layout(
                barmode="relative", title="Strong / weak tags (weighted, net)",
                xaxis_title="Tag", yaxis_title="Weight (failed shown as negative)",
            )
            fig4.add_hline(y=0, line_color="black", line_width=1)
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(
                ranking_sorted.rename(
                    columns={"tag": "Tag", "green_weight": "Solved (Weight)", "red_weight": "Failed (Weight)"}
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.write("Not enough data yet to rank tags -- participate in a few contests and check back.")

    # ======================================================================
    # TAB 4: Deep Dive
    # ======================================================================
    with tab_deep_dive:
        st.subheader("Elo breakdown by tag")
        with st.expander("What does this chart show?"):
            st.markdown(
                """
Each **elo bucket** is a rating range around your own current rating --
problems get grouped into buckets based on how far their rating is from
yours, so you can see whether a tag trips you up specifically at your
level or across the board.

Within each bucket, one bar per selected tag is stacked into four
categories:

- **Solved (1st attempt)** -- you got it right away.
- **Solved (2nd+ attempt)** -- you got there, but not on the first try.
- **Unsolved (attempted)** -- you tried and didn't solve it.
- **Not done in contest** -- it was fair game in one of your recent
  contests (within a reasonable rating range) but you never attempted it,
  even later.
"""
            )
        selected_tags = st.multiselect(
            "Tags", all_tags, default=all_tags[:1] if all_tags else [], key="deep_dive_tags"
        )
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
                        marker_color=COLORS[category],
                    )
                fig.update_layout(
                    barmode="stack",
                    title="Attempts by elo bucket, one bar per tag",
                    xaxis_title="Elo bucket / tag",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Recommended problems for the selected tag(s)**")
                elo_recs = _compute_recommendations(handle, sync_marker, tuple(recent_ids), tuple(selected_tags))
                _display_recommendations_table(elo_recs)
            else:
                st.write("Nothing to show for the selected tag(s). Try a different tag, or check the Tag Overview tab for ones with more activity.")
        else:
            st.write("Select at least one tag to see the breakdown.")

        st.divider()

        st.subheader("Browse more on Codeforces")
        st.caption(
            "CF's own filter uses ALL selected tags (a problem must have every one), "
            "unlike the recommendation lists above which match ANY selected tag — "
            "different tools, different logic."
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

    # ======================================================================
    # TAB 5: Practice Session
    # ======================================================================
    with tab_practice:
        with get_connection() as conn:
            active = sessions_db.get_active_session(conn, handle)

        if active is None:
            st.write("No active session. Start one to track problems, timing, and notes as you practice.")
            if st.button("Start Session", type="primary"):
                with get_connection() as conn:
                    sessions_db.start_session(conn, handle)
                st.rerun()
        else:
            session_id = int(active["id"])
            started_dt = pd.to_datetime(active["started_at"], unit="s")
            st.subheader(f"Active session -- started {started_dt.strftime('%Y-%m-%d %H:%M')}")

            with st.expander("Add a problem to this session", expanded=True):
                add_mode = st.radio(
                    "Pick from",
                    ["Weak-tag suggestions", "Virtual contest suggestions", "Enter manually"],
                    horizontal=True,
                    key="practice_add_mode",
                )
                if add_mode == "Weak-tag suggestions":
                    weak_recs = _compute_recommendations(handle, sync_marker, tuple(recent_ids), tuple())
                    if weak_recs.empty:
                        st.write("No weak-tag suggestions available right now.")
                    else:
                        options = (weak_recs["problem_id"] + " -- " + weak_recs["name"]).tolist()
                        choice = st.selectbox("Problem", options, key="practice_weak_choice")
                        if st.button("Add to session", key="practice_add_weak"):
                            row = weak_recs.iloc[options.index(choice)]
                            with get_connection() as conn:
                                sessions_db.add_attempt(
                                    conn, session_id, int(row["contest_id"]), row["problem_index"]
                                )
                            st.rerun()
                elif add_mode == "Virtual contest suggestions":
                    vc = _compute_function2(handle, sync_marker, current_rating)
                    if vc.empty:
                        st.write("No virtual contest suggestions available right now.")
                    else:
                        st.caption(
                            "Pick a suggested contest, then add its problems one at a time by index "
                            "(e.g. A, B, C) -- we suggest the contest, not individual problems within it."
                        )
                        vc_options = (vc["contest_id"].astype(str) + " -- " + vc["name"]).tolist()
                        vc_choice = st.selectbox("Contest", vc_options, key="practice_vc_choice")
                        vc_contest_id = int(vc_choice.split(" -- ")[0])
                        vc_index = st.text_input("Problem index (e.g. A, B, C1)", key="practice_vc_index")
                        if st.button("Add to session", key="practice_add_vc") and vc_index.strip():
                            with get_connection() as conn:
                                sessions_db.add_attempt(conn, session_id, vc_contest_id, vc_index.strip().upper())
                            st.rerun()
                else:
                    manual_cols = st.columns(2)
                    manual_contest = manual_cols[0].number_input(
                        "Contest ID", min_value=1, step=1, key="practice_manual_contest"
                    )
                    manual_index = manual_cols[1].text_input(
                        "Problem index (e.g. A, B, C1)", key="practice_manual_index"
                    )
                    if st.button("Add to session", key="practice_add_manual") and manual_index.strip():
                        with get_connection() as conn:
                            sessions_db.add_attempt(
                                conn, session_id, int(manual_contest), manual_index.strip().upper()
                            )
                        st.rerun()

            st.divider()

            with get_connection() as conn:
                attempts_df = sessions_db.get_session_attempts(conn, session_id)

            if attempts_df.empty:
                st.write("No problems added to this session yet -- add one above.")
            else:
                for _, arow in attempts_df.iterrows():
                    attempt_id = int(arow["id"])
                    problem_label = f"{arow['contest_id']}{arow['problem_index']}"
                    elapsed = sessions_db.elapsed_seconds(arow)

                    with st.container(border=True):
                        header_cols = st.columns([2, 2, 3])
                        header_cols[0].markdown(f"**{problem_label}**")
                        header_cols[0].caption(f"status: {arow['status']}")
                        header_cols[1].write(sessions_db.format_duration(elapsed))
                        header_cols[2].link_button(
                            "Open on Codeforces",
                            f"https://codeforces.com/problemset/problem/{arow['contest_id']}/{arow['problem_index']}",
                        )

                        if arow["status"] == sessions_db.STATUS_IN_PROGRESS:
                            timer_cols = st.columns(3)
                            if arow["timer_state"] == sessions_db.TIMER_RUNNING:
                                if timer_cols[0].button("Pause", key=f"pause_{attempt_id}"):
                                    with get_connection() as conn:
                                        sessions_db.pause_timer(conn, attempt_id)
                                    st.rerun()
                            else:
                                start_label = "Start" if elapsed == 0 else "Resume"
                                if timer_cols[0].button(start_label, key=f"start_{attempt_id}"):
                                    with get_connection() as conn:
                                        sessions_db.start_timer(conn, attempt_id)
                                    st.rerun()

                            felt = st.selectbox(
                                "Felt difficulty",
                                ["", "Easier than rated", "As expected", "Harder than rated"],
                                key=f"felt_{attempt_id}",
                            )
                            notes = st.text_area("Notes", key=f"notes_{attempt_id}", height=80)
                            action_cols = st.columns(2)
                            if action_cols[0].button("Mark solved", key=f"solved_{attempt_id}", type="primary"):
                                with get_connection() as conn:
                                    sessions_db.mark_solved(conn, attempt_id, felt or None, notes or None)
                                st.rerun()
                            if action_cols[1].button("Give up", key=f"giveup_{attempt_id}"):
                                with get_connection() as conn:
                                    sessions_db.mark_gave_up(conn, attempt_id, felt or None, notes or None)
                                st.rerun()
                        else:
                            if arow["felt_difficulty"]:
                                st.caption(f"Felt: {arow['felt_difficulty']}")
                            if arow["notes"]:
                                st.write(arow["notes"])

            st.divider()
            end_notes = st.text_area("Session notes (optional)", key="practice_end_notes")
            if st.button("End Session"):
                with get_connection() as conn:
                    sessions_db.end_session(conn, session_id, notes=end_notes or None)
                st.rerun()

    # ======================================================================
    # TAB 6: Previous Sessions
    # ======================================================================
    with tab_prev_sessions:
        with get_connection() as conn:
            history_df = sessions_db.get_session_history(conn, handle)

        if history_df.empty:
            st.write("No past sessions yet -- finish one in the Practice Session tab and it'll show up here.")
        else:
            for _, srow in history_df.iterrows():
                started = pd.to_datetime(srow["started_at"], unit="s")
                ended = pd.to_datetime(srow["ended_at"], unit="s")
                with st.expander(f"{started.strftime('%Y-%m-%d %H:%M')}  ({ended - started})"):
                    with get_connection() as conn:
                        past_attempts = sessions_db.get_session_attempts(conn, int(srow["id"]))
                    if srow["notes"]:
                        st.write(srow["notes"])
                    if past_attempts.empty:
                        st.caption("No problems logged in this session.")
                    else:
                        display_df = past_attempts.copy()
                        display_df["Problem"] = display_df["contest_id"].astype(str) + display_df["problem_index"]
                        display_df["Time"] = display_df["active_seconds"].apply(sessions_db.format_duration)
                        st.dataframe(
                            display_df[["Problem", "status", "Time", "felt_difficulty", "notes"]].rename(
                                columns={
                                    "status": "Status",
                                    "felt_difficulty": "Felt",
                                    "notes": "Notes",
                                }
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )

    # ======================================================================
    # TAB 7: Logged Questions
    # ======================================================================
    with tab_logged:
        with get_connection() as conn:
            all_attempts = sessions_db.get_all_attempts(conn, handle)

        if all_attempts.empty:
            st.write("No problems logged yet -- add some in the Practice Session tab.")
        else:
            merged = all_attempts.merge(
                problems_df[["contest_id", "problem_index", "rating"]],
                on=["contest_id", "problem_index"], how="left",
            )
            tag_lists = (
                tags_df.groupby(["contest_id", "problem_index"])["tag"]
                .apply(list).reset_index().rename(columns={"tag": "tags"})
            )
            merged = merged.merge(tag_lists, on=["contest_id", "problem_index"], how="left")
            merged["tags"] = merged["tags"].apply(lambda t: t if isinstance(t, list) else [])
            merged["problem"] = merged["contest_id"].astype(str) + merged["problem_index"]
            merged["time_taken"] = merged["active_seconds"].apply(sessions_db.format_duration)

            filter_cols = st.columns([2, 1])
            all_logged_tags = sorted({t for tl in merged["tags"] for t in tl})
            tag_filter = filter_cols[0].multiselect("Filter by tag", all_logged_tags, key="logged_tag_filter")
            sort_by = filter_cols[1].radio(
                "Sort by", ["Most recent", "Rating", "Tag"], horizontal=True, key="logged_sort_by"
            )

            view = merged
            if tag_filter:
                view = view[view["tags"].apply(lambda tl: any(t in tl for t in tag_filter))]

            if sort_by == "Rating":
                view = view.sort_values("rating", ascending=True, na_position="last")
            elif sort_by == "Tag":
                view = view.assign(_first_tag=view["tags"].apply(lambda tl: tl[0] if tl else "~")).sort_values("_first_tag")
            else:
                view = view.sort_values("created_at", ascending=False)

            if view.empty:
                st.write("No logged problems match that tag filter.")
            else:
                for _, row in view.iterrows():
                    with st.container(border=True):
                        info_cols = st.columns([2, 1, 3, 2])
                        info_cols[0].markdown(f"**{row['problem']}**")
                        info_cols[0].caption(f"status: {row['status']}")
                        info_cols[1].write(f"{int(row['rating'])}" if pd.notna(row["rating"]) else "—")
                        info_cols[2].write(", ".join(row["tags"]) if row["tags"] else "—")
                        info_cols[3].write(row["time_taken"])

                        if row["felt_difficulty"]:
                            st.caption(f"Felt: {row['felt_difficulty']}")
                        if row["notes"]:
                            st.write(row["notes"])

                        if row["tags"]:
                            if st.button("View in Deep Dive", key=f"jump_{row['id']}"):
                                st.session_state["pending_deep_dive_tags"] = row["tags"]
                                st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # st.stop() raises Streamlit's own StopException, which IS a
        # regular Exception subclass -- a naive broad catch here would
        # silently swallow every st.stop() call in the app (the "enter a
        # handle" guard, the error branches, etc.), breaking their
        # intended early-exit behavior. Let it through unchanged; only
        # genuinely unexpected bugs get the friendly-error treatment.
        if type(exc).__name__ == "StopException":
            raise
        logger.exception("Unexpected error rendering the dashboard")
        st.error("Something went wrong -- try refreshing the page.")
