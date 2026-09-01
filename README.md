# CF Analytics

A local dashboard for analyzing your [Codeforces](https://codeforces.com) performance: which topics you're weak in, past contests worth attempting as a virtual contest, tag-level solve/fail breakdowns, and targeted problem recommendations — all built from your public submission history.

Everything runs locally. Data is pulled from the public Codeforces API and cached in a local SQLite database (`cf_data.db`) so repeated views don't re-hit the API.

## Requirements

- Python 3.10+
- The packages in `requirements.txt`

## Setup

```bash
git clone https://github.com/op-poyo/cf_tool.git
cd cf_tool
pip install -r requirements.txt
```

## Usage

Launch the dashboard:

```bash
python main.py
```

This opens the Streamlit app in your browser. Type in a Codeforces handle and click **Load / Refresh**.

You can also pass a handle directly so it loads automatically:

```bash
python main.py your_handle
```

Force a fresh pull from the Codeforces API (bypassing the cache) instead of using cached data:

```bash
python main.py your_handle --refresh
```

| Argument     | Description                                              |
|--------------|------------------------------------------------------------|
| `handle`     | (optional) Codeforces handle to load automatically on start |
| `--refresh`  | Force-refresh cached data instead of using what's cached     |

The **Load / Refresh** button in the dashboard itself always force-refreshes, regardless of these flags.

### Caching

- Global data (contest list, problemset) is refreshed at most once a day.
- Per-user data (rating history, submissions) is refreshed at most once an hour, or immediately if you click **Load / Refresh**.
- If a refresh fails (e.g. Codeforces is unreachable) and cached data already exists, the dashboard falls back to the cached data rather than failing outright.

## What's in the dashboard

- **Weakness flagging** — problems from your recent contests that were within a fair range of your rating at the time but that you never solved (not even later as an upsolve), aggregated by tag.
- **Contest history** — for the same set of recent contests: problems solved during the contest window vs. solved overall (including later upsolves).
- **Virtual contest suggestions** — past, fully-rated contests you haven't attempted yet that are a reasonable fit for your current rating (or any Div. 3/4).
- **Tag breakdowns** — overall solved-count per tag, plus an elo-bucketed chart per selected tag showing solved-on-first-attempt vs. solved-later vs. unsolved vs. never-attempted-but-should-have.
- **Recommended problems** — a combined list of problems you've attempted and failed, or should have attempted in a recent contest, filterable by tag, plus a direct link to Codeforces' own problemset filter.
- **Strong / weak tags** — a solved-vs-failed breakdown per tag, shown both as raw counts and as a rating-weighted score (solving/failing further from your own rating counts for more).

## Development

`scripts/test_run.py` runs the full processing pipeline against synthetic fixture data (`fixtures/generate_fixtures.py`) and prints the output of every function — useful for sanity-checking changes to the processing logic without needing a live Codeforces handle:

```bash
python scripts/test_run.py
```

## Project layout

```
main.py              CLI entry point — launches the Streamlit dashboard
dashboard/app.py      Streamlit UI
ingestion/            Codeforces API client, rate limiting/retry, sync logic
db/                    SQLite schema and connection helpers
processing/            Pure data-transform functions (weakness flagging, tag stats, recommendations, etc.)
fixtures/              Synthetic data for local testing
scripts/test_run.py    Manual pipeline smoke test
```
