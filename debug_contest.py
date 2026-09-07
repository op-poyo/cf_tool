"""Diagnostic for the "wrong number of problems" bug -- run this against
your real cf_data.db to see exactly what's happening for a specific
contest, rather than guessing.

Usage:
    python debug_contest.py 2250
    python debug_contest.py 2250 --handle artyom.k08   # also re-checks live via the API

What it checks, in order:
  1. Does cache_meta have a 'contest_problems:<id>' marker? If NOT, the
     backfill has never run for this contest at all -- meaning you're
     looking at data from before the fix, or a sync hasn't happened
     since it was applied. Fix: click "Load / Refresh" in the app.
  2. How many problems does the LOCAL db currently have for this
     contest, and what are their indices?
  3. (only with --handle) What does contest.standings return RIGHT NOW,
     live, for this contest -- bypassing everything cached. If this also
     comes back short, the bug is in how we call/parse that endpoint,
     not in the backfill trigger logic.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db.database import get_connection, get_last_refresh, DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("contest_id", type=int)
    parser.add_argument("--handle", help="If given, also fetches contest.standings live via the API")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    with get_connection(Path(args.db)) as conn:
        cache_key = f"contest_problems:{args.contest_id}"
        last_refresh = get_last_refresh(conn, cache_key)
        print(f"[1] cache_meta['{cache_key}'] = {last_refresh}")
        if last_refresh is None:
            print("    -> Backfill has NEVER run for this contest.")
            print("       This means the fix hasn't had a chance to fire yet --")
            print("       click 'Load / Refresh' in the app (not just reload the page),")
            print("       then re-run this script.")
        else:
            import datetime
            print(f"    -> Last backfilled at {datetime.datetime.fromtimestamp(last_refresh)}")

        rows = conn.execute(
            "SELECT problem_index, name, rating FROM problems WHERE contest_id = ? ORDER BY problem_index",
            (args.contest_id,),
        ).fetchall()
        print(f"\n[2] Local DB currently has {len(rows)} problem(s) for contest {args.contest_id}:")
        for r in rows:
            print(f"    {r['problem_index']}  {r['name']}  (rating={r['rating']})")

    if args.handle:
        from ingestion.client import CFClient
        client = CFClient()
        print(f"\n[3] Fetching contest.standings LIVE for contest {args.contest_id} (bypassing all cache)...")
        try:
            live_problems = client.get_contest_problems(args.contest_id)
            print(f"    -> API returned {len(live_problems)} problem(s) right now:")
            for p in live_problems:
                print(f"    {p.get('index')}  {p.get('name')}  (rating={p.get('rating')})")
            if len(live_problems) != len(rows):
                print(
                    f"\n    MISMATCH: live API has {len(live_problems)}, "
                    f"local DB has {len(rows)}. The fix hasn't actually run "
                    "for this contest yet -- force-refresh in the app."
                )
            else:
                print("\n    Local DB matches what the API returns right now.")
        except Exception as exc:
            print(f"    -> API call failed: {exc}")
            print("       If this fails, it points to a real bug in how the app")
            print("       calls contest.standings for this contest specifically --")
            print("       share this error.")


if __name__ == "__main__":
    main()
