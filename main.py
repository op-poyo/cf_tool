"""CLI entry point: `python main.py [handle] [--refresh]`.

Launches the Streamlit dashboard, optionally pre-filled with a handle
(triggering an immediate load) and/or forcing a cache refresh.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Codeforces analytics dashboard")
    parser.add_argument("handle", nargs="?", default=None, help="Codeforces handle to analyze")
    parser.add_argument(
        "--refresh", action="store_true", help="Force-refresh cached data before displaying"
    )
    args = parser.parse_args()

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    cmd = ["streamlit", "run", str(app_path)]

    trailing = []
    if args.handle:
        trailing += ["--handle", args.handle]
    if args.refresh:
        trailing += ["--refresh"]
    if trailing:
        cmd += ["--"] + trailing

    subprocess.run(cmd)


if __name__ == "__main__":
    sys.exit(main())
