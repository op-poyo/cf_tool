"""Thin wrapper around the public Codeforces API.

Every method goes through the shared RateLimiter and call_with_retry so
callers never have to think about throttling.
"""

import requests
import logging

from .rate_limiter import RateLimiter, call_with_retry, TransientAPIError

logger = logging.getLogger(__name__)

BASE_URL = "https://codeforces.com/api"
REQUEST_TIMEOUT = 15  # seconds


class InvalidHandleError(Exception):
    """Raised when CF reports the handle doesn't exist / user not found."""


class CFClient:
    def __init__(self):
        self._limiter = RateLimiter()

    # -- low-level request plumbing ------------------------------------

    def _get(self, method: str, params: dict | None = None) -> dict:
        def do_request():
            self._limiter.wait()
            try:
                resp = requests.get(
                    f"{BASE_URL}/{method}", params=params, timeout=REQUEST_TIMEOUT
                )
            except requests.RequestException as exc:
                raise TransientAPIError(str(exc)) from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise TransientAPIError(f"HTTP {resp.status_code}")

            data = resp.json()
            if data.get("status") != "OK":
                comment = data.get("comment", "")
                if "not found" in comment.lower():
                    raise InvalidHandleError(comment)
                # Other CF-side errors (bad params etc.) aren't retryable.
                raise ValueError(f"CF API error for {method}: {comment}")

            return data["result"]

        return call_with_retry(do_request)

    # -- per-user endpoints -----------------------------------------------

    def get_user_info(self, handle: str) -> dict:
        result = self._get("user.info", {"handles": handle})
        return result[0]

    def get_user_rating(self, handle: str) -> list[dict]:
        """Full rating change history, chronological."""
        return self._get("user.rating", {"handle": handle})

    def get_user_status(self, handle: str, batch_size: int = 1000) -> list[dict]:
        """All submissions for a user, fetched in batches for consistency
        even though CF will usually return everything in one call."""
        all_submissions = []
        start = 1
        while True:
            batch = self._get(
                "user.status",
                {"handle": handle, "from": start, "count": batch_size},
            )
            all_submissions.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size
        return all_submissions

    # -- global endpoints ---------------------------------------------------

    def get_contest_list(self, gym: bool = False) -> list[dict]:
        return self._get("contest.list", {"gym": str(gym).lower()})

    def get_problemset_problems(self) -> dict:
        """Returns {'problems': [...], 'problemStatistics': [...]}."""
        return self._get("problemset.problems")

    def get_contest_problems(self, contest_id: int) -> list[dict]:
        """Authoritative, always-immediate full problem list for ONE
        contest -- unlike problemset.problems (a daily-cached bulk
        endpoint that can lag behind a contest that just happened),
        contest.standings reflects a contest's real problem set right
        away. count=1 keeps the payload tiny; we only want result.problems,
        not the standings rows themselves."""
        result = self._get(
            "contest.standings",
            {"contestId": contest_id, "from": 1, "count": 1, "showUnofficial": "false"},
        )
        problems = result["problems"]
        for p in problems:
            p["contestId"] = contest_id  # contest.standings problems omit this; upsert needs it
        return problems
