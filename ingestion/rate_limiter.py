"""Throttling + retry logic for hitting the Codeforces API politely.

CF has no hard documented rate limit, but ~1 req/sec is the informal
community norm. We use a slightly more conservative fixed delay plus
exponential backoff on 429/5xx so we don't get soft-banned mid-ingestion.
"""

import time
import logging

logger = logging.getLogger(__name__)

MIN_DELAY_SECONDS = 1.75
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2


class RateLimiter:
    """Sequential fixed-delay throttle. Not a token bucket -- ingestion
    is inherently sequential here, so we don't need burst capacity."""

    def __init__(self, min_delay: float = MIN_DELAY_SECONDS):
        self.min_delay = min_delay
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


class IngestionError(Exception):
    """Raised when a CF API call fails after exhausting retries."""


def call_with_retry(fn, *args, **kwargs):
    """Call `fn(*args, **kwargs)`, retrying with exponential backoff on
    failures that look transient (429 / 5xx / network errors).

    `fn` is expected to raise `TransientAPIError` for retryable failures
    and any other exception for non-retryable ones (e.g. bad handle).
    """
    delay = BACKOFF_BASE_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except TransientAPIError as exc:
            last_exc = exc
            logger.warning(
                "Transient CF API error (attempt %d/%d): %s -- retrying in %ds",
                attempt, MAX_RETRIES, exc, delay,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise IngestionError(
        f"CF API call failed after {MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


class TransientAPIError(Exception):
    """Raised by the client for retryable failures (429, 5xx, timeouts)."""
