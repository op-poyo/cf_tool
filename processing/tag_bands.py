"""Buckets every RATED problem this app has already synced into its
official Codeforces rank tier, and counts how often each tag appears
within each tier -- the same analysis as the standalone cf_tag_bands
tool, but computed directly from data already in this app's local db
(problems_df / tags_df from the global problemset sync). No separate
scrape, no extra API calls: it's a pure aggregation over data the app
already has for other tabs.

Only rated problems are included, since unrated ones can't be placed
into a tier. A problem with multiple tags contributes to each of its
tags -- same convention used everywhere else in this app (Tag Overview,
Deep Dive, etc.).
"""

import pandas as pd

# Official Codeforces rank tiers, by rating. Upper bound of the last
# tier is open-ended (a large int stands in for "and above").
RANK_TIERS = [
    ("Newbie", 0, 1199),
    ("Pupil", 1200, 1399),
    ("Specialist", 1400, 1599),
    ("Expert", 1600, 1899),
    ("Candidate Master", 1900, 2099),
    ("Master", 2100, 2299),
    ("International Master", 2300, 2399),
    ("Grandmaster", 2400, 2599),
    ("International Grandmaster", 2600, 2899),
    ("Legendary Grandmaster", 2900, 10_000),
]

_TIER_ORDER = {name: i for i, (name, _, _) in enumerate(RANK_TIERS)}
_TIER_RANGE_LABEL = {
    name: (f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}") for name, lo, hi in RANK_TIERS
}


def tier_for_rating(rating) -> str | None:
    if rating is None or pd.isna(rating):
        return None
    rating = int(rating)
    for name, lo, hi in RANK_TIERS:
        if lo <= rating <= hi:
            return name
    return None


def tag_counts_by_band(
    problems_df: pd.DataFrame, tags_df: pd.DataFrame, top_n: int | None = None
) -> pd.DataFrame:
    """Returns one row per (band, tag): elo_band, rating_range, tag,
    tag_count, band_problem_count, pct_of_band. Sorted by tier order
    (Newbie -> Legendary Grandmaster), then by tag_count descending
    within each tier. Pass top_n to keep only the N most common tags
    per band; None (default) returns all of them."""
    rated = problems_df[problems_df["rating"].notna()].copy()
    if rated.empty:
        return pd.DataFrame(
            columns=["elo_band", "rating_range", "tag", "tag_count", "band_problem_count", "pct_of_band"]
        )

    rated["band"] = rated["rating"].apply(tier_for_rating)
    rated = rated.dropna(subset=["band"])

    merged = rated.merge(tags_df, on=["contest_id", "problem_index"], how="inner")

    band_totals = rated.groupby("band").size().rename("band_problem_count")
    tag_counts = merged.groupby(["band", "tag"]).size().rename("tag_count").reset_index()
    tag_counts = tag_counts.merge(band_totals, on="band")
    tag_counts["pct_of_band"] = (tag_counts["tag_count"] / tag_counts["band_problem_count"] * 100).round(1)
    tag_counts["rating_range"] = tag_counts["band"].map(_TIER_RANGE_LABEL)

    tag_counts["_tier_order"] = tag_counts["band"].map(_TIER_ORDER)
    tag_counts = (
        tag_counts.sort_values(["_tier_order", "tag_count"], ascending=[True, False])
        .drop(columns="_tier_order")
    )

    if top_n is not None:
        tag_counts = tag_counts.groupby("band", sort=False).head(top_n).reset_index(drop=True)

    return tag_counts.rename(columns={"band": "elo_band"})[
        ["elo_band", "rating_range", "tag", "tag_count", "band_problem_count", "pct_of_band"]
    ]
