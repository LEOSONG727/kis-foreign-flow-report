"""Pre-registered hypotheses, frozen before the data that will test them exists.

Registered on 2026-09-28 after the first DART financials run showed, on the
2026-04-06 ~ 2026-08-26 signal dates only, that point-in-time operating yield
and book-to-market ranked 20-session excess returns even after size,
volatility and momentum were removed.

Those dates are the *discovery window*: the hypotheses were shaped by them,
so results there prove nothing. Every other date is a *confirmation window*:
the deep history backfill (2025) adds dates never looked at, and every day
after registration adds forward dates. A hypothesis is confirmed only by the
confirmation window.

Nothing here may be edited after registration. A changed idea is a new
hypothesis with a new id and a new registration date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REGISTERED_ON = "2026-09-28"
DISCOVERY_START = "2026-04-06"
DISCOVERY_END = "2026-08-26"
MIN_CONFIRMATION_DATES = 60


@dataclass(frozen=True)
class Hypothesis:
    id: str
    title: str
    weights: dict[str, float]


# Fixed weights on within-date percentile ranks (missing = neutral). No fitting.
HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        id="H1-value-quality-lowvol",
        title="영업이익 수익률 + 순자산 비율 + 저변동성, 외인 수급 보조",
        weights={"op_yield": 1.0, "book_to_market": 0.5, "atr20_pct": -1.0, "foreign_20_adtv": 0.5},
    ),
    Hypothesis(
        id="H1b-op-yield-only",
        title="영업이익 수익률 단독",
        weights={"op_yield": 1.0},
    ),
)
# H2 (filing veto) lives in jev_events: removing names with an adverse filing
# in the last 30 days from the discovery top 3 lowers severe-loss share.


def hypothesis_score(frame: pd.DataFrame, hypothesis: Hypothesis, date_column: str = "as_of") -> pd.Series:
    """Weighted sum of within-date percentile ranks, centred at 0."""

    total = pd.Series(0.0, index=frame.index)
    for column, weight in hypothesis.weights.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        ranks = values.groupby(frame[date_column]).rank(pct=True) - 0.5
        total = total + weight * ranks.fillna(0.0)
    return total


def add_hypothesis_scores(frame: pd.DataFrame, date_column: str = "as_of") -> pd.DataFrame:
    result = frame.copy()
    for hypothesis in HYPOTHESES:
        result[f"hyp_{hypothesis.id}"] = hypothesis_score(result, hypothesis, date_column)
    return result


def window_of(dates: pd.Series) -> pd.Series:
    """'discovery' for the dates that shaped the hypotheses, else 'confirmation'."""

    stamp = pd.to_datetime(dates)
    inside = (stamp >= pd.Timestamp(DISCOVERY_START)) & (stamp <= pd.Timestamp(DISCOVERY_END))
    return pd.Series(np.where(inside, "discovery", "confirmation"), index=dates.index)


def verdict(confirmation_ic: dict[str, Any], confirmation_partial: dict[str, Any]) -> str:
    if (confirmation_ic.get("dates") or 0) < MIN_CONFIRMATION_DATES:
        return "INSUFFICIENT_DATA"
    low = confirmation_ic.get("low")
    partial_low = confirmation_partial.get("low")
    if low is not None and low > 0 and partial_low is not None and partial_low > 0:
        return "CONFIRMED"
    high = confirmation_ic.get("high")
    if high is not None and high < 0:
        return "REVERSED"
    return "NOT_CONFIRMED"


VERDICT_LABELS = {
    "INSUFFICIENT_DATA": "검증 표본 부족",
    "CONFIRMED": "확인됨",
    "NOT_CONFIRMED": "확인 안 됨",
    "REVERSED": "반대로 나타남",
}
