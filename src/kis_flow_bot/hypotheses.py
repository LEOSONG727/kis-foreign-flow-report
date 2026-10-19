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
# H4 (registered 2026-09-30) lives in research.evaluate_h4 and paper.py:
# the walk-forward learned score ranks the whole eligible universe (OOS IC
# +0.078, lower bound +0.033 over 231 dates) even though its top-5 lost money,
# so the *top quintile* is tested instead of the top few names:
#   (a) mean D+20 excess return of the top 20% by learned score > 0, and
#   (b) top-20% minus bottom-20% excess spread > 0,
# both with 90% block-bootstrap lower bounds > 0 over >= 60 confirmation dates.
# Every signal date up to the registration date is discovery (the OOS IC and
# the failed top-5 were seen); only later dates count. Paper trading holds a
# seeded random 2-per-day sample of the quintile across 40 slots.
# H2 (filing veto) lives in jev_events: removing names with an adverse filing
# in the last 30 days from the discovery top 3 lowers severe-loss share.
# H3 (earnings quality) lives in earnings_quality: dropping H1 candidates whose
# latest profit explanation Jev judges one-off beats both plain H1 and the
# same veto by keywords. H3 was registered before any answer was seen, so
# every labelled date counts as evidence.


H4_REGISTERED_ON = "2026-09-30"


def h4_window(dates: pd.Series) -> pd.Series:
    """H4 discovery = every signal date up to registration; confirmation = later ones."""

    stamp = pd.to_datetime(dates)
    return pd.Series(
        np.where(stamp <= pd.Timestamp(H4_REGISTERED_ON), "discovery", "confirmation"), index=dates.index
    )


# ---------------------------------------------------------------------------
# H5 (registered 2026-10-19): foreign accumulation that price has *confirmed*
# ---------------------------------------------------------------------------
# Found on the stored history 2025-04-14 ~ 2026-10-16 (every date was looked at,
# so all of it is discovery). Inside the top 20% by 60-session foreign net buy
# divided by market cap, names whose 20-session return was in the top third beat
# names in the bottom third by +2.5%p of D+20 excess on average (90% block
# bootstrap +0.9%p ~ +4.7%p); the bottom-third "underreaction" basket lost in
# every variant tried. This is the opposite of the current buy gate, which
# requires price to have *not* reacted. Flow is normalised by market cap
# (ownership change), not by turnover.
#
# Frozen rule, per signal date, over the eligible universe:
#   flow  = foreign_net60 / market cap (KRW)
#   keep  = within-date pct rank(flow) >= 0.80
#   H5    = keep & pct rank(price_return20) >= 2/3, ordered by flow, top k
#   twin  = keep & pct rank(price_return20) <  1/3, ordered by flow, top k
# Confirmed only by dates after registration: >= 60 dates where the H5 top-10
# mean D+20 excess has a 90% lower bound > 0 AND (H5 - twin) has a lower
# bound > 0. Changing any number here means a new id and a new date.

H5_ID = "H5-flow-mcap-price-confirm"
H5_TITLE = "60일 외인 순매수/시총 상위 20% 중 20일 가격확인 상위 1/3"
H5_REGISTERED_ON = "2026-10-19"
H5_FLOW_MIN_RANK = 0.80
H5_CONFIRM_MIN_RANK = 2.0 / 3.0
H5_LAG_MAX_RANK = 1.0 / 3.0
H5_TOP_K = 10
H5_FLOW_COLUMN = "h5_flow_mcap60"


def h5_window(dates: pd.Series) -> pd.Series:
    """Every date up to registration shaped H5 (discovery); later ones confirm it."""

    stamp = pd.to_datetime(dates)
    return pd.Series(
        np.where(stamp <= pd.Timestamp(H5_REGISTERED_ON), "discovery", "confirmation"), index=dates.index
    )


def h5_flow(foreign_net60: pd.Series, market_cap_krw: pd.Series) -> pd.Series:
    """60-session foreign net buy as a share of market cap (missing when cap is unknown)."""

    net = pd.to_numeric(foreign_net60, errors="coerce")
    cap = pd.to_numeric(market_cap_krw, errors="coerce")
    return (net / cap.where(cap > 0)).astype(float)


def h5_cap_scale(universe: pd.DataFrame, closes: pd.DataFrame) -> pd.Series:
    """Shares implied by the master market cap: cap (KRW) / last stored close, per symbol.

    Multiplying by a date's close gives that date's cap with today's share count.
    Live and research use the same scale so H5 ranks the same universe in both;
    the holdings-snapshot cap covers only a subset of names. A share-count drift
    is a small error for a flow denominator.
    """

    if universe.empty or "market_cap_eok" not in universe or closes.empty:
        return pd.Series(dtype=float)
    symbols = universe["symbol"].astype(str).str.zfill(6)
    cap = pd.Series(pd.to_numeric(universe["market_cap_eok"], errors="coerce").to_numpy() * 100_000_000.0, index=symbols)
    cap = cap[~cap.index.duplicated(keep="last")]
    last = closes.ffill().iloc[-1]
    last.index = last.index.astype(str).str.zfill(6)
    scale = cap / last.reindex(cap.index)
    return scale[np.isfinite(scale) & (scale > 0)].astype(float)


def attach_h5_flow(frame: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Add ``h5_flow_mcap60`` from ``foreign_net60`` and the scaled signal-date cap."""

    result = frame.copy()
    if result.empty or "foreign_net60" not in result or "close" not in result or scale.empty:
        result[H5_FLOW_COLUMN] = np.nan
        return result
    caps = result["symbol"].astype(str).str.zfill(6).map(scale) * pd.to_numeric(result["close"], errors="coerce")
    result[H5_FLOW_COLUMN] = h5_flow(result["foreign_net60"], caps).to_numpy()
    return result


def h5_select(frame: pd.DataFrame, side: str = "confirm", top_k: int = H5_TOP_K, date_column: str = "as_of") -> pd.DataFrame:
    """H5 picks (``side='confirm'``) or its underreaction twin (``side='lag'``) per date.

    Ranks are taken within each date over the rows given, which must already be
    the eligible universe. Rows without flow or 20-session return are ignored.
    """

    if side not in {"confirm", "lag"}:
        raise ValueError(f"unknown H5 side: {side}")
    needed = {date_column, H5_FLOW_COLUMN, "price_return20"}
    if frame.empty or not needed.issubset(frame.columns):
        return frame.iloc[0:0]
    data = frame.copy()
    data["_flow"] = pd.to_numeric(data[H5_FLOW_COLUMN], errors="coerce")
    data["_r20"] = pd.to_numeric(data["price_return20"], errors="coerce")
    data = data.dropna(subset=["_flow", "_r20"])
    if data.empty:
        return data.drop(columns=["_flow", "_r20"])
    grouped = data.groupby(date_column)
    flow_rank = grouped["_flow"].rank(pct=True)
    price_rank = grouped["_r20"].rank(pct=True)
    keep = flow_rank >= H5_FLOW_MIN_RANK
    keep &= price_rank >= H5_CONFIRM_MIN_RANK if side == "confirm" else price_rank < H5_LAG_MAX_RANK
    chosen = (
        data[keep]
        .sort_values([date_column, "_flow"], ascending=[True, False], kind="stable")
        .groupby(date_column, sort=True)
        .head(top_k)
    )
    return chosen.drop(columns=["_flow", "_r20"])


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
