from __future__ import annotations

import pandas as pd

from .config import ScreeningConfig


def apply_candidate_eligibility(
    scored: pd.DataFrame,
    screening: ScreeningConfig,
    report_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Annotate the single discovery-universe policy used live and in tests.

    A stale row must never compete with today's settled data. Watchlist rows
    may still be retained by the caller for display, but
    ``discovery_eligible`` remains false so they cannot be proposed.
    """
    if scored.empty:
        return scored.copy()

    frame = scored.copy()
    as_of_raw = frame["as_of"] if "as_of" in frame else pd.Series(pd.NaT, index=frame.index)
    as_of = pd.to_datetime(as_of_raw, errors="coerce")
    resolved_date = pd.to_datetime(report_date, errors="coerce")
    if pd.isna(resolved_date):
        resolved_date = as_of.max()
    report_day = resolved_date.date() if pd.notna(resolved_date) else None

    frame["data_fresh"] = as_of.dt.date == report_day if report_day else False
    adtv_raw = frame["adtv20"] if "adtv20" in frame else pd.Series(0.0, index=frame.index)
    points_raw = frame["data_points"] if "data_points" in frame else pd.Series(0.0, index=frame.index)
    adtv20 = pd.to_numeric(adtv_raw, errors="coerce").fillna(0.0)
    points = pd.to_numeric(points_raw, errors="coerce").fillna(0.0)
    frame["liquidity_eligible"] = adtv20 >= screening.min_median_turnover_krw
    frame["history_eligible"] = points >= screening.min_data_points
    frame["discovery_eligible"] = (
        frame["data_fresh"]
        & frame["liquidity_eligible"]
        & frame["history_eligible"]
    )

    reasons: list[str] = []
    for fresh, liquid, history in zip(
        frame["data_fresh"],
        frame["liquidity_eligible"],
        frame["history_eligible"],
    ):
        blocked: list[str] = []
        if not fresh:
            blocked.append("당일 수급 데이터 미수신")
        if not liquid:
            blocked.append("20일 중앙 거래대금 부족")
        if not history:
            blocked.append("분석 이력 부족")
        reasons.append(" · ".join(blocked) if blocked else "후보 조건 통과")
    frame["eligibility_reason"] = reasons
    return frame
