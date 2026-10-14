"""Structural shadow snapshot checks (promotion/performance gates are separate)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ml_shadow import SHADOW_MODEL_VERSION, SHADOW_TOP_K


def _boolean(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.lower().str.strip()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Invalid boolean values in {series.name}")
    return values.isin(["true", "1"])


def validate_latest_shadow_snapshot(report: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    if report.empty or ledger.empty:
        raise ValueError("Shadow report and ledger must be nonempty")
    report_dates = pd.to_datetime(report["as_of"], errors="coerce").dt.normalize()
    if report_dates.isna().any():
        raise ValueError("Report contains invalid as_of dates")
    latest = report_dates.max()
    available = report.loc[report_dates.eq(latest) & _boolean(report["ml_shadow_available"])]
    if not 1 <= len(available) <= SHADOW_TOP_K:
        raise ValueError("Latest report has no shadow candidates or exceeds the candidate cap")
    if not available["ml_shadow_model"].eq(SHADOW_MODEL_VERSION).all():
        raise ValueError("Unexpected shadow model version in latest report")
    dates = pd.to_datetime(ledger["signal_date"], errors="coerce").dt.normalize()
    current = ledger.loc[ledger["model_version"].eq(SHADOW_MODEL_VERSION) & dates.eq(latest)].copy()
    if current["symbol"].duplicated().any() or set(current["symbol"]) != set(available["symbol"]):
        raise ValueError("Latest shadow ledger candidates do not match the report")
    expected_ranks = set(range(1, len(current) + 1))
    for column in ("ml_shadow_rank", "discovery_rank"):
        if set(pd.to_numeric(current[column], errors="coerce")) != expected_ranks:
            raise ValueError(f"Latest snapshot has invalid {column}")
    if not np.isfinite(pd.to_numeric(current["ml_shadow_score"], errors="coerce")).all():
        raise ValueError("Latest snapshot has nonfinite model scores")
    for flag, rank in (("ml_top1", "ml_shadow_rank"), ("baseline_top1", "discovery_rank")):
        selected = _boolean(current[flag])
        if selected.sum() != 1 or not selected.equals(pd.to_numeric(current[rank]).eq(1)):
            raise ValueError(f"Latest snapshot must have exactly one {flag} at rank 1")
    return current.sort_values("ml_shadow_rank")
