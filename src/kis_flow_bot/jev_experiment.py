"""Prospective Jev vs quant/ML comparison on the same daily shortlist."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .storage import read_csv, write_csv

VERSION = "jev-challenge-v1"
COLUMNS = [
    "signal_date", "symbol", "name", "experiment_version", "baseline_rank",
    "ml_rank", "jev_rank", "jev_review", "jev_fragile_flow",
    "jev_price_trap", "jev_chase_risk", "signal_close", "entry_date",
    "entry_open", "return_5", "return_20", "max_drawdown_20",
]


def _new_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "jev_review" not in scored:
        return pd.DataFrame(columns=COLUMNS)
    frame = scored.head(20).copy()
    if frame["jev_review"].isna().any():
        # A missing API judgment must not silently change a contender list.
        return pd.DataFrame(columns=COLUMNS)
    frame["baseline_rank"] = range(1, len(frame) + 1)
    if "ml_shadow_rank" in frame:
        frame["ml_rank"] = pd.to_numeric(frame["ml_shadow_rank"], errors="coerce")
    else:
        frame["ml_rank"] = pd.NA
    frame["_review_order"] = frame["jev_review"].map(
        {"DEEP_REVIEW": 0, "WATCH": 1, "REJECT": 2}
    )
    risk_columns = ["jev_fragile_flow", "jev_price_trap", "jev_chase_risk"]
    frame["_risk"] = frame[risk_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    ranked = frame.sort_values(["_review_order", "_risk", "baseline_rank"])
    frame["jev_rank"] = pd.Series(range(1, len(frame) + 1), index=ranked.index)
    frame["signal_date"] = pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
    frame["experiment_version"] = VERSION
    frame["signal_close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    for col in ("entry_date", "entry_open", "return_5", "return_20", "max_drawdown_20"):
        frame[col] = pd.NA
    return frame.rename(columns={"symbol": "symbol"})[COLUMNS]


def update_jev_experiment(scored: pd.DataFrame, history: pd.DataFrame, path: Path) -> pd.DataFrame:
    incoming = _new_rows(scored)
    existing = read_csv(path)
    if existing.empty:
        ledger = incoming
    elif incoming.empty:
        ledger = existing
    else:
        key = ["signal_date", "symbol", "experiment_version"]
        known = pd.MultiIndex.from_frame(existing[key].astype(str))
        novel = ~pd.MultiIndex.from_frame(incoming[key].astype(str)).isin(known)
        ledger = pd.concat([existing, incoming.loc[novel]], ignore_index=True)
    if ledger.empty or history.empty:
        return ledger
    prices = history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype("string").str.zfill(6)
    prices = prices.dropna(subset=["date"]).sort_values(["symbol", "date"])
    ledger["symbol"] = ledger["symbol"].astype("string").str.zfill(6)
    for index, row in ledger[ledger["return_20"].isna()].iterrows():
        day = pd.to_datetime(row["signal_date"], errors="coerce")
        future = prices[(prices["symbol"] == row["symbol"]) & (prices["date"] > day)].head(20)
        if future.empty:
            continue
        opening = pd.to_numeric(future["open"].iloc[0], errors="coerce")
        if pd.isna(opening) or opening <= 0:
            continue
        ledger.at[index, "entry_date"] = future["date"].iloc[0].strftime("%Y-%m-%d")
        ledger.at[index, "entry_open"] = opening
        for horizon in (5, 20):
            if len(future) >= horizon:
                closing = pd.to_numeric(future["close"].iloc[horizon - 1], errors="coerce")
                if pd.notna(closing):
                    ledger.at[index, f"return_{horizon}"] = closing / opening - 1
        if len(future) >= 20:
            lows = pd.to_numeric(future["low"], errors="coerce").dropna()
            if not lows.empty:
                ledger.at[index, "max_drawdown_20"] = (lows / opening - 1).min()
    ledger = ledger.sort_values(["signal_date", "baseline_rank", "symbol"]).reset_index(drop=True)
    write_csv(ledger[COLUMNS], path)
    return ledger[COLUMNS]
