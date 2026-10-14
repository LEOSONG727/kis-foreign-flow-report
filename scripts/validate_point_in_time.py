"""Read-only audit of the cached PIT evidence and real trading-session split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kis_flow_bot.backtest import split_time_holdout
from kis_flow_bot.fundamentals import load_financials
from kis_flow_bot.point_in_time import PIT_VERSION
from kis_flow_bot.storage import read_csv
from kis_flow_bot.temporal_validation import label_ends, trading_calendar


def audit(root: Path, horizon: int = 20) -> dict:
    history = read_csv(root / "data/history.csv")
    if history.empty:
        raise ValueError("history.csv is required")
    assert history["symbol"].dropna().str.fullmatch(r"\d{6}").all()
    sessions = trading_calendar(history["date"])
    if len(sessions) <= horizon + 3:
        raise ValueError("Not enough trading sessions to test a purged split")
    frame = pd.DataFrame({"as_of": sessions[:-horizon]})
    frame["label_end_date"] = label_ends(frame, horizon, sessions)
    train, test, cut = split_time_holdout(frame, horizon=horizon, trading_dates=sessions)
    if not train.empty:
        assert train["label_end_date"].max() < test["as_of"].min()
        assert sessions.get_loc(cut) - sessions.get_loc(train["as_of"].max()) >= horizon + 1
    financials = load_financials(root / "data")
    receipts = int(financials.get("_pit_verified", pd.Series(dtype=bool)).sum())
    source = financials.get("pit_source_version", pd.Series("", index=financials.index)).eq(PIT_VERSION)
    usable = int((source & financials.get("_pit_verified", pd.Series(False, index=financials.index))).sum())
    training = read_csv(root / "reports/ml_rich_features.csv")
    shadow_ready = bool(not training.empty and "pit_provenance_version" in training
                        and training["pit_provenance_version"].eq(PIT_VERSION).all())
    return {"pit_version": PIT_VERSION, "history_rows": len(history), "sessions": len(sessions),
            "verified_receipt_dates": receipts, "verified_financial_sources": usable,
            "quarantined_financial_rows": len(financials) - usable,
            "shadow_table_has_pit_provenance": shadow_ready,
            "train_dates": len(train), "test_dates": len(test),
            "purged_dates": len(frame) - len(train) - len(test),
            "holdout_start": str(cut.date()) if cut is not None else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.horizon), ensure_ascii=False, indent=2))
