from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = [
    "date",
    "symbol",
    "market",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "foreign_net_qty",
    "foreign_net_amount",
    "institution_net_qty",
    "institution_net_amount",
]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"symbol": "string"}, **kwargs)


def merge_history(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    retention_days: int,
    as_of: date,
) -> pd.DataFrame:
    frames = [frame for frame in (current, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["symbol"] = merged["symbol"].astype("string").str.zfill(6)
    cutoff = pd.Timestamp(as_of - timedelta(days=retention_days))
    merged = merged[merged["date"].notna() & (merged["date"] >= cutoff)]
    merged = merged.drop_duplicates(["date", "symbol"], keep="last")
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
    for column in HISTORY_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    return merged[HISTORY_COLUMNS]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def merge_snapshots(current: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (current, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["symbol"] = merged["symbol"].astype("string").str.zfill(6)
    return (
        merged.dropna(subset=["date", "symbol"])
        .drop_duplicates(["date", "symbol"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"symbols": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"symbols": {}}


def write_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

