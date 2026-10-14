from __future__ import annotations

import json
import os
import tempfile
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

# Identifiers must be strings BEFORE pandas infers numbers (receipt numbers can
# also lose precision when a column contains blanks). Other columns keep their
# existing numeric/boolean inference for compatibility with scoring and ledgers.
IDENTIFIER_WIDTHS = {"symbol": 6, "stock_code": 6, "corp_code": 8}
TEXT_COLUMNS = (*IDENTIFIER_WIDTHS, "rcept_no", "rcept_dt", "reprt_code", "bsns_year")


def normalize_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, width in IDENTIFIER_WIDTHS.items():
        if column not in result:
            continue
        values = result[column].astype("string").str.strip().replace("", pd.NA)
        # Repair identifiers already inferred as floats, without inventing an
        # identifier for missing data or changing synthetic alphanumeric keys.
        values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        numeric = values.str.fullmatch(r"\d+", na=False)
        values.loc[numeric] = values.loc[numeric].str.zfill(width)
        result[column] = values
    for column in set(TEXT_COLUMNS) - set(IDENTIFIER_WIDTHS):
        if column in result:
            result[column] = result[column].astype("string")
    return result


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    requested = kwargs.pop("dtype", None)
    if requested is None or isinstance(requested, dict):
        dtype = {**(requested or {}), **dict.fromkeys(TEXT_COLUMNS, "string")}
    else:
        # A scalar dtype=str/string is safe. Any other scalar must not cast IDs.
        if requested not in (str, "str", "string", object):
            raise ValueError("Use a dtype mapping for numeric columns; identifiers must remain strings")
        dtype = requested
    converters = kwargs.get("converters", {})
    if set(converters) & set(TEXT_COLUMNS):
        raise ValueError("Identifier converters would bypass string schema protection")
    return normalize_identifiers(pd.read_csv(path, dtype=dtype, **kwargs))


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
    merged = normalize_identifiers(merged)
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
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        normalize_identifiers(frame).to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_snapshots(current: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (current, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = normalize_identifiers(merged)
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

