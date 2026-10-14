"""Trading-session purge/embargo, with actual label endpoints when known."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .point_in_time import local_dates


def trading_calendar(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(local_dates(pd.Series(values)).dropna().unique()).sort_values()


def label_ends(frame: pd.DataFrame, horizon: int, sessions=None) -> pd.Series:
    if horizon < 1:
        raise ValueError("Label holding period must be positive")
    if "label_end_date" in frame:
        return local_dates(frame["label_end_date"])
    if sessions is None:
        # Sparse signal dates or a weekday calendar cannot identify KRX label
        # maturity; unknown endpoints must not enter training/calibration.
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    calendar = trading_calendar(sessions)
    signals = local_dates(frame["as_of"])
    positions = calendar.get_indexer(signals)
    good = (positions >= 0) & (positions + horizon < len(calendar))
    ends = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    ends.loc[good] = calendar.take(positions[good] + horizon).to_numpy()
    return ends


def mature_training_mask(frame: pd.DataFrame, validation_start, horizon: int,
                         *, sessions=None, embargo_sessions: int = 1) -> pd.Series:
    if embargo_sessions < 1:
        raise ValueError("At least one embargo session is required")
    signals = local_dates(frame["as_of"])
    start = local_dates(pd.Series([validation_start])).iloc[0]
    ends = label_ends(frame, horizon, sessions)
    # Calendar alone is insufficient for malformed/reversed endpoints.
    valid = signals.notna() & ends.notna() & ends.gt(signals)
    if sessions is None:
        # One strict boundary gap is always safe; a larger configured gap
        # requires a real trading calendar instead of guessing holidays.
        if embargo_sessions > 1:
            return pd.Series(False, index=frame.index)
        return valid & signals.lt(start) & ends.lt(start)
    calendar = trading_calendar(sessions)
    start_index = calendar.searchsorted(start)
    cutoff_index = start_index - embargo_sessions
    if cutoff_index < 0:
        return pd.Series(False, index=frame.index)
    cutoff = calendar[cutoff_index]
    signal_index = calendar.get_indexer(signals)
    signal_cutoff = start_index - horizon - embargo_sessions
    return valid & ends.le(cutoff) & (signal_index >= 0) & (signal_index <= signal_cutoff)
