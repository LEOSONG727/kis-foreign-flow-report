from __future__ import annotations

import numpy as np
import pandas as pd

from kis_flow_bot.ml_rich_history import RAW_FEATURES
from kis_flow_bot.ml_shadow import (
    apply_rich_shadow,
    prepare_shadow_candidates,
    update_shadow_tracking,
)


class _DummyModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["foreign_5_adtv"].to_numpy(dtype=float)


def _scored() -> pd.DataFrame:
    rows = []
    for index in range(3):
        row = {
            "as_of": "2026-09-18",
            "symbol": f"{index + 1:06d}",
            "name": f"후보{index + 1}",
            "market": "KOSPI",
            "close": 100.0 + index,
            "discovery_eligible": True,
            "discovery_score": 90.0 - index * 10,
            "timing_score": 60.0,
            "entry_score": 55.0,
        }
        for feature_index, feature in enumerate(RAW_FEATURES):
            row[feature] = float(feature_index) / 100.0
        row["foreign_5_adtv"] = float(index)
        rows.append(row)
    return pd.DataFrame(rows)


def test_shadow_reranks_only_historical_top_k_pool(tmp_path):
    scored = _scored()
    candidates = prepare_shadow_candidates(scored, top_k=2)
    assert list(candidates["symbol"]) == ["000001", "000002"]

    ranked = apply_rich_shadow(
        scored,
        tmp_path / "unused.csv",
        top_k=3,
        model=_DummyModel(),
    )
    assert ranked["ml_shadow_available"].all()
    assert set(ranked["ml_shadow_rank"].astype(int)) == {1, 2, 3}
    assert ranked.loc[ranked["ml_shadow_top"], "symbol"].item() == "000003"
    assert ranked.loc[ranked["ml_shadow_baseline_rank"].eq(1), "symbol"].item() == "000001"


def test_shadow_tracking_is_immutable_and_resolves_20_sessions(tmp_path):
    ranked = apply_rich_shadow(
        _scored(),
        tmp_path / "unused.csv",
        top_k=3,
        model=_DummyModel(),
    )
    dates = pd.bdate_range("2026-09-21", periods=20)
    history_rows = []
    for symbol in ranked["symbol"]:
        for offset, day in enumerate(dates):
            history_rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 102.0 + offset,
                    "low": 98.0,
                    "close": 101.0 + offset,
                }
            )
    history = pd.DataFrame(history_rows)
    path = tmp_path / "shadow.csv"

    first = update_shadow_tracking(ranked, history, path)
    second = update_shadow_tracking(ranked, history, path)

    assert len(first) == 3
    assert len(second) == 3
    assert second["resolved"].all()
    assert second["entry_date"].eq("2026-09-21").all()
    assert second["exit_date"].eq("2026-10-16").all()
    assert second["forward_return_20"].notna().all()
