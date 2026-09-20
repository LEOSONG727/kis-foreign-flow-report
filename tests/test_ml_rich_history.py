from __future__ import annotations

import numpy as np
import pandas as pd

from kis_flow_bot.ml_rich_history import (
    MODEL_FEATURES,
    RAW_FEATURES,
    _paired_bootstrap,
    add_blended_score,
    prepare_rich_frame,
)


def _frame() -> pd.DataFrame:
    rows = []
    for day_index, day in enumerate(pd.bdate_range("2026-01-02", periods=8)):
        for candidate in range(4):
            row = {
                "as_of": day,
                "symbol": f"{candidate + 1:06d}",
                "discovery_score": 90.0 - candidate * 5,
                "forward_return": 0.08 - candidate * 0.02,
                "excess_forward_return": 0.06 - candidate * 0.02,
                "max_drawdown": -0.02 - candidate * 0.01,
            }
            for index, feature in enumerate(RAW_FEATURES):
                row[feature] = float(candidate + day_index / 10 + index / 1000)
            rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_rich_frame_builds_day_ranks_and_target_rank():
    prepared = prepare_rich_frame(_frame())
    assert set(MODEL_FEATURES).issubset(prepared.columns)
    first = prepared[prepared["as_of"] == prepared["as_of"].min()]
    assert first["target_rank"].min() == 0.25
    assert first["target_rank"].max() == 1.0
    assert first["symbol"].str.len().eq(6).all()


def test_blended_score_respects_zero_and_full_weight():
    prepared = prepare_rich_frame(_frame())
    prediction = -prepared["discovery_score"].to_numpy()
    baseline = add_blended_score(prepared, prediction, 0.0)
    model_only = add_blended_score(prepared, prediction, 1.0)
    baseline_pick = baseline.sort_values(
        ["as_of", "rich_rank_score"], ascending=[True, False]
    ).groupby("as_of").head(1)
    model_pick = model_only.sort_values(
        ["as_of", "rich_rank_score"], ascending=[True, False]
    ).groupby("as_of").head(1)
    assert baseline_pick["symbol"].eq("000001").all()
    assert model_pick["symbol"].eq("000004").all()


def test_paired_bootstrap_detects_consistent_improvement():
    prepared = prepare_rich_frame(_frame())
    dates = sorted(prepared["as_of"].unique())
    model = pd.DataFrame(
        {"as_of": dates, "excess_forward_return": np.repeat(0.03, len(dates))}
    )
    baseline = pd.DataFrame(
        {"as_of": dates, "excess_forward_return": np.repeat(0.01, len(dates))}
    )
    result = _paired_bootstrap(model, baseline, samples=500)
    assert result["probability_positive"] == 1.0
    assert result["ci_low"] > 0
