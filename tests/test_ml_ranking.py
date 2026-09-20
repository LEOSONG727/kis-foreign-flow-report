from __future__ import annotations

import numpy as np
import pandas as pd

from kis_flow_bot.ml_alpha import prepare_training_frame
from kis_flow_bot.ml_ranking import (
    MODEL_FEATURES,
    add_point_in_time_features,
    build_pairwise_ranker,
    make_pairwise_training_set,
    score_candidates,
)


def _frame() -> pd.DataFrame:
    rows = []
    for day in pd.bdate_range("2026-01-02", periods=12):
        for index, (score, foreign, excess) in enumerate(
            ((90.0, 3.0, 0.08), (80.0, 2.0, 0.03), (70.0, 1.0, -0.02))
        ):
            rows.append(
                {
                    "as_of": day,
                    "symbol": f"{index + 1:06d}",
                    "market": "KOSPI" if index < 2 else "KOSDAQ",
                    "discovery_score": score,
                    "foreign_20_adtv": foreign,
                    "price_return20": -0.02 + index * 0.01,
                    "rebound_from_low20": 0.01 + index * 0.01,
                    "ma20_distance": -0.01 + index * 0.01,
                    "cheap_accumulation": index == 0,
                    "dip_absorption": index == 1,
                    "forward_return": excess + 0.01,
                    "excess_forward_return": excess,
                    "max_drawdown": -0.03 - index * 0.01,
                }
            )
    return prepare_training_frame(pd.DataFrame(rows))


def test_point_in_time_ranks_are_calculated_within_each_day():
    featured = add_point_in_time_features(_frame())
    first_day = featured[featured["as_of"] == featured["as_of"].min()]
    assert first_day["discovery_score_day_rank"].min() == 1 / 3
    assert first_day["discovery_score_day_rank"].max() == 1.0
    assert set(first_day["market_kosdaq"]) == {0.0, 1.0}


def test_pairwise_set_is_symmetric_and_balanced():
    pair_x, pair_y = make_pairwise_training_set(_frame())
    assert list(pair_x.columns) == list(MODEL_FEATURES)
    assert len(pair_x) > 0
    assert pair_y.mean() == 0.5
    values = pair_x.to_numpy(dtype=float)
    assert np.allclose(values[0], -values[1], equal_nan=True)


def test_ranker_learns_obvious_within_day_order():
    frame = _frame()
    pair_x, pair_y = make_pairwise_training_set(frame)
    model = build_pairwise_ranker()
    model.fit(pair_x, pair_y)
    scored = score_candidates(model, frame)
    picked = (
        scored.sort_values(["as_of", "pairwise_rank_score"], ascending=[True, False])
        .groupby("as_of")
        .head(1)
    )
    assert picked["symbol"].eq("000001").all()
    assert model.score(pair_x, pair_y) > 0.90
