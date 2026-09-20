from __future__ import annotations

import pandas as pd

from kis_flow_bot.ml_alpha import (
    prepare_training_frame,
    purged_train_test_split,
    select_daily_top,
    summarize_selection,
)


def _sample_frame() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-02", periods=60)
    for index, date in enumerate(dates):
        for suffix, score, excess in (
            ("1", 80.0, 0.01),
            ("2", 70.0, 0.03),
        ):
            rows.append(
                {
                    "as_of": date,
                    "symbol": f"{index:05d}{suffix}"[-6:],
                    "market": "KOSPI",
                    "bucket": "오늘 신규매수 검토",
                    "discovery_score": score,
                    "foreign_20_adtv": 2.0 + index / 100.0,
                    "price_return20": 0.01,
                    "rebound_from_low20": 0.02,
                    "ma20_distance": 0.01,
                    "cheap_accumulation": suffix == "1",
                    "dip_absorption": suffix == "2",
                    "forward_return": excess + 0.01,
                    "excess_forward_return": excess,
                    "max_drawdown": -0.05,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_training_frame_normalizes_types_and_symbols():
    frame = _sample_frame().head(2).copy()
    frame["cheap_accumulation"] = ["True", "False"]
    prepared = prepare_training_frame(frame)
    assert list(prepared["cheap_accumulation"]) == [1.0, 0.0]
    assert prepared["symbol"].str.len().eq(6).all()
    assert pd.api.types.is_datetime64_any_dtype(prepared["as_of"])


def test_purged_split_removes_overlapping_forward_window():
    frame = prepare_training_frame(_sample_frame())
    calendar = pd.DatetimeIndex(pd.bdate_range("2026-01-02", periods=80))
    train, test, info = purged_train_test_split(
        frame,
        calendar,
        test_fraction=0.30,
        horizon_sessions=20,
    )
    assert not train.empty
    assert not test.empty
    assert train["as_of"].max() <= info.train_end
    assert test["as_of"].min() >= info.test_start
    train_end_pos = calendar.get_loc(info.train_end)
    test_start_pos = calendar.get_loc(info.test_start)
    assert test_start_pos - train_end_pos > 20


def test_select_daily_top_uses_requested_score():
    frame = prepare_training_frame(_sample_frame().head(4))
    frame["ml_alpha_pred"] = [0.1, 0.2, 0.3, 0.1]
    selected = select_daily_top(frame, "ml_alpha_pred")
    assert len(selected) == 2
    assert list(selected["ml_alpha_pred"]) == [0.2, 0.3]


def test_summary_reports_excess_and_drawdown():
    frame = prepare_training_frame(_sample_frame().head(4))
    stats = summarize_selection(frame)
    assert stats["signals"] == 4
    assert stats["dates"] == 2
    assert stats["mean_excess"] > 0
    assert stats["mean_drawdown"] == -0.05
