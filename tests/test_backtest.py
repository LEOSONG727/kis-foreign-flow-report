from __future__ import annotations

import pandas as pd

from kis_flow_bot.backtest import (
    build_calibration,
    evaluate_forward_ohlc,
    evaluate_forward_path,
    summarize_variants,
)


def test_target_before_stop_is_a_win():
    result = evaluate_forward_path(100.0, pd.Series([102.0, 109.0, 94.0]), 0.08, 0.06)
    assert result.outcome == "target"
    assert result.days_to_event == 2


def test_stop_before_target_is_a_loss():
    result = evaluate_forward_path(100.0, pd.Series([93.0, 110.0]), 0.08, 0.06)
    assert result.outcome == "stop"
    assert result.days_to_event == 1


def test_calibration_stays_inactive_with_small_sample():
    results = pd.DataFrame([{"score": 75.0, "outcome": "target"}] * 10)
    calibration = build_calibration(results)
    assert not calibration["active"]
    assert calibration["resolved_samples"] == 10


def test_calibration_rejects_large_but_loss_making_sample():
    dates = pd.bdate_range("2025-01-02", periods=320)
    results = pd.DataFrame(
        [
            {
                "as_of": dates[index],
                "entry_score": 74.0,
                "outcome": "target" if index % 3 == 0 else "stop",
                "realized_return": 0.077 if index % 3 == 0 else -0.063,
            }
            for index in range(320)
        ]
    )
    calibration = build_calibration(results)
    assert not calibration["active"]
    assert calibration["quality_gate_version"] == 2
    assert not calibration["activation_checks"]["overall_positive_expectancy"]


def test_backtest_enters_next_open_and_uses_intraday_path_conservatively():
    future = pd.DataFrame(
        [
            {"open": 100.0, "high": 109.0, "low": 93.0, "close": 105.0},
            {"open": 105.0, "high": 110.0, "low": 104.0, "close": 109.0},
        ]
    )
    result = evaluate_forward_ohlc(100.0, future, target=0.08, stop=0.06, roundtrip_cost=0.003)
    assert result.outcome == "stop"
    assert result.days_to_event == 1
    assert result.ambiguous_same_day
    assert round(result.forward_return, 3) == 0.087


def test_variant_summary_reports_signal_quality():
    results = pd.DataFrame(
        [
            {"variant": "v1_balanced", "production": True, "entry_min_return3_pct": -3.0,
             "entry_max_return20_pct": 3.0, "entry_max_rebound20_pct": 5.0, "requires_reacceleration": True,
             "outcome": "target", "forward_return": 0.10, "realized_return": 0.077, "max_upside": 0.12, "max_drawdown": -0.03,
             "ambiguous_same_day": False},
            {"variant": "v1_balanced", "production": True, "entry_min_return3_pct": -3.0,
             "entry_max_return20_pct": 3.0, "entry_max_rebound20_pct": 5.0, "requires_reacceleration": True,
             "outcome": "stop", "forward_return": -0.07, "realized_return": -0.063, "max_upside": 0.02, "max_drawdown": -0.08,
             "ambiguous_same_day": True},
        ]
    )
    row = summarize_variants(results).iloc[0]
    assert row["signals"] == 2
    assert row["resolved_win_rate"] == 0.5
    assert row["ambiguous_days"] == 1
