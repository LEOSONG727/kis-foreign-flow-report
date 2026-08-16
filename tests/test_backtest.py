from __future__ import annotations

import pandas as pd

from kis_flow_bot.backtest import build_calibration, evaluate_forward_path


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
