from __future__ import annotations

import pandas as pd
import pytest

from kis_flow_bot.ml_shadow import SHADOW_MODEL_VERSION
from kis_flow_bot.shadow_validation import validate_latest_shadow_snapshot


def snapshots():
    report = pd.DataFrame({"symbol": ["000001", "000002"], "as_of": ["2026-10-13"] * 2,
                           "ml_shadow_available": [True, True], "ml_shadow_model": [SHADOW_MODEL_VERSION] * 2})
    latest = pd.DataFrame({"symbol": ["000001", "000002"], "signal_date": ["2026-10-13"] * 2,
                           "model_version": [SHADOW_MODEL_VERSION] * 2,
                           "ml_shadow_rank": [2, 1], "discovery_rank": [1, 2],
                           "ml_shadow_score": [50.0, 100.0], "ml_top1": [False, True], "baseline_top1": [True, False]})
    older = latest.assign(signal_date="2026-10-12")
    return report, pd.concat([older, latest], ignore_index=True)


def test_validation_checks_latest_date_and_accepts_smaller_candidate_pool():
    report, ledger = snapshots()
    current = validate_latest_shadow_snapshot(report, ledger)
    assert len(current) == 2 and current.symbol.iloc[0] == "000002"
    assert current.signal_date.eq("2026-10-13").all()


@pytest.mark.parametrize("column,value", [("ml_top1", True), ("ml_top1", "invalid"),
                                            ("ml_shadow_rank", 1), ("ml_shadow_score", float("nan")),
                                            ("signal_date", "2026-10-11"), ("symbol", "000003")])
def test_validation_rejects_broken_latest_snapshot(column, value):
    report, ledger = snapshots()
    if isinstance(value, str):
        ledger[column] = ledger[column].astype(object)
    ledger.loc[2, column] = value
    with pytest.raises(ValueError):
        validate_latest_shadow_snapshot(report, ledger)
