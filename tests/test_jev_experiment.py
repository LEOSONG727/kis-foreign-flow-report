import pandas as pd

from kis_flow_bot.jev_experiment import update_jev_experiment


def test_challenger_and_baseline_use_same_cohort_and_future_prices(tmp_path):
    scored = pd.DataFrame([
        {"as_of": "2026-09-01", "symbol": "000001", "name": "A", "close": 100,
         "jev_review": "WATCH", "jev_fragile_flow": .8, "jev_price_trap": .7,
         "jev_chase_risk": .3, "ml_shadow_rank": 1},
        {"as_of": "2026-09-01", "symbol": "000002", "name": "B", "close": 100,
         "jev_review": "DEEP_REVIEW", "jev_fragile_flow": .1, "jev_price_trap": .2,
         "jev_chase_risk": .1, "ml_shadow_rank": 2},
    ])
    dates = pd.bdate_range("2026-09-02", periods=20)
    history = pd.DataFrame([
        {"date": day, "symbol": symbol, "open": 100, "close": close, "low": 90}
        for symbol, close in [("000001", 95), ("000002", 110)]
        for day in dates
    ])
    path = tmp_path / "jev_experiment.csv"
    first = update_jev_experiment(scored, history, path)
    second = update_jev_experiment(scored, history, path)
    assert len(first) == len(second) == 2
    assert first["baseline_rank"].tolist() == [1, 2]
    assert first["jev_rank"].tolist() == [2, 1]
    assert first["return_20"].astype(float).tolist() == [-.05, .10]
    assert first["entry_date"].tolist() == ["2026-09-02", "2026-09-02"]


def test_partial_api_results_are_not_selected(tmp_path):
    scored = pd.DataFrame([
        {"as_of": "2026-09-01", "symbol": "000001", "jev_review": "WATCH"},
        {"as_of": "2026-09-01", "symbol": "000002", "jev_review": None},
    ])
    result = update_jev_experiment(scored, pd.DataFrame(), tmp_path / "ledger.csv")
    assert result.empty
