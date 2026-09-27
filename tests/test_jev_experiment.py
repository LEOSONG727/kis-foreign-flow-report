import pandas as pd
import pytest

from kis_flow_bot.jev_experiment import _new_rows, update_jev_experiment, write_jev_comparison
from kis_flow_bot.jev_shadow import MODEL, POLICY, PROMPT_HASH


def cohort():
    return pd.DataFrame([dict(as_of="2026-09-01", symbol=f"{i:06d}", name=str(i),
        close=100, discovery_score=90-i, ml_shadow_rank=i, jev_status="OK", jev_review="WATCH",
        jev_fragile_flow=.9 if i == 1 else .1, jev_price_trap=.1, jev_chase_risk=.1,
        jev_policy=POLICY, jev_prompt_hash=PROMPT_HASH, jev_model=MODEL,
        jev_captured_at="2026-09-01T12:00:00+00:00") for i in range(1, 4)])


def history():
    return pd.DataFrame([dict(date=day, symbol=f"{i:06d}", open=100, close=110, low=90)
                         for i in range(1, 4) for day in pd.bdate_range("2026-09-02", periods=20)])


def test_prospective_cost_and_duplicate_price_dates(tmp_path):
    path = tmp_path / "ledger.csv"
    prices = history()
    prices = pd.concat([prices, prices])
    result = update_jev_experiment(cohort(), prices, path, roundtrip_cost_pct=.3)
    assert result.loc[0, "jev_rank"] == 3
    assert result.loc[0, "return_20"] == pytest.approx(.1)
    assert result.loc[0, "net_return_20"] == pytest.approx(.097)
    assert result.loc[0, "worst_entry_loss_20"] == pytest.approx(-.1)
    assert result.loc[0, "entry_date"] == "2026-09-02"


def test_whole_daily_cohort_frozen_on_rerun(tmp_path):
    path = tmp_path / "ledger.csv"
    first = update_jev_experiment(cohort(), pd.DataFrame(), path)
    changed = cohort()
    changed.loc[0, "symbol"] = "999999"
    second = update_jev_experiment(changed, pd.DataFrame(), path)
    assert len(first) == len(second) == 3
    assert "999999" not in set(second["symbol"])


def test_late_capture_never_gets_forward_outcomes(tmp_path):
    frame = cohort()
    frame["jev_captured_at"] = "2026-10-01T12:00:00Z"
    result = update_jev_experiment(frame, history(), tmp_path / "ledger.csv")
    assert result["return_20"].isna().all()
    assert result["late_capture"].all()


def test_partial_results_saved_but_not_ranked():
    frame = cohort()
    frame.loc[0, "jev_status"] = "API_ERROR"
    result = _new_rows(frame)
    assert len(result) == 3
    assert result["jev_rank"].isna().all()


def test_only_common_dates_and_versions_compared(tmp_path):
    result = _new_rows(cohort())
    result["net_return_20"] = .1
    result["worst_entry_loss_20"] = -.05
    result.loc[0, "ml_rank"] = float("nan")
    path = tmp_path / "report.md"
    write_jev_comparison(result, path)
    assert "공통 날짜 0일" in path.read_text()
