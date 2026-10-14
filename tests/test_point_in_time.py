from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from kis_flow_bot import fundamentals, ml_shadow, research
from kis_flow_bot.backtest import split_time_holdout, build_calibration
from kis_flow_bot.config import load_settings
from kis_flow_bot.earnings_quality import build_state, state_hash, review_rows, PROMPT_HASH, MODEL
from kis_flow_bot.fundamentals import attach_fundamentals, parse_multi_accounts, update_financials
from kis_flow_bot.jev_replay import load_jev_verdict
from kis_flow_bot.ml_shadow import RAW_FEATURES, ShadowTrainingUnavailable, fit_shadow_model
from kis_flow_bot.point_in_time import PIT_VERSION, backward_asof, bind_receipts, observable_answers
from kis_flow_bot.storage import read_csv, write_csv
from kis_flow_bot.temporal_validation import label_ends, mature_training_mask
from kis_flow_bot.shadow_validation import validate_latest_shadow_snapshot
from kis_flow_bot.features import build_base_features
from kis_flow_bot.scoring import score_features
from kis_flow_bot.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def financial(receipt="20260101000001", op=100):
    return {"symbol": "000001", "bsns_year": "2025", "reprt_code": "11011", "fs_div": "CFS",
            "rcept_no": receipt, "available": "2020-01-01", "revenue": 1000, "revenue_prev": 900,
            "op": op, "op_prev": 50, "ni": 10, "equity": 500, "pit_source_version": PIT_VERSION}


def filing(receipt="20260101000001", received="20260504"):
    return {"symbol": "000001", "rcept_no": receipt, "rcept_dt": received,
            "corp_name": "회사", "report_nm": "손익구조 변동"}


def test_actual_receipt_date_overrides_year_end_prefix_and_unverified_available():
    frame = pd.DataFrame({"symbol": ["000001"] * 4,
                          "as_of": ["2026-01-02", "2026-05-01", "2026-05-04", "2026-05-06"], "pit_size": np.log(10000)})
    result = attach_fundamentals(frame, pd.DataFrame([financial()]), filings=pd.DataFrame([filing()]))
    assert result.op_yield.iloc[:3].isna().all()
    assert result.op_yield.iloc[3] == pytest.approx(0.01)
    assert result.financial_available.iloc[3] == pd.Timestamp("2026-05-04")
    assert result.financial_rcept_no.iloc[3] == "20260101000001"


def test_missing_receipt_or_unversioned_financials_are_not_backfilled():
    frame = pd.DataFrame({"symbol": ["000001"], "as_of": ["2026-06-01"], "pit_size": np.log(10000)})
    for fin, filings in [(financial(), pd.DataFrame()),
                          ({k: v for k, v in financial().items() if k != "pit_source_version"}, pd.DataFrame([filing()]))]:
        result = attach_fundamentals(frame, pd.DataFrame([fin]), filings=filings)
        assert result.op_yield.isna().all()
        assert result.operating_profit.isna().all()


def test_corrections_keep_their_own_receipt_and_values():
    rows = []
    for receipt, amount in [("20260310000001", "100"), ("20260610000001", "999")]:
        for account in ("영업이익", "매출액"):
            rows.append({"stock_code": "000001", "bsns_year": "2025", "reprt_code": "11011",
                         "fs_div": "CFS", "rcept_no": receipt, "account_nm": account,
                         "thstrm_amount": amount, "frmtrm_amount": "50"})
    fin = parse_multi_accounts(rows, {})
    assert len(fin) == 2
    assert fin.set_index("rcept_no").op.to_dict() == {"20260310000001": 100, "20260610000001": 999}
    filings = pd.DataFrame([filing("20260310000001", "20260310"), filing("20260610000001", "20260610")])
    frame = pd.DataFrame({"symbol": ["000001"] * 3, "as_of": ["2026-04-01", "2026-06-10", "2026-06-11"], "pit_size": np.log(10000)})
    result = attach_fundamentals(frame, fin, filings=filings)
    np.testing.assert_allclose(result.op_yield, [0.01, 0.01, 0.0999])


def test_financial_cache_preserves_original_and_adds_correction(tmp_path, monkeypatch):
    write_csv(pd.DataFrame([financial("20260310000001", 100)]), tmp_path / "dart_financials.csv")
    write_csv(pd.DataFrame([filing("20260310000001", "20260310"), filing("20260610000001", "20260610")]), tmp_path / "dart_filings.csv")
    monkeypatch.setattr(fundamentals, "load_corp_codes", lambda *args: pd.DataFrame({"symbol": ["000001"], "corp_code": ["00000001"]}))
    monkeypatch.setattr(fundamentals, "REPORT_CODES", {"11011": "年"})
    raw = [{"stock_code": "000001", "bsns_year": "2025", "reprt_code": "11011", "fs_div": "CFS", "account_nm": "영업이익",
            "rcept_no": receipt, "thstrm_amount": amount, "frmtrm_amount": "50"}
           for receipt, amount in [("20260310000001", "777"), ("20260610000001", "999")]]
    monkeypatch.setattr(fundamentals.FinancialsClient, "multi_accounts", lambda *args: raw)
    result = update_financials(Mock(), tmp_path, ["000001"], [2025], refresh_years=[2025])
    assert result.set_index("rcept_no").op.to_dict() == {"20260310000001": 100, "20260610000001": 999}
    assert len(read_csv(tmp_path / "dart_financials.csv")) == 2


def test_asof_preserves_unsorted_rows_duplicate_indices_and_invalid_dates():
    base = pd.DataFrame({"symbol": ["000001", "000002", "000001", "000001"],
                         "as_of": ["2026-05-06", "2026-05-06", "bad", "2026-05-04"]}, index=[7, 7, 3, 9])
    events = pd.DataFrame({"symbol": ["000001"], "available": ["2026-05-04"], "value": [100]})
    result = backward_asof(base, events)
    assert list(result.index) == [7, 7, 3, 9]
    assert result.value.iloc[0] == 100
    assert result.value.iloc[1:].isna().all()


def test_conflicting_receipt_dates_do_not_make_information_available_early():
    rows = bind_receipts(pd.DataFrame([financial()]), pd.DataFrame([filing(received="20260501"), filing(received="20260508")]))
    assert rows.available.item() == pd.Timestamp("2026-05-08")


def test_answer_capture_is_korean_date_and_never_backcast():
    rows = pd.DataFrame({"as_of": ["2026-05-01", "2026-05-02", "2026-05-04"], "state_hash": ["a"] * 3})
    answers = pd.DataFrame({"state_hash": ["a"], "captured_at": ["2026-05-01T16:00:00Z"], "eq_one_off": [0.9]})
    out = observable_answers(rows, answers, ["eq_one_off"])
    assert out.eq_one_off.iloc[:2].isna().all()
    assert out.eq_one_off.iloc[2] == 0.9
    assert out.answer_available.iloc[2] == pd.Timestamp("2026-05-02")


def test_earliest_answer_is_immutable_and_unknown_capture_is_unavailable():
    rows = pd.DataFrame({"as_of": ["2026-05-06"], "state_hash": ["a"]})
    answers = pd.DataFrame({"state_hash": ["a", "a"], "captured_at": ["2026-05-04", "2026-05-01"], "eq_one_off": [0.9, 0.1]})
    assert observable_answers(rows, answers, ["eq_one_off"]).eq_one_off.item() == 0.1
    assert observable_answers(rows, answers.drop(columns="captured_at"), ["eq_one_off"]).eq_one_off.isna().all()


def test_document_and_answer_mapping_obeys_both_receipt_and_capture(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text())
    settings = load_settings(tmp_path)
    f = filing("20260501000001", "20260501")
    write_csv(pd.DataFrame([f]), settings.data_dir / "dart_filings.csv")
    text = "유형자산 처분이익"
    write_csv(pd.DataFrame({"rcept_no": [f["rcept_no"]], "text": [text]}), settings.data_dir / "dart_documents.csv")
    digest = state_hash(build_state(f["report_nm"], text, f["corp_name"]))
    write_csv(pd.DataFrame({"state_hash": [digest], "prompt_hash": [PROMPT_HASH], "model": [MODEL], "status": ["OK"],
                           "eq_one_off": [0.9], "eq_recurring": [0.1], "captured_at": ["2026-05-04T00:00:00Z"]}), settings.data_dir / "earnings_quality_answers.csv")
    rows = pd.DataFrame({"symbol": ["000001"] * 4, "as_of": pd.to_datetime(["2026-04-30", "2026-05-01", "2026-05-04", "2026-05-06"])})
    out = review_rows(rows, settings, client=None, jev_key="", max_calls=0, max_docs=0)
    assert not out.has_text.iloc[:2].any()
    assert out.has_text.iloc[2:].all()
    assert pd.isna(out.eq_one_off.iloc[2]) and out.eq_one_off.iloc[3] == 0.9
    assert out.kw_veto.iloc[2]  # public text may already support a code-only flag


def test_future_global_verdict_is_not_a_historical_feature(tmp_path):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps({"verdict": "ADDS_VALUE", "generated_at": "2026-05-04T22:00:00+09:00"}))
    assert load_jev_verdict(path, as_of="2026-05-04") is None
    assert load_jev_verdict(path, as_of="2026-05-06")["verdict"] == "ADDS_VALUE"


def test_purge_uses_real_sessions_and_actual_delayed_endpoints():
    sessions = pd.bdate_range("2026-05-01", periods=25).difference(pd.DatetimeIndex(["2026-05-05"]))
    frame = pd.DataFrame({"as_of": sessions[:15]})
    frame["label_end_date"] = label_ends(frame, 5, sessions)
    frame.loc[0, "label_end_date"] = sessions[18]  # delayed/halted stock
    mask = mature_training_mask(frame, sessions[15], 5, sessions=sessions, embargo_sessions=2)
    assert not mask.iloc[0]
    assert frame.loc[mask, "label_end_date"].max() <= sessions[13]
    assert frame.loc[mask, "as_of"].max() <= sessions[8]
    assert not mature_training_mask(frame.drop(columns="label_end_date"), sessions[15], 5).any()


@pytest.mark.parametrize("embargo", [1, 3])
def test_holdout_labels_do_not_overlap_train_or_embargo(embargo):
    sessions = pd.bdate_range("2026-01-02", periods=120)
    frame = pd.DataFrame({"as_of": sessions[:100], "outcome": "target"})
    frame["label_end_date"] = label_ends(frame, 20, sessions)
    train, test, cut = split_time_holdout(frame, 0.30, horizon=20, embargo_sessions=embargo, trading_dates=sessions)
    assert not train.empty and not test.empty
    assert train.label_end_date.max() <= sessions[sessions.get_loc(cut) - embargo]
    assert len(train) + len(test) < len(frame)
    assert train.as_of.max() <= sessions[sessions.get_loc(cut) - 20 - embargo]


def test_calibration_bands_never_fit_holdout_outcomes():
    sessions = pd.bdate_range("2026-01-02", periods=180)
    frame = pd.DataFrame({"as_of": sessions[:150], "entry_score": 74.0,
                          "outcome": "target", "realized_return": 0.08, "excess_forward_return": 0.03,
                          "pit_provenance_version": PIT_VERSION})
    frame["label_end_date"] = label_ends(frame, 20, sessions)
    first = build_calibration(frame, trading_dates=sessions)
    altered = frame.copy()
    altered.loc[altered.as_of >= pd.Timestamp(first["holdout_start"]), "outcome"] = "stop"
    second = build_calibration(altered, trading_dates=sessions)
    assert first["bands"] == second["bands"]
    assert first["training_resolved_samples"] == 80 and first["purged_samples"] == 20
    assert first["activation_checks"]["purged_time_boundary"]
    assert not build_calibration(frame.drop(columns="label_end_date"))["active"]


def test_shadow_fits_only_mature_pit_targets_and_refuses_legacy(tmp_path, monkeypatch):
    sessions = pd.bdate_range("2026-01-02", periods=110)
    rows = []
    for day in range(100):
        for n in range(12):
            rows.append({**{name: float(n) for name in RAW_FEATURES}, "as_of": sessions[day],
                         "symbol": f"{n:06d}", "excess_forward_return": float(n),
                         "label_end_date": sessions[day + 5], "pit_provenance_version": PIT_VERSION})
    path = tmp_path / "training.csv"
    write_csv(pd.DataFrame(rows), path)
    model = Mock()
    monkeypatch.setattr(ml_shadow, "_build_shadow_model", lambda: model)
    fit_shadow_model(path, as_of=sessions[70], horizon=5, trading_dates=sessions)
    features, target = model.fit.call_args.args
    assert len(features) == 65 * 12
    assert model.train_signal_end_ == sessions[64]
    legacy = read_csv(path).drop(columns="pit_provenance_version")
    write_csv(legacy, path)
    with pytest.raises(ShadowTrainingUnavailable, match="legacy"):
        fit_shadow_model(path, as_of=sessions[70], trading_dates=sessions)


def test_shadow_pending_state_is_explicit_and_not_an_activation():
    report = pd.DataFrame({"as_of": ["2026-05-06"], "symbol": ["000001"],
                           "ml_shadow_available": [False], "ml_shadow_status": ["pit_training_required"]})
    assert validate_latest_shadow_snapshot(report, pd.DataFrame()).empty
    report["ml_shadow_status"] = "prediction_failed"
    with pytest.raises(ValueError):
        validate_latest_shadow_snapshot(report, pd.DataFrame())


def test_historical_scores_are_invariant_to_current_master_financials_and_shares():
    dates = pd.bdate_range("2026-01-02", periods=65)
    history = pd.DataFrame({"date": dates, "symbol": "000001", "market": "KOSPI", "name": "S",
                            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                            "volume": 100000, "turnover": 1e9, "foreign_net_qty": 10000,
                            "foreign_net_amount": 1e8, "institution_net_qty": 0, "institution_net_amount": 0})
    universe = pd.DataFrame({"symbol": ["000001"], "market": ["KOSPI"], "name": ["S"],
                             "listed_shares": [1000], "roe": [5.0], "operating_profit": [-5], "net_income": [-5]})
    first = build_base_features(history, universe, point_in_time=True)
    changed = universe.assign(listed_shares=9999999999, roe=80.0, operating_profit=5000, net_income=5000)
    second = build_base_features(history, changed, point_in_time=True)
    pd.testing.assert_frame_equal(first, second)
    assert first.pit_size.isna().all()
    one = score_features(first, ModelConfig())
    two = score_features(second, ModelConfig())
    pd.testing.assert_frame_equal(one, two)


def test_historical_size_uses_only_observable_quote_shares():
    dates = pd.bdate_range("2026-01-02", periods=65)
    history = pd.DataFrame({"date": dates, "symbol": "000001", "open": 100.0, "high": 101.0,
                            "low": 99.0, "close": 100.0, "volume": 100000, "turnover": 1e9,
                            "foreign_net_qty": 10000, "foreign_net_amount": 1e8,
                            "institution_net_qty": 0, "institution_net_amount": 0})
    quotes = pd.DataFrame({"date": [dates[-2], dates[-1] + pd.Timedelta(days=1)], "symbol": "000001",
                           "current_price": 100.0, "market_cap_eok": [10.0, 10000.0]})
    universe = pd.DataFrame({"symbol": ["000001"], "listed_shares": [99999]})
    feature = build_base_features(history, universe, quotes, point_in_time=True)
    assert np.exp(feature.pit_size.item()) == pytest.approx(1e9)
    assert feature.listed_shares.item() == pytest.approx(1e7)


def test_walk_forward_future_target_perturbation_cannot_change_prior_predictions():
    dates = pd.bdate_range("2026-01-02", periods=40)
    rows = [{"as_of": dates[d], "date_index": d, "symbol": f"{n:06d}",
             "foreign_5_adtv": float(n), "excess5": float(n + d), "label_end_5": dates[d + 5]}
            for d in range(30) for n in range(25)]
    panel = pd.DataFrame(rows)
    features = ("foreign_5_adtv",)
    one = research.walk_forward_predictions(panel, 5, 1, 3, 20, features)
    altered = panel.copy()
    altered.loc[altered.date_index >= 10, "excess5"] = -1e10
    two = research.walk_forward_predictions(altered, 5, 1, 3, 20, features)
    pd.testing.assert_frame_equal(one.loc[one.date_index < 16].reset_index(drop=True),
                                  two.loc[two.date_index < 16].reset_index(drop=True))


def test_research_shadow_export_carries_receipt_and_endpoint_provenance():
    dates = pd.bdate_range("2026-01-02", periods=25)
    frame = pd.DataFrame([{**{name: float(n) for name in RAW_FEATURES}, "as_of": dates[0],
                           "symbol": f"{n:06d}", "discovery_score": 100 - n,
                           "fwd20": 0.02, "excess20": 0.01, "label_end_20": dates[20],
                           "entry_date": dates[1], "pit_provenance_version": PIT_VERSION}
                          for n in range(20)])
    exported = research.build_shadow_training(frame)
    assert len(exported) == 12
    assert exported.pit_provenance_version.eq(PIT_VERSION).all()
    assert exported.label_end_date.eq(dates[20]).all()
    assert set(exported.symbol) == {f"{n:06d}" for n in range(12)}


def test_unobservable_label_cannot_change_mature_targets_rank_in_same_day(tmp_path, monkeypatch):
    sessions = pd.bdate_range("2026-01-02", periods=110)
    rows = [{**{name: float(n) for name in RAW_FEATURES}, "as_of": sessions[d],
             "symbol": f"{n:06d}", "excess_forward_return": float(n),
             "label_end_date": sessions[100] if (d, n) == (60, 11) else sessions[d + 5],
             "pit_provenance_version": PIT_VERSION} for d in range(100) for n in range(12)]
    frame = pd.DataFrame(rows)
    path = tmp_path / "training.csv"
    one, two = Mock(), Mock()
    monkeypatch.setattr(ml_shadow, "_build_shadow_model", Mock(side_effect=[one, two]))
    write_csv(frame, path)
    fit_shadow_model(path, as_of=sessions[70], horizon=5, trading_dates=sessions)
    frame.loc[frame.as_of.eq(sessions[60]) & frame.symbol.eq("000011"), "excess_forward_return"] = -1e10
    write_csv(frame, path)
    fit_shadow_model(path, as_of=sessions[70], horizon=5, trading_dates=sessions)
    np.testing.assert_array_equal(one.fit.call_args.args[1], two.fit.call_args.args[1])


def test_unverified_old_year_cache_is_refetched_not_marked_complete(tmp_path, monkeypatch):
    old = financial("20260310000001", 100)
    old.pop("pit_source_version")
    write_csv(pd.DataFrame([old]), tmp_path / "dart_financials.csv")
    write_csv(pd.DataFrame([filing("20260310000001", "20260310")]), tmp_path / "dart_filings.csv")
    monkeypatch.setattr(fundamentals, "load_corp_codes", lambda *args: pd.DataFrame({"symbol": ["000001"], "corp_code": ["00000001"]}))
    monkeypatch.setattr(fundamentals, "REPORT_CODES", {"11011": "annual"})
    download = Mock(return_value=[{"stock_code": "000001", "bsns_year": "2025", "reprt_code": "11011", "fs_div": "CFS",
                                  "account_nm": "영업이익", "rcept_no": "20260310000001", "thstrm_amount": "777"}])
    monkeypatch.setattr(fundamentals.FinancialsClient, "multi_accounts", download)
    updated = update_financials(Mock(), tmp_path, ["000001"], [2025])
    assert download.call_count == 1
    assert updated.op.item() == 777 and updated.pit_source_version.item() == PIT_VERSION


def test_research_export_to_shadow_fit_runs_with_pit_sources(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text())
    settings = load_settings(tmp_path)
    monkeypatch.setenv("DART_API_KEY", "")
    sessions = pd.bdate_range("2026-01-02", periods=140)
    history = pd.DataFrame([{"date": day, "symbol": f"{n:06d}", "name": f"S{n}", "market": "KOSPI",
                             "open": 100.0 + n, "high": 102.0 + n, "low": 99.0 + n, "close": 100.0 + n,
                             "volume": 100000, "turnover": 2e9, "foreign_net_qty": 10000,
                             "foreign_net_amount": 1e8 * (1 + n / 20), "institution_net_qty": 0, "institution_net_amount": 0}
                            for day in sessions for n in range(12)])
    universe = pd.DataFrame({"symbol": [f"{n:06d}" for n in range(12)], "name": [f"S{n}" for n in range(12)], "market": "KOSPI"})
    write_csv(history, settings.data_dir / "history.csv")
    write_csv(universe, settings.data_dir / "universe.csv")
    write_csv(pd.DataFrame([financial("20251201000001")]), settings.data_dir / "dart_financials.csv")
    write_csv(pd.DataFrame([filing("20251201000001", "20251201")]), settings.data_dir / "dart_filings.csv")
    panel = research.build_research_panel(settings, workers=1)
    assert panel.pit_provenance_version.eq(PIT_VERSION).all()
    assert panel.label_end_20.notna().any()
    table = research.build_shadow_training(panel)
    path = settings.reports_dir / "ml_rich_features.csv"
    write_csv(table, path)
    model = Mock()
    monkeypatch.setattr(ml_shadow, "_build_shadow_model", lambda: model)
    trained = fit_shadow_model(path, as_of=sessions[-1], trading_dates=sessions)
    assert trained.pit_version_ == PIT_VERSION
    assert len(model.fit.call_args.args[0]) >= 500
    assert trained.train_signal_end_ <= sessions[-22]
