from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from kis_flow_bot import fundamentals
from kis_flow_bot.brief import annotate_watch_brief, inventory_summary, select_daily_brief
from kis_flow_bot.dart import DartClient, DartError, DartUnavailable, update_filings
from kis_flow_bot.features import build_base_features
from kis_flow_bot.notify import build_headline
from kis_flow_bot.config import BriefConfig
from kis_flow_bot.pipeline import _resolve_investor_query_date, _validate_scan_freshness
from kis_flow_bot.storage import read_csv, write_csv
from test_brief import _row
from test_features import _history
from test_point_in_time import financial, filing


def client_with(*responses, **kwargs):
    session = Mock()
    session.get.side_effect = responses
    return DartClient("private-key", session=session, pause=lambda _: None, min_interval=0, **kwargs)


def test_dart_transient_connection_is_retried_and_recovers():
    response = SimpleNamespace(status_code=200)
    client = client_with(requests.ConnectionError("private-key"), response)
    assert client._get("test", {}) is response
    assert client.calls == 2 and client.consecutive_failures == 0


def test_dart_outage_has_bounded_network_calls_and_redacts_error():
    client = client_with(*[requests.Timeout("url?crtfc_key=private-key")] * 20)
    for _ in range(2):
        with pytest.raises(DartError, match="Timeout") as error:
            client._get("test", {})
        assert "private-key" not in str(error.value)
    with pytest.raises(DartUnavailable):
        client._get("test", {})
    assert client.calls == 9
    with pytest.raises(DartUnavailable):
        client._get("another-endpoint", {})
    assert client.calls == 9


@pytest.mark.parametrize("status", [401, 403, 429])
def test_dart_auth_or_rate_limit_defers_remaining_calls(status):
    client = client_with(SimpleNamespace(status_code=status))
    with pytest.raises(DartUnavailable):
        client._get("test", {})
    assert client.calls == 1


def test_dart_server_failure_streak_resets_on_success():
    bad, good = SimpleNamespace(status_code=503), SimpleNamespace(status_code=200)
    client = client_with(bad, good, bad, good, max_attempts=1, failure_limit=2)
    for _ in range(2):
        with pytest.raises(DartError):
            client._get("test", {})
        assert client._get("test", {}) is good
    assert not client.unavailable


def test_dart_json_daily_quota_opens_circuit():
    response = Mock(status_code=200)
    response.json.return_value = {"status": "020", "message": "daily quota"}
    client = client_with(response)
    with pytest.raises(DartUnavailable):
        fundamentals.FinancialsClient(client).multi_accounts(["00000001"], 2025, "11011")
    with pytest.raises(DartUnavailable):
        client._get("list.json", {})
    assert client.calls == 1


def test_financial_outage_saves_partial_success_without_filling_unknowns(tmp_path, monkeypatch):
    codes = pd.DataFrame({"symbol": [f"{i:06}" for i in range(1, 202)],
                          "corp_code": [f"{i:08}" for i in range(1, 202)]})
    monkeypatch.setattr(fundamentals, "load_corp_codes", lambda *args: codes)
    monkeypatch.setattr(fundamentals, "REPORT_CODES", {"11011": "annual", "11013": "Q1"})
    raw = [{"stock_code": "000001", "bsns_year": "2025", "reprt_code": "11011", "fs_div": "CFS",
            "rcept_no": "20260310000001", "account_nm": "영업이익", "thstrm_amount": "100"}]
    calls = Mock(side_effect=[raw, DartUnavailable("outage")])
    monkeypatch.setattr(fundamentals.FinancialsClient, "multi_accounts", calls)
    write_csv(pd.DataFrame([filing("20260310000001", "20260310")]), tmp_path / "dart_filings.csv")
    result = fundamentals.update_financials(Mock(), tmp_path, codes.symbol, [2025, 2026])
    assert calls.call_count == 2
    assert list(result.symbol) == ["000001"]
    assert result.op.iloc[0] == 100
    assert read_csv(tmp_path / "dart_financials.csv").op.iloc[0] == 100


def test_partial_financial_cache_still_fetches_missing_companies(tmp_path, monkeypatch):
    write_csv(pd.DataFrame([financial()]), tmp_path / "dart_financials.csv")
    codes = pd.DataFrame({"symbol": ["000001", "000002"], "corp_code": ["00000001", "00000002"]})
    monkeypatch.setattr(fundamentals, "load_corp_codes", lambda *args: codes)
    monkeypatch.setattr(fundamentals, "REPORT_CODES", {"11011": "annual"})
    calls = Mock(return_value=[])
    monkeypatch.setattr(fundamentals.FinancialsClient, "multi_accounts", calls)
    fundamentals.update_financials(Mock(), tmp_path, codes.symbol, [2025])
    assert calls.call_args.args[0] == ["00000002"]


def test_financial_collection_handles_no_cache_and_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(fundamentals, "load_corp_codes", lambda *args: pd.DataFrame({"symbol": ["000001"], "corp_code": ["00000001"]}))
    monkeypatch.setattr(fundamentals.FinancialsClient, "multi_accounts", lambda *args: [])
    assert fundamentals.update_financials(Mock(), tmp_path, ["000001"], [2025]).empty


def test_filing_outage_does_not_advance_coverage(tmp_path):
    write_csv(pd.DataFrame({"symbol": ["000001"], "corp_code": ["00000001"]}), tmp_path / "dart_corp_codes.csv")
    client = Mock(calls=0)
    client.filings.side_effect = DartUnavailable("outage")
    assert update_filings(client, tmp_path, [("000001", date(2025, 1, 1), date(2026, 1, 1))]).empty
    assert (tmp_path / "dart_coverage.json").read_text().strip() == "{}"


def test_probe_uses_actual_trading_date_not_accepted_calendar_date():
    client = Mock()
    client.investor_daily.return_value = [{"stck_bsop_date": "20261016"}]
    assert _resolve_investor_query_date(client, date(2026, 10, 18)) == date(2026, 10, 16)


def test_probe_rejects_future_date():
    client = Mock()
    client.investor_daily.return_value = [{"stck_bsop_date": "20261019"}]
    with pytest.raises(RuntimeError, match="future"):
        _resolve_investor_query_date(client, date(2026, 10, 18))


def test_probe_rejects_outdated_session():
    client = Mock()
    client.investor_daily.return_value = [{"stck_bsop_date": "20260901"}]
    with pytest.raises(RuntimeError, match="outdated"):
        _resolve_investor_query_date(client, date(2026, 10, 18))


def test_scan_requires_distinct_current_symbols_not_nonempty_old_responses():
    universe = pd.DataFrame({"symbol": ["000001", "000002", "000003"]})
    rows = pd.DataFrame({"symbol": ["000001"] * 10 + ["000002", "000003"],
                         "date": ["2026-10-16"] * 10 + ["2026-10-15"] * 2})
    with pytest.raises(RuntimeError, match="fresh=1/3"):
        _validate_scan_freshness(rows, universe, date(2026, 10, 16))
    rows.loc[10:, "date"] = "2026-10-16"
    _validate_scan_freshness(rows, universe, date(2026, 10, 16))


def test_rejected_steady_and_risky_dips_never_fill_shortlist():
    rows = pd.DataFrame([
        _row(symbol="000001", name="stale", steady_accumulation=True, discovery_eligible=False),
        _row(symbol="000002", name="risk", steady_accumulation=True, dip_absorption=True,
             discovery_score=90, action="위험흡수 구조 확인 대기", price_return20=-0.14, discovery_eligible=True),
        _row(symbol="000003", name="valid", discovery_eligible=True),
    ])
    annotated = annotate_watch_brief(rows)
    assert list(select_daily_brief(annotated).symbol) == ["000003"]
    assert annotated.set_index("symbol").in_daily_brief.to_dict() == {"000001": False, "000002": False, "000003": True}
    assert [x.name for x in build_headline(annotated).wait_items] == ["valid"]


def test_empty_shortlist_is_valid_when_every_candidate_is_rejected():
    frame = annotate_watch_brief(pd.DataFrame([_row(steady_accumulation=True, discovery_eligible=False)]))
    assert select_daily_brief(frame).empty
    assert not build_headline(frame).items and not build_headline(frame).wait_items


def test_headline_counts_full_buy_pool_even_when_shortlist_is_smaller():
    rows = [_row(symbol=f"{i:06}", name=f"buy{i}", action="초기 분할매수",
                 decision_bucket="신규매수", discovery_score=90) for i in range(6)]
    brief = BriefConfig(size=1)
    frame = annotate_watch_brief(pd.DataFrame(rows), brief=brief)
    headline = build_headline(frame, brief)
    assert "6개" in headline.verdict
    assert len(headline.items) == 3


def test_stale_saved_shortlist_flag_cannot_reintroduce_rejected_row():
    frame = annotate_watch_brief(pd.DataFrame([_row(steady_accumulation=True, discovery_eligible=False)]))
    frame["in_daily_brief"] = True
    assert not build_headline(frame).items and not build_headline(frame).wait_items


def test_reentry_discloses_long_selling_and_incomplete_windows():
    row = _row(campaign_regime="신규재진입", data_points=120,
               foreign_net60=-61_400_000_000, foreign_net120=-86_860_000_000)
    assert "60일 -614억" in inventory_summary(row)
    assert "장기 순매도 잔존" in inventory_summary(row)
    frame = annotate_watch_brief(pd.DataFrame([row]))
    assert "장기 순매도 잔존" in frame.flow_summary.iloc[0]
    row["data_points"] = 60
    assert "120일 이력 미확보" in inventory_summary(row)


def test_ten_day_flow_uses_exactly_ten_rows_and_does_not_mutate_inputs():
    history = _history("000001", "test", 1_000_000, 100, 100)
    before = history.copy(deep=True)
    universe = pd.DataFrame({"symbol": ["000001"]})
    features = build_base_features(history, universe)
    assert features.foreign_net10.iloc[0] == 10_000_000
    pd.testing.assert_frame_equal(history, before)
