import pandas as pd
import pytest

from kis_flow_bot.jev_shadow import MODEL, _parse, _state, apply_jev_shadow


def candidate():
    return dict(symbol="005930", discovery_score=80, foreign_net5=1e8, foreign_net20=4e8,
                foreign_buy_days10=8, sellback_ratio20=.1, foreign_20_adtv=.5,
                price_return5=.01, price_return20=.02, market_return20=.03,
                ma20_distance=.01, rebound_from_low20=.04, atr20_pct=.02)


class Response:
    def __init__(self, status=200, risk=.2):
        self.status_code = status
        self.risk = risk
    def json(self):
        return {"model": MODEL, "answers": {key: {"type": "noul", "noul": self.risk}
                for key in ("fragile_flow", "price_trap", "chase_risk")}}


class Session:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or [Response()]
    def post(self, url, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls)-1, len(self.responses)-1)]


def test_explicit_units_and_no_model_anchor():
    state = _state(candidate())
    assert state["price"]["return_20_sessions"] == "+2.00%"
    assert state["flow"]["net_buy_5_sessions"] == "+1.00 KRW hundred-million"
    assert "discovery_score" not in str(state)
    assert "005930" not in str(state)


def test_shadow_is_independent_and_limited():
    frame = pd.DataFrame([candidate(), {**candidate(), "symbol": "068270"}])
    original = frame.copy(deep=True)
    session = Session()
    result = apply_jev_shadow(frame, api_key="test", limit=1, session=session)
    pd.testing.assert_frame_equal(frame, original)
    assert result.loc[0, "jev_review"] == "DEEP_REVIEW"
    assert result.loc[1, "jev_status"] == "NOT_SELECTED"
    assert len(session.calls) == 1
    assert session.calls[0]["json"]["model"] == MODEL


def test_missing_data_abstains_and_missing_key_skips():
    session = Session()
    frame = pd.DataFrame([{"symbol": "000001"}])
    assert apply_jev_shadow(frame, api_key="test", session=session).loc[0, "jev_status"] == "INSUFFICIENT_DATA"
    assert apply_jev_shadow(frame, api_key="", session=session).loc[0, "jev_status"] == "DISABLED"
    assert not session.calls


def test_auth_error_opens_circuit_and_transient_error_retries():
    frame = pd.DataFrame([candidate(), candidate()])
    session = Session([Response(401)])
    result = apply_jev_shadow(frame, api_key="test", session=session)
    assert len(session.calls) == 1
    assert result.loc[1, "jev_status"] == "CIRCUIT_OPEN"
    session = Session([Response(429), Response()])
    result = apply_jev_shadow(frame.head(1), api_key="test", session=session, pause=lambda _: None)
    assert result.loc[0, "jev_status"] == "OK"
    assert len(session.calls) == 2


def test_high_risk_cannot_be_averaged_away_and_malformed_response_rejected():
    payload = Response().json()
    payload["answers"]["price_trap"]["noul"] = .9
    assert _parse(payload)["jev_review"] == "REJECT"
    payload["answers"]["price_trap"]["noul"] = True
    with pytest.raises(ValueError):
        _parse(payload)
    payload["model"] = "unexpected"
    with pytest.raises(ValueError):
        _parse(payload)


def test_ineligible_rows_not_sent():
    frame = pd.DataFrame([{**candidate(), "discovery_eligible": False},
                          {**candidate(), "data_fresh": False, "discovery_eligible": True}])
    session = Session()
    apply_jev_shadow(frame, api_key="test", session=session)
    assert not session.calls


def test_report_and_csv_accept_new_diagnostics(tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from kis_flow_bot import pipeline
    frame = pd.DataFrame([{**candidate(), "name": "Example", "as_of": "2026-09-01"}])
    scored = apply_jev_shadow(frame, api_key="test", session=Session())
    monkeypatch.setattr(pipeline, "render_report", lambda *a, **k:
        SimpleNamespace(html="<main>report</main>", markdown="report"))
    settings = SimpleNamespace(screening=SimpleNamespace(top_n=3), watchlist=(),
                               brief=None, model=None, reports_dir=tmp_path)
    output = pipeline._write_outputs(scored, settings, datetime(2026, 9, 1))
    assert "Example" in output.read_text()
    assert "DEEP_REVIEW" in output.read_text()
    saved = pd.read_csv(tmp_path / "latest.csv")
    assert "jev_prompt_hash" in saved
    assert "jev_input_hash" in saved
    assert saved.columns.is_unique
