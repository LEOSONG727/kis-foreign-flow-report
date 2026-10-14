from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import kis_flow_bot.earnings_quality as quality

from kis_flow_bot.config import load_settings
from kis_flow_bot.dart import xml_to_text
from kis_flow_bot.earnings_quality import (
    apply_earnings_quality,
    build_state,
    eq_section,
    keyword_flags,
    latest_explanation,
    run_earnings_quality,
)
from kis_flow_bot.jev_shadow import MODEL

ROOT = Path(__file__).resolve().parents[1]
ONE_OFF_TEXT = "당기순이익 증가. 변동 주요원인: 유형자산 처분이익 발생에 따른 영업외수익 증가"
RECURRING_TEXT = "영업이익 증가. 변동 주요원인: 판매량 증가 및 원가 절감에 따른 수익성 개선"


@pytest.fixture
def historical_jev_answers(monkeypatch):
    """Synthetic classifier answers were known before the simulated signals."""
    actual_call = quality.call_jev
    def historical_call(*args, **kwargs):
        ledger = actual_call(*args, **kwargs)
        ledger["captured_at"] = "2026-02-28T00:00:00+00:00"
        return ledger
    monkeypatch.setattr(quality, "call_jev", historical_call)


def _settings(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


def test_xml_to_text_strips_tags_and_entities():
    assert xml_to_text("<P>매출&amp;이익 <SPAN>증가</SPAN></P>") == "매출&이익 증가"


def test_latest_explanation_is_point_in_time():
    filings = pd.DataFrame(
        {
            "symbol": ["000001"] * 3,
            "rcept_no": ["1", "2", "3"],
            "rcept_dt": ["20260301", "20260501", "20260701"],
            "report_nm": ["매출액또는손익구조30%(대규모법인은15%)이상변동", "임원ㆍ주요주주특정증권등소유상황보고서", "연결재무제표기준영업(잠정)실적(공정공시)"],
            "corp_name": ["알파"] * 3,
        }
    )
    assert latest_explanation(filings, "000001", pd.Timestamp("2026-06-01"))["rcept_no"] == "1"
    assert latest_explanation(filings, "000001", pd.Timestamp("2026-07-01"))["rcept_no"] == "1"
    assert latest_explanation(filings, "000001", pd.Timestamp("2026-07-02"))["rcept_no"] == "3"
    assert latest_explanation(filings, "000001", pd.Timestamp("2026-02-01")) is None


def test_state_hides_identity_and_years_and_keywords_split_one_off_from_recurring():
    state = build_state("알파 손익구조 변동", "알파의 2026년 실적 " + ONE_OFF_TEXT + " 005930", "알파")
    text = str(state)
    assert "알파" not in text and "2026" not in text and "005930" not in text
    assert keyword_flags(ONE_OFF_TEXT) == {"one_off": True, "recurring": False}
    assert keyword_flags(RECURRING_TEXT) == {"one_off": False, "recurring": True}


class FakeDart:
    """Symbols 0-9 file an explanation; even ones read as one-off."""

    def __init__(self, symbols):
        self.symbols = symbols
        self.calls = 0
        self.docs = 0

    def corp_codes(self):
        self.calls += 1
        return pd.DataFrame({"symbol": self.symbols, "corp_code": [f"{i:08d}" for i in range(len(self.symbols))], "corp_name": "x"})

    def filings(self, corp_code, start, end):
        self.calls += 1
        index = int(corp_code)
        if index >= 10:
            return []
        return [{"corp_code": corp_code, "corp_name": f"회사{index}", "rcept_no": f"R{index:04d}", "rcept_dt": "20260210",
                 "report_nm": "매출액또는손익구조30%(대규모법인은15%)이상변동", "flr_nm": f"회사{index}", "rm": ""}]

    def document_text(self, rcept_no):
        self.docs += 1
        return ONE_OFF_TEXT if int(rcept_no[1:]) % 2 == 0 else RECURRING_TEXT


class KeywordJev:
    def __init__(self):
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        one_off = "처분이익" in kwargs["json"]["state"]["text"]

        class Response:
            status_code = 200

            def json(self_inner):
                return {"model": MODEL, "answers": {
                    "one_off": {"type": "noul", "noul": 0.9 if one_off else 0.1},
                    "recurring": {"type": "noul", "noul": 0.1 if one_off else 0.9}}}

        return Response()

    def close(self):
        pass


def _panel(dates=45, names=24, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(dates):
        for n in range(names):
            rows.append({
                "as_of": pd.Timestamp("2026-03-02") + pd.Timedelta(days=d),
                "symbol": f"{n:06d}", "market": "KOSPI", "name": f"n{n}",
                "op_yield": 1.0 / (n + 1), "book_to_market": 0.5, "atr20_pct": 0.02, "foreign_20_adtv": 0.1,
                "excess20": rng.normal(0, 0.05) - (0.03 if n < 10 and n % 2 == 0 else 0.0),
                "worst20": -abs(rng.normal(0, 0.08)),
            })
    return pd.DataFrame(rows)


def test_replay_caches_by_text_and_equal_to_keywords_cannot_win(tmp_path, historical_jev_answers):
    settings = _settings(tmp_path)
    panel = _panel()
    dart = FakeDart(sorted(panel["symbol"].unique()))
    jev = KeywordJev()
    result = run_earnings_quality(settings, panel=panel, dart_client=dart, jev_key="k", session=jev)
    assert result["rows_with_text"] > 0
    assert jev.calls == 2  # two distinct texts after redaction, cached across 45 days
    assert dart.docs == 10
    assert result["verdict"] in {"KEYWORDS_ENOUGH", "NO_VALUE"}
    assert (tmp_path / "reports" / "earnings_quality.md").exists()
    run_earnings_quality(settings, panel=panel, dart_client=dart, jev_key="k", session=jev)
    assert jev.calls == 2 and dart.docs == 10


def test_daily_flags_h1_candidates_and_section(tmp_path, historical_jev_answers):
    settings = _settings(tmp_path)
    today = _panel(dates=1).assign(discovery_eligible=True)
    dart = FakeDart(sorted(today["symbol"].unique()))
    flagged = apply_earnings_quality(today, settings, dart_client=dart, jev_key="k", session=KeywordJev())
    assert flagged.loc[flagged["symbol"] == "000000", "eq_one_off"].iloc[0] >= 0.5
    assert flagged.loc[flagged["symbol"] == "000001", "eq_one_off"].iloc[0] < 0.5
    markdown, _ = eq_section(flagged, None)
    assert "일회성" in markdown and "검증 전" in markdown


def test_daily_without_dart_key_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)
    settings = _settings(tmp_path)
    result = apply_earnings_quality(_panel(dates=1), settings, jev_key="")
    assert result["eq_one_off"].isna().all()
