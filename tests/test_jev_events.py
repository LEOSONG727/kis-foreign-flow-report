from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.dart import update_filings
from kis_flow_bot.jev_events import (
    EVENTS,
    apply_event_review,
    build_state,
    event_section,
    filing_window,
    keyword_flags,
    run_event_replay,
    state_hash,
)
from kis_flow_bot.jev_shadow import MODEL

ROOT = Path(__file__).resolve().parents[1]


def _filings():
    rows = [
        ("000001", "20260301", "주요사항보고서(유상증자결정)", "알파"),
        ("000001", "20260305", "임원ㆍ주요주주특정증권등소유상황보고서", "홍길동"),
        ("000001", "20260320", "알파 횡령ㆍ배임혐의발생", "알파"),  # after the signal date below
        ("000002", "20260302", "주요사항보고서(무상증자결정)", "베타"),
        ("000003", "20260303", "단일판매ㆍ공급계약체결", "감마"),
    ]
    names = {"000001": "알파", "000002": "베타", "000003": "감마"}
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "corp_code": "0" * 8,
                "corp_name": names[symbol],
                "rcept_no": f"{symbol}{day}",
                "rcept_dt": day,
                "report_nm": title,
                "flr_nm": filer,
                "rm": "",
            }
            for symbol, day, title, filer in rows
        ]
    )


def test_window_is_point_in_time_and_state_is_anonymous():
    window = filing_window(_filings(), "000001", pd.Timestamp("2026-03-10"))
    assert len(window) == 2  # the 3/20 filing is not yet public
    state = build_state(window, pd.Timestamp("2026-03-10"))
    text = str(state)
    assert "알파" not in text and "000001" not in text and "2026" not in text
    assert {item["filer"] for item in state["filings"]} == {"the company itself", "another party"}


def test_same_filings_on_consecutive_days_share_one_state():
    filings = _filings()
    a = build_state(filing_window(filings, "000001", pd.Timestamp("2026-03-14")), pd.Timestamp("2026-03-14"))
    b = build_state(filing_window(filings, "000001", pd.Timestamp("2026-03-15")), pd.Timestamp("2026-03-15"))
    assert state_hash(a) == state_hash(b)


def test_keyword_baseline_handles_bonus_issue_and_catalyst():
    filings = _filings()
    assert keyword_flags(filing_window(filings, "000001", pd.Timestamp("2026-03-10")))["dilution"]
    assert not keyword_flags(filing_window(filings, "000002", pd.Timestamp("2026-03-10")))["dilution"]
    assert keyword_flags(filing_window(filings, "000003", pd.Timestamp("2026-03-10")))["catalyst"]
    assert keyword_flags(filing_window(filings, "000001", pd.Timestamp("2026-03-25")))["adverse"]


class FakeDart:
    def __init__(self, filings):
        self.store = filings
        self.calls = 0
        self.requested = []

    def corp_codes(self):
        self.calls += 1
        symbols = sorted(self.store["symbol"].unique())
        return pd.DataFrame({"symbol": symbols, "corp_code": [f"{i:08d}" for i in range(len(symbols))], "corp_name": ["x"] * len(symbols)})

    def filings(self, corp_code, start, end):
        self.calls += 1
        self.requested.append((corp_code, start, end))
        symbol = sorted(self.store["symbol"].unique())[int(corp_code)]
        rows = self.store[self.store["symbol"] == symbol]
        days = pd.to_datetime(rows["rcept_dt"], format="%Y%m%d").dt.date
        return rows[(days >= start) & (days <= end)].drop(columns="symbol").to_dict("records")


def test_filing_cache_skips_covered_ranges(tmp_path):
    client = FakeDart(_filings())
    needs = [("000001", date(2026, 2, 1), date(2026, 3, 31))]
    first = update_filings(client, tmp_path, needs)
    calls = client.calls
    second = update_filings(client, tmp_path, needs)
    assert client.calls == calls
    assert len(first) == len(second) == 3


class EchoKeywordJev:
    """A Jev that simply agrees with the keyword rule."""

    def __init__(self):
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        titles = " ".join(item["title"] for item in kwargs["json"]["state"]["filings"])
        answers = {
            "dilution": 0.9 if "유상증자" in titles and "무상" not in titles else 0.1,
            "adverse": 0.9 if "횡령" in titles else 0.1,
            "catalyst": 0.9 if "공급계약" in titles else 0.1,
        }

        class Response:
            status_code = 200

            def json(self_inner):
                return {"model": MODEL, "answers": {k: {"type": "noul", "noul": v} for k, v in answers.items()}}

        return Response()

    def close(self):
        pass


def _panel(dates=45, names=14, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(dates):
        as_of = pd.Timestamp("2026-03-02") + pd.Timedelta(days=d)
        for n in range(names):
            rows.append(
                {
                    "as_of": as_of,
                    "date_index": d,
                    "symbol": f"{n % 3 + 1:06d}" if n < 3 else f"{100 + n:06d}",
                    "market": "KOSPI",
                    "discovery_score": 100 - n,
                    "timing_score": 50.0,
                    "entry_score": 50.0,
                    "excess5": rng.normal(0, 0.02),
                    "excess20": rng.normal(0, 0.05),
                    "worst20": -abs(rng.normal(0, 0.08)),
                }
            )
    return pd.DataFrame(rows)


def test_replay_caches_answers_and_jev_equal_to_keywords_cannot_win(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    jev = EchoKeywordJev()
    result = run_event_replay(
        settings,
        panel=_panel(),
        dart_client=FakeDart(_filings()),
        jev_key="k",
        session=jev,
    )
    assert result["rows_with_filings"] > 0
    # Distinct filing states, not rows: consecutive days reuse answers.
    assert jev.calls == result["distinct_states"] < result["rows_with_filings"]
    assert result["verdict"] in {"NO_VALUE", "KEYWORDS_ENOUGH"}
    agreement = [row["agreement"] for row in result["flags"] if row["source"] == "kw"]
    assert all(value == 1.0 for value in agreement)
    assert (tmp_path / "reports" / "jev_events.md").exists()
    calls = jev.calls
    run_event_replay(settings, panel=_panel(), dart_client=FakeDart(_filings()), jev_key="k", session=jev)
    assert jev.calls == calls


def test_daily_review_flags_without_reordering(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    scored = _panel(dates=1).assign(as_of=pd.Timestamp("2026-03-10"), name="n")
    result = apply_event_review(scored, settings, dart_client=FakeDart(_filings()), jev_key="k", session=EchoKeywordJev())
    assert list(result["symbol"]) == list(scored["symbol"])
    flagged = result[result["symbol"] == "000001"].iloc[0]
    assert flagged["event_dilution"] >= 0.5 and flagged["kw_dilution"] is True
    assert (tmp_path / "data" / "jev_event_live.csv").exists()
    markdown, html = event_section(result, None)
    assert "희석" in markdown and "검증 전" in markdown


def test_daily_review_without_dart_key_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    scored = _panel(dates=1)
    result = apply_event_review(scored, settings, jev_key="")
    assert result["event_filings"].isna().all()
    assert all(f"event_{name}" in result for name in EVENTS)
