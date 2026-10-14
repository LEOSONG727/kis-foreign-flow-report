import numpy as np
import pytest
import pandas as pd

from kis_flow_bot.fundamentals import attach_fundamentals, parse_multi_accounts
from kis_flow_bot.point_in_time import PIT_VERSION, bind_receipts


def _row(account, current, prior, rcept, reprt="11013", fs="CFS", year="2026"):
    return {
        "rcept_no": rcept,
        "bsns_year": year,
        "reprt_code": reprt,
        "stock_code": "000001",
        "fs_div": fs,
        "account_nm": account,
        "thstrm_amount": f"{current:,}",
        "frmtrm_amount": f"{prior:,}",
    }


def test_parse_prefers_consolidated_and_reads_receipt_date():
    rows = [
        _row("영업이익", 300, 100, "20260515000123"),
        _row("매출액", 2000, 1000, "20260515000123"),
        _row("자본총계", 5000, 4000, "20260515000123"),
        _row("영업이익", 999, 1, "20260515000123", fs="OFS"),
    ]
    parsed = parse_multi_accounts(rows, {})
    assert len(parsed) == 1
    row = parsed.iloc[0]
    assert row["op"] == 300 and row["op_prev"] == 100 and row["equity"] == 5000
    assert pd.isna(row["available"])
    filings = pd.DataFrame({"symbol": ["000001"], "rcept_no": ["20260515000123"], "rcept_dt": ["20260516"]})
    assert bind_receipts(parsed, filings)["available"].iloc[0] == pd.Timestamp("2026-05-16")


def test_features_use_only_reports_public_by_each_date():
    financials = pd.DataFrame(
        [
            {"symbol": "000001", "bsns_year": "2025", "reprt_code": "11011", "rcept_no": "20260310000001",
             "available": "20260310", "revenue": 4000, "revenue_prev": 3000, "op": 400, "op_prev": 300,
             "ni": 1, "equity": 1000},
            {"symbol": "000001", "bsns_year": "2026", "reprt_code": "11013", "rcept_no": "20260515000001",
             "available": "20260515", "revenue": 1100, "revenue_prev": 1000, "op": 150, "op_prev": 50,
             "ni": 1, "equity": 1200},
        ]
    )
    mcap = 10_000.0
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-15", "2026-05-18"]),
            "symbol": "000001",
            "pit_size": np.log(mcap),
        }
    )
    financials["pit_source_version"] = PIT_VERSION
    filings = financials[["symbol", "rcept_no", "available"]].rename(columns={"available": "rcept_dt"})
    out = attach_fundamentals(frame, financials, filings=filings)
    assert out["op_yield"].isna().iloc[0]  # nothing public yet
    assert out.loc[1, "op_yoy_mcap"] == pytest.approx((400 - 300) / mcap)  # annual report only
    assert out.loc[2, "book_to_market"] == pytest.approx(1000 / mcap)  # Q1 not public until 5/15
    assert out.loc[3, "op_yoy_mcap"] == pytest.approx((150 - 50) / mcap)
    assert out.loc[3, "op_yield"] == pytest.approx(400 / mcap)  # annual yield still from the annual report
    assert out.loc[3, "report_age"] == pytest.approx(3)
