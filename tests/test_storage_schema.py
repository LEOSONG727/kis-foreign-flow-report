from __future__ import annotations

import pandas as pd
import pytest

from kis_flow_bot.storage import read_csv, write_csv, normalize_identifiers
from kis_flow_bot.portfolio import load_portfolio


def test_csv_roundtrip_preserves_ids_numbers_booleans_and_missing(tmp_path):
    path = tmp_path / "dart.csv"
    path.write_text("symbol,corp_code,rcept_no,rcept_dt,amount,resolved\n"
                    "005930,00126380,20261014001234,20261014,1.5,True\n"
                    "660,264738,,,2.0,False\n"
                    ",,,,,False\n")
    frame = read_csv(path, dtype={"amount": float, "corp_code": int})
    assert list(frame.symbol[:2]) == ["005930", "000660"]
    assert list(frame.corp_code[:2]) == ["00126380", "00264738"]
    assert frame.rcept_no[0] == "20261014001234"
    assert frame.rcept_dt[0] == "20261014"
    assert pd.isna(frame.symbol[2]) and pd.isna(frame.corp_code[2])
    assert frame.amount.dtype.kind == "f" and frame.resolved.dtype.kind == "b"
    write_csv(frame, path)
    pd.testing.assert_frame_equal(frame, read_csv(path))
    assert path.read_text().splitlines()[1].startswith("005930,00126380,20261014001234")


def test_normalize_numeric_float_ids_keeps_alphanumeric_and_missing():
    raw = pd.DataFrame({"symbol": [5930.0, None, " A ", ""],
                        "corp_code": [126380.0, pd.NA, "TEST", ""]})
    normalized = normalize_identifiers(raw)
    assert normalized.symbol[0] == "005930"
    assert normalized.corp_code[0] == "00126380"
    assert normalized.symbol[2] == "A" and normalized.corp_code[2] == "TEST"
    assert normalized.symbol.iloc[[1, 3]].isna().all()
    assert raw.symbol[0] == 5930.0


def test_scalar_string_dtype_and_numeric_dtype_guard(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("symbol,close\n005930,100.0\n")
    assert read_csv(path, dtype=str).symbol.item() == "005930"
    with pytest.raises(ValueError):
        read_csv(path, dtype=float)
    with pytest.raises(ValueError):
        read_csv(path, converters={"symbol": int})


def test_atomic_write_failure_retains_original(tmp_path, monkeypatch):
    path = tmp_path / "history.csv"
    path.write_text("original\n")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError):
        write_csv(pd.DataFrame({"symbol": ["005930"]}), path)
    assert path.read_text() == "original\n"
    assert len(list(tmp_path.iterdir())) == 1


def test_portfolio_identifiers_use_shared_schema(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("symbol,quantity,average_price\n5930,5,100\n")
    assert load_portfolio(path).symbol.item() == "005930"
    assert load_portfolio(path).quantity.item() == 5
