from __future__ import annotations

from datetime import date

import pytest

from kis_flow_bot.kis_client import KISAPIError
from kis_flow_bot.pipeline import _resolve_investor_query_date


class DateProbeClient:
    def __init__(self, accepted: date):
        self.accepted = accepted
        self.calls: list[date] = []

    def investor_daily(self, symbol: str, as_of: date):
        self.calls.append(as_of)
        if as_of > self.accepted:
            raise KISAPIError("TIME LIMIT", code="OPSQ2001")
        return [{"stck_bsop_date": self.accepted.strftime("%Y%m%d")}]


def test_investor_query_date_falls_back_across_holiday():
    client = DateProbeClient(date(2026, 8, 16))

    resolved = _resolve_investor_query_date(client, date(2026, 8, 17))

    assert resolved == date(2026, 8, 16)
    assert client.calls == [date(2026, 8, 17), date(2026, 8, 16)]


def test_investor_query_date_does_not_hide_other_api_errors():
    class BrokenClient:
        def investor_daily(self, symbol: str, as_of: date):
            raise KISAPIError("authentication failed", code="EGW00123")

    with pytest.raises(KISAPIError, match="authentication failed"):
        _resolve_investor_query_date(BrokenClient(), date(2026, 8, 17))
