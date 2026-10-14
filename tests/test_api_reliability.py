from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import requests

from kis_flow_bot.config import APIConfig
from kis_flow_bot import kis_client as api
from kis_flow_bot.kis_client import KISAPIError, KISClient


def response(status=200, payload=None, headers=None):
    result = Mock(status_code=status, ok=200 <= status < 300, headers=headers or {})
    result.json.side_effect = ValueError("html") if payload is None else None
    result.json.return_value = payload
    result.text = "never-log-this-secret-response"
    return result


@pytest.fixture
def clock(monkeypatch):
    now = [1_800_000_000.0]
    sleeps = []
    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay
    # Replace this module's clock only; never alter pytest/threading clocks.
    monkeypatch.setattr(api, "time", SimpleNamespace(time=lambda: now[0],
                                                   monotonic=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(api.random, "uniform", lambda low, high: high)
    return now, sleeps


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("KIS_TOKEN_CACHE_DIR", raising=False)
    api._LIMITERS.clear()
    return KISClient("key", "secret", APIConfig(token_cache_dir=str(tmp_path)))


def token(value="token"):
    return response(payload={"access_token": value, "expires_in": 86400})


def test_token_reused_across_clients_and_credentials_isolated(client, clock):
    client.session.post = Mock(return_value=token())
    assert client.authenticate() == "token"
    other = KISClient("key", "secret", client.config)
    other.session.post = Mock(side_effect=AssertionError("Should reuse disk cache"))
    assert other.authenticate() == "token"
    assert other.limiter is client.limiter
    rotated = KISClient("key", "rotated-secret", client.config)
    assert rotated._cache_path != client._cache_path
    if os.name != "nt":
        assert client._cache_path.stat().st_mode & 0o777 == 0o600
    assert "secret" not in client._cache_path.read_text()


def test_token_rate_limit_enforces_minute_between_attempts(client, clock, caplog):
    now, sleeps = clock
    issued = []
    replies = iter([response(429), token()])
    def post(*args, **kwargs):
        issued.append(now[0])
        return next(replies)
    client.session.post = post
    assert client.authenticate() == "token"
    assert issued[1] - issued[0] >= 65
    assert "never-log" not in caplog.text


def test_network_auth_failure_retries_and_permanent_failure_is_bounded(client, clock):
    client.session.post = Mock(side_effect=[requests.Timeout(), token()])
    assert client.authenticate() == "token"
    client.session.post = Mock(return_value=response(400, {"error_code": "BAD_KEY"}))
    with pytest.raises(KISAPIError, match="BAD_KEY"):
        client.authenticate(force=True)
    assert client.session.post.call_count == 1


def test_refresh_reuses_another_clients_new_token(client, clock):
    other = KISClient("key", "secret", client.config)
    client.session.post = Mock(side_effect=[token("old"), token("new")])
    assert client.authenticate() == other.authenticate() == "old"
    assert client.authenticate(force=True) == "new"
    other.session.post = Mock(side_effect=AssertionError("Already refreshed"))
    assert other.authenticate(force=True) == "new"
    assert client.session.post.call_count == 2


def test_simultaneous_token_issuance_is_serialized(client):
    other = KISClient("key", "secret", client.config)
    post = Mock(return_value=token())
    client.session.post = other.session.post = post
    barrier = threading.Barrier(2)
    def issue(subject):
        barrier.wait(timeout=5)
        return subject.authenticate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(issue, [client, other])) == ["token", "token"]
    assert post.call_count == 1


def test_non_json_429_honors_retry_after_without_reissuing(client, clock):
    client.session.post = Mock(return_value=token())
    client.session.get = Mock(side_effect=[response(429, headers={"Retry-After": "120"}),
                                          response(payload={"rt_cd": "0", "output": {}})])
    assert client.get("/quotes", "ID", {})["rt_cd"] == "0"
    assert 120 in clock[1]
    assert client.session.post.call_count == 1


def test_non_json_401_refreshes_and_http_date_retry_after_is_supported(client, clock):
    client.session.post = Mock(side_effect=[token("old"), token("new")])
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    client.session.get = Mock(side_effect=[response(401), response(503, headers={"Retry-After": future}),
                                          response(payload={"rt_cd": "0"})])
    client.get("/quotes", "ID", {})
    assert client.session.post.call_count == 2
    assert any(110 < delay <= 120 for delay in clock[1])


def test_auth_retry_budget_is_not_multiplied_by_get(client, clock):
    client.session.post = Mock(return_value=response(429))
    client.session.get = Mock(side_effect=AssertionError("No token, no GET"))
    with pytest.raises(KISAPIError, match="exhausted retries"):
        client.get("/quotes", "ID", {})
    assert client.session.post.call_count == client.config.max_retries + 1


def test_expired_get_refreshes_once_then_stops_at_retry_budget(client, clock):
    client.session.post = Mock(side_effect=[token("old"), token("new")])
    client.session.get = Mock(return_value=response(401, {"msg_cd": "EGW00123", "rt_cd": "1"}))
    with pytest.raises(KISAPIError):
        client.get("/quotes", "ID", {})
    assert client.session.post.call_count == 2
    assert client.session.get.call_count == client.config.max_retries + 1


def test_get_backoff_and_nontransient_error(client, clock):
    client.session.post = Mock(return_value=token())
    client.session.get = Mock(side_effect=[response(503), response(200, {"rt_cd": "1", "msg_cd": "EGW00201"}),
                                          response(payload={"rt_cd": "0"})])
    client.get("/quotes", "ID", {})
    assert 1.5 in clock[1] and 3.0 in clock[1]
    client.session.get = Mock(return_value=response(400, {"rt_cd": "1", "msg_cd": "INVALID_INPUT"}))
    with pytest.raises(KISAPIError):
        client.get("/quotes", "ID", {})
    assert client.session.get.call_count == 1


def test_corrupt_cache_and_zero_retry_budget(client, clock):
    client._cache_path.write_text("broken-json")
    client.session.post = Mock(return_value=token())
    assert client.authenticate() == "token"
    assert json.loads(client._cache_path.read_text())["value"] == "token"
    client.config = replace(client.config, max_retries=0)
    client.session.get = Mock(return_value=response(429))
    with pytest.raises(KISAPIError):
        client.get("/quotes", "ID", {})
    assert client.session.get.call_count == 1
