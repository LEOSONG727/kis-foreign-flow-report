from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from .config import APIConfig
from .token_cache import locked_cache, read_cache, write_cache

LOGGER = logging.getLogger(__name__)
_LIMITERS: dict[str, "RateLimiter"] = {}
_LIMITERS_LOCK = threading.Lock()


class KISAPIError(RuntimeError):
    def __init__(self, message: str, code: str = "", transient: bool = False):
        super().__init__(message)
        self.code = code
        self.transient = transient


@dataclass
class Token:
    value: str
    expires_at: datetime

    @property
    def valid(self) -> bool:
        return datetime.now(timezone.utc) < self.expires_at - timedelta(minutes=5)


class RateLimiter:
    def __init__(self, calls_per_second: float):
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")
        self.interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.interval - (now - self._last_call)
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()


class KISClient:
    BASE_URL = "https://openapi.koreainvestment.com:9443"

    def __init__(self, app_key: str, app_secret: str, config: APIConfig):
        if not app_key.strip() or not app_secret.strip():
            raise ValueError("KIS_APP_KEY and KIS_APP_SECRET are required")
        self.app_key = app_key.strip()
        self.app_secret = app_secret.strip()
        self.config = config
        if config.max_retries < 0 or config.retry_base_seconds <= 0 or config.retry_max_seconds <= 0:
            raise ValueError("API retries must be nonnegative and retry delays positive")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kis-foreign-flow-report/0.1"})
        identity = hashlib.sha256(
            f"{self.BASE_URL}\0{self.app_key}\0{self.app_secret}".encode()
        ).hexdigest()
        cache_dir = os.getenv("KIS_TOKEN_CACHE_DIR", config.token_cache_dir)
        self._cache_path = Path(cache_dir).expanduser() / f"{identity}.json"
        with _LIMITERS_LOCK:
            if identity not in _LIMITERS:
                _LIMITERS[identity] = RateLimiter(config.calls_per_second)
            self.limiter = _LIMITERS[identity]
            if config.calls_per_second <= 0:
                raise ValueError("calls_per_second must be positive")
            self.limiter.interval = max(self.limiter.interval, 1.0 / config.calls_per_second)
        self._token: Token | None = None

    @classmethod
    def from_env(cls, config: APIConfig) -> "KISClient":
        return cls(
            os.getenv("KIS_APP_KEY", ""),
            os.getenv("KIS_APP_SECRET", ""),
            config,
        )

    def authenticate(self, force: bool = False) -> str:
        if not force and self._token and self._token.valid:
            return self._token.value
        rejected_value = self._token.value if force and self._token else None
        # Lock spans the cooldown and issuance, so other processes recheck the
        # cache after the first issuer finishes rather than issuing in parallel.
        with locked_cache(self._cache_path):
            state = read_cache(self._cache_path)
            try:
                if not isinstance(state["value"], str) or not state["value"]:
                    raise ValueError("Invalid cached token")
                cached = Token(str(state["value"]), datetime.fromisoformat(state["expires_at"]))
                if cached.valid and (not force or (rejected_value is not None and cached.value != rejected_value)):
                    self._token = cached
                    return cached.value
            except (KeyError, TypeError, ValueError):
                pass
            self._token = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    last_issue = float(state.get("last_issue_at", 0))
                except (TypeError, ValueError):
                    last_issue = 0.0
                delay = max(60.0, self.config.token_min_interval_seconds) - (time.time() - last_issue)
                if delay > 0:
                    time.sleep(delay)
                state = {"last_issue_at": time.time()}
                # Keep the cooldown even if the request fails or is interrupted.
                write_cache(self._cache_path, state)
                response = None
                try:
                    response = self.session.post(
                        f"{self.BASE_URL}/oauth2/tokenP",
                        json={"grant_type": "client_credentials", "appkey": self.app_key,
                              "appsecret": self.app_secret},
                        timeout=self.config.timeout_seconds,
                    )
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    if not isinstance(payload, dict):
                        payload = {}
                    code = str(payload.get("error_code", payload.get("msg_cd", "")))
                    message = str(payload.get("error_description", payload.get("msg1", "")))
                    if not response.ok or not payload.get("access_token"):
                        raise KISAPIError(
                            f"KIS token issuance failed (HTTP {response.status_code}, code={code})",
                            code=code,
                            transient=self._is_transient(code, message, response.status_code),
                        )
                    expires_in = int(payload.get("expires_in", 86_400))
                    if expires_in <= 0:
                        raise KISAPIError("KIS returned a nonpositive token lifetime")
                    self._token = Token(str(payload["access_token"]),
                        datetime.now(timezone.utc) + timedelta(seconds=expires_in))
                    state.update(value=self._token.value, expires_at=self._token.expires_at.isoformat())
                    write_cache(self._cache_path, state)
                    return self._token.value
                except (requests.RequestException, KISAPIError) as exc:
                    transient = isinstance(exc, requests.RequestException) or exc.transient
                    if not transient or attempt >= self.config.max_retries:
                        # Auth has already consumed its own budget; GET must not
                        # multiply it or log a response containing credentials.
                        raise KISAPIError("KIS token issuance exhausted retries" if transient else str(exc),
                                          code=getattr(exc, "code", "")) from exc
                    self._sleep_retry(attempt, response)
        raise KISAPIError("KIS token issuance exhausted retries")

    def _sleep_retry(self, attempt: int, response=None) -> None:
        base = min(self.config.retry_max_seconds, self.config.retry_base_seconds * 2**attempt)
        delay = random.uniform(base * 0.75, base)
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                try:
                    wait = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    wait = 0.0
            delay = max(delay, wait)
        LOGGER.warning("Transient KIS error; retry %d in %.1fs", attempt + 1, delay)
        time.sleep(delay)

    def _headers(self, tr_id: str, tr_cont: str = "") -> dict[str, str]:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.authenticate()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if tr_cont:
            headers["tr_cont"] = tr_cont
        return headers

    @staticmethod
    def _is_transient(code: str, message: str, status: int) -> bool:
        text = f"{code} {message}".lower()
        return (
            status in {408, 425, 429, 500, 502, 503, 504}
            or code in {"EGW00201", "EGW00123", "EGW00133"}
            or "초당 거래건수" in text
            or "1분당" in text
            or "1 minute" in text
            or "rate" in text
            or "tempor" in text
        )

    def get(self, path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        refreshed = False
        for attempt in range(self.config.max_retries + 1):
            response = None
            try:
                headers = self._headers(tr_id)
                self.limiter.wait()
                response = self.session.get(
                    f"{self.BASE_URL}{path}",
                    headers=headers,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 401 and not refreshed and attempt < self.config.max_retries:
                    refreshed = True
                    self.authenticate(force=True)
                    continue
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise KISAPIError(
                        f"Non-JSON KIS response ({response.status_code})",
                        transient=self._is_transient("", "", response.status_code),
                    ) from exc
                if not isinstance(payload, dict):
                    raise KISAPIError("KIS response must be a JSON object",
                        transient=self._is_transient("", "", response.status_code))

                code = str(payload.get("msg_cd", ""))
                message = str(payload.get("msg1", ""))
                if response.ok and str(payload.get("rt_cd", "0")) == "0":
                    return payload

                transient = self._is_transient(code, message, response.status_code)
                if code == "EGW00123" and not refreshed and attempt < self.config.max_retries:
                    refreshed = True
                    self.authenticate(force=True)
                    continue
                raise KISAPIError(
                    f"KIS call failed {path} (HTTP {response.status_code}, code={code})",
                    code=code,
                    transient=transient,
                )
            except (requests.RequestException, KISAPIError) as exc:
                last_error = exc
                transient = isinstance(exc, requests.RequestException) or getattr(exc, "transient", False)
                if attempt >= self.config.max_retries or not transient:
                    raise
                self._sleep_retry(attempt, response)
        raise KISAPIError(f"KIS request exhausted retries: {last_error}")

    def investor_daily(self, symbol: str, as_of: date) -> list[dict[str, Any]]:
        payload = self.get(
            "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
            "FHPTJ04160001",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": as_of.strftime("%Y%m%d"),
                "FID_ORG_ADJ_PRC": "",
                "FID_ETC_CLS_CODE": "",
            },
        )
        output = payload.get("output2") or []
        return output if isinstance(output, list) else [output]

    def current_price(self, symbol: str) -> dict[str, Any]:
        payload = self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        output = payload.get("output") or {}
        return output if isinstance(output, dict) else {}

    def invest_opinion(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        payload = self.get(
            "/uapi/domestic-stock/v1/quotations/invest-opinion",
            "FHKST663300C0",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "16633",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
            },
        )
        output = payload.get("output") or []
        return output if isinstance(output, list) else [output]

