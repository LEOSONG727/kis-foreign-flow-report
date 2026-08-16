from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from .config import APIConfig

LOGGER = logging.getLogger(__name__)


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
        if not app_key or not app_secret:
            raise ValueError("KIS_APP_KEY and KIS_APP_SECRET are required")
        self.app_key = app_key.strip()
        self.app_secret = app_secret.strip()
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kis-foreign-flow-report/0.1"})
        self.limiter = RateLimiter(config.calls_per_second)
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

        response = self.session.post(
            f"{self.BASE_URL}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
            timeout=self.config.timeout_seconds,
        )
        if not response.ok:
            raise KISAPIError(
                f"KIS token issuance failed ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        value = payload.get("access_token")
        if not value:
            raise KISAPIError(f"KIS token response did not contain access_token: {payload}")
        expires_in = int(payload.get("expires_in", 86_400))
        self._token = Token(
            value=value,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        )
        return value

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
            or code in {"EGW00201", "EGW00123"}
            or "초당 거래건수" in text
            or "rate" in text
            or "tempor" in text
        )

    def get(self, path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(
                    f"{self.BASE_URL}{path}",
                    headers=self._headers(tr_id),
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise KISAPIError(
                        f"Non-JSON KIS response ({response.status_code}): {response.text[:300]}",
                        transient=response.status_code >= 500,
                    ) from exc

                code = str(payload.get("msg_cd", ""))
                message = str(payload.get("msg1", ""))
                if response.ok and str(payload.get("rt_cd", "0")) == "0":
                    return payload

                transient = self._is_transient(code, message, response.status_code)
                if code == "EGW00123" and attempt == 0:
                    self.authenticate(force=True)
                    continue
                raise KISAPIError(
                    f"KIS call failed {path} [{code}]: {message or response.text[:300]}",
                    code=code,
                    transient=transient,
                )
            except (requests.RequestException, KISAPIError) as exc:
                last_error = exc
                transient = isinstance(exc, requests.RequestException) or getattr(exc, "transient", False)
                if attempt >= self.config.max_retries or not transient:
                    raise
                delay = min(30.0, 1.5 * (2**attempt))
                LOGGER.warning("Transient KIS error; retrying in %.1fs: %s", delay, exc)
                time.sleep(delay)
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

