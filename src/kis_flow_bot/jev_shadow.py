"""Optional Jev shadow judgments. Never changes ranking or trading decisions."""
from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
QUESTIONS = {
    "fragile_flow": {"type": "noul", "instructions": "Is apparent foreign accumulation fragile, concentrated in recent days, or offset by sellback or inventory reversals? Judge only supplied fields."},
    "price_trap": {"type": "noul", "instructions": "Does sustained price weakness despite foreign buying look more like an unresolved falling-price trap than benign underreaction? Judge only supplied fields."},
    "chase_risk": {"type": "noul", "instructions": "Does extension from moving averages or a sharp rebound make a new entry vulnerable to chasing? Judge only supplied fields."},
    "review": {"type": "choice", "instructions": "Choose an independent research queue without predicting returns or issuing a buy instruction.", "criteria": {
        "DEEP_REVIEW": "No obvious false-positive pattern; investigate business, filings and valuation separately.",
        "WATCH": "Some ambiguity; await fresher flow or price confirmation.",
        "REJECT": "A material false-positive pattern warrants investigation.",
    }},
}
FIELDS = (
    "foreign_net1", "foreign_net3", "foreign_net5", "foreign_net20", "foreign_net60",
    "foreign_1_adtv", "foreign_5_adtv", "foreign_20_adtv",
    "foreign_buy_days10", "foreign_buy_days20", "foreign_inventory_up_days10",
    "foreign_inventory_down_days10", "foreign_flow_concentration20",
    "sellback_ratio20", "foreign_acceleration3", "foreign_acceleration5",
    "price_return1", "price_return3", "price_return5", "price_return20",
    "price_residual20", "market_return20", "rebound_from_low20",
    "ma20_distance", "atr20_pct", "volume_ratio5_20",
)
OUTPUTS = ("jev_fragile_flow", "jev_price_trap", "jev_chase_risk",
           "jev_review", "jev_review_confidence", "jev_model", "jev_error")


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _state(row):
    return {
        "units": "returns and ratios are in the bot's native units; foreign_net values are KRW",
        "market": str(row.get("market", "")),
        "features": {field: _number(row.get(field)) for field in FIELDS},
    }


def _parse(payload):
    answers = payload["answers"]
    result = {}
    for key in ("fragile_flow", "price_trap", "chase_risk"):
        value = _number(answers[key]["noul"])
        if value is None or not 0 <= value <= 1:
            raise ValueError(f"Invalid Jev {key} response")
        result[f"jev_{key}"] = value
    review = answers["review"]
    if review["choice"] not in ("DEEP_REVIEW", "WATCH", "REJECT"):
        raise ValueError("Invalid Jev review choice")
    result["jev_review"] = review["choice"]
    result["jev_review_confidence"] = _number(review.get("confidence"))
    result["jev_model"] = str(payload.get("model", MODEL))
    return result


def apply_jev_shadow(scored: pd.DataFrame, *, api_key: str | None = None, limit: int = 20, session=None, pause=time.sleep) -> pd.DataFrame:
    """Annotate at most limit top rows; failure never blocks the daily report."""
    result = scored.copy()
    for col in OUTPUTS:
        result[col] = None
    key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")
    if not key or result.empty:
        LOGGER.info("Jev shadow skipped (key absent or no candidates)")
        return result
    http = session or requests.Session()
    for index in result.head(limit).index:
        row = result.loc[index]
        symbol = str(row.get("symbol", ""))
        body = {"model": MODEL, "state": _state(row), "questions": QUESTIONS}
        for attempt in range(3):
            try:
                response = http.post(ENDPOINT, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=12)
                if response.status_code in (429, 529) and attempt < 2:
                    pause(2 ** attempt)
                    continue
                response.raise_for_status()
                parsed = _parse(response.json())
                for col, value in parsed.items():
                    result.at[index, col] = value
                break
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                # Avoid logging response bodies or auth headers.
                LOGGER.warning("Jev shadow failed for %s (%s)", symbol, type(exc).__name__)
                result.at[index, "jev_error"] = type(exc).__name__
                break
    return result
