"""Independent pattern judgments; arithmetic and routing stay in Python."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
POLICY = "jev-diagnostics-v2"
LIMIT = 12  # Same discovery shortlist as the existing ML shadow.
QUESTIONS = {
    "fragile_flow": {
        "type": "noul",
        "instructions": "Using only flow evidence, does the recent buying pattern undermine the thesis of sustained foreign accumulation?",
        "criteria": {"true": "Buying looks episodic or is being substantially reversed.", "false": "Buying is persistent with little reversal; do not infer future returns."},
    },
    "price_trap": {
        "type": "noul",
        "instructions": "Using only price and flow evidence, does persistent price deterioration undermine the thesis of benign price underreaction?",
        "criteria": {"true": "Ongoing weakness despite buying gives little evidence of stabilization.", "false": "Price is stable or recovering while flow persists. Flat price alone is not deterioration."},
    },
    "chase_risk": {
        "type": "noul",
        "instructions": "Using only entry evidence, does the price pattern indicate an extended new entry?",
        "criteria": {"true": "A sharp rebound or sustained extension makes the entry look stretched.", "false": "Price is near its recent trend with limited rebound. Do not infer future returns."},
    },
}
PROMPT_HASH = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:16]
OUTPUTS = (
    "jev_fragile_flow", "jev_price_trap", "jev_chase_risk", "jev_review",
    "jev_model", "jev_error", "jev_status", "jev_policy", "jev_prompt_hash",
    "jev_input_hash", "jev_state_json", "jev_captured_at", "jev_input_tokens",
)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def select_candidates(scored, limit=LIMIT):
    eligible = pd.Series(True, index=scored.index)
    for col in ("discovery_eligible", "data_fresh"):
        if col in scored:
            eligible &= scored[col].fillna(False).eq(True)
    cols = [c for c in ("discovery_score", "timing_score", "entry_score") if c in scored]
    selected = scored.loc[eligible]
    if cols:
        selected = selected.sort_values(cols, ascending=False, kind="stable")
    return selected.head(min(max(int(limit), 0), LIMIT))


def _state(row):
    # Code performs unit conversions. Missing observations stay explicit.
    def pct(key):
        value = _number(row.get(key))
        return "unknown" if value is None else f"{value * 100:+.2f}%"

    def amount(key):
        value = _number(row.get(key))
        return "unknown" if value is None else f"{value / 100_000_000:+.2f} KRW hundred-million"

    def count(key, total):
        value = _number(row.get(key))
        return "unknown" if value is None else f"{value:g} of {total} trading sessions"

    def ratio(key):
        value = _number(row.get(key))
        return "unknown" if value is None else f"{value:.3f} times"

    return {
        "scope": "Observed end-of-day pattern only. No business, news or future outcome evidence. Unknown is missing, not zero.",
        "flow": {
            "net_buy_5_sessions": amount("foreign_net5"),
            "net_buy_20_sessions": amount("foreign_net20"),
            "buying_sessions": count("foreign_buy_days10", 10),
            "sellback_fraction": pct("sellback_ratio20"),
            "net20_relative_to_daily_turnover": ratio("foreign_20_adtv"),
        },
        "price": {
            "return_5_sessions": pct("price_return5"),
            "return_20_sessions": pct("price_return20"),
            "market_return_20_sessions": pct("market_return20"),
        },
        "entry": {
            "distance_from_20_session_average": pct("ma20_distance"),
            "rebound_from_20_session_low": pct("rebound_from_low20"),
            "daily_atr_fraction_of_price": pct("atr20_pct"),
        },
    }


REQUIRED = ("foreign_net5", "foreign_net20", "foreign_buy_days10", "sellback_ratio20",
            "price_return5", "price_return20", "ma20_distance", "rebound_from_low20", "atr20_pct")


def _parse(payload):
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise ValueError("Unexpected Jev model")
    answers = payload["answers"]
    result = {}
    for key in QUESTIONS:
        answer = answers[key]
        value = _number(answer.get("noul"))
        if answer.get("type") != "noul" or value is None or not 0 <= value <= 1:
            raise ValueError("Invalid Jev response")
        result[f"jev_{key}"] = value
    # Experimental policy, not calibrated profit probabilities.
    worst = max(result.values())
    result["jev_review"] = "REJECT" if worst >= .65 else "WATCH" if worst > .35 else "DEEP_REVIEW"
    result["jev_model"] = payload["model"]
    result["jev_input_tokens"] = (payload.get("usage") or {}).get("input_tokens")
    return result


def apply_jev_shadow(scored, *, api_key=None, limit=LIMIT, session=None, pause=time.sleep):
    result = scored.copy()
    for col in OUTPUTS:
        result[col] = None
    key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")
    result["jev_status"] = "NOT_SELECTED"
    if not key:
        result["jev_status"] = "DISABLED"
        return result
    http = session or requests.Session()
    circuit_open = False
    for index, row in select_candidates(result, limit).iterrows():
        state = _state(row)
        encoded = json.dumps(state, sort_keys=True, ensure_ascii=False)
        for col, value in {
            "jev_policy": POLICY, "jev_prompt_hash": PROMPT_HASH,
            "jev_state_json": encoded, "jev_input_hash": hashlib.sha256(encoded.encode()).hexdigest(),
            "jev_captured_at": datetime.now(timezone.utc).isoformat(),
        }.items():
            result.at[index, col] = value
        if circuit_open:
            result.at[index, "jev_status"] = "CIRCUIT_OPEN"
            continue
        if any(_number(row.get(field)) is None for field in REQUIRED):
            result.at[index, "jev_status"] = "INSUFFICIENT_DATA"
            continue
        result.at[index, "jev_status"] = "API_ERROR"
        for attempt in range(3):
            try:
                response = http.post(ENDPOINT, headers={"Authorization": f"Bearer {key}"},
                    json={"model": MODEL, "state": state, "questions": QUESTIONS},
                    timeout=(5, 12), allow_redirects=False)
                if response.status_code in (401, 403, 422):
                    circuit_open = True
                if response.status_code in (429, 500, 502, 503, 504, 529) and attempt < 2:
                    pause(2 ** attempt)
                    continue
                if not 200 <= response.status_code < 300:
                    raise ValueError(f"HTTP_{response.status_code}")
                for col, value in _parse(response.json()).items():
                    result.at[index, col] = value
                result.at[index, "jev_status"] = "OK"
                break
            except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
                LOGGER.warning("Jev unavailable for %s (%s)", row.get("symbol"), type(exc).__name__)
                result.at[index, "jev_error"] = type(exc).__name__
                break
    if session is None:
        http.close()
    return result
