from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import BriefConfig, ModelConfig
from .scoring import is_steady_accumulation


WATCH_TIERS = (
    "즉시매수",
    "추가매수",
    "관심후보",
    "고확신눌림",
    "급락진정",
    "눌림대기",
    "감시",
    "위험확인",
    "제외",
)

TIER_RANK = {tier: index for index, tier in enumerate(WATCH_TIERS)}
TIER_BASE = {
    "즉시매수": 400.0,
    "추가매수": 350.0,
    "관심후보": 300.0,
    "고확신눌림": 240.0,
    "급락진정": 180.0,
    "눌림대기": 160.0,
    "감시": 110.0,
    "위험확인": 80.0,
    "제외": 0.0,
}

BUY_ACTIONS = {"강한 신규매수", "초기 분할매수"}
BUY_CAMPAIGNS = {"지속매집", "장기매집·재가속"}
WATCH_CAMPAIGNS = BUY_CAMPAIGNS | {"신규재진입", "장기매집·휴식", "초기매집"}
ACCUM_STAGES = {"매집·눌림", "초기매집", "매집휴식"}
TERMINAL_CAMPAIGNS = {"분산·이탈", "이벤트·구조확인", "이상대량·확인"}
TERMINAL_STAGES = {"분산·이탈", "이벤트확인", "수익실현"}

PROPOSAL_STATUS_RANK = {
    "오늘 신규매수 검토": 3,
    "강한 후보·가격 대기": 2,
    "관찰": 1,
    "제외": 0,
}


def _float(row: pd.Series | dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else default


def _text(row: pd.Series | dict, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value)


def _bool(row: pd.Series | dict, key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _flow_metric(
    row: pd.Series | dict,
    canonical: str,
    legacy: str,
    default: float = 0.0,
) -> float:
    """Read the corrected flow-path name with one-cycle CSV compatibility."""
    if canonical in row:
        return _float(row, canonical, default)
    return _float(row, legacy, default)


def discovery_grade(
    row: pd.Series | dict,
    brief: BriefConfig | None = None,
) -> str:
    brief = brief or BriefConfig()
    score = _float(row, "discovery_score")
    if score >= brief.high_conviction_score:
        return "A"
    if score >= brief.buy_review_score:
        return "B"
    if score >= brief.min_discovery_score:
        return "C"
    return "제외"


def proposal_status(
    row: pd.Series | dict,
    brief: BriefConfig | None = None,
) -> str:
    """Separate stock quality from whether today's price is buyable."""
    brief = brief or BriefConfig()
    if "discovery_eligible" in row and not _bool(row, "discovery_eligible"):
        return "제외"
    tier = _text(row, "watch_tier")
    grade = discovery_grade(row, brief)
    if tier in {"제외", "위험확인"} or grade == "제외":
        return "제외"
    if tier == "즉시매수" and grade in {"A", "B"}:
        return "오늘 신규매수 검토"
    if grade == "A":
        return "강한 후보·가격 대기"
    return "관찰"


def format_eok(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    amount = value / 100_000_000
    if abs(amount) < 0.05:
        return "0억"
    if abs(amount) < 1.0:
        return f"{amount:+,.1f}억"
    return f"{amount:+,.0f}억"


def format_pct(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value * 100:+.{digits}f}%"


def _scale(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def infer_decision_bucket(row: pd.Series | dict) -> str:
    bucket = _text(row, "decision_bucket")
    if bucket:
        return bucket
    action = _text(row, "action")
    if action in BUY_ACTIONS:
        return "신규매수"
    if "추가매수 검토" in action:
        return "추가매수"
    if any(token in action for token in ("익절", "비중 축소", "매수 금지", "이벤트")):
        return "수익실현·중단"
    if any(token in action for token in ("눌림", "확인", "추격 금지", "급락 멈춤", "재가속", "대기")):
        return "눌림·확인대기"
    return "관찰"


def is_cheap_accumulation(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> bool:
    """Unextended foreign accumulation. Intensity over absolute won amount."""
    model = model or ModelConfig()
    brief = brief or BriefConfig()
    campaign = _text(row, "campaign_regime", "중립")
    stage = _text(row, "stage", "중립")
    if campaign not in WATCH_CAMPAIGNS or stage not in ACCUM_STAGES:
        return False
    if _float(row, "discovery_score") < brief.min_discovery_score:
        return False
    if _text(row, "absorption_signal") == "주의":
        return False
    if any(token in _text(row, "action") for token in ("위험흡수", "이벤트", "매수 금지")):
        return False
    if _float(row, "distribution_score") > model.entry_max_distribution_score:
        return False
    if _float(row, "foreign_20_adtv") < 0.20:
        return False
    if _float(row, "price_return20") >= model.cheap_max_return20_pct / 100.0:
        return False
    if _float(row, "price_return20") <= -0.15:
        return False
    if _float(row, "rebound_from_low20") >= model.cheap_max_rebound20_pct / 100.0:
        return False
    if "ma20_distance" in getattr(row, "index", []) or (
        isinstance(row, dict) and "ma20_distance" in row
    ):
        if _float(row, "ma20_distance") > model.cheap_max_ma20_distance_pct / 100.0:
            return False
    if "distance_high60" in getattr(row, "index", []) or (
        isinstance(row, dict) and "distance_high60" in row
    ):
        if _float(row, "distance_high60") > model.cheap_max_distance_high60_pct / 100.0:
            return False
    return True


def is_steady_price_ok(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> bool:
    """Staircase accumulation that is not too expensive to keep on the buy list.

    This is not the executable buy gate. It asks whether persistent daily
    foreign buying is occurring at a price close enough to the 20-day line
    and not already extended.
    """
    model = model or ModelConfig()
    brief = brief or BriefConfig()
    if not (_bool(row, "steady_accumulation") or is_steady_accumulation(row, model)):
        return False
    if _text(row, "watch_tier") in {"제외", "위험확인"}:
        return False
    if _text(row, "campaign_regime") in TERMINAL_CAMPAIGNS | {"상승진행"}:
        return False
    if _text(row, "stage") in TERMINAL_STAGES:
        return False
    if _text(row, "absorption_signal") == "주의":
        return False
    if _float(row, "distribution_score") > model.entry_max_distribution_score:
        return False
    if _float(row, "price_return3") < brief.watch_min_return3_pct / 100.0:
        return False
    if _float(row, "price_return20") >= brief.watch_max_return20_pct / 100.0:
        return False
    if _float(row, "price_return20") <= -0.12:
        return False
    if _float(row, "rebound_from_low20") >= brief.watch_max_rebound20_pct / 100.0:
        return False
    ma20 = _float(row, "ma20_distance")
    if ma20 > brief.steady_max_ma20_distance_pct / 100.0 and not (
        _bool(row, "ma20_support") or _bool(row, "near_ma20")
    ):
        return False
    return True


def screen_bucket(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> str:
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model, brief)
    if tier == "즉시매수":
        return "즉시매수"
    if _bool(row, "steady_accumulation") or is_steady_accumulation(row, model):
        if is_steady_price_ok(row, model, brief):
            return "차곡차곡가격"
        if _float(row, "price_return1") < 0:
            return "차곡차곡눌림"
        return "차곡차곡"
    if is_cheap_accumulation(row, model, brief):
        return "싼매집"
    if _bool(row, "dip_absorption"):
        return "하락흡수감시" if tier == "감시" else "하락흡수"
    if tier in {"관심후보", "고확신눌림"}:
        return "관찰"
    return tier


def _quality_accumulation(row: pd.Series | dict, model: ModelConfig) -> bool:
    campaign = _text(row, "campaign_regime", "중립")
    stage = _text(row, "stage", "중립")
    return (
        campaign in WATCH_CAMPAIGNS
        and stage in ACCUM_STAGES
        and _float(row, "distribution_score") <= model.entry_max_distribution_score
        and _float(row, "discovery_score") > 0
    )


def classify_watch_tier(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> str:
    model = model or ModelConfig()
    brief = brief or BriefConfig()
    bucket = infer_decision_bucket(row)
    action = _text(row, "action")
    campaign = _text(row, "campaign_regime", "중립")
    stage = _text(row, "stage", "중립")
    readiness = _text(row, "entry_readiness")
    discovery = _float(row, "discovery_score")
    price3 = _float(row, "price_return3")
    price20 = _float(row, "price_return20")
    rebound = _float(row, "rebound_from_low20")

    if "discovery_eligible" in row and not _bool(row, "discovery_eligible"):
        return "제외"

    if bucket == "신규매수" or action in BUY_ACTIONS:
        return "즉시매수"
    if bucket == "추가매수":
        return "추가매수"
    if campaign in TERMINAL_CAMPAIGNS or stage in TERMINAL_STAGES:
        return "제외"
    if discovery < brief.min_discovery_score:
        return "제외"
    if (
        "위험흡수" in action
        or "이벤트" in action
        or "블록" in action
        or (
            _text(row, "absorption_signal") == "주의"
            and price20 <= -0.08
        )
    ):
        return "위험확인"
    # A live campaign buying the dip is the setup, not a crash-to-wait-on.
    if _bool(row, "dip_absorption") and campaign in WATCH_CAMPAIGNS | {"상승진행"}:
        far_above_ma20 = _float(row, "ma20_distance") > model.dip_watch_ma20_distance_pct / 100.0
        if far_above_ma20 and not _bool(row, "ma20_support"):
            return "감시"
        if _bool(row, "ma20_support"):
            return "관심후보"
        if discovery >= brief.high_conviction_score:
            return "고확신눌림"
        return "관심후보"
    if readiness == "반전확인" or "급락 멈춤" in action:
        return "급락진정"
    if readiness == "추세확인":
        return "눌림대기"

    near_entry = (
        _quality_accumulation(row, model)
        and price3 >= brief.watch_min_return3_pct / 100.0
        and price20 < brief.watch_max_return20_pct / 100.0
        and rebound < brief.watch_max_rebound20_pct / 100.0
    )
    if near_entry:
        return "관심후보"
    if discovery >= brief.high_conviction_score and campaign in WATCH_CAMPAIGNS | {"상승진행"}:
        return "고확신눌림"
    if readiness == "눌림대기" or any(token in action for token in ("눌림", "추격 금지", "재개 확인")):
        return "눌림대기"
    if campaign in WATCH_CAMPAIGNS or stage in ACCUM_STAGES:
        return "눌림대기"
    return "제외"


def flow_freshness(row: pd.Series | dict) -> float:
    """Prefer still-alive foreign buying over a stale 20-day total."""
    net1 = _float(row, "foreign_net1")
    net3 = _float(row, "foreign_net3")
    net5 = _float(row, "foreign_net5")
    score = 0.0
    if net3 > 0 and net5 > 0:
        score += 6.0
    elif net3 > 0:
        score += 3.0
    if net1 > 0:
        score += 2.0
    elif net1 < 0 and net3 < 0:
        score -= 6.0
    score += 4.0 * _scale(_float(row, "foreign_3_adtv"), 0.0, 0.40)
    score += 3.0 * _scale(_float(row, "buy_days20"), 6.0, 14.0)
    if _float(row, "current_buy_streak") >= 2:
        score += 2.0
    score -= 8.0 * _scale(_float(row, "sellback_ratio20"), 0.40, 1.00)
    score += 3.0 * _scale(_float(row, "foreign_acceleration3"), -0.03, 0.10)
    score += 3.0 * _scale(_float(row, "foreign_20_adtv"), 0.30, 2.00)
    return score


def _ma_slope_tag(row: pd.Series | dict) -> str:
    if _bool(row, "ma20_rising"):
        return "20일선 우상향"
    if _bool(row, "ma20_falling"):
        return "20일선 우하향"
    return ""


def _divergence_tag(row: pd.Series | dict) -> str:
    value = _text(row, "flow_price_divergence")
    if value == "bullish":
        return "가격↓수급↑"
    if value == "bearish":
        return "가격↑수급↓"
    return ""


def flow_summary(row: pd.Series | dict) -> str:
    parts = [
        f'외인 20일 {format_eok(_float(row, "foreign_net20"))}',
        f'3일 {format_eok(_float(row, "foreign_net3"))}',
        f'당일 {format_eok(_float(row, "foreign_net1"))}',
    ]
    if _bool(row, "large_flow_signal"):
        parts.append("500억+")
    if _bool(row, "steady_accumulation") or is_steady_accumulation(row):
        up_days = int(_flow_metric(row, "foreign_buy_days10", "foreign_roll20_up_days10"))
        if up_days > 0:
            parts.append(f"최근 10일 중 외인 순매수 {up_days}일")
        else:
            parts.append("차곡차곡")
    if _bool(row, "dip_absorption"):
        parts.append("하락흡수")
    slope_tag = _ma_slope_tag(row)
    if slope_tag:
        parts.append(slope_tag)
    div_tag = _divergence_tag(row)
    if div_tag:
        parts.append(div_tag)
    if _bool(row, "touched_ma20"):
        parts.append("20일선 접촉")
    elif _bool(row, "near_ma20") or _bool(row, "ma20_support"):
        parts.append("20일선 근접")
    campaign = _text(row, "campaign_regime", "중립")
    if campaign:
        parts.append(campaign)
    buy_days = _float(row, "buy_days20")
    if buy_days > 0:
        parts.append(f"매수 {int(buy_days)}일/20일")
    return " · ".join(parts)


def block_reason(row: pd.Series | dict, model: ModelConfig | None = None) -> str:
    model = model or ModelConfig()
    if "discovery_eligible" in row and not _bool(row, "discovery_eligible"):
        return _text(row, "eligibility_reason", "후보 데이터 조건 미충족")
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model)
    if tier == "즉시매수":
        return f"{flow_summary(row)} · 즉시매수 게이트 통과"
    if tier == "추가매수":
        return f"{flow_summary(row)} · 추가매수 조건 충족"

    bits: list[str] = []
    price1 = _float(row, "price_return1")
    price3 = _float(row, "price_return3")
    price20 = _float(row, "price_return20")
    rebound = _float(row, "rebound_from_low20")
    floor3 = model.entry_min_return3_pct / 100.0
    cap20 = model.entry_max_return20_pct / 100.0
    rebound_cap = model.entry_max_rebound20_pct / 100.0
    if _bool(row, "dip_absorption"):
        bits.append(
            f'당일 {format_pct(price1)}에 외인 {format_eok(_float(row, "foreign_net1"))} 흡수'
        )
    if _bool(row, "touched_ma20") or _bool(row, "ma20_support"):
        bits.append(f'20일선 {format_pct(_float(row, "ma20_distance"))} · 접촉')
    elif _float(row, "ma20_distance") > 0.01:
        bits.append(f'20일선 {format_pct(_float(row, "ma20_distance"))} · 접촉 전')
    if _bool(row, "ma20_falling") and _text(row, "flow_price_divergence") != "bullish":
        bits.append("20일선 우하향 · 우상향 또는 가격↓수급↑ 필요")
    if _text(row, "flow_price_divergence") == "bullish":
        bits.append("가격↓수급↑ 다이버전스")
    if price3 < floor3:
        bits.append(f"3일 {format_pct(price3)} · {model.entry_min_return3_pct:+.0f}% 이상 필요")
    if price20 >= cap20:
        bits.append(f"20일 {format_pct(price20)} · +{model.entry_max_return20_pct:.0f}% 미만 필요")
    if rebound >= rebound_cap:
        bits.append(f"저점반등 {format_pct(rebound)} · +{model.entry_max_rebound20_pct:.0f}% 미만 필요")
    if _text(row, "absorption_signal") == "주의":
        bits.append("급락+외인매수 구조 미확인")
    if _float(row, "distribution_score") > model.entry_max_distribution_score:
        bits.append(f'분산 {_float(row, "distribution_score"):.0f}점 · {model.entry_max_distribution_score:.0f}점 이하 필요')
    if not bits:
        gate = _text(row, "entry_gate_reason")
        if gate and not _bool(row, "entry_gate_passed"):
            bits.append(gate)
        elif _text(row, "action"):
            bits.append(_text(row, "action"))
    return " · ".join(bits[:3]) if bits else "추가 확인 필요"


def next_trigger(row: pd.Series | dict, model: ModelConfig | None = None) -> str:
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model)
    falling_without_div = (
        _bool(row, "ma20_falling") and _text(row, "flow_price_divergence") != "bullish"
    )
    if falling_without_div:
        if _bool(row, "dip_absorption"):
            return "20일선 우상향 회복 + 하락일 외인 매수 유지"
        return "20일선 우상향 회복 또는 가격↓수급↑ 다이버전스"
    if _bool(row, "steady_accumulation") or is_steady_accumulation(row):
        if _bool(row, "entry_gate_passed"):
            return "차곡차곡 유지되면 분할 신규 검토"
        if is_steady_price_ok(row):
            return (
                "20일 +3% 미만·저점반등 +5% 미만·3일 -3% 이상 + "
                "20일선 접촉 + 당일 외인 순매수 시 분할"
            )
        if _float(row, "price_return1") < 0:
            return "20일선까지 더 눌리면서 외인 순매수가 이어지는지"
        return "가격 과열 진정 + 외인 순매수 지속"
    if _bool(row, "dip_absorption"):
        if not _bool(row, "ma20_support"):
            return "20일선 접촉 + 하락일 외인 매수 유지"
        if tier == "고확신눌림":
            return "20일선 지지 + 하락일 외인 매수 유지 + 과열 진정 시 분할 검토"
        return "20일선 지지 유지 후 즉시매수 게이트 재확인"
    promotion = _text(row, "promotion_condition")
    if promotion and promotion != "현재 조건 충족":
        return promotion
    if tier == "즉시매수":
        return "현재 조건 충족 · 분할 신규 검토"
    if tier == "관심후보":
        return "소폭 눌림 후 즉시매수 게이트 재확인"
    if tier == "고확신눌림":
        return "가격 과열 진정 + 외인 20일 순매수 유지"
    if tier == "급락진정":
        return "3일 급락 진정 후 매집 신호 유지 확인"
    if tier == "감시":
        return "20일선 접근 후 싼 매집 조건 재확인"
    if tier == "위험확인":
        return "가격구조·이벤트 확인 전에는 관망"
    return promotion or "매집 신호 유지 여부 확인"


def watch_priority(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> float:
    model = model or ModelConfig()
    brief = brief or BriefConfig()
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model, brief)
    rebound_over = max(0.0, _float(row, "rebound_from_low20") - model.entry_max_rebound20_pct / 100.0)
    return_over = max(0.0, _float(row, "price_return20") - model.entry_max_return20_pct / 100.0)
    fall_over = max(0.0, model.entry_min_return3_pct / 100.0 - _float(row, "price_return3"))
    closeness = 20.0 - 140.0 * rebound_over - 90.0 * return_over - 80.0 * fall_over
    knife = 0.0
    price20 = _float(row, "price_return20")
    if price20 < -0.10:
        knife = 30.0 * min(1.0, (-price20 - 0.10) / 0.10)
    return (
        TIER_BASE.get(tier, 0.0)
        + _float(row, "discovery_score")
        + flow_freshness(row)
        + (2.0 if _bool(row, "large_flow_signal") else 0.0)
        + (8.0 if _bool(row, "dip_absorption") and _bool(row, "ma20_support") else 0.0)
        + (8.0 if _text(row, "flow_price_divergence") == "bullish" and _bool(row, "ma20_support") else 0.0)
        + (4.0 if _bool(row, "ma20_rising") else 0.0)
        + (6.0 if _bool(row, "ma20_support") else 0.0)
        + (12.0 if _bool(row, "steady_accumulation") or is_steady_accumulation(row, model) else 0.0)
        + (
            10.0
            if (
                (_bool(row, "steady_accumulation") or is_steady_accumulation(row, model))
                and _float(row, "price_return1") < 0
            )
            else 0.0
        )
        + 8.0 * _scale(
            _flow_metric(row, "foreign_buy_days10", "foreign_roll20_up_days10"),
            6.0,
            10.0,
        )
        + 6.0 * _scale(_float(row, "foreign_20_adtv"), 0.25, 2.0)
        + closeness
        - knife
        - (12.0 if _bool(row, "ma20_falling") and _text(row, "flow_price_divergence") != "bullish" else 0.0)
        - (
            6.0
            if _bool(row, "ma_fan_up") and _float(row, "price_return20") >= 0.03
            else 0.0
        )
    )


def annotate_watch_brief(
    scored: pd.DataFrame,
    model: ModelConfig | None = None,
    brief: BriefConfig | None = None,
) -> pd.DataFrame:
    """Add daily-brief columns without changing buy/sell action gates."""
    model = model or ModelConfig()
    brief = brief or BriefConfig()
    if scored.empty:
        return scored
    frame = scored.copy()
    tiers: list[str] = []
    for _, row in frame.iterrows():
        tiers.append(classify_watch_tier(row, model, brief))
    frame["watch_tier"] = tiers
    frame["high_conviction"] = frame["discovery_score"].fillna(0.0) >= brief.high_conviction_score
    frame["flow_summary"] = [_summary_from_row(row) for _, row in frame.iterrows()]
    frame["flow_freshness"] = [flow_freshness(row) for _, row in frame.iterrows()]
    frame["block_reason"] = [block_reason(row, model) for _, row in frame.iterrows()]
    frame["next_trigger"] = [next_trigger(row, model) for _, row in frame.iterrows()]
    frame["watch_priority"] = [watch_priority(row, model, brief) for _, row in frame.iterrows()]
    frame["cheap_accumulation"] = [is_cheap_accumulation(row, model, brief) for _, row in frame.iterrows()]
    if "steady_accumulation" not in frame:
        frame["steady_accumulation"] = [is_steady_accumulation(row, model) for _, row in frame.iterrows()]
    else:
        missing = frame["steady_accumulation"].isna()
        if missing.any():
            frame.loc[missing, "steady_accumulation"] = [
                is_steady_accumulation(row, model) for _, row in frame.loc[missing].iterrows()
            ]
        frame["steady_accumulation"] = frame["steady_accumulation"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
    frame["steady_price_ok"] = [is_steady_price_ok(row, model, brief) for _, row in frame.iterrows()]
    frame["screen_bucket"] = [screen_bucket(row, model, brief) for _, row in frame.iterrows()]
    frame["discovery_grade"] = [discovery_grade(row, brief) for _, row in frame.iterrows()]
    frame["proposal_status"] = [proposal_status(row, brief) for _, row in frame.iterrows()]
    frame["proposal_priority"] = [
        1_000.0 * PROPOSAL_STATUS_RANK.get(status, 0) + priority
        for status, priority in zip(frame["proposal_status"], frame["watch_priority"])
    ]
    selected = set(select_daily_brief(frame, brief).get("symbol", pd.Series(dtype="string")).astype(str).str.zfill(6))
    frame["in_daily_brief"] = frame["symbol"].astype(str).str.zfill(6).isin(selected)
    return frame


def _summary_from_row(row: pd.Series) -> str:
    return flow_summary(row)


def select_daily_brief(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
) -> pd.DataFrame:
    """Pick a short, mixed list a human can review tonight."""
    brief = brief or BriefConfig()
    if scored.empty:
        return scored
    frame = scored.copy()
    if "watch_tier" not in frame:
        frame = annotate_watch_brief(frame, brief=brief)
    if "flow_freshness" not in frame:
        frame["flow_freshness"] = [flow_freshness(row) for _, row in frame.iterrows()]
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    picked: list[pd.DataFrame] = []

    def take(tier: str, limit: int) -> None:
        remaining = brief.size - sum(len(part) for part in picked)
        if remaining <= 0 or limit <= 0:
            return
        used = set(pd.concat(picked)["symbol"]) if picked else set()
        pool = frame[(frame["watch_tier"] == tier) & (~frame["symbol"].isin(used))]
        if pool.empty:
            return
        pool = pool.sort_values(
            ["watch_priority", "flow_freshness", "discovery_score"],
            ascending=False,
        ).head(min(limit, remaining))
        if not pool.empty:
            picked.append(pool)

    def take_mask(mask: pd.Series, limit: int, sort_by: list[str] | None = None) -> None:
        remaining = brief.size - sum(len(part) for part in picked)
        if remaining <= 0 or limit <= 0:
            return
        used = set(pd.concat(picked)["symbol"]) if picked else set()
        pool = frame[mask & (~frame["symbol"].isin(used))]
        if pool.empty:
            return
        columns = [
            column
            for column in (sort_by or ["watch_priority", "flow_freshness", "discovery_score"])
            if column in pool.columns
        ]
        if not columns:
            columns = ["discovery_score"] if "discovery_score" in pool.columns else []
        if columns:
            pool = pool.sort_values(columns, ascending=False)
        pool = pool.head(min(limit, remaining))
        if not pool.empty:
            picked.append(pool)

    cheap = frame.apply(lambda row: is_cheap_accumulation(row, brief=brief), axis=1)
    if "steady_accumulation" in frame:
        steady = frame["steady_accumulation"].map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        steady = frame.apply(lambda row: is_steady_accumulation(row), axis=1)
    price_down = pd.to_numeric(frame.get("price_return1", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0) < 0
    if "dip_absorption" in frame:
        dip = frame["dip_absorption"].map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        dip = pd.Series(False, index=frame.index)
    # Buyable names lead. Within every setup, prefer relative intensity and
    # persistence; the 500억 absolute amount remains a badge, not a rank key.
    if "steady_price_ok" in frame:
        priced = frame["steady_price_ok"].map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        priced = frame.apply(lambda row: is_steady_price_ok(row), axis=1)
    steady_sort = [
        "watch_priority",
        "flow_freshness",
        "foreign_20_adtv",
        "foreign_buy_days10",
        "discovery_score",
    ]
    if "proposal_status" in frame:
        take_mask(frame["proposal_status"] == "오늘 신규매수 검토", brief.size)
        take_mask(
            frame["proposal_status"] == "강한 후보·가격 대기",
            max(1, brief.leftover_slots),
        )
    take_mask(steady & priced, brief.steady_priced_slots, steady_sort)
    take_mask(steady & price_down & ~priced, brief.steady_slots, steady_sort)
    take_mask(steady, brief.steady_slots, steady_sort)
    take_mask(cheap & ~steady, brief.cheap_slots)
    selected_dips = 0
    if picked and "dip_absorption" in pd.concat(picked).columns:
        selected_dips = int(
            pd.concat(picked)["dip_absorption"].map(
                lambda value: bool(value) if pd.notna(value) else False
            ).sum()
        )
    take_mask(dip & ~cheap & ~steady, max(0, brief.dip_slots - selected_dips))
    take("즉시매수", brief.leftover_slots)
    take("추가매수", 1)
    take("관심후보", brief.leftover_slots)
    take("고확신눌림", brief.leftover_slots)
    take("급락진정", 1)
    take("눌림대기", 1)
    take("감시", 1)
    if sum(len(part) for part in picked) < min(3, brief.size):
        take("위험확인", 1)
    take("관심후보", brief.size)
    take("고확신눌림", brief.size)

    if not picked:
        fallback = frame[frame["watch_tier"] != "제외"].sort_values(
            ["watch_priority", "flow_freshness", "discovery_score"], ascending=False
        ).head(brief.size)
        return fallback.reset_index(drop=True)
    return (
        pd.concat(picked)
        .drop_duplicates("symbol")
        .head(brief.size)
        .reset_index(drop=True)
    )


def iter_brief_rows(scored: pd.DataFrame, brief: BriefConfig | None = None) -> Iterable[pd.Series]:
    selected = scored[scored.get("in_daily_brief", False)] if "in_daily_brief" in scored else pd.DataFrame()
    if selected.empty:
        selected = select_daily_brief(scored, brief)
    else:
        sort_cols = [
            column
            for column in ("proposal_priority", "watch_priority", "flow_freshness", "foreign_20_adtv")
            if column in selected.columns
        ]
        if sort_cols:
            selected = selected.sort_values(sort_cols, ascending=False)
    for _, row in selected.iterrows():
        yield row
