from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import BriefConfig, ModelConfig


WATCH_TIERS = (
    "즉시매수",
    "추가매수",
    "관심후보",
    "고확신눌림",
    "급락진정",
    "눌림대기",
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
    "위험확인": 80.0,
    "제외": 0.0,
}

BUY_ACTIONS = {"강한 신규매수", "초기 분할매수"}
BUY_CAMPAIGNS = {"지속매집", "장기매집·재가속"}
WATCH_CAMPAIGNS = BUY_CAMPAIGNS | {"신규재진입", "장기매집·휴식", "초기매집"}
ACCUM_STAGES = {"매집·눌림", "초기매집", "매집휴식"}
TERMINAL_CAMPAIGNS = {"분산·이탈", "이벤트·구조확인", "이상대량·확인"}
TERMINAL_STAGES = {"분산·이탈", "이벤트확인", "수익실현"}


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
    if readiness == "반전확인" or "급락 멈춤" in action:
        return "급락진정"

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


def flow_summary(row: pd.Series | dict) -> str:
    parts = [f'외인 20일 {format_eok(_float(row, "foreign_net20"))}']
    if _bool(row, "large_flow_signal"):
        parts.append("500억+")
    campaign = _text(row, "campaign_regime", "중립")
    if campaign:
        parts.append(campaign)
    buy_days = _float(row, "buy_days20")
    if buy_days > 0:
        parts.append(f"매수 {int(buy_days)}일/20일")
    return " · ".join(parts)


def block_reason(row: pd.Series | dict, model: ModelConfig | None = None) -> str:
    model = model or ModelConfig()
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model)
    if tier == "즉시매수":
        return f"{flow_summary(row)} · 즉시매수 게이트 통과"
    if tier == "추가매수":
        return f"{flow_summary(row)} · 추가매수 조건 충족"

    bits: list[str] = []
    price3 = _float(row, "price_return3")
    price20 = _float(row, "price_return20")
    rebound = _float(row, "rebound_from_low20")
    floor3 = model.entry_min_return3_pct / 100.0
    cap20 = model.entry_max_return20_pct / 100.0
    rebound_cap = model.entry_max_rebound20_pct / 100.0
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
    promotion = _text(row, "promotion_condition")
    if promotion and promotion != "현재 조건 충족":
        return promotion
    tier = _text(row, "watch_tier") or classify_watch_tier(row, model)
    if tier == "즉시매수":
        return "현재 조건 충족 · 분할 신규 검토"
    if tier == "관심후보":
        return "소폭 눌림 후 즉시매수 게이트 재확인"
    if tier == "고확신눌림":
        return "가격 과열 진정 + 외인 20일 순매수 유지"
    if tier == "급락진정":
        return "3일 급락 진정 후 매집 신호 유지 확인"
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
        + (8.0 if _bool(row, "large_flow_signal") else 0.0)
        + closeness
        - knife
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
    frame["block_reason"] = [block_reason(row, model) for _, row in frame.iterrows()]
    frame["next_trigger"] = [next_trigger(row, model) for _, row in frame.iterrows()]
    frame["watch_priority"] = [watch_priority(row, model, brief) for _, row in frame.iterrows()]
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
            ["watch_priority", "discovery_score"],
            ascending=False,
        ).head(min(limit, remaining))
        if not pool.empty:
            picked.append(pool)

    take("즉시매수", 3)
    take("추가매수", 2)
    take("관심후보", brief.size)
    take("고확신눌림", brief.size)
    take("급락진정", 1)
    take("눌림대기", brief.size)
    if sum(len(part) for part in picked) < min(3, brief.size):
        take("위험확인", 1)

    if not picked:
        fallback = frame[frame["watch_tier"] != "제외"].sort_values(
            ["watch_priority", "discovery_score"], ascending=False
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
        selected = selected.sort_values(["watch_priority", "discovery_score"], ascending=False)
    for _, row in selected.iterrows():
        yield row
