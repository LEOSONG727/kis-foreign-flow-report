"""Mid- and long-horizon boards that do not churn with the daily price gate.

The daily brief answers "is the price buyable *today*?". This module answers
"what should I actually own for weeks or months?", and explains yesterday's
tactical dropout so a name falling off the buy list is not read as a sell.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .brief import (
    _bool,
    _float,
    _text,
    discovery_grade,
    format_eok,
    format_pct,
    select_daily_brief,
)
from .config import BriefConfig, HorizonConfig, ModelConfig


TERMINAL_CAMPAIGNS = {"분산·이탈", "이벤트·구조확인", "이상대량·확인"}
TERMINAL_STAGES = {"분산·이탈", "이벤트확인", "수익실현"}
MID_CAMPAIGNS = {"지속매집", "장기매집·재가속", "장기매집·휴식", "신규재진입"}
LONG_CAMPAIGNS = MID_CAMPAIGNS | {"초기매집"}


@dataclass(frozen=True)
class HorizonName:
    symbol: str
    name: str
    role: str
    score: float
    discovery_score: float
    grade: str
    proposal_status: str
    campaign: str
    stage: str
    since: str
    streak_days: int
    why: str
    hold_rule: str
    foreign_net20: float
    foreign_net60: float
    foreign_net120: float
    price_return20: float
    rebound_from_low20: float
    buy_days10: int
    appearance_days: int
    is_new: bool
    today_buyable: bool


@dataclass(frozen=True)
class ContinuityNote:
    kind: str
    symbol: str
    name: str
    text: str


@dataclass(frozen=True)
class HorizonBoard:
    as_of: str
    verdict_lines: tuple[str, ...]
    mid_core: tuple[HorizonName, ...]
    long_core: tuple[HorizonName, ...]
    continuity: tuple[ContinuityNote, ...]
    yesterday_tactical: tuple[dict[str, str], ...] = ()
    decision_lines: tuple[str, ...] = ()

    def to_state(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "mid_core": [
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "since": item.since,
                    "score": round(item.score, 2),
                }
                for item in self.mid_core
            ],
            "long_core": [
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "since": item.since,
                    "score": round(item.score, 2),
                }
                for item in self.long_core
            ],
            "tactical": list(self.yesterday_tactical),
        }


def _scale(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _buy_days20(row: pd.Series | Mapping[str, Any]) -> float:
    if "foreign_buy_days20" in row:
        value = _float(row, "foreign_buy_days20")
        if value:
            return value
    return _float(row, "buy_days20")


def _buy_days10(row: pd.Series | Mapping[str, Any]) -> float:
    if "foreign_buy_days10" in row:
        value = _float(row, "foreign_buy_days10")
        if value:
            return value
    return _float(row, "foreign_roll20_up_days10")


def appearance_counts(
    snapshots: pd.DataFrame | None,
    lookback: int = 10,
) -> dict[str, int]:
    """How many recent sessions a name was an A/B candidate."""
    if snapshots is None or snapshots.empty or "symbol" not in snapshots:
        return {}
    frame = snapshots.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    date_col = "signal_date" if "signal_date" in frame.columns else "date"
    if date_col not in frame.columns:
        return {}
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    dates = sorted(stamp.date() for stamp in frame[date_col].dropna().unique())
    if not dates:
        return {}
    keep = set(dates[-lookback:])
    recent = frame[frame[date_col].dt.date.isin(keep)]
    if "tracking_cohort" in recent.columns:
        recent = recent[recent["tracking_cohort"].fillna("candidate") == "candidate"]
    if "discovery_grade" in recent.columns:
        recent = recent[recent["discovery_grade"].isin(["A", "B"])]
    if recent.empty:
        return {}
    return (
        recent.groupby("symbol")[date_col]
        .nunique()
        .astype(int)
        .to_dict()
    )


def _disqualified(
    row: pd.Series | Mapping[str, Any],
    campaigns: set[str],
) -> bool:
    if "discovery_eligible" in row and not _bool(row, "discovery_eligible"):
        return True
    if _text(row, "campaign_regime") not in campaigns:
        return True
    if _text(row, "stage") in TERMINAL_STAGES:
        return True
    if _text(row, "campaign_regime") in TERMINAL_CAMPAIGNS:
        return True
    if _text(row, "watch_tier") in {"위험확인"}:
        return True
    if _text(row, "absorption_signal") == "주의":
        return True
    return False


def mid_horizon_score(
    row: pd.Series | Mapping[str, Any],
    appearance_days: int = 0,
    lookback: int = 10,
    cfg: HorizonConfig | None = None,
) -> float:
    """Quality of the 2–8 week campaign. Ignores today's price gate."""
    cfg = cfg or HorizonConfig()
    if _disqualified(row, MID_CAMPAIGNS):
        return 0.0
    discovery = _float(row, "discovery_score")
    if discovery < cfg.mid_floor_discovery:
        return 0.0
    score = 0.0
    score += 0.46 * discovery
    score += 12.0 * _scale(_float(row, "accumulation_score"), 18.0, 40.0)
    score += 10.0 * _scale(_buy_days10(row), 5.0, 10.0)
    score += 8.0 * _scale(_buy_days20(row), 8.0, 16.0)
    score += 8.0 * _scale(_float(row, "foreign_20_adtv"), 0.7, 3.5)
    score += 6.0 * _scale(_float(row, "foreign_net20"), 8_000_000_000, 150_000_000_000)
    score += 8.0 * _scale(float(appearance_days), 1.0, float(lookback))
    if _bool(row, "large_flow_signal"):
        score += 3.0
    if _bool(row, "steady_accumulation"):
        score += 4.0
    score -= 0.28 * _float(row, "distribution_score")
    rebound = _float(row, "rebound_from_low20")
    if rebound > 0.12:
        score -= 16.0 * _scale(rebound, 0.12, 0.30)
    ret20 = _float(row, "price_return20")
    if ret20 > 0.08:
        score -= 10.0 * _scale(ret20, 0.08, 0.24)
    if _text(row, "proposal_status") == "제외":
        score -= 12.0
    return float(np.clip(score, 0.0, 100.0))


def long_horizon_score(
    row: pd.Series | Mapping[str, Any],
    cfg: HorizonConfig | None = None,
) -> float:
    """60–120 day inventory that is still being built, not a one-week timing."""
    cfg = cfg or HorizonConfig()
    if _disqualified(row, LONG_CAMPAIGNS):
        return 0.0
    discovery = _float(row, "discovery_score")
    if discovery < cfg.long_min_discovery:
        return 0.0
    net60 = _float(row, "foreign_net60")
    net120 = _float(row, "foreign_net120")
    if net60 < cfg.long_min_net60_krw and net120 < cfg.long_min_net120_krw:
        return 0.0
    score = 0.0
    score += 0.22 * discovery
    score += 22.0 * _scale(net60, 20_000_000_000, 400_000_000_000)
    score += 18.0 * _scale(net120, 30_000_000_000, 250_000_000_000)
    if net120 < 0:
        score -= 12.0
    score += 10.0 * _scale(_buy_days20(row), 8.0, 16.0)
    score += 8.0 * _scale(1.0 - _float(row, "sellback_ratio20"), 0.50, 0.95)
    if _bool(row, "large_flow_signal"):
        score += 4.0
    campaign = _text(row, "campaign_regime")
    if campaign in {"지속매집", "장기매집·재가속"}:
        score += 6.0
    elif campaign == "장기매집·휴식":
        score += 3.0
    score -= 0.16 * _float(row, "distribution_score")
    rebound = _float(row, "rebound_from_low20")
    if rebound > 0.20:
        score -= 14.0 * _scale(rebound, 0.20, 0.40)
    ret20 = _float(row, "price_return20")
    if ret20 > 0.15:
        score -= 10.0 * _scale(ret20, 0.15, 0.32)
    if _text(row, "stage") == "가격상승" or campaign == "상승진행":
        score -= 18.0
    return float(np.clip(score, 0.0, 100.0))


def _why_mid(row: pd.Series | Mapping[str, Any], appearance_days: int) -> str:
    bits = [
        f"발견 {_text(row, 'discovery_grade') or discovery_grade(row)}",
        f"20일 {format_eok(_float(row, 'foreign_net20'))}",
    ]
    buy10 = int(_buy_days10(row))
    if buy10:
        bits.append(f"10일 중 순매수 {buy10}일")
    if appearance_days >= 2:
        bits.append(f"최근 {appearance_days}일 후보 유지")
    if _bool(row, "large_flow_signal"):
        bits.append("500억+")
    bits.append(_text(row, "campaign_regime", "매집"))
    status = _text(row, "proposal_status")
    if status == "오늘 신규매수 검토":
        bits.append("오늘 가격도 통과")
    elif status == "강한 후보·가격 대기":
        bits.append("품질은 충분 · 가격만 대기")
    return " · ".join(bits)


def _why_long(row: pd.Series | Mapping[str, Any]) -> str:
    bits = [
        f"60일 {format_eok(_float(row, 'foreign_net60'))}",
        f"120일 {format_eok(_float(row, 'foreign_net120'))}",
        f"발견 {discovery_grade(row)}",
    ]
    if _bool(row, "large_flow_signal"):
        bits.append("500억+")
    bits.append(_text(row, "campaign_regime", "매집"))
    ret20 = _float(row, "price_return20")
    if ret20 >= 0.08:
        bits.append(f"20일 {format_pct(ret20)} · 추격 금지")
    else:
        bits.append("재고가 아직 쌓이는 구간")
    return " · ".join(bits)


def _hold_rule(role: str) -> str:
    if role == "중기 코어":
        return "2~8주 관점. 오늘 타이밍 리스트에서 빠져도 팔지 않는다. 분산·이탈 전까지 유지."
    return "3~6개월 관점. 60·120일 재고가 줄거나 분산 단계가 되기 전에는 코어에서 빼지 않는다."


def _as_of_text(scored: pd.DataFrame) -> str:
    if scored.empty or "as_of" not in scored:
        return ""
    stamp = pd.to_datetime(scored["as_of"].max(), errors="coerce")
    if pd.isna(stamp):
        return ""
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")


def _lookup(scored: pd.DataFrame) -> dict[str, pd.Series]:
    table: dict[str, pd.Series] = {}
    if scored.empty or "symbol" not in scored:
        return table
    for _, row in scored.iterrows():
        table[str(row["symbol"]).zfill(6)] = row
    return table


def _select_sticky(
    scored: pd.DataFrame,
    scores: pd.Series,
    previous: list[dict[str, Any]],
    slots: int,
    min_score: float,
    stay_buffer: float,
    replace_margin: float,
    as_of: str,
) -> list[dict[str, Any]]:
    """Keep yesterday's names unless they actually broke. Do not churn for 2 points."""
    lookup = _lookup(scored)
    kept: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in previous:
        symbol = str(item.get("symbol", "")).zfill(6)
        if not symbol or symbol in used:
            continue
        row = lookup.get(symbol)
        if row is None:
            continue
        score = float(scores.get(symbol, 0.0) or 0.0)
        if score < max(0.0, min_score - stay_buffer):
            continue
        since = str(item.get("since") or as_of)
        kept.append({"symbol": symbol, "name": _text(row, "name"), "since": since, "score": score})
        used.add(symbol)
        if len(kept) >= slots:
            return kept

    ranked = (
        scores.sort_values(ascending=False)
        if not scores.empty
        else pd.Series(dtype="float64")
    )
    weakest = min((item["score"] for item in kept), default=0.0)
    for symbol, score in ranked.items():
        symbol = str(symbol).zfill(6)
        if symbol in used or float(score) < min_score:
            continue
        if kept and float(score) < weakest + replace_margin and len(kept) >= slots:
            continue
        row = lookup.get(symbol)
        if row is None:
            continue
        kept.append(
            {
                "symbol": symbol,
                "name": _text(row, "name"),
                "since": as_of,
                "score": float(score),
            }
        )
        used.add(symbol)
        if len(kept) >= slots:
            break
    kept.sort(key=lambda item: item["score"], reverse=True)
    return kept[:slots]


def _to_horizon_name(
    row: pd.Series,
    *,
    role: str,
    score: float,
    since: str,
    as_of: str,
    appearance_days: int,
    is_new: bool,
) -> HorizonName:
    streak = 1
    if since and as_of and since <= as_of:
        try:
            streak = max(1, (pd.Timestamp(as_of) - pd.Timestamp(since)).days + 1)
        except (TypeError, ValueError):
            streak = 1
    status = _text(row, "proposal_status")
    return HorizonName(
        symbol=str(row.get("symbol", "")).zfill(6),
        name=_text(row, "name"),
        role=role,
        score=round(float(score), 1),
        discovery_score=round(_float(row, "discovery_score"), 1),
        grade=_text(row, "discovery_grade") or discovery_grade(row),
        proposal_status=status,
        campaign=_text(row, "campaign_regime", "중립"),
        stage=_text(row, "stage", "중립"),
        since=since,
        streak_days=streak,
        why=_why_mid(row, appearance_days) if role == "중기 코어" else _why_long(row),
        hold_rule=_hold_rule(role),
        foreign_net20=_float(row, "foreign_net20"),
        foreign_net60=_float(row, "foreign_net60"),
        foreign_net120=_float(row, "foreign_net120"),
        price_return20=_float(row, "price_return20"),
        rebound_from_low20=_float(row, "rebound_from_low20"),
        buy_days10=int(_buy_days10(row)),
        appearance_days=int(appearance_days),
        is_new=is_new,
        today_buyable=status == "오늘 신규매수 검토",
    )


def _tactical_records(scored: pd.DataFrame, brief: BriefConfig) -> list[dict[str, str]]:
    selected = select_daily_brief(scored, brief)
    records: list[dict[str, str]] = []
    if selected.empty:
        return records
    for _, row in selected.iterrows():
        records.append(
            {
                "symbol": str(row.get("symbol", "")).zfill(6),
                "name": _text(row, "name"),
                "status": _text(row, "proposal_status", "관찰"),
            }
        )
    return records


def _topic_particle(name: str) -> str:
    """Attach 은/는 using the last Hangul syllable's batchim."""
    if not name:
        return name
    code = ord(name[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return name + ("는" if (code - 0xAC00) % 28 == 0 else "은")
    return name + "는"


def _continuity(
    today_tactical: list[dict[str, str]],
    yesterday_tactical: list[dict[str, Any]],
    mid_core: list[HorizonName],
    long_core: list[HorizonName],
    lookup: dict[str, pd.Series],
) -> list[ContinuityNote]:
    notes: list[ContinuityNote] = []
    today_listed = {item["symbol"] for item in today_tactical}
    today_buy = {
        item["symbol"]
        for item in today_tactical
        if item.get("status") == "오늘 신규매수 검토"
    }
    mid_symbols = {item.symbol for item in mid_core}
    long_symbols = {item.symbol for item in long_core}
    core_names = {item.symbol: item.name for item in (*mid_core, *long_core)}

    def dropout_note(symbol: str, name: str, row: pd.Series | None, was_buy: bool) -> ContinuityNote:
        titled = _topic_particle(name)
        grade = _text(row, "discovery_grade") if row is not None else ""
        status = _text(row, "proposal_status") if row is not None else "제외"
        extra = f" 오늘은 {status}" if status else ""
        if grade:
            extra += f" · 발견 {grade}등급"
        if symbol in mid_symbols:
            return ContinuityNote(
                kind="core_wait",
                symbol=symbol,
                name=name,
                text=(
                    f"{titled} 어제 타이밍 리스트에서 빠졌지만 중기 코어입니다. "
                    "가격 게이트가 닫힌 것이지 종목이 나빠진 것이 아닙니다. 팔지 말고 눌림을 기다리세요."
                ),
            )
        if symbol in long_symbols:
            return ContinuityNote(
                kind="long_hold",
                symbol=symbol,
                name=name,
                text=(
                    f"{titled} 오늘 살 타이밍은 아니지만 장기 매집 코어입니다. "
                    "당일 리스트 교체를 매도 신호로 읽지 마세요."
                ),
            )
        if was_buy:
            return ContinuityNote(
                kind="tactical_dropout",
                symbol=symbol,
                name=name,
                text=(
                    f"{titled} 어제 ‘지금 살 타이밍’이었습니다. 오늘 리스트에서 빠진 것은 "
                    "당일 가격 게이트가 닫힌 것이지, 중기·장기 코어에서 강등된 것이 아닙니다. "
                    "이미 샀다면 당일 타이밍 포지션으로 보고 분산 신호가 나오기 전에는 기계적으로 팔지 마세요. "
                    "아직 안 샀다면 억지로 따라가지 않아도 됩니다."
                    + extra
                ),
            )
        return ContinuityNote(
            kind="watch_dropout",
            symbol=symbol,
            name=name,
            text=(
                f"{titled} 어제 숏리스트 관찰이었고 오늘은 빠졌습니다. "
                "중기 코어가 아니면 따라가지 않아도 됩니다."
                + extra
            ),
        )

    for item in yesterday_tactical:
        symbol = str(item.get("symbol", "")).zfill(6)
        name = str(item.get("name") or core_names.get(symbol) or symbol)
        if not symbol:
            continue
        row = lookup.get(symbol)
        was_buy = item.get("status") == "오늘 신규매수 검토"
        still_buy = symbol in today_buy
        still_listed = symbol in today_listed
        if was_buy and not still_buy:
            notes.append(dropout_note(symbol, name, row, True))
        elif not was_buy and not still_listed:
            notes.append(dropout_note(symbol, name, row, False))

    yesterday_symbols = {str(item.get("symbol", "")).zfill(6) for item in yesterday_tactical}
    for item in today_tactical:
        if item.get("status") != "오늘 신규매수 검토":
            continue
        symbol = item["symbol"]
        if symbol in yesterday_symbols:
            yesterday_status = next(
                (
                    str(prev.get("status") or "")
                    for prev in yesterday_tactical
                    if str(prev.get("symbol", "")).zfill(6) == symbol
                ),
                "",
            )
            if yesterday_status == "오늘 신규매수 검토":
                continue
        name = item["name"]
        titled = _topic_particle(name)
        if symbol in mid_symbols or symbol in long_symbols:
            notes.append(
                ContinuityNote(
                    kind="core_now_buyable",
                    symbol=symbol,
                    name=name,
                    text=(
                        f"{titled} 중기·장기 코어가 오늘 가격 게이트를 통과했습니다. "
                        "매일 바뀌는 단타가 아니라, 갖고 가기로 한 종목의 분할 신규 자리입니다."
                    ),
                )
            )
        else:
            notes.append(
                ContinuityNote(
                    kind="fresh_timing",
                    symbol=symbol,
                    name=name,
                    text=(
                        f"{titled} 오늘 타이밍입니다. 중기 코어에 없으면 포지션 전체를 이 종목으로 "
                        "갈아타지 마세요. 현금이 있을 때만 분할로 검토합니다."
                    ),
                )
            )
    return notes[:6]


def _verdict(
    tactical: list[dict[str, str]],
    mid_core: list[HorizonName],
    long_core: list[HorizonName],
    continuity: list[ContinuityNote],
) -> tuple[str, ...]:
    buy = [item for item in tactical if item.get("status") == "오늘 신규매수 검토"]
    if buy:
        line1 = "지금 현금이 있으면: " + " → ".join(item["name"] for item in buy)
    else:
        line1 = "오늘은 당장 살 자리 없음. 중기 코어의 가격만 기다리세요."
    line2 = (
        "중기 코어(2~8주, 매일 바꾸지 말 것): " + " · ".join(item.name for item in mid_core)
        if mid_core
        else "중기 코어: 오늘은 유지할 고확신 매집이 없습니다."
    )
    line3 = (
        "장기 매집(3~6개월): " + " · ".join(item.name for item in long_core)
        if long_core
        else "장기 매집: 60·120일 재고가 아직 쌓이는 코어가 없습니다."
    )
    drop = next((note.text for note in continuity if note.kind == "tactical_dropout"), "")
    extra = (drop,) if drop else ()
    return (line1, line2, line3, *extra)


def _decision(
    tactical: list[dict[str, str]],
    mid_core: list[HorizonName],
    long_core: list[HorizonName],
    continuity: list[ContinuityNote],
) -> tuple[str, ...]:
    """Concrete do / wait / hold for tonight's report. Not a second ranking."""
    buy = [item for item in tactical if item.get("status") == "오늘 신규매수 검토"]
    observe = [item for item in tactical if item.get("status") in {"관찰", "제외"}]
    seen: set[str] = set()
    cores: list[HorizonName] = []
    for item in (*mid_core, *long_core):
        if item.symbol in seen:
            continue
        seen.add(item.symbol)
        cores.append(item)
    waiting = [item for item in cores if not item.today_buyable]
    open_cores = [item for item in cores if item.today_buyable]
    drops = [note for note in continuity if note.kind == "tactical_dropout"]

    lines: list[str] = []
    if buy:
        names = " → ".join(item["name"] for item in buy)
        lines.append(f"오늘 살 것: {names}  · 현금 있을 때만 분할. 코어를 팔아 갈아타지 말 것")
    else:
        lines.append("오늘 살 것: 없음. 게이트가 열린 종목이 없으면 현금을 유지")
    if open_cores:
        lines.append(
            "코어가 오늘 열림: "
            + " · ".join(item.name for item in open_cores)
            + "  · 갖고 가기로 한 종목의 분할 신규"
        )
    if waiting:
        lines.append(
            "기다릴 것: "
            + " · ".join(item.name for item in waiting)
            + "  · 20일 +3% 미만·저점반등 +5% 미만이 될 때까지"
        )
    if drops:
        lines.append(
            "팔지 말 것: "
            + " · ".join(note.name for note in drops)
            + "  · 리스트 이탈은 매도가 아님"
        )
    if observe:
        lines.append("추격 금지: " + " · ".join(item["name"] for item in observe))
    return tuple(lines)


def build_horizon_board(
    scored: pd.DataFrame,
    previous: Mapping[str, Any] | None = None,
    snapshots: pd.DataFrame | None = None,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
    horizon: HorizonConfig | None = None,
) -> HorizonBoard:
    brief = brief or BriefConfig()
    model = model or ModelConfig()
    horizon = horizon or HorizonConfig()
    previous = previous or {}
    if scored is None or scored.empty:
        return HorizonBoard(
            as_of="",
            verdict_lines=("오늘 데이터가 없습니다.",),
            mid_core=(),
            long_core=(),
            continuity=(),
            decision_lines=("오늘 살 것: 없음.",),
        )

    frame = scored.copy()
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    as_of = _as_of_text(frame)
    looks = appearance_counts(snapshots, horizon.appearance_lookback)
    mid_scores: dict[str, float] = {}
    long_scores: dict[str, float] = {}
    for _, row in frame.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        appeared = int(looks.get(symbol, 0))
        mid_scores[symbol] = mid_horizon_score(row, appeared, horizon.appearance_lookback, horizon)
        long_scores[symbol] = long_horizon_score(row, horizon)
    frame["mid_horizon_score"] = frame["symbol"].map(mid_scores).fillna(0.0)
    frame["long_horizon_score"] = frame["symbol"].map(long_scores).fillna(0.0)

    mid_kept = _select_sticky(
        frame,
        frame.set_index("symbol")["mid_horizon_score"],
        list(previous.get("mid_core") or []),
        horizon.mid_slots,
        horizon.mid_min_score,
        horizon.stay_buffer,
        horizon.replace_margin,
        as_of,
    )
    long_kept = _select_sticky(
        frame,
        frame.set_index("symbol")["long_horizon_score"],
        list(previous.get("long_core") or []),
        horizon.long_slots,
        horizon.long_min_score,
        horizon.stay_buffer,
        horizon.replace_margin,
        as_of,
    )
    lookup = _lookup(frame)
    previous_mid = {
        str(item.get("symbol", "")).zfill(6): str(item.get("since") or as_of)
        for item in (previous.get("mid_core") or [])
    }
    previous_long = {
        str(item.get("symbol", "")).zfill(6): str(item.get("since") or as_of)
        for item in (previous.get("long_core") or [])
    }
    mid_names = [
        _to_horizon_name(
            lookup[item["symbol"]],
            role="중기 코어",
            score=item["score"],
            since=item["since"],
            as_of=as_of,
            appearance_days=int(looks.get(item["symbol"], 0)),
            is_new=item["symbol"] not in previous_mid,
        )
        for item in mid_kept
        if item["symbol"] in lookup
    ]
    long_names = [
        _to_horizon_name(
            lookup[item["symbol"]],
            role="장기 매집",
            score=item["score"],
            since=item["since"],
            as_of=as_of,
            appearance_days=int(looks.get(item["symbol"], 0)),
            is_new=item["symbol"] not in previous_long,
        )
        for item in long_kept
        if item["symbol"] in lookup
    ]
    tactical = _tactical_records(frame, brief)
    yesterday = [
        {
            "symbol": str(item.get("symbol", "")).zfill(6),
            "name": str(item.get("name", "")),
            "status": str(item.get("status", "")),
        }
        for item in (previous.get("tactical") or [])
        if item.get("symbol")
    ]
    continuity = _continuity(tactical, yesterday, mid_names, long_names, lookup)
    verdict = _verdict(tactical, mid_names, long_names, continuity)
    decision = _decision(tactical, mid_names, long_names, continuity)
    return HorizonBoard(
        as_of=as_of,
        verdict_lines=verdict,
        mid_core=tuple(mid_names),
        long_core=tuple(long_names),
        continuity=tuple(continuity),
        yesterday_tactical=tuple(tactical),
        decision_lines=decision,
    )


def annotate_horizon(
    scored: pd.DataFrame,
    previous: Mapping[str, Any] | None = None,
    snapshots: pd.DataFrame | None = None,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
    horizon: HorizonConfig | None = None,
) -> tuple[pd.DataFrame, HorizonBoard]:
    board = build_horizon_board(scored, previous, snapshots, brief, model, horizon)
    frame = scored.copy()
    if frame.empty or "symbol" not in frame:
        return frame, board
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    mid_map = {item.symbol: item.score for item in board.mid_core}
    long_map = {item.symbol: item.score for item in board.long_core}
    looks = appearance_counts(snapshots, (horizon or HorizonConfig()).appearance_lookback)
    cfg = horizon or HorizonConfig()
    if "mid_horizon_score" not in frame:
        frame["mid_horizon_score"] = [
            mid_horizon_score(row, int(looks.get(str(row["symbol"]).zfill(6), 0)), cfg.appearance_lookback, cfg)
            for _, row in frame.iterrows()
        ]
    if "long_horizon_score" not in frame:
        frame["long_horizon_score"] = [
            long_horizon_score(row, cfg) for _, row in frame.iterrows()
        ]
    frame["in_mid_core"] = frame["symbol"].isin(mid_map)
    frame["in_long_core"] = frame["symbol"].isin(long_map)
    return frame, board


def empty_horizon_state() -> dict[str, Any]:
    return {"as_of": "", "mid_core": [], "long_core": [], "tactical": []}


def render_horizon_plain(board: HorizonBoard) -> list[str]:
    if not board.verdict_lines:
        return []
    lines = ["오늘 판단", *board.decision_lines, "", "그래서 뭘 사나", *board.verdict_lines]
    if board.mid_core:
        lines.append("")
        lines.append("중기 코어")
        for index, item in enumerate(board.mid_core, 1):
            tag = "NEW" if item.is_new else f"{item.streak_days}일째"
            lines.append(f"{index}. {item.name}  {item.symbol}  ·  {tag}  ·  {item.why}")
    if board.long_core:
        lines.append("")
        lines.append("장기 매집")
        for index, item in enumerate(board.long_core, 1):
            tag = "NEW" if item.is_new else f"{item.streak_days}일째"
            lines.append(f"{index}. {item.name}  {item.symbol}  ·  {tag}  ·  {item.why}")
    drop = [note for note in board.continuity if note.kind in {"tactical_dropout", "core_wait", "long_hold"}]
    if drop:
        lines.append("")
        lines.append("어제 산 것")
        for note in drop[:3]:
            lines.append(f"- {note.text}")
    return lines


def render_horizon_markdown(board: HorizonBoard) -> str:
    if not board.verdict_lines:
        return ""
    lines = [
        "### 오늘 판단",
        "",
        "「오늘 신규매수 검토」만 사고, 코어는 그 뱃지가 다시 켜질 때까지 기다립니다. 리스트에서 빠졌다고 팔지 않습니다.",
        "",
    ]
    for line in board.decision_lines:
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            "### 그래서 뭘 사나",
            "",
            "매일 타이밍은 위 리스트를 보고, 실제로 들고 갈 자리는 아래를 봅니다.",
            "",
        ]
    )
    for line in board.verdict_lines:
        lines.append(f"- {line}")
    if board.mid_core:
        lines.extend(["", "### 중기 코어 · 2~8주", ""])
        for index, item in enumerate(board.mid_core, 1):
            tag = "신규 편입" if item.is_new else f"{item.streak_days}일째 유지"
            lines.append(
                f"{index}. **{item.name}** ({item.symbol}) · {tag} · 중기 {item.score:.0f} · "
                f"발견 {item.discovery_score:.1f} ({item.grade}) · {item.why}"
            )
            lines.append(f"   - {item.hold_rule}")
    if board.long_core:
        lines.extend(["", "### 장기 매집 · 3~6개월", ""])
        for index, item in enumerate(board.long_core, 1):
            tag = "신규 편입" if item.is_new else f"{item.streak_days}일째 유지"
            lines.append(
                f"{index}. **{item.name}** ({item.symbol}) · {tag} · 장기 {item.score:.0f} · "
                f"60일 {format_eok(item.foreign_net60)} · 120일 {format_eok(item.foreign_net120)} · {item.why}"
            )
            lines.append(f"   - {item.hold_rule}")
    if board.continuity:
        lines.extend(["", "### 어제와 오늘", ""])
        for note in board.continuity:
            lines.append(f"- {note.text}")
    return "\n".join(lines)


def render_horizon_html(board: HorizonBoard) -> str:
    if not board.verdict_lines:
        return ""

    def items(names: tuple[HorizonName, ...]) -> str:
        if not names:
            return '<div class="empty">해당 없음</div>'
        cards = []
        for item in names:
            tag = "신규 편입" if item.is_new else f"{item.streak_days}일째"
            buy = (
                '<span class="badge green">오늘 가격 통과</span>'
                if item.today_buyable
                else '<span class="badge yellow">가격 대기</span>'
            )
            cards.append(
                "<article class='horizon-card'>"
                f"<div class='horizon-card-head'><strong>{html.escape(item.name)}</strong>"
                f"<small>{html.escape(item.symbol)} · {html.escape(tag)}</small></div>"
                f"<div class='horizon-score'>{item.score:.0f}<span>점</span></div>"
                f"<p>{html.escape(item.why)}</p>"
                f"<p class='muted'>{html.escape(item.hold_rule)}</p>"
                f"{buy}</article>"
            )
        return "<div class='horizon-cards'>" + "".join(cards) + "</div>"

    verdict = "".join(f"<li>{html.escape(line)}</li>" for line in board.verdict_lines)
    decision = "".join(f"<li>{html.escape(line)}</li>" for line in board.decision_lines)
    notes = "".join(f"<li>{html.escape(note.text)}</li>" for note in board.continuity)
    notes_block = (
        f"<h3>어제와 오늘</h3><ul class='horizon-notes'>{notes}</ul>" if notes else ""
    )
    return (
        '<section class="horizon">'
        '<div class="horizon-kicker">시간축 보드 · 매일 바꾸지 않는 자리</div>'
        "<h2>오늘 판단</h2>"
        "<p class='muted'>날짜가 아니라 조건입니다. 「오늘 신규매수 검토」일 때만 사고, "
        "코어는 게이트가 다시 열릴 때까지 기다립니다. 빠진 종목은 매도가 아닙니다.</p>"
        f"<ul class='horizon-verdict'>{decision}</ul>"
        "<h2>그래서 뭘 사나</h2>"
        "<p class='muted'>위는 오늘 할 일, 아래는 중기·장기로 들고 갈 코어입니다. "
        "어제의 최선이 오늘 빠졌다고 종목을 갈아타지 마세요.</p>"
        f"<ul class='horizon-verdict'>{verdict}</ul>"
        "<div class='horizon-split'>"
        "<div><h3>중기 코어 · 2~8주</h3>"
        "<p class='muted'>발견 품질이 높은 매집. 오늘 게이트가 닫혀도 유지.</p>"
        f"{items(board.mid_core)}</div>"
        "<div><h3>장기 매집 · 3~6개월</h3>"
        "<p class='muted'>60·120일 재고가 아직 쌓이는 자리. 눌림만 기다립니다.</p>"
        f"{items(board.long_core)}</div>"
        f"</div>{notes_block}</section>"
    )


def render_horizon_telegram(board: HorizonBoard) -> list[str]:
    if not board.verdict_lines:
        return []
    lines = ["<b>오늘 판단</b>"]
    for line in board.decision_lines[:4]:
        lines.append(html.escape(line))
    lines.append("<b>그래서 뭘 사나</b>")
    for line in board.verdict_lines[:4]:
        lines.append(html.escape(line))
    if board.mid_core:
        compact = " · ".join(
            f"{html.escape(item.name)}"
            + ("*" if item.today_buyable else "")
            for item in board.mid_core
        )
        lines.append(f"중기 코어  {compact}")
    if board.long_core:
        compact = " · ".join(html.escape(item.name) for item in board.long_core)
        lines.append(f"장기 매집  {compact}")
    return lines
