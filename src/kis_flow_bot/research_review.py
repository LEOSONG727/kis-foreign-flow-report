"""Conservative research queue, separate from tracked trading signals.

Thresholds are explicit heuristics, not calibrated return predictions.
Amounts retain the upstream CSV's KRW units. No orders or notifications.
"""
from datetime import date
from math import isfinite


def _true(value):
    return str(value).strip().lower() in {"true", "1"}


def _number(row, key):
    try:
        value = float(row[key])
        return value if isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def review_rows(rows, as_of, event_blocks=None):
    """Audit all rows; event_blocks maps symbol to a dated human-reviewed reason.

    Caller must supply the intended market session. Missing values fail closed.
    Event blocks never expire silently: remove only after a fresh human review.
    """
    date.fromisoformat(as_of)
    result = []
    keys = ("foreign_net3", "foreign_net5", "foreign_net20",
            "sellback_ratio20", "price_return3", "price_return20",
            "foreign_buy_days10", "ma20_slope")
    for row in rows:
        symbol = str(row.get("symbol", "")).zfill(6)
        reasons = []
        if str(row.get("as_of", ""))[:10] != as_of:
            reasons.append("기준일 불일치")
        if not all(_true(row.get(k)) for k in ("data_fresh", "discovery_eligible")):
            reasons.append("데이터/적격성 미확인")
        if row.get("proposal_status") != "오늘 신규매수 검토":
            reasons.append("기존 후보 아님")
        nums = {k: _number(row, k) for k in keys}
        if any(v is None for v in nums.values()):
            reasons.append("필수 수치 누락")
        else:
            if any(nums[k] <= 0 for k in keys[:3]):
                reasons.append("3·5·20일 외인 동반 순매수 미충족")
            if not 0 <= nums["sellback_ratio20"] <= 0.4:
                reasons.append("매수 대비 재매도 과다")
            if nums["price_return3"] < -0.03 or nums["price_return20"] < -0.15:
                reasons.append("가격 하락 위험")
            if not 6 <= nums["foreign_buy_days10"] <= 10:
                reasons.append("매수 지속성 부족")
        if symbol in (event_blocks or {}):
            reasons.append("중요 뉴스 재검토: " + event_blocks[symbol])
        passed = not reasons
        score = None
        if passed:
            score = round(40 * nums["foreign_buy_days10"] / 10
                          + 35 * (1 - nums["sellback_ratio20"])
                          + 25 * max(0, min(1, 1 + nums["ma20_slope"] / 0.05)), 2)
        confirmed = passed and _true(row.get("above_ma5")) and (
            _true(row.get("above_ma10")) or nums["ma20_slope"] >= -0.005)
        result.append({"symbol": symbol, "name": row.get("name", ""),
                       "as_of": as_of, "passed": passed, "review_score": score,
                       "status": "제외" if not passed else (
                           "가격 확인·기업 검토 필요" if confirmed else "가격 반전 대기·기업 검토 필요"),
                       "reason": "; ".join(reasons)})
    return sorted(result, key=lambda r: (not r["passed"], -(r["review_score"] or 0), r["symbol"]))


def render_review(rows, as_of, limit=3):
    selected = [r for r in rows if r["passed"]][:max(0, limit)]
    lines = ["## 별도 리서치 검토", "", f"수급 기준일: {as_of}", "",
             "수익률 미검증 보수적 필터입니다. 매수 지시가 아니며 현금흐름·부채·최신 공시를 별도로 확인해야 합니다.",
             "기존 신호 및 성과 추적을 변경하지 않습니다. 후보가 부족하면 채우지 않습니다.", "",
             "| 종목 | 검토 점수 | 상태 |", "|---|---:|---|"]
    for r in selected:
        lines.append(f"| {r['name']} ({r['symbol']}) | {r['review_score']} | {r['status']} |")
    if not selected:
        lines.append("| 통과 후보 없음 | — | 대기 |")
    return "\n".join(lines) + "\n"
