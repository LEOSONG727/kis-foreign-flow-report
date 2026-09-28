"""Model-health triage computed in code, sent as a bounded status note."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .jev_experiment import VERSION

LABELS = {"OBSERVE": "계속 관찰", "DIAGNOSE": "작동 상태 점검", "REVIEW": "개선 검토 필요"}


def summarize(scored, ledger):
    counts = scored.get("jev_status", pd.Series(dtype=str)).value_counts().to_dict()
    problems = sum(int(counts.get(k, 0)) for k in ("API_ERROR", "CIRCUIT_OPEN", "INSUFFICIENT_DATA"))
    rows = []
    cohort = "none"
    recorded = 0
    if not ledger.empty and "cohort_id" in ledger:
        current = ledger[ledger["experiment_version"].eq(VERSION)].copy()
        if not current.empty:
            latest = current.sort_values("signal_date").iloc[-1]
            cohort = str(latest["cohort_id"])
            current = current[current["cohort_id"].eq(cohort)]
            recorded = current["signal_date"].nunique()
            for day, group in current.groupby("signal_date"):
                if (len(group) != int(group["cohort_size"].iloc[0]) or group["symbol"].nunique() != len(group)
                        or not group["jev_status"].eq("OK").all()
                        or not group["net_return_20"].notna().all()
                        or not group["worst_entry_loss_20"].notna().all()):
                    continue
                values = {}
                for name, col in (("baseline", "baseline_rank"), ("ml", "ml_rank"), ("jev", "jev_rank")):
                    top = group[pd.to_numeric(group[col], errors="coerce").between(1, 3)]
                    if len(top) != 3:
                        break
                    values[name] = (float(pd.to_numeric(top["net_return_20"]).mean()),
                                    float(pd.to_numeric(top["worst_entry_loss_20"]).mean()))
                if len(values) == 3:
                    rows.append(values)
    n = len(rows)
    ret_delta = sum(r["jev"][0] - r["baseline"][0] for r in rows) / n if n else None
    loss_delta = sum(r["jev"][1] - r["baseline"][1] for r in rows) / n if n else None
    evidence = {
        "operational_status": "issues_present" if problems else "no_reported_issues",
        "status_counts": {k: int(v) for k, v in counts.items()},
        "cohort": cohort, "recorded_signal_days": int(recorded), "completed_common_days": n,
        "sample_status": "eligible_for_exploratory_review" if n >= 20 else "insufficient_for_performance_review",
        "return_difference": "unavailable" if n < 20 else f"Jev minus baseline: {ret_delta*100:+.2f} percentage points, after configured costs",
        "entry_loss_difference": "unavailable" if n < 20 else f"Jev minus baseline: {loss_delta*100:+.2f} percentage points; positive means smaller worst entry-relative loss",
        "limitations": "Overlapping holdings; no significance claim, no automatic promotion; this is not portfolio MDD.",
    }
    return evidence


def monitor(scored, ledger, *, verdict=None, registry=None):
    """Code-only model-health triage.

    The earlier version asked Jev to pick a queue and then overrode it with
    code guards in every case that mattered, so the call added cost without
    changing the outcome. The decision now comes from computed evidence:
    operational failures, the weekly self-validation registry and the Jev
    replay verdict.
    """
    evidence = summarize(scored, ledger)
    current = (registry or {}).get("current") or {}
    champion = (current.get("champion") or {}).get("status")
    challenger = (current.get("challenger") or {}).get("status")
    replay = (verdict or {}).get("verdict")
    evidence.update(
        champion_status=champion or "unavailable",
        challenger_status=challenger or "unavailable",
        jev_replay_verdict=replay or "unavailable",
    )
    if evidence["operational_status"] == "issues_present":
        decision = "DIAGNOSE"
    elif challenger == "PROMOTE" or champion == "NEGATIVE" or replay == "ADDS_VALUE":
        decision = "REVIEW"
    else:
        decision = "OBSERVE"
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "evidence": evidence,
              "decision": decision, "source": "code", "model": None, "error": None}
    labels = {
        "INSUFFICIENT_DATA": "표본 부족", "NO_EDGE": "예측력 미확인", "NOT_BETTER": "현행 대비 우위 없음",
        "PROMOTE": "승격 기준 통과", "VALIDATED": "실전 예측력 확인", "NEGATIVE": "역방향 예측",
        "NO_VALUE": "추가 가치 없음", "ADDS_VALUE": "추가 가치 확인",
    }
    note = f"모델 감독: {LABELS[decision]}"
    if champion or challenger:
        note += (f"\n자가검증: 현행 {labels.get(champion, champion or '-')}"
                 f" · 도전자 {labels.get(challenger, challenger or '-')}")
    if replay:
        note += f"\nJev 과거 검증: {labels.get(replay, replay)}"
        if replay == "NO_VALUE":
            note += " (일일 호출 중지)"
    if evidence["operational_status"] == "issues_present":
        note += "\n진단 상태: " + str(evidence["status_counts"])
    if decision == "REVIEW":
        note += "\n검증 결과가 바뀌었습니다. 사람이 검토하세요. 자동 수정은 하지 않습니다."
    result["note"] = note
    return result


def save_monitor(result, path):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
