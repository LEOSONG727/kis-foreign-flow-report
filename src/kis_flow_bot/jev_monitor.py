"""Use Jev to triage model-health evidence and send a bounded status note."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from .jev_experiment import VERSION
from .jev_shadow import ENDPOINT, MODEL

CHOICES = {
    "OBSERVE": "Continue collecting prospective evidence; no review is warranted yet.",
    "DIAGNOSE": "Inspect missing data, API failures, or a tracking problem.",
    "REVIEW": "Prepare a human review of the challenger; do not change production rules.",
}
LABELS = {"OBSERVE": "계속 관찰", "DIAGNOSE": "작동 상태 점검", "REVIEW": "개선 검토 필요"}


def summarize(scored, ledger):
    counts = scored.get("jev_status", pd.Series(dtype=str)).value_counts().to_dict()
    problems = sum(int(counts.get(k, 0)) for k in ("API_ERROR", "CIRCUIT_OPEN", "INSUFFICIENT_DATA", "DISABLED"))
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


def monitor(scored, ledger, *, api_key=None, session=None):
    evidence = summarize(scored, ledger)
    fallback = "DIAGNOSE" if evidence["operational_status"] == "issues_present" else "OBSERVE"
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "evidence": evidence,
              "decision": fallback, "source": "code_fallback", "model": None, "error": None}
    key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")
    if key:
        http = session or requests.Session()
        try:
            response = http.post(ENDPOINT, headers={"Authorization": f"Bearer {key}"},
                json={"model": MODEL, "state": evidence, "questions": {
                    "next_step": {"type": "choice", "instructions": "Which review queue fits the computed model-health evidence? Respect sample_status. Return a review queue, not code changes or trading instructions.", "criteria": CHOICES}}},
                timeout=(5, 12), allow_redirects=False)
            if not 200 <= response.status_code < 300:
                raise ValueError("HTTP failure")
            body = response.json()
            answer = body["answers"]["next_step"]
            confidence = answer["confidence"]
            if (body.get("model") != MODEL or answer.get("type") != "choice"
                or answer.get("choice") not in CHOICES or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
                raise ValueError("Invalid monitor response")
            result.update(model=body["model"], model_decision=answer["choice"], confidence=confidence,
                          decision=answer["choice"] if confidence >= .6 else fallback,
                          source="jev" if confidence >= .6 else "code_low_confidence")
            if evidence["operational_status"] == "issues_present":
                result.update(decision="DIAGNOSE", source="code_guard")
            elif evidence["completed_common_days"] < 20 and result["decision"] == "REVIEW":
                result.update(decision="OBSERVE", source="code_sample_guard")
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
            result["error"] = type(exc).__name__
        finally:
            if session is None:
                http.close()
    else:
        result["error"] = "missing_api_key"
    evidence_days = evidence["completed_common_days"]
    label = LABELS[result["decision"]]
    origin = "Jev 판단" if result["source"] == "jev" else "코드 기준 판단"
    note = f"모델 감독: {label} ({origin})\n20일 성과 비교 가능: {evidence_days}일 / 기록 {evidence['recorded_signal_days']}일"
    if evidence_days < 20:
        note += "\n성과 표본 축적 중이며 기준 변경은 보류합니다."
    else:
        note += "\n" + evidence["return_difference"]
        note += "\n" + evidence["entry_loss_difference"]
    if evidence["operational_status"] == "issues_present":
        note += "\n진단 상태: " + str(evidence["status_counts"])
    if result["error"]:
        note += "\n감독 API 미확인: " + result["error"]
    if result["decision"] == "REVIEW":
        note += "\n입력·질문·위험 기준의 검토가 필요합니다. 자동 수정은 하지 않습니다."
    result["note"] = note
    return result


def save_monitor(result, path):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
