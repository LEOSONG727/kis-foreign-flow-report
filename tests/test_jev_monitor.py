import pandas as pd

from kis_flow_bot.jev_monitor import monitor, summarize
from kis_flow_bot.jev_shadow import MODEL


class Response:
    status_code = 200
    def json(self):
        return {"model": MODEL, "answers": {"next_step": {
            "type": "choice", "choice": "REVIEW", "confidence": .9}}}


class Session:
    def post(self, *a, **k):
        return Response()


def test_sample_guard_blocks_premature_improvement():
    result = monitor(pd.DataFrame({"jev_status": ["OK"]}), pd.DataFrame(), api_key="test", session=Session())
    assert result["decision"] == "OBSERVE"
    assert result["model_decision"] == "REVIEW"
    assert result["source"] == "code_sample_guard"


def test_operational_errors_take_priority():
    result = monitor(pd.DataFrame({"jev_status": ["API_ERROR", "CIRCUIT_OPEN"]}), pd.DataFrame(), api_key="test", session=Session())
    assert result["decision"] == "DIAGNOSE"
    assert "작동 상태 점검" in result["note"]


def test_missing_key_is_explicit():
    result = monitor(pd.DataFrame({"jev_status": ["DISABLED"]}), pd.DataFrame(), api_key="")
    assert result["error"] == "missing_api_key"
    assert "코드 기준" in result["note"]


def test_complete_paired_evidence_allows_review():
    from kis_flow_bot.jev_experiment import VERSION
    ledger = pd.DataFrame([{
        "signal_date": day.strftime("%Y-%m-%d"), "experiment_version": VERSION,
        "cohort_id": "v2", "cohort_size": 3, "symbol": str(i), "jev_status": "OK",
        "net_return_20": -.03, "worst_entry_loss_20": -.1,
        "baseline_rank": i, "ml_rank": i, "jev_rank": i,
    } for day in pd.bdate_range("2026-06-01", periods=20) for i in range(1, 4)])
    result = monitor(pd.DataFrame({"jev_status": ["OK"]}), ledger, api_key="test", session=Session())
    assert result["evidence"]["completed_common_days"] == 20
    assert result["decision"] == "REVIEW"
    assert result["source"] == "jev"


def test_notes_reach_both_notification_builders(monkeypatch):
    from types import SimpleNamespace
    from kis_flow_bot import notify
    frame = pd.DataFrame({"symbol": ["000001"]})
    frame.attrs["jev_monitor_note"] = "모델 감독: 계속 관찰"
    monkeypatch.setattr(notify, "_brief_frame", lambda *a: frame)
    monkeypatch.setattr(notify, "build_headline", lambda *a: SimpleNamespace())
    monkeypatch.setattr(notify, "_title", lambda *a: "title")
    monkeypatch.setattr(notify, "_telegram_headline", lambda *a: [])
    monkeypatch.setattr(notify, "render_headline_plain", lambda *a: [])
    monkeypatch.setattr(notify, "render_headline_html", lambda *a: "")
    assert frame.attrs["jev_monitor_note"] in notify.build_telegram_text(frame)
    plain, html = notify.build_email_bodies(frame)
    assert frame.attrs["jev_monitor_note"] in plain
    assert frame.attrs["jev_monitor_note"] in html
