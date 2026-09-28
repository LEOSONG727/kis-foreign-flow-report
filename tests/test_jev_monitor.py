import pandas as pd

from kis_flow_bot.jev_monitor import monitor, summarize


def test_default_is_observe_from_code():
    result = monitor(pd.DataFrame({"jev_status": ["OK"]}), pd.DataFrame())
    assert result["decision"] == "OBSERVE"
    assert result["source"] == "code"


def test_operational_errors_take_priority():
    registry = {"current": {"challenger": {"status": "PROMOTE"}}}
    result = monitor(pd.DataFrame({"jev_status": ["API_ERROR", "CIRCUIT_OPEN"]}), pd.DataFrame(), registry=registry)
    assert result["decision"] == "DIAGNOSE"
    assert "작동 상태 점검" in result["note"]


def test_validation_changes_request_review():
    registry = {"current": {"champion": {"status": "NEGATIVE"}, "challenger": {"status": "NO_EDGE"}}}
    result = monitor(pd.DataFrame({"jev_status": ["OK"]}), pd.DataFrame(), registry=registry)
    assert result["decision"] == "REVIEW"
    assert "역방향 예측" in result["note"]


def test_replay_verdict_is_reported_and_validation_off_is_not_an_error():
    frame = pd.DataFrame({"jev_status": ["VALIDATION_OFF"] * 3})
    result = monitor(frame, pd.DataFrame(), verdict={"verdict": "NO_VALUE"})
    assert result["decision"] == "OBSERVE"
    assert "일일 호출 중지" in result["note"]


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
