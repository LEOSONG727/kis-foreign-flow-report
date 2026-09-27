import pandas as pd

from kis_flow_bot.jev_shadow import apply_jev_shadow


class Response:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "model": "jev-1.13.0",
            "answers": {
                "accumulation": {"type": "noul", "noul": 0.81},
                "underreaction": {"type": "noul", "noul": 0.72},
                "overheated": {"type": "noul", "noul": 0.18},
                "review": {"type": "choice", "choice": "WATCH", "confidence": 0.66},
            },
        }


class Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()


def test_shadow_is_independent_and_limited():
    frame = pd.DataFrame([
        {"symbol": "005930", "discovery_score": 80, "foreign_net20": 123},
        {"symbol": "068270", "discovery_score": 79, "foreign_net20": 99},
    ])
    session = Session()
    result = apply_jev_shadow(frame, api_key="test-key", limit=1, session=session)
    assert frame.columns.tolist() == ["symbol", "discovery_score", "foreign_net20"]
    assert result["symbol"].tolist() == frame["symbol"].tolist()
    assert result.loc[0, "jev_review"] == "WATCH"
    assert pd.isna(result.loc[1, "jev_review"])
    assert len(session.calls) == 1
    assert "discovery_score" not in session.calls[0][1]["json"]["state"]
    assert session.calls[0][1]["json"]["state"]["features"]["foreign_net20"] == 123


def test_missing_key_skips_network():
    result = apply_jev_shadow(pd.DataFrame([{"symbol": "005930"}]), api_key="")
    assert pd.isna(result.loc[0, "jev_review"])
