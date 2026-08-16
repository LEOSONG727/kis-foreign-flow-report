from __future__ import annotations

from kis_flow_bot.config import ModelConfig
from kis_flow_bot.scoring import score_features

from test_features import synthetic_features


def test_accumulation_candidate_gets_actionable_stage():
    scored = score_features(synthetic_features(), ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] == "매집·눌림"
    assert candidate["absorption_signal"] in {"강함", "보통"}
    assert candidate["score"] >= 72
    assert candidate["action"] in {"초기 분할매수", "강한 신규매수"}


def test_distribution_overrides_buy_action():
    features = synthetic_features()
    index = features[features["symbol"] == "000010"].index[0]
    features.loc[index, "foreign_20_adtv"] = -1.0
    features.loc[index, "foreign_5_adtv"] = -0.5
    features.loc[index, "foreign_flow_share20"] = -0.04
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] == "분산·이탈"
    assert candidate["action"] == "비중 축소·신규매수 금지"

