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


def test_fast_absolute_rise_is_not_called_pressed_accumulation():
    features = synthetic_features()
    index = features[features["symbol"] == "000010"].index[0]
    features.loc[index, "price_return20"] = 0.11
    features.loc[index, "price_return5"] = 0.06
    features.loc[index, "price_residual20"] = 0.01
    features.loc[index, "foreign_1_adtv"] = 0.10
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] == "가격상승"
    assert candidate["action"] == "추격 금지·보유"


def test_today_flow_drives_profit_taking_instead_of_five_day_flow():
    features = synthetic_features()
    index = features[features["symbol"] == "000010"].index[0]
    features.loc[index, "price_return20"] = 0.12
    features.loc[index, "price_return5"] = 0.02
    features.loc[index, "foreign_5_adtv"] = 0.50
    features.loc[index, "foreign_1_adtv"] = -0.20
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] == "수익실현"
    assert candidate["action"] == "절반 익절·잔여 추적"


def test_structural_break_blocks_new_buy():
    features = synthetic_features()
    index = features[features["symbol"] == "000010"].index[0]
    features.loc[index, "price_return20"] = -0.30
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] == "이벤트확인"
    assert candidate["action"] == "기업 이벤트·급락 확인 전 매수 금지"
