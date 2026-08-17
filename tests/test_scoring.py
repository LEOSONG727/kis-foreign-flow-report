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
    assert candidate["action"] == "외국인 재가속 확인 후 분할매수"
    assert candidate["entry_readiness"] == "재가속확인"


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


def test_low_to_current_rebound_blocks_buy_even_when_endpoint_return_is_small():
    baseline_features, baseline_index = _candidate_features()
    baseline = score_features(baseline_features, ModelConfig())
    baseline_entry = baseline[baseline["symbol"] == "000010"].iloc[0]["entry_score"]

    features, index = _candidate_features()
    for column in ("price_return1", "price_return3", "price_return5", "price_return10", "price_return20"):
        features.loc[index, column] = 0.01
    features.loc[index, "price_residual20"] = 0.01
    features.loc[index, "rebound_from_low10"] = 0.03
    features.loc[index, "rebound_from_low20"] = 0.20
    features.loc[index, "rebound_atr_distance20"] = 5.0
    features.loc[index, "ma20_distance"] = 0.02
    features.loc[index, "ma20_atr_distance"] = 0.7
    features.loc[index, "distance_high20"] = -0.03
    features.loc[index, "up_days5"] = 2
    features.loc[index, "up_days10"] = 5
    features.loc[index, "price_up_streak"] = 0
    features.loc[index, "rebound_volume_ratio20"] = 1.5

    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]

    assert candidate["stage"] == "가격상승"
    assert candidate["campaign_regime"] == "상승진행"
    assert candidate["action"] == "추격 금지·보유"
    assert bool(candidate["price_heat_flag"])
    assert "20일 저점반등" in candidate["price_heat_reason"]
    assert candidate["price_heat_penalty"] > 0
    assert candidate["entry_score"] < baseline_entry


def test_atr_distance_alone_confirms_heat_but_does_not_block_entry():
    features, index = _candidate_features()
    for column in ("price_return1", "price_return3", "price_return5", "price_return10", "price_return20"):
        features.loc[index, column] = 0.01
    features.loc[index, "rebound_from_low10"] = 0.05
    features.loc[index, "rebound_from_low20"] = 0.10
    features.loc[index, "rebound_atr_distance20"] = 5.0
    features.loc[index, "ma20_atr_distance"] = 3.0
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["stage"] != "가격상승"
    assert not bool(candidate["price_heat_flag"])
    assert candidate["price_heat_reason"] == "과열 없음"


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


def _candidate_features():
    features = synthetic_features()
    index = features[features["symbol"] == "000010"].index[0]
    return features, index


def test_large_flow_signal_requires_20_day_500_eok():
    features, index = _candidate_features()
    features.loc[index, "foreign_net20"] = 60_000_000_000
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert bool(candidate["large_flow_signal"])
    assert candidate["campaign_score"] > 0


def test_long_inventory_pause_is_not_mistaken_for_distribution():
    features, index = _candidate_features()
    features.loc[index, "foreign_60_adtv"] = 3.0
    features.loc[index, "foreign_120_adtv"] = 6.0
    features.loc[index, "foreign_20_adtv"] = 0.8
    features.loc[index, "foreign_5_adtv"] = -0.1
    features.loc[index, "foreign_3_adtv"] = -0.05
    features.loc[index, "foreign_acceleration5"] = -0.03
    features.loc[index, "sellback_ratio20"] = 0.36
    features.loc[index, "price_return20"] = 0.07
    features.loc[index, "price_return5"] = 0.01
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] == "장기매집·휴식"
    assert candidate["action"] == "외국인 재개 확인 후 분할매수"


def test_new_reentry_after_long_distribution_waits_for_first_pullback_when_hot():
    features, index = _candidate_features()
    features.loc[index, "foreign_60_adtv"] = -3.0
    features.loc[index, "foreign_120_adtv"] = -6.0
    features.loc[index, "foreign_20_adtv"] = 0.6
    features.loc[index, "foreign_5_adtv"] = 0.4
    features.loc[index, "foreign_3_adtv"] = 0.3
    features.loc[index, "price_return20"] = 0.06
    features.loc[index, "price_return5"] = 0.12
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] == "신규재진입"
    assert candidate["action"] == "첫 눌림 대기·추격 금지"


def test_recent_three_day_drop_blocks_fresh_buy_until_stable():
    features, index = _candidate_features()
    features.loc[index, "price_return3"] = -0.04
    features.loc[index, "foreign_60_adtv"] = 3.0
    features.loc[index, "foreign_120_adtv"] = 6.0
    features.loc[index, "foreign_acceleration5"] = 0.10
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["entry_readiness"] == "반전확인"
    assert candidate["action"] == "급락 멈춤 확인·신규매수 대기"


def test_low_to_current_five_percent_rebound_waits_for_pullback():
    features, index = _candidate_features()
    features.loc[index, "rebound_from_low20"] = 0.06
    features.loc[index, "foreign_60_adtv"] = 3.0
    features.loc[index, "foreign_120_adtv"] = 6.0
    features.loc[index, "foreign_acceleration5"] = 0.10
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["entry_readiness"] == "눌림대기"
    assert candidate["action"] == "첫 눌림 대기·추격 금지"


def test_institution_led_rally_waits_for_foreign_confirmation():
    features, index = _candidate_features()
    features.loc[index, "foreign_20_adtv"] = 0.1
    features.loc[index, "foreign_5_adtv"] = -0.1
    features.loc[index, "institution_20_adtv"] = 0.5
    features.loc[index, "foreign_60_adtv"] = 3.0
    features.loc[index, "foreign_120_adtv"] = 6.0
    features.loc[index, "sellback_ratio20"] = 0.9
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] == "기관주도·외인확인"
    assert candidate["action"] == "기관 주도·외국인 확인 대기"


def test_long_inventory_handoff_is_not_called_distribution():
    features, index = _candidate_features()
    features.loc[index, "foreign_20_adtv"] = -0.5
    features.loc[index, "foreign_5_adtv"] = -0.15
    features.loc[index, "foreign_3_adtv"] = 0.08
    features.loc[index, "foreign_60_adtv"] = 3.6
    features.loc[index, "foreign_120_adtv"] = 9.0
    features.loc[index, "institution_20_adtv"] = 1.8
    features.loc[index, "price_return20"] = 0.10
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] == "손바뀜·재확인"
    assert candidate["stage"] != "분산·이탈"
    assert candidate["action"] == "기관 흡수 후 외국인 재확인"


def test_lumpy_flow_is_flagged_instead_of_called_accumulation():
    features, index = _candidate_features()
    features.loc[index, "foreign_20_adtv"] = 0.1
    features.loc[index, "foreign_60_adtv"] = 0.0
    features.loc[index, "foreign_120_adtv"] = 0.0
    features.loc[index, "foreign_flow_concentration20"] = 0.5
    features.loc[index, "tolerant_buy_ratio20"] = 0.4
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] == "이상대량·확인"
    assert candidate["action"] == "블록·이벤트성 수급 확인"


def test_absorption_caution_does_not_become_a_buy_action():
    features, index = _candidate_features()
    features.loc[index, "price_return20"] = -0.10
    features.loc[index, "price_return60"] = -0.30
    features.loc[index, "distance_high60"] = -0.35
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["absorption_signal"] == "주의"
    assert candidate["action"] == "위험흡수 구조 확인 대기"


def test_recent_three_and_five_day_selling_requires_restart_confirmation():
    features, index = _candidate_features()
    features.loc[index, "foreign_60_adtv"] = 0.0
    features.loc[index, "foreign_120_adtv"] = 0.0
    features.loc[index, "foreign_20_adtv"] = 0.8
    features.loc[index, "foreign_3_adtv"] = -0.20
    features.loc[index, "foreign_5_adtv"] = -0.10
    features.loc[index, "absorption_signal"] = "없음"
    scored = score_features(features, ModelConfig())
    candidate = scored[scored["symbol"] == "000010"].iloc[0]
    assert candidate["campaign_regime"] in {"지속매집", "장기매집·재가속"}
    assert candidate["action"] == "외국인 재개 확인 후 분할매수"


def test_high_discovery_but_low_timing_is_not_a_buy():
    features, _ = _candidate_features()
    model = ModelConfig(persistent_entry_requires_reacceleration=False, timing_buy_score=80.0)
    candidate = score_features(features, model).query("symbol == '000010'").iloc[0]
    assert candidate["discovery_score"] >= model.discovery_candidate_score
    assert candidate["timing_score"] < model.timing_buy_score
    assert candidate["action"] == "가격 반전·수급 재가속 확인 대기"


def test_distribution_score_can_warn_before_full_campaign_exit():
    features, index = _candidate_features()
    features.loc[index, "price_return20"] = 0.10
    features.loc[index, "rebound_from_low20"] = 0.25
    features.loc[index, "foreign_1_adtv"] = -0.05
    features.loc[index, "foreign_3_adtv"] = -0.80
    features.loc[index, "sellback_ratio20"] = 1.20
    features.loc[index, "upper_wick"] = 0.60
    features.loc[index, "foreign_flow_concentration20"] = 0.55
    candidate = score_features(features, ModelConfig()).query("symbol == '000010'").iloc[0]
    assert candidate["stage"] == "가격상승"
    assert candidate["distribution_score"] >= ModelConfig().distribution_exit_score
    assert candidate["action"] == "수익실현 신호·절반 익절 검토"
