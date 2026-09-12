from __future__ import annotations

from datetime import datetime

from kis_flow_bot.config import ModelConfig
from kis_flow_bot.report import eok, render_report
from kis_flow_bot.scoring import score_features

from test_features import synthetic_features


def test_report_contains_ranking_and_action():
    scored = score_features(synthetic_features(), ModelConfig())
    rendered = render_report(scored, 20, ("000010",), datetime(2026, 8, 16, 18, 41))
    assert "외국인 매집 레이더" in rendered.html
    assert "매집후보" in rendered.html
    assert "초기 분할매수" in rendered.html
    assert "오늘 볼 종목" in rendered.html
    assert "오늘 결론" in rendered.html
    assert "살 만한 종목이 있나?" in rendered.html
    assert rendered.html.index("살 만한 종목이 있나?") < rendered.html.index("다음 후보 · 아직 사지 않음")
    assert "강한 후보 · 가격 대기" in rendered.html
    assert "오늘 신규매수 검토" in rendered.html
    assert "참고 · 기존 보유/후속 관리" in rendered.html
    assert "가격·단기열기" in rendered.html
    assert "20일 저점반등" in rendered.html
    assert "외인 순매수" in rendered.html
    assert "분산위험" in rendered.html
    assert "다음 거래일 시가 +1% 초과" in rendered.html
    assert "외국인 매집 액션 리포트" in rendered.markdown
    assert "오늘 결론 · 살 만한 종목이 있나?" in rendered.markdown
    assert "저점반등 20일" in rendered.markdown
    assert "다음 후보 · 아직 사지 않음" in rendered.markdown


def test_eok_does_not_render_negative_zero():
    assert eok(-1.0) == "0억"
    assert eok(-40_000_000.0) == "-0.4억"
