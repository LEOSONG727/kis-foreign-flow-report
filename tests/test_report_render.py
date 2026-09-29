from datetime import datetime
from pathlib import Path

import pandas as pd

from kis_flow_bot import pipeline
from kis_flow_bot.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_real_report_renders_with_paper_summary(tmp_path):
    """Regression: daily run #40 died when a DataFrame sat in DataFrame.attrs.

    Rendering the real stored ranking exercises the brief's pd.concat calls,
    which compare attrs; the paper summary must travel as an argument.
    """
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    scored = pd.read_csv(ROOT / "reports" / "latest.csv", dtype={"symbol": str})
    scored.attrs["jev_monitor_note"] = "모델 감독: 계속 관찰"
    summary = pd.DataFrame(
        {"strategy": ["discovery_top5"], "return": [0.01], "excess": [0.0], "open_positions": [3]}
    )
    output = pipeline._write_outputs(scored, settings, datetime(2026, 9, 29), None, summary)
    assert "모의 운용" in output.read_text(encoding="utf-8")
    assert all(not isinstance(value, pd.DataFrame) for value in scored.attrs.values())
