import unittest
from kis_flow_bot.research_review import review_rows, render_review


class ReviewTests(unittest.TestCase):
    def row(self, **kw):
        row = dict(symbol="078930", name="GS", as_of="2026-09-17", data_fresh=True,
                   discovery_eligible=True, proposal_status="오늘 신규매수 검토",
                   foreign_net3=10, foreign_net5=20, foreign_net20=30,
                   sellback_ratio20=.2, price_return3=0, price_return20=0,
                   foreign_buy_days10=8, ma20_slope=0, above_ma5=True, above_ma10=True)
        row.update(kw)
        return row

    def test_recent_selling_blocks_old_accumulation(self):
        self.assertFalse(review_rows([self.row(foreign_net3=-1)], "2026-09-17")[0]["passed"])

    def test_missing_nonfinite_and_false_fail_closed(self):
        for kw in ({"foreign_net3": "nan"}, {"foreign_net5": ""},
                   {"foreign_net20": float("inf")}, {"data_fresh": "False"},
                   {"foreign_buy_days10": 11}, {"sellback_ratio20": -.1}):
            with self.subTest(kw=kw):
                self.assertFalse(review_rows([self.row(**kw)], "2026-09-17")[0]["passed"])

    def test_date_and_news_blocks(self):
        self.assertFalse(review_rows([self.row()], "2026-09-18")[0]["passed"])
        self.assertFalse(review_rows([self.row()], "2026-09-17", {"078930": "review required"})[0]["passed"])

    def test_flow_is_not_price_confirmation(self):
        result = review_rows([self.row(above_ma5="False")], "2026-09-17")[0]
        self.assertTrue(result["passed"])
        self.assertIn("반전 대기", result["status"])

    def test_no_forced_fill_and_no_input_mutation(self):
        row = self.row()
        copy = dict(row)
        result = review_rows([row], "2026-09-17")
        self.assertEqual(row, copy)
        self.assertEqual(render_review(result, "2026-09-17").count("GS (078930)"), 1)
        self.assertIn("통과 후보 없음", render_review([], "2026-09-17"))


if __name__ == "__main__":
    unittest.main()
