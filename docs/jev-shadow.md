# Jev review — diagnostics v2

## Roles and model limits

Quant code owns arithmetic, input validity, eligibility, and trading gates. Existing ML shadow estimates a historical ranking. Jev independently judges three narrow observed patterns: episodic/reversing flow, persistent price deterioration, and an extended entry. It receives explicit percent and money units grouped into short flow/price/entry evidence, without stock identity, quant score, ML rank, or a return label.

Official documentation (https://docs.typesafe.ai/model-jaggedness/jev-1.13) recommends arithmetic in code, short relevant context, literal atomic questions, and code-enforced relationships. All three Noul questions share one request. Python derives the review route from the highest diagnostic signal; .35/.65 are frozen experimental boundaries, not calibrated financial probabilities. Missing essential observations produce INSUFFICIENT_DATA. The provider model is pinned to jev-1.13.0; prompt hash, policy, input hash, exact evidence, capture time, actual model, status and token usage are recorded. Authentication/validation errors stop remaining calls for that run.

## Prospective comparison

All methods use the same discovery-eligible, fresh top-12 shortlist (matching the ML shadow limit). The existing report's ranking and trading gates are unaffected. Independent diagnostics appear in the report. A high value is agreement with a risk-pattern statement, not a loss probability.

The whole first cohort of each signal date is immutable: reruns cannot insert replacement stocks. Incomplete API cohorts are retained for coverage accounting but receive no Jev rank. Successful Jev ranks use the maximum diagnostic value, with discovery rank as tie-breaker. This unvalidated aggregation cannot cancel a severe risk by averaging benign answers.

The ledger persists before any outcomes mature. Duplicate price dates are removed. Outcomes require capture before the next-session opening time, preventing hindsight runs from being called prospective. Returns are measured from next available session open to 5th/20th session close. Both gross and configured roundtrip-cost-adjusted returns are kept. Worst entry-relative loss is explicitly not portfolio maximum drawdown. Corporate-action adjustment, missing/suspended sessions, delisted names and execution slippage still need a richer price source for investment-grade backtesting.

Comparison reports isolate policy/model/prompt/cost cohorts and use only dates with complete outcomes and all three top-3 portfolios. It displays paired differences after 20 common dates. Overlapping holding periods are not independent observations; these summaries do not imply statistical significance or trigger automatic promotion.

## Competitiveness assessment (2026-09-27)

The stored ML report has 44 holdout signal dates: -2.26% average 20-day return versus -5.10% baseline; +3.50 percentage-point improvement. Its reported 95% improvement interval is -2.23 to +9.59 percentage points, including zero. Average entry-relative downside is worse (-17.65% versus -15.99%). Thus there is a research signal, but neither stable positive absolute returns nor a reliable advantage has been established. Current-universe survivorship bias and overlapping horizons remain. The historical PASS label should not be interpreted as production-readiness.

The separate rule-policy report shows +0.69% average 20-day return and +4.94% market excess on its own cohort. That cohort and entry policy differ from the ML holdout; comparing the headline figures directly is invalid. Jev has no live performance record yet.

Before production use: verify actual API responses; collect versioned forward outcomes; compare net paired returns and downside across multiple regimes with block-based uncertainty estimates; include costs, capacity, turnover, corporate actions and delisted stocks; require benefit versus the existing rules, not merely plausible-sounding model judgments. Jev may be removed if its incremental value is absent. Source-backed filing/news classification could add new information later, but no such feed is currently connected.

## Automated model-health notification

Every daily run now computes the latest cohort's paired performance summary in Python and asks Jev Choice for OBSERVE / DIAGNOSE / REVIEW. The notification uses fixed Korean templates. Numerical calculations, missing-data and operational-error guards remain in code; performance review is blocked before 20 common completed signal dates. Provider failures and low confidence are explicitly labeled as code fallback. The JSON record includes the evidence, proposed model decision, enforced decision and source in reports/jev_monitor_latest.json. The daily HTML, existing email and Telegram messages include the status. This is automatic triage and notification, not automatic code editing or promotion. It depends on the configured GitHub Actions job and notification secrets.
