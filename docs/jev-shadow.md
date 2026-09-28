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

## Historical replay (2026-09-28)

Jev's input has no stock identity, date or score, so asking it about past signal dates cannot leak later prices. `python -m kis_flow_bot research` (weekly, and once via `validate-self-research.yml`) sends every labelled date's live-equivalent 12-name shortlist through the same request/parse code, stores answers in `data/jev_replay.csv` (resumable, keyed by date, symbol and prompt hash) and writes `reports/jev_replay.md` and `data/jev_verdict.json`.

It measures three things:

1. **Code replicability.** An out-of-fold code model of the same ten inputs (rank, clipped raw value and squares; contiguous date folds) predicts Jev's answer. High R² means code can reproduce Jev.
2. **Jev's own judgment.** Answer minus that replica, correlated with the next 20-session excess return within each day's shortlist (date-block bootstrap).
3. **Use.** Top 3 by lowest Jev risk and "discovery top 3 without REJECT" versus discovery top 3 on the same dates, with empty slots as cash.

Pre-registered verdict: `ADDS_VALUE` requires the own-judgment correlation's upper bound < 0 **and** the lowest-risk top-3 improvement's lower bound > 0 over at least 40 complete dates. Otherwise `NO_VALUE`. With `NO_VALUE` for the current prompt hash and model, the daily pipeline stops calling Jev (`VALIDATION_OFF`). A new prompt or model version restarts validation automatically.

## Model-health note

The daily note is computed in code. The previous Jev choice call was overridden by code guards in every case that mattered. The note combines operational status, the weekly self-validation registry and the replay verdict. It requests human review when the challenger passes promotion, the rule champion turns negative after the freeze date, or the replay finds added value. It never edits code or rules.

## Where Jev could add information

The ten numbers above are already available to code, and the self-validation shows they carry little cross-sectional signal. Jev is most likely to add value on inputs code cannot read: for example, classifying filings or news around abnormal foreign flow (index rebalancing, block deals, rights issues) versus organic accumulation. That needs a point-in-time filing/news source (DART or KIS news titles) and can be evaluated with the same replay and gate.

## Filing review (jev-events-v1)

Jev now reads what code does not: DART filing titles received during the 30 days up to the signal date. Company names become `회사`, dates become "within 7 days" or "8-30 days before", and the filer is reduced to "the company itself" or "another party". Three narrow questions ask about dilution supply (rights issues, CB/BW/EB, treasury-share disposal), serious adverse events (litigation, embezzlement, audit, suspension, delisting review) and positive catalysts (supply contracts, buybacks and cancellation, dividends). Answers are cached per distinct filing state, so consecutive days with the same filings cost one call.

A keyword rule reads the same titles. `python -m kis_flow_bot event-replay` (also part of the weekly `research`) compares, on each labelled date's 12-name shortlist, discovery top 3 against top 3 after removing dilution/adverse names by Jev and by keywords (next-ranked names fill in; empty slots are cash). It also reports excess return and the share of holdings falling 15% below entry. Verdicts: `ADDS_VALUE` only when Jev's veto beats the keyword veto (lower bound > 0) and beats the baseline on average; `KEYWORDS_ENOUGH` when the keyword veto alone beats the baseline, in which case code should use it and Jev is unnecessary; otherwise `NO_VALUE`.

Daily runs record the shortlist's filing flags in `data/jev_event_live.csv` before outcomes exist and show them in the report as reference only. Ranking and buy gates do not change until a verdict supports it.
