# Exploration Report — Search-and-Pair (5 Agents)

> Panel throughout: `generate(seed=11, n_stocks=1000)`, 783 days 2020-01-01–2022-12-30, refs from `run.py` REF_EXPRS, all via existing API. No combined alpha built — master authorizes H1–H3 for a separate follow-up build step only.

---

## Explorer Logs (every test, nulls included)

### Explorer_Sentiment_Fundamentals — 22 tests

Coverage thr = max(5, 1%) = 10; early-2020 min=0 = window warmup, not a break.

| # | Expression | S / F / TO% | Yearly S (Ret%) | Coverage | Shape → break |
|---|すすめ---|---|---|---|---|
| 01 | `rank(ebitda)` | +0.89 / +0.25 / 0.14 | +0.88 / −0.89 / +2.47 | full | V-shape → breaks Jan-21 and Jan-22 |
| 02 | `rank(sales)` | −0.93 / −0.26 / 0.06 | −1.33 / −2.06 / +0.42 | full | neg thru 2021, flat after Jan-22 |
| 03 | `rank(debt)` | −0.11 / −0.01 / 0.05 | −0.38 / −1.44 / +1.46 | full | worst 2021, pos only after Jan-22 |
| 04 | `rank(assets)` | −0.60 / −0.14 / 0.06 | −1.10 / −1.73 / +0.89 | full | bleeds thru 2021 |
| 05 | `ebitda/sales` raw | +2.57 / +1.26 / 0.24 | +3.62 / +0.80 / +3.28 | ok | dip Jan-21 (4.2→0.9%), recovers Jan-22; TO fails |
| 06 | `rank(ebitda/sales)` | +2.67 / +1.28 / 0.23 | +3.85 / +1.01 / +3.13 | 500/500 | same dip, milder; TO fails |
| 07 | `debt/assets` raw | +0.10 / +0.01 / 0.00 | +0.78 / −0.79 / +0.32 | full | flat wiggle, null; TO fails |
| 08 | `rank(debt/assets)` | +0.37 / +0.07 / 0.00 | +0.61 / −0.26 / +0.80 | full | null; TO fails |
| 09 | `rank(ts_backfill(debt,32)/ts_backfill(assets,32))` unsigned | +0.37 / +0.07 / 0.00 | same as 08 | full | identical to 08 — backfill no-op on dense fundamentals. FAILs here (note: user reports UNSIGNED passes on real Brain — simulator/real sign discrepancy flagged) |
| 10 | negated twin | −0.37 / −0.07 / 0.00 | mirror | full | exact mirror, equally null |
| 11 | `ts_delta(ebitda/sales,32)` | +2.94 / +1.50 / 1.29 PASS | +3.69 / +2.05 / +3.12 | warmup flag only | steady climb, no break |
| 12 | `ts_delta(ebitda/sales,63)` | +2.94 / +1.50 / 1.29 PASS | same (quarterly steps → 32≈63) | warmup flag only | no break |
| 13 | `ts_delta(ebitda/sales,126)` | +0.86 / +0.23 / 0.98 | +1.55 / +1.04 / +0.30 | warmup flag | decaying, flat by 2022 — long window kills edge |
| 14 | `ts_delta(rank(ebitda/sales),32)` | +2.83 / +1.46 / 1.29 PASS | +3.92 / +2.10 / +2.53 | warmup flag | steady, no break |
| 15 | `ts_delta(rank(debt/assets),32)` | +0.21 / +0.10 / 0.68 | +1.16 / +0.02 / −0.29 (Ret +10.8/+0.3/−3.3) | means ~0–1 FLAG | one-year wonder, dead after Jan-21 |
| 16 | `vec_avg(nws12…)` raw | +1.36 / +0.36 / 69.7 | +2.70 / +1.22 / +1.01 | single-digit finite FLAG | choppy, TO fails |
| 17 | `ts_mean(backfill(vec_avg,20),20)` | +3.44 / +1.87 / 3.40 PASS | +4.07 / +1.56 / +4.76 | ok post-warmup | U-shape: dip Jan-21, strongest 2022 |
| 18 | `…ts_mean(…,60)` | +2.63 / +1.22 / 1.60 PASS | +2.82 / +1.26 / +4.03 | ok | same U, weaker |
| 19 | `ts_zscore(vec_avg,20)` no backfill | +0.67 / +0.11 / 76.5 | +1.86 / +0.22 / +0.55 | thin FLAG | fades after 2020; TO fails |
| 20 | `rank(backfill(buzz,20))` | −0.16 / −0.01 / 21.7 | +0.56 / −0.69 / −0.30 | ok | pos only 2020, neg after Jan-21; null |
| 21 | `ts_zscore(buzz,20)` raw | −0.58 / −0.08 / 76.7 | −0.38 / +0.89 / −2.74 (Ret +2.9→−6.7) | thin FLAG | sharpest flip: Jan-22 collapse |
| 22 | `ts_zscore(backfill(buzz,20),60)` | −0.11 / −0.01 / 23.1 | +0.42 / −0.43 / −0.26 | ok | flat null |

Sharpest for master: Jan-22 buzz collapse (#21); Jan-21 multi-family sag (#01/06/17/20, REGIME 1.5→0.5); leverage-change death (#15).

### Explorer_Microstructure_Volatility — 16 tests

Refs: mom60 +1.11, rev5 +1.59, quality +1.82, vol-spike −0.92, mom120 +0.86, lowvol +0.95, asset-turn −0.68, highvol −0.95.

| ID | Expression | S / F / TO% | Yearly S | Shape → break |
|----|---|---|---|---|
| T01 | `-ts_zscore(returns,5)` | 1.59 / 0.27 / 70.7 FAIL | 1.23 / 1.13 / 2.35 | steady-up, 21 sag |
| T02 | `-ts_zscore(returns,10)` | 3.70 / 0.97 / 70.0 FAIL fit/TO | 4.00 / 2.69 / 4.35 | same, larger |
| T03 | `-ts_zscore(returns,21)` | 4.51 / 1.32 / 69.2 PASS | 5.30 / 3.11 / 5.09 | kink-flat in 21; N-sweep works (5<10<21) |
| T04/T05 | `±rank(ts_std_dev(returns,20))` | ∓0.95 / ∓0.30 / 6.5 FAIL | +2.74 / +0.74 / −0.48 (long leg) | lowvol dies AND flips sign Jan-22 |
| T06/T07 | `±rank(ts_std_dev(returns,60))` | ∓0.55 / ∓0.13 / 2.25 FAIL | +2.92 / +0.03 / −0.87 | longer window damps 2021 to ~0, confirms flip |
| T08 | `ts_mean(returns,60)` | 1.11 / 0.43 / 8.5 FAIL | 0.73 / 1.04 / 1.53 | grind-up, NO 21 dip (control) |
| T09 | `ts_mean(returns,120)` | 2.61 / 1.53 / 5.4 PASS | 2.80 / 2.49 / 2.69 | straight line, no dip (control) |
| T10/T11 | `ts_decay_linear(-zscore(returns,5),10/30)` | 6.71 / 3.13 and 5.89 / 2.82 PASS | 9.24 / 4.08 / 6.98 | decay boosts; same 21 halving (w=10>30) |
| T12 | `ts_covariance(returns,volume,20)` | −0.71 / −0.31 / 12.7 FAIL + WC 11.3% | −1.10 / −1.41 / +0.26 | only conc violator; skewed coverage artifact |
| T13 | `ts_corr(returns,volume,20)` | −1.43 / −0.49 / 13.1 FAIL | −2.60 / −0.75 / −1.04 | persistently neg, no rescue |
| T14 | `group_neutralize(-zscore(returns,5),sector)` | 1.51 / 0.24 / 70.8 FAIL | 1.49 / 1.07 / 1.92 | preserves shape, small haircut |
| T15 | `rank(volume/ts_mean(volume,20))` | −0.92 / −0.11 / 64.7 FAIL | −1.46 / −0.89 / −0.48 | steady bleed, no break |
| T16 | `trade_when(volume>adv20,−zscore(returns,5),−1)` | 0.0 NULL | 0 / 0 / 0 | 0.2% finite — threshold never true, needs rescale |

Sharpest: Jan-21 collapse 1.5→0.5 (all reversal/vol legs halve; absent in T08/T09); Jan-22 recovery + lowvol inversion; warmups (01-07/01-28/03-24/06-16) explain all 2020 min-0 flags.

### Explorer_Derivatives_IV — 18 tests

Only 10/60 tenors exist (20/120 → Unknown field). Raw coverage identical all legs: 37.4% every year (shared mask — no tenor sparsity edge; backfill 10→90.9%, 15→95.2%, 20→97.3%).

| ID | Expression | S / F / TO% | Yearly S | Shape |
|----|---|---|---|---|
| T01–T04 | single-leg levels `-rank(backfill(put/call_10/60,15))` | puts ~1.28–1.33 FAIL fit; calls ~0.65–0.67 FAIL | puts +4.0/+0.4/−0.3; calls +3.0/+0.3/−1.1 | 2020 spike, die/flat-to-down 2021–22 |
| T05/T07 | raw diffs (no backfill) | −1.00 / −1.28, TO 69–75%, conc 8–9.5% FAIL | neg/choppy | NULL + concentration flags |
| T06 | `-rank(backfill(put_10,15)−backfill(call_10,15))` | 2.62 / 1.11 / 17.5 PASS | +4.48 / +0.01 / +3.39 | barbell: strong 20+22, flat 21 |
| T08 | 60-diff version | 1.81 / 0.57 / 22.0 FAIL fit | +2.16 / +0.19 / +3.33 | stall then rally; mid-21 DD |
| T09 BRAIN | `-rank(backfill(put_10,15)/backfill(call_10,15))` | 2.72 / 1.13 / 17.8 PASS | +4.56 / −0.08 / +3.70 | same barbell, 21 dead flat; DD 2021-04-13→12-30, worst 60d 09-22→12-14 (−0.81), worst month Nov-21 (−0.44) |
| T10–T14 | 60-ratio + backfill 10/20 variants | 1.91–2.68 | same barbell, weaker (60) | 60 breaks earlier (DD from 2020-11-02) |
| T15–T17 | delta/zscore on IV | \|S\|<0.4 FAIL | ~0 | flat nulls |
| T18 | smoothed call_60 level | −0.47 FAIL | −2.47 / +0.17 / +0.68 | bleed then flat |

Division audit (all ratios): denom median ~37.5, p1 ~21.7, min 0.56–0.73; |ratio|>5 only 0.11–0.14% cells, top-1% mass 3–4% — stable, rank-capped, no blow-up.

Sharpest: 10-tenor flatline Apr→Dec 2021; 60-tenor DD Nov-20→Aug-21 (lead); levels selloff May→Aug 2022 where pairs rally.

### Explorer_RegimeClassifier — 20 conditions

Clean #1: `ts_mean(close,21)>ts_mean(close,63)` (≡ cap version): TRUE 2020-06-16→2021-07-28 (292d bull), FALSE 2021-07-29, TRUE 2021-08-31; fracs .795/.835/.758. Clean #2: `ts_mean(returns,63)>ts_delay(ts_mean(returns,63),63)`: TRUE 2021-02-09→05-14, FALSE 2021-05-17→08-30 (Q3 1/66 TRUE), TRUE 08-31; fracs .774/.471/.465. Backup #3: `ts_mean(returns,63)>0.001`: TRUE 2021-01-27→05-27, FALSE from 05-28; Q2 .969→Q3 .197. Rejected: fast/noisy (21d trend 80 flips; vol-vs-trailing 55–83 flips), degenerate always-TRUE/FALSE (120d trend, rank>0.5 by construction, volume*close>adv20 breadth 0.43, rescaled always-TRUE, composite with warmup artifact), bear mirror (no independent value).

---

## Master Report

### Matched clusters

- **A — Jan-21 sag (1.5→0.5), strong:** micro reversal halving (exact 12-31→01-01) + sentiment multi-family dip (exact month) + 60-tenor IV DD onset 11-02 (approx lead) + high-bar trend TRUE from 01-27 (approx lag). CLEAN #1 still TRUE → stall, not flip.
- **B — Mid-21 stall Apr→Dec 2021, strongest:** 10-tenor IV DD 04-13 (exact anchor) + CLEAN #2 FALSE 05-17 + BACKUP #3 FALSE 05-28 (~4–6 wk) + CLEAN #1 flip 07-29 + worst 60d 09-22→12-14 + micro/sentiment yearly sag (loose).
- **C — Jan-22 recovery/inversion, moderate-strong:** buzz +2.89→−6.71 + reversal 5.39→9.62 + lowvol sign-flip (+0.74→−0.48) all exact at boundary; IV pairs rally vs levels selloff (approx, ~5 mo lag). No classifier flip — trigger must be proxied.
- **Stabilizers (no break anywhere):** `ts_delta(ebitda/sales,32/63)` (~2.94) and `ts_mean(returns,120)` (2.61) — ideal complement legs.

### Ranked pairings (handoff for separate build/test — no code here)

1. **Trend-switch: short reversal ↔ long momentum.** TRUE (bull/expansion) → 120d momentum; FALSE (stall/bear) → 21d reversal (N=21 dominates 10/5). Corroborated in all 3 clusters. *Why:* canonical horizon diversification — drift crowds out reversion in trend, pays liquidity provision in dispersion; condition measures the drift it switches on. Cleanest factors, no conc/coverage defects.
2. **Stall-switch: IV skew ↔ fundamental drift.** Acceleration TRUE (e.g. 02-09→05-14) → backfilled 10-tenor put/call skew (ratio; diff backup); FALSE (05-17→08-30, Apr→Dec flatline) → slow `ts_delta(ebitda/sales,32/63)`. 4-way exact in cluster B (skew +4.56→−0.08→+3.70 vs drift +3.9/+2.1/+3.1). *Why:* skew needs directional trend to trend (flatlines −0.08/yr sideways); earnings diffusion grinds through digestion; audited tradable (denom median 37.5, extremes 0.1%, rank-capped, backfill required).
3. **Bear-switch: low-vol ↔ reversion.** TRUE (bull/calm) → low-vol; FALSE (bear/dispersion) → reversal, buzz collapse as confirm. Exact mirror Jan-22 (lowvol flips sign as reversal recovers). *Why:* defensive crowding unwinds in shock (long-duration selloff) while dispersion pays reversion; attention exhaustion consistent. Weaker only from trigger imprecision (no Jan-22 classifier flip — needs robustness check).

### Discarded (reasons)

Tenor-internal IV switch (same family, offset windows, no cross-asset story); single-leg IV levels (die after 2020, dominated by pairs); buzz-vs-buzz internal switch (same input, self-cancelling); debt-delta one-year wonder (single-year artifact, DD 18.9%); unsigned debt/assets rank (simulator FAIL +0.37 vs real-Brain PASS — validity unresolved, excluded pending reconciliation); raw news / volume>adv20 / static debt (NULL: TO ~70%, 0.2% finite, 0.00% TO); fast/degenerate classifiers (whipsaw or always-TRUE/FALSE incl. warmup artifact); flat IV delta/zscore (no break); N=5 reversal alone (dominated by N=21).

**Authorization:** follow-up build/test authorized for H1–H3 in rank order only.
