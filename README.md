# REGIME-SHIFT: Macro-Aware Tactical Asset Allocation Engine

**FEC IIT Guwahati — DIY '26 — Quant · Easy**

A dynamic asset allocation engine that detects hidden market regimes (Bull / Bear /
Crisis) with a walk-forward Hidden Markov Model and shifts capital across
equities, bonds, and a safe haven using a regime-conditioned convex optimizer —
benchmarked against static 60/40 and equal-weight portfolios.

## 1. Pipeline

```
Data (yfinance/Treasury) → Feature Engineering → Walk-Forward HMM →
Dynamic CVXPY Optimizer → Backtest vs. Static Benchmarks → Tear Sheet + Charts
```

All of this runs top-to-bottom inside `RegimeShift_Engine.ipynb`.

## 2. Architecture Decisions

**Walk-forward HMM, not a single global fit.** The HMM is refit at every
month-end rebalance date using only data strictly before that date
(`X[:t]`, trailing μ/Σ window ending at `t-1`). A single in-sample fit over the
whole history would leak future regime information into early trading
decisions — the classic look-ahead bias that makes backtests lie.

**Auto-labelling by volatility rank, not manual labels.** Hidden states are
unordered by construction, so after each fit the states are ranked by their
equity-return variance and mapped to Bull → Bear → Crisis (low → high vol).
This keeps regime labels consistent across every walk-forward refit without
hand-picking dates.

**Majority-vote persistence filter.** A single noisy day can flip the raw
Viterbi-decoded state. A 10-day trailing majority vote (60% confirmation
threshold) is applied before a regime change is acted on, which is what keeps
the strategy from thrashing between regimes.

**Concave utility instead of the textbook Cornuejols max-Sharpe
reformulation.** The standard convex max-Sharpe substitution (`y = w/κ`) does
not survive an L1 turnover penalty — that mapping is nonlinear and silently
breaks convexity once transaction costs are added to the objective. Instead,
every regime branch solves a strictly concave mean-variance utility
`μᵀw − λ·wᵀΣw` (low λ in Bull, high λ in Bear/Crisis), with Crisis collapsing
to pure `Minimize(wᵀΣw)` plus an equity cap and safe-haven floor. This stays
convex in every branch, including with the turnover term embedded directly in
the objective rather than bolted on after solving.

**Transaction costs inside the objective, not after.** `tc_bps × ‖w − w_prev‖₁`
sits inside the CVXPY objective for every branch, so the optimizer is aware
of turnover cost while it chooses weights — not just penalized for it after
the fact.

**Expanding z-score, not a global scaler.** Feature standardization at time
`t` uses only rows up to `t` (expanding mean/std), so scaling itself can't
leak future distribution information into the HMM inputs.

**Live data with a synthetic fallback.** `fetch_data()` tries `yfinance`
(SPY/TLT/GLD + ^VIX + ^TNX/^IRX as a yield-spread proxy for FRED) first and
only falls back to a synthetic GBM + 3-state Markov-chain generator if the
network is unreachable. The notebook logs which path it took — **the
submitted version should be the run against live data**, not the synthetic
fallback.

## 3. Repository Structure

```
regime-shift/
├── RegimeShift_Engine.ipynb   # full pipeline, run end-to-end with outputs
├── README.md                  # this file
└── requirements.txt
```

## 4. Reproducibility / How to Run

```bash
git clone <your-repo-url>
cd regime-shift
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook RegimeShift_Engine.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`). Requires network
access for the `yfinance` pull in Section 2 — without it, the notebook falls
back to synthetic data and prints a `[data] yfinance/FRED unreachable`
warning, which should not appear in the final submitted run.

## 5. Known Limitations

- The synthetic fallback intentionally cycles regimes every few weeks to
  stress-test the anti-thrash filter; real macro regimes persist for
  quarters-to-years, so live turnover is materially lower than the synthetic
  path would suggest.
- `^TNX`/`^IRX` are used as a proxy for a CPI/yield-curve macro feature; a
  production version would pull this directly from the FRED API
  (`fredapi` + API key).
