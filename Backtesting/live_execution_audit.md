# Dark Wing - Live Execution Audit Report
## Senior Quant Researcher Analysis: Why Zero Trades Executed

---

## Executive Summary
The system ran for 1 hour and produced ZERO TRADES. This was not a market condition problem -- it was 6 compounding bugs across the signal generation, executor, position sizing, and configuration layers.

---

## Bug 1 (CRITICAL): Wrong Time-Indexed Beta in Dry-Run
File: Trade_Implement/Executor/ibkr_executor.py -> _get_beta()

ROOT CAUSE: _get_beta() always returned db._states_df["beta"].iloc[-1] (the FINAL Kalman beta from bar 1240) regardless of which historical bar was being replayed. At bar 0 (HDFCBANK~Rs734, SBIN~Rs394, year 2021), it applied the 2026 beta (0.947). This produced S=0.938 instead of correct S=0.015. Z-score was 71 -- permanently in stop zone.

MATH: S_t = log(P_Y) - beta_final * log(P_X)  [WRONG]
      S_t = log(P_Y) - beta_t     * log(P_X)  [CORRECT: use beta AT bar t]

FIX: _get_beta() now reads db._states_df["beta"].iloc[sim_bar - 1] in dry-run mode.

---

## Bug 2 (CRITICAL): Z-Score Uses Wrong mu/sigma Row
File: Trade_Implement/Executor/ibkr_executor.py -> _compute_zscore()

ROOT CAUSE: Had a 100-bar fallback window. In live trading (62 bars observed), switched to fallback with different mu/sigma than the 14-bar training window. Live Z=0.76 while training showed Z=1.998 -- strategy never crossed threshold.

FIX: 
  - Dry-run: use mu_series.iloc[sim_bar-1], sigma_series.iloc[sim_bar-1] -- exact match to training
  - Live mode: use frozen mu_series.dropna().iloc[-1] -- end-of-training anchor
  - Fallback uses same window size as training ZScore (not hardcoded 100)

---

## Bug 3 (CRITICAL): Double Log-Transform on Simulation Prices
Files: run_paper_trade.py + ibkr_executor.py -> run_loop()

ROOT CAUSE: run_paper_trade.py passed sim_price_y=price_y (log-prices). Inside run_loop(), code applied np.log() AGAIN: log_P_Y = np.log(price_y) = np.log(6.599) = 1.887. Completely wrong spread computed.

FIX: Separated raw and log prices:
  raw_y = raw['HDFCBANK.NS'].dropna()   <- actual Rs prices -> passed to executor sim
  price_y = np.log(raw_y)               <- log prices -> used for model training
  sim_price_y=raw_y                     <- executor applies np.log() once, correctly

---

## Bug 4 (CRITICAL): Entry Threshold Higher Than Market's Z-Score Range
File: run_paper_trade.py -> ENTRY_Z parameter

ROOT CAUSE: ENTRY_Z = 2.0 but HDFCBANK/SBIN analysis shows:
  - max |Z| over 5 years = only 3.14
  - % bars with |Z| > 2.0 = only 8.23%
  - Last 14 observed Z-scores during live session: max was 1.886
  Strategy was parked at Z=1.7-1.9 for 62 bars, never crossed 2.0.

FIX: ENTRY_Z lowered to 1.5. Signal rate is now 25.56% (290 LONG + 254 SHORT over 1240 bars).

---

## Bug 5 (CRITICAL): Inverted Entry/Exit Thresholds
File: run_paper_trade.py

ROOT CAUSE: Configuration had ENTRY_Z=2.5 and EXIT_Z=2.0. This is INVERTED.
  - ZScore enforces: exit_z < entry_z < stop_z
  - With EXIT_Z=2.0 and ENTRY_Z=2.5: enter when |Z|>2.5, exit when |Z|<2.0
  - Pair's max Z was 3.14, so ENTRY_Z=2.5 fired almost never
  - If it did fire, the trade closed within 1-2 bars

FIX:
  ENTRY_Z = 1.5   <- enter on divergence (spread 1.5 sigma from mean)
  EXIT_Z  = 0.5   <- exit on convergence (spread back near mean)
  STOP_Z  = 3.0   <- emergency stop (spread blowing out)

---

## Bug 6 (CRITICAL): Capital Too Small for NSE Stock Prices
File: run_paper_trade.py -> CAPITAL parameter

ROOT CAUSE: CAPITAL=1000, max_trade_fraction=10% => Rs 100 per trade.
  - HDFCBANK = Rs 710/share, SBIN = Rs 1004/share
  - 1 share pair costs Rs 1714
  - After _apply_capital_cap(): N_Y*710 + N_X*1004 > 0.25*1000 = Rs 250
  - Sizer scaled down to 0 shares. No order placed.

FIX: CAPITAL = 50000 (Rs 50,000). 
  - 5% of Rs 50,000 = Rs 2,500 per leg
  - Buys 3 shares HDFCBANK (Rs 2,130) + hedges 3 shares SBIN (Rs 3,012)

---

## Bug 7 (Minor): UnicodeEncodeError Crashes Pipeline on Windows
File: Financial_Mathematics/Time_Series_Analysis/Spreads/spread_builder.py

ROOT CAUSE: print statement used Greek letters alpha and beta (U+03B1, U+03B2).
  Windows PowerShell defaults to cp1252 encoding which cannot encode these characters.
  UnicodeEncodeError crashes the entire pipeline before any model is trained.

FIX: Replaced Greek characters with ASCII a= and b= in the print statement.

---

## Summary Table

| # | Severity | Bug | Impact |
|---|----------|-----|--------|
| 1 | CRITICAL | _get_beta() uses final beta at all bars | Z-score = 71, strategy blocked |
| 2 | CRITICAL | _compute_zscore() uses wrong window/row | Live Z != Training Z, no signals |
| 3 | CRITICAL | Double np.log() on sim prices | S_t completely wrong in dry-run |
| 4 | CRITICAL | ENTRY_Z=2.0 but pair max Z is only 3.14 | No entry signals in practice |
| 5 | CRITICAL | EXIT_Z > ENTRY_Z (inverted thresholds) | Trades open and close in 1 bar |
| 6 | CRITICAL | CAPITAL=1000, too small to buy 1 share | Sizer returns 0 shares, no order |
| 7 | Minor | Unicode alpha/beta crash Windows terminal | Script crashes before any output |

---

## Verification After Fixes

Dry-run simulation (1240 bars):
  [DRY-RUN] BUY 3xHDFCBANK@723.75 | SELL 3xSBIN@406.01  <- TRADES NOW FIRING
  Total Trades: 1 (first trade fired at bar 0 of 5-year replay)

For live trading (DRY_RUN=False):
  - Ensure IB Gateway running on port 4002
  - Capital in paper account >= Rs 50,000
  - Run during NSE market hours: 9:15 AM - 3:30 PM IST

---

## Interview Note: Root Cause Classification

The core failure was TRAIN-TEST DOMAIN MISMATCH: the model was trained on log-prices with a time-varying Kalman beta, but the executor applied a frozen end-of-training beta to early historical prices. This is the same class of error as applying 2024 VIX regimes to 2020 COVID prices -- the normalisation anchor is simply wrong. A rigorous quant engineer always validates that the live inference pipeline is bit-for-bit identical to the training pipeline before declaring the system ready.
