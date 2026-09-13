# Signal Generation Module — Implementation Plan
## Dark Wing Pairs Trading System | Step 8

---

## Goal

Convert the stationary spread $S_t$ and its Z-score $Z_t$ into **actionable trading signals** for the pairs-trading engine.
This is **Step 8** in the pipeline — it sits directly after the Kalman Filter and Forecasting modules and feeds into the backtest and live-execution layers.

---

## Pipeline Position

```
STEP 1: Data Ingestion
STEP 2: Stationarity Tests (ADF, KPSS)
STEP 3: Cointegration Tests (Johansen, EG, Rolling)
STEP 4: VECM Fit + Diagnostics
STEP 5: Spread Construction (SpreadBuilder, HalfLife, ZScore)
STEP 6: Kalman Filter (KalmanFilterPairs, Smoother, DynamicBeta)
STEP 7: Forecasting (VECMForecaster, KalmanForecaster)
======================================================
STEP 8: THIS MODULE — Trade_Implement/Signals/
======================================================
STEP 9: Risk Management  (Trade_Implement/Risk_Management/)
STEP 10: Backtesting     (backtest/)
STEP 11: Live Execution  (Angel_1_Implement/, IBKR_Implement/)
```

---

## Files to Create

### 1. `pair_signal.py` — Core Signal Engine

**Class:** `PairSignalGenerator`

Primary signal source. Wraps `ZScore` output and `KalmanForecaster` / `DynamicBeta` outputs
into a discrete signal stream (+1, 0, -1, +2 for stop) with position-level metadata.

#### Mathematical Foundation

**Z-score signal rule (threshold method):**

$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

| Signal | Condition | Action |
|--------|-----------|--------|
| **LONG** spread | $Z_t < -\theta_{entry}$ | Buy Y, Sell X |
| **SHORT** spread | $Z_t > +\theta_{entry}$ | Sell Y, Buy X |
| **EXIT** | $|Z_t| < \theta_{exit}$ | Close position |
| **STOP** | $|Z_t| > \theta_{stop}$ | Emergency flat |

**Kalman Forecast Confirmation Filter:**

$$\text{Confirm LONG if } \hat{S}_{t+h|t} > S_t$$

$$\text{Confirm SHORT if } \hat{S}_{t+h|t} < S_t$$

**Regime filter (from DynamicBeta.regime_detect()):**

$$\text{CUSUM}_{t} = \max(0,\ \text{CUSUM}_{t-1} + |\Delta\beta_t| - k)$$

If $\text{CUSUM}_t > h_{\text{threshold}}$: suppress signals (pair breakdown).

**Position Sizing (Kelly-fractional):**

$$f_t = \frac{\mu_{Z}}{\sigma_{Z}^2} \cdot f_{max}$$

where $f_{max}$ is a cap (e.g. 0.25 of capital per leg).

#### Methods

```
PairSignalGenerator
│
├── __init__(zscore, dynamic_beta, kalman_forecaster, ...)
├── generate()              → pd.DataFrame [signal, position, z, spread, beta, regime]
├── current_signal()        → dict: latest signal + metadata
├── signal_series()         → pd.Series of {-1, 0, 1}
├── position_series()       → pd.Series (stateful: entered/exited/stopped)
├── _apply_thresholds()     → private: pure Z-score threshold logic
├── _apply_forecast_filter()→ private: Kalman forecast confirmation
├── _apply_regime_filter()  → private: CUSUM regime suppression
├── plot()                  → 3-panel: spread, Z-score with bands, signals
└── summary()               → print/dict of stats
```

---

### 2. `multi_pair_signal.py` — Portfolio Signal Aggregator

**Class:** `MultiPairSignalAggregator`

Runs `PairSignalGenerator` across N pairs simultaneously and aggregates into a
portfolio-level signal frame. Handles cross-pair capital allocation and signal conflicts.

#### Mathematical Foundation

**Capital allocation per pair (inverse-variance weighting):**

$$w_i = \frac{1/\sigma_i^2}{\sum_{j=1}^{N} 1/\sigma_j^2}$$

**Signal consensus (majority vote):**

$$\text{Signal}_{consensus} = \text{sign}\left(\sum_{i=1}^{N} w_i \cdot s_{i,t}\right)$$

**Correlation-adjusted exposure cap:**

$$\text{Exposure}_{adj} = \sqrt{w^T \Sigma w} \leq \text{MaxExposure}$$

#### Methods

```
MultiPairSignalAggregator
│
├── __init__(pairs_config, capital, max_pairs_open, correlation_cap)
├── add_pair(name, zscore, dynamic_beta, kalman_forecaster)
├── run_all()              → dict[str -> pd.DataFrame] per-pair signals
├── portfolio_signals()    → pd.DataFrame [pair x time] signal matrix
├── capital_weights()      → pd.Series of pair weights
├── aggregate_signal()     → pd.Series: portfolio-level net signal
├── active_pairs()         → list of pairs with open signals
├── plot_all()             → grid of per-pair signal charts
└── summary()              → portfolio-level stats table
```

---

## Interface Contracts

| Source | What it provides | Used in |
|--------|-----------------|---------|
| `ZScore.zscore_series()` | $Z_t$ pd.Series | `PairSignalGenerator.__init__` |
| `ZScore.generate_signals()` | Raw threshold signals | `_apply_thresholds()` |
| `DynamicBeta.regime_detect()` | Regime flag Series | `_apply_regime_filter()` |
| `KalmanForecaster.forecast_spread()` | $\hat{S}_{t+h}$ | `_apply_forecast_filter()` |
| `DynamicBeta.current_beta()` | Latest beta_t | `current_signal()` |

---

## Additional File Suggestions

| File | Purpose |
|------|---------|
| `regime_signal.py` | Dedicated HMM/CUSUM regime detection to override Z-score signals |
| `signal_logger.py` | Append each signal event to CSV/SQLite for audit trail |
| `signal_validator.py` | Pre-trade checks: min spread age, liquidity, correlation check |

---

## Verification Plan

1. Unit test `pair_signal.py` on synthetic spread (sine wave): verify LONG/SHORT/EXIT fires correctly.
2. Integration: Wire `DynamicBeta` -> `KalmanForecaster` -> `PairSignalGenerator` on real pair.
3. Plot check: 3-panel plot shows entry/exit markers at correct Z-score crossings.
4. Multi-pair: `MultiPairSignalAggregator.portfolio_signals()` produces clean DataFrame for 3 pairs.
