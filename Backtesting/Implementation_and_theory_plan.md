# Backtesting Module — Implementation & Theory Plan
## Dark Wing Pairs Trading System | Step 10

---

## Pipeline Position

```
STEP 1-4  → Financial_Mathematics/  (Stationarity, Cointegration, VECM)
STEP 5    → Spreads/                 (SpreadBuilder, HalfLife, ZScore)
STEP 6-7  → Kalman_Filter/           (KalmanFilterPairs, DynamicBeta, VECMForecaster, KalmanForecaster)
STEP 8    → Trade_Implement/Signals/ (PairSignalGenerator, MultiPairSignalAggregator)
STEP 9    → Trade_Implement/Risk_Management/ (PositionSizer, StopLossManager)
══════════════════════════════════════════════════
STEP 10   → Backtesting/             ← THIS MODULE
══════════════════════════════════════════════════
STEP 11   → Trade_Implement/Angel_1_Implement/ | IBKR_Implement/
```

---

## Directory & Import Integration

### Path Resolution Strategy

`Backtesting/` sits at the **project root level**, one level above `Trade_Implement/` and `Financial_Mathematics/`. The path setup block used in all files of this project is followed here exactly:

```python
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../Backtesting/
_ROOT_DIR    = os.path.dirname(_THIS_DIR)                    # .../Dark Wing/
_FM_DIR      = os.path.join(_ROOT_DIR, "Financial_Mathematics")
_TSA_DIR     = os.path.join(_FM_DIR, "Time_Series_Analysis")
_SPREADS_DIR = os.path.join(_TSA_DIR, "Spreads")
_KALMAN_DIR  = os.path.join(_TSA_DIR, "Kalman_Filter")
_TRADE_DIR   = os.path.join(_ROOT_DIR, "Trade_Implement")
_SIGNALS_DIR = os.path.join(_TRADE_DIR, "Signals")
_RISK_DIR    = os.path.join(_TRADE_DIR, "Risk_Management")
```

All directories are injected into `sys.path` using the standard project pattern:
```python
for _p in [_FM_DIR, _TSA_DIR, _SPREADS_DIR, _KALMAN_DIR, _SIGNALS_DIR, _RISK_DIR, _THIS_DIR]:
    if _p not in sys.path: sys.path.insert(0, _p)
```

### Imports Used

| Import | Source File | Used In |
|--------|-------------|---------|
| `ZScore` | `Spreads/zscore.py` | engine.py (threshold access) |
| `HalfLife` | `Spreads/half_life.py` | engine.py (time stop via HL) |
| `PairSignalGenerator` | `Signals/pair_signal.py` | engine.py (signal generation per bar) |
| `MultiPairSignalAggregator` | `Signals/multi_pair_signal.py` | engine.py (portfolio signals) |
| `PositionSizer` | `Risk_Management/position_sizing.py` | engine.py (N_Y, N_X per trade) |
| `StopLossManager` | `Risk_Management/stoploss.py` | engine.py (stop monitoring) |

`performance.py` only depends on `pandas`, `numpy`, `matplotlib` — **no internal imports needed** — it only consumes the `trade_log` DataFrame produced by `engine.py`.

---

## File 1: `engine.py` — `BacktestEngine`

### Theory

A pairs-trading backtest is a **bar-by-bar simulation** of the full signal→size→stop→fill cycle on historical data. The engine replays data sequentially (never looking ahead) and produces a `trade_log` DataFrame — one row per closed trade.

#### Core Accounting

Each bar, the P&L on an open position is:

$$\text{PnL}_t^{\text{LONG}}  = N_Y \cdot (S_t - S_{\text{entry}}) - \text{TC}$$
$$\text{PnL}_t^{\text{SHORT}} = N_Y \cdot (S_{\text{entry}} - S_t) - \text{TC}$$

where TC = transaction cost (commission + slippage) per leg.

**Capital evolution:**

$$C_t = C_{t-1} + \Delta\text{PnL}_t$$

**Drawdown:**

$$DD_t = \frac{\max_{s \leq t}(C_s) - C_t}{\max_{s \leq t}(C_s)}$$

#### Slippage Model

Slippage is modelled as a fixed fraction of price per leg:

$$\text{Fill price}_Y = P_Y \cdot (1 + \text{slippage} \cdot \text{direction})$$

#### Transaction Cost per Trade

$$\text{TC} = (N_Y \cdot P_Y + N_X \cdot P_X) \cdot \text{commission\_pct}$$

### BacktestEngine Class Structure

```
BacktestEngine
├── __init__(signal_df, price_y, price_x, zscore, half_life, capital,
│            commission_pct, slippage_pct, method, ...)
├── run()                  → executes bar-by-bar loop → trade_log DataFrame
├── _open_trade(bar)       → enters position, calls PositionSizer + StopLossManager.enter_position()
├── _check_exit(bar)       → checks signal==0 or stop, closes trade
├── _compute_pnl(bar)      → unrealised P&L at current bar
├── _apply_costs(N_Y,N_X)  → returns dollar TC from commission + slippage
├── equity_curve()         → pd.Series of capital over time
├── drawdown_series()      → pd.Series of rolling drawdown
├── trade_log()            → pd.DataFrame of all closed trades
├── summary()              → prints key stats
└── plot()                 → 3-panel: equity, drawdown, spread+signals
```

---

## File 2: `performance.py` — `PerformanceAnalytics`

### Theory

Performance analytics converts the raw `trade_log` from `BacktestEngine` into standardised quantitative metrics used by every systematic fund.

#### Sharpe Ratio (annualised)

$$\text{Sharpe} = \frac{\sqrt{252} \cdot \bar{r}}{\sigma_r}$$

where $\bar{r}$ and $\sigma_r$ are mean and std of **daily** P&L returns.

#### Sortino Ratio (downside risk only)

$$\text{Sortino} = \frac{\sqrt{252} \cdot \bar{r}}{\sigma_{\text{downside}}}$$

$$\sigma_{\text{downside}} = \sqrt{\frac{1}{N}\sum_{r_t < 0} r_t^2}$$

#### Calmar Ratio

$$\text{Calmar} = \frac{\text{Ann. Return}}{\text{Max Drawdown}}$$

#### Maximum Drawdown

$$\text{MDD} = \max_t \frac{\max_{s \leq t}(C_s) - C_t}{\max_{s \leq t}(C_s)}$$

#### Win Rate & Profit Factor

$$\text{Win Rate} = \frac{N_{\text{win}}}{N_{\text{total}}} \quad \text{Profit Factor} = \frac{\sum_{r>0} r}{\sum_{r<0} |r|}$$

#### Average Trade Duration

$$\bar{\tau} = \frac{1}{N}\sum_{i=1}^{N} \tau_i$$

### PerformanceAnalytics Class Structure

```
PerformanceAnalytics
├── __init__(equity_curve, trade_log, capital, risk_free_rate)
├── sharpe()              → float
├── sortino()             → float
├── calmar()              → float
├── max_drawdown()        → float
├── win_rate()            → float
├── profit_factor()       → float
├── avg_trade_duration()  → float
├── annualised_return()   → float
├── all_metrics()         → dict of all above
├── summary()             → prints formatted table
└── plot()                → 4-panel: equity, drawdown, monthly P&L heatmap, trade PnL dist
```

---

## Connection Map to Other Modules

```
Financial_Mathematics/
│
├── Spreads/zscore.py         → ZScore.entry_z, exit_z, stop_z         → engine.py
├── Spreads/half_life.py      → HalfLife.half_life_days                → engine.py (time stop)
│
└── Kalman_Filter/
    └── dynamic_beta.py       → DynamicBeta.current_beta()             → PositionSizer β_t

Trade_Implement/
│
├── Signals/pair_signal.py    → PairSignalGenerator.generate()         → engine.py signal loop
│
└── Risk_Management/
    ├── position_sizing.py    → PositionSizer.compute()                → engine.py N_Y, N_X
    └── stoploss.py           → StopLossManager.check_stops()          → engine.py exit logic

Backtesting/
├── engine.py                 → BacktestEngine.run() → trade_log, equity_curve
└── performance.py            → PerformanceAnalytics(equity_curve, trade_log) → metrics
```

---

## Verification Plan

1. **Flat test** — feed a flat Z-score series (no signals), confirm 0 trades, equity stays at initial capital.
2. **Single trade test** — manually construct a LONG entry + exit, verify P&L = N_Y × (S_exit − S_entry) − TC exactly.
3. **Stop test** — construct a spread that diverges, confirm StopLossManager fires at correct bar.
4. **Sharpe sanity** — on random-walk P&L series, Sharpe should be close to 0.
5. **Plot check** — `BacktestEngine.plot()` and `PerformanceAnalytics.plot()` render without errors.
