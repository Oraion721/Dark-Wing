# Executor Module — Implementation & Operating Plan
## Dark Wing Pairs Trading System | Step 11 (Live Execution)

---

## Pipeline Position

```
STEP 1-4  → Financial_Mathematics/       (Stationarity, Cointegration, VECM)
STEP 5    → Spreads/                     (SpreadBuilder, HalfLife, ZScore)
STEP 6-7  → Kalman_Filter/               (DynamicBeta, KalmanForecaster)
STEP 8    → Trade_Implement/Signals/     (PairSignalGenerator, MultiPairSignalAggregator)
STEP 9    → Trade_Implement/Risk_Management/ (PositionSizer, StopLossManager)
STEP 10   → Backtesting/                 (BacktestEngine, PerformanceAnalytics)
══════════════════════════════════════════════════════════
STEP 11   → Trade_Implement/Executor/    ← THIS MODULE
══════════════════════════════════════════════════════════
         ├── ibkr_executor.py  → IBKR TWS / IB Gateway (paper + live)
         └── angel_executor.py → Angel One SmartAPI (paper + live)
```

---

## Directory & Import Integration

### Path Resolution

`Executor/` sits inside `Trade_Implement/`, one level below the project root.

```python
_THIS_DIR    = .../Trade_Implement/Executor/
_TRADE_DIR   = .../Trade_Implement/
_ROOT_DIR    = .../Dark Wing/
_FM_DIR      = .../Financial_Mathematics/
_TSA_DIR     = .../Financial_Mathematics/Time_Series_Analysis/
_SPREADS_DIR = .../Spreads/
_KALMAN_DIR  = .../Kalman_Filter/
_SIGNALS_DIR = .../Trade_Implement/Signals/
_RISK_DIR    = .../Trade_Implement/Risk_Management/
_IBKR_DIR   = .../Trade_Implement/IBKR_Implement/
_BACKTEST_DIR= .../Backtesting/
```

### Cross-Module Imports

| Import | Source | Used in |
|--------|--------|---------|
| `PairSignalGenerator` | `Signals/pair_signal.py` | Both executors — signal generation loop |
| `PositionSizer` | `Risk_Management/position_sizing.py` | Both executors — N_Y, N_X per trade |
| `StopLossManager` | `Risk_Management/stoploss.py` | Both executors — stop monitoring |
| `ZScore` | `Spreads/zscore.py` | Both executors — for stop_z threshold |
| `HalfLife` | `Spreads/half_life.py` | Both executors — for time stop |
| `ConnectionManager` | `IBKR_Implement/IBKR_Connection.py` | `ibkr_executor.py` — socket layer |
| `MarketDataHandler` | `IBKR_Implement/IBKR_MarketData.py` | `ibkr_executor.py` — live prices |
| `PerformanceAnalytics` | `Backtesting/performance.py` | Both — live metrics computation |

---

## File 1: `ibkr_executor.py` — `IBKRExecutor`

### What it does

`IBKRExecutor` wraps the existing `MarketDataHandler` (IBKR_Implement) and adds the live **signal → order → stop** execution loop on top. It:

1. Connects to TWS / IB Gateway using `ConnectionManager`
2. Subscribes to real-time tick data for both legs of the pair
3. On every new tick — computes Z-score live, calls `PairSignalGenerator`
4. On a new non-zero signal — calls `PositionSizer` for share quantities, places MKT orders via `placeOrder()`
5. Each bar — calls `StopLossManager.check_stops()`, places closing orders if triggered
6. Tracks live P&L, writes to `live_trade_log.csv`

### IBKR Order Flow

```
Signal = +1 (LONG spread)
  → BUY  N_Y shares of Y  at market
  → SELL N_X shares of X  at market

Signal = -1 (SHORT spread)
  → SELL N_Y shares of Y  at market
  → BUY  N_X shares of X  at market

Exit / Stop
  → SELL N_Y shares of Y  (close long)  / BUY N_Y  (close short)
  → BUY  N_X shares of X  (close short) / SELL N_X (close long)
```

### IBKRExecutor Class Structure

```
IBKRExecutor
├── __init__(pair_signal_gen, position_sizer, stop_loss_mgr, 
│            symbol_y, symbol_x, exchange, currency, capital,
│            host, port, client_id, paper_trading)
├── connect()                → ConnectionManager.connect()
├── subscribe_prices()       → MarketDataHandler.request_market_data() for Y and X
├── run_loop(interval_sec)   → main event loop, runs indefinitely
│    ├── _fetch_prices()     → get latest bid/ask mid from market_data dict
│    ├── _compute_signal()   → calls PairSignalGenerator (or ZScore directly)
│    ├── _place_order(side, symbol, qty) → EClient.placeOrder() with MKT Order
│    ├── _check_stops()      → StopLossManager.check_stops()
│    └── _log_trade()        → appends row to live_trade_log.csv
├── close_all()              → flatten both legs immediately
├── live_pnl()               → float, current unrealised PnL
├── performance()            → PerformanceAnalytics(equity_curve, trade_log)
└── disconnect()             → cancel subscriptions, ConnectionManager.disconnect()
```

---

## File 2: `angel_executor.py` — `AngelExecutor`

### What it does

`AngelExecutor` uses the **SmartAPI** REST client (Angel One) to place NSE equity orders for Indian market pairs (e.g. RELIANCE / TCS). Logic is identical to IBKR at the signal level — only the order placement and data subscription layer differs.

- Uses `smartapi-python` (`SmartConnect`) for authentication via `api_key + TOTP`
- Fetches LTP (Last Traded Price) via `get_ltp()` on a polling loop
- Places `MARKET` orders via `place_order()` with product type `MIS` (intraday) or `CNC` (delivery)
- Tracks order status via `order_book()` polling

### AngelExecutor Class Structure

```
AngelExecutor
├── __init__(pair_signal_gen, position_sizer, stop_loss_mgr,
│            symbol_y, token_y, symbol_x, token_x,
│            api_key, client_id, totp_secret, capital,
│            exchange, product_type)
├── connect()                → SmartConnect.generateSession()
├── _get_prices()            → SmartConnect.get_ltp() for Y and X
├── run_loop(interval_sec)   → polling loop at interval_sec cadence
│    ├── _compute_signal()
│    ├── _place_order(symbol, token, side, qty, product)
│    ├── _check_stops()
│    └── _log_trade()
├── close_all()
├── live_pnl()
├── performance()            → PerformanceAnalytics
└── disconnect()             → SmartConnect.terminateSession()
```

---

## Live Trade Performance Measurement

### Metrics Updated in Real-Time

| Metric | How computed | Update frequency |
|--------|-------------|-----------------|
| Unrealised PnL | `N_Y*(S_t - S_entry)` | Every price tick |
| Realised PnL | Cumulative from closed trades | On each close |
| Live Sharpe | Rolling 252-bar daily returns | Daily |
| Live Drawdown | Peak equity − current equity | Continuous |
| Win Rate | n_winners / n_trades | On each close |
| Slippage realised | Fill price vs mid-price | On each fill |

### Live vs Backtest Comparison

After each paper-trade session, run:
```python
perf = PerformanceAnalytics(equity_curve=executor.equity_series(),
                             trade_log=executor.closed_trades_df(),
                             capital=capital)
perf.summary()
```
Compare `perf.sharpe()` and `perf.max_drawdown()` against backtest values.

---

## IBKR Paper Trading Setup (Step-by-Step)

See the end of this document for the step-by-step guide.

---

## Operating Checklist

### Before Starting Live Loop

- [ ] TWS / IB Gateway open and Paper Trading account logged in  
- [ ] `port=7497` (TWS paper) or `port=4002` (IB Gateway paper)  
- [ ] API settings in TWS: `Enable ActiveX and Socket Clients` ticked  
- [ ] `PairSignalGenerator` fitted on historical data and `generate()` called  
- [ ] `PositionSizer` and `StopLossManager` initialised with correct `capital`, `zscore`, `half_life`  
- [ ] `symbol_y`, `symbol_x`, `exchange`, `currency` confirmed in IB Contract search  

### During Live Loop

- Logs written every bar to `live_trade_log.csv` in `Executor/` folder  
- Console prints: current Z-score, signal, position, unrealised PnL  
- Graceful exit via `Ctrl+C` → triggers `close_all()` before disconnect  

### After Session

- Call `executor.performance()` → `PerformanceAnalytics.summary()` + `.plot()`  
- Compare against backtest: Sharpe, MDD, win rate, avg trade duration  
- Adjust `entry_z`, `stop_z`, `max_loss_pct` if live metrics diverge significantly  
