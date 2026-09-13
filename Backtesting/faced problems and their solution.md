# Faced Problems and Their Solutions

During the redevelopment and verification of the algorithmic trading project, several key issues were identified and resolved to ensure the strategy is mathematically sound and execution-ready.

## 1. Lack of Out-Of-Sample Validation (Look-Ahead Bias)
**Problem**: The original backtest evaluated performance over the entire dataset using parameters (like OLS beta and Z-score threshold bounds) fitted on that exact same dataset. This introduces look-ahead bias, leading to an artificially inflated in-sample Sharpe ratio that would fail in live trading.
**Solution**: Redeveloped the backtesting suite by implementing an expanding-window **Walk-Forward Backtest** (`walkforward.py`). The dataset is split into training and out-of-sample (OOS) testing windows. `DynamicBeta` and `HalfLife` are fitted on the train window, and these *frozen* parameters are used to generate trades on the OOS test window. `run_paper_trade.py` was updated with a `WALK_FORWARD = True/False` toggle to run this rigorous, unbiased validation.

## 2. Stop-Loss Whip-Sawing (Missing Cooldown)
**Problem**: The `StopLossManager` correctly closed positions when risk limits were breached, but allowed the engine to immediately re-enter a trade on the very next bar if the signal condition (e.g. $|Z| > 2$) was still active. In extreme market events, this leads to chain-stopping (whip-sawing) because the algorithm re-enters while the spread is still highly volatile.
**Solution**: Implemented a **cooldown period** in `StopLossManager` (`stoploss.py`). Added a `cooldown_bars` parameter (default 5) and an `in_cooldown` property. Updated `BacktestEngine.run()` (`engine.py`) to block any new position entries while the cooldown is active, allowing the spread volatility to settle before re-engaging.

## 3. Fragile Mathematics (Lack of Unit Testing)
**Problem**: Advanced quantitative models (Kalman Filters, Kelly Criterion, Conditional Value at Risk) were implemented without automated regression tests. Any refactoring could silently break the underlying mathematics (e.g., the sign error previously found in the OLS spread formula).
**Solution**: Created a comprehensive testing suite (`tests/test_core.py`) using `pytest`. The suite simulates deterministic scenarios (like a synthetic Ornstein-Uhlenbeck process with a known mean-reversion speed) to mathematically assert that:
*   The OLS spread has the correct sign and near-zero mean.
*   The delta-method 95% Confidence Interval successfully brackets the estimated Half-Life.
*   VaR and CVaR maintain coherence ($CVaR \ge VaR \ge 0$).
*   The Kelly fraction returns a positive size within leverage caps.
*   The StopLoss cooldown strictly blocks re-entry.
*All tests currently pass.*

## 4. IB Gateway Live Execution Hangs / Connection Errors
**Problem**: When running `run_paper_trade.py` with `DRY_RUN = False` to connect to the IB Gateway, the script may hang or fail to stream data.
**Solution**: This is a common environment issue with Interactive Brokers' API. To ensure smooth live execution:
*   Ensure **IB Gateway** (or TWS) is open and logged into the paper trading account.
*   Go to **Settings > API > Settings** and verify that **"Enable ActiveX and Socket Clients"** is checked.
*   Verify the port: TWS paper usually uses `7497`, while IB Gateway paper uses `4002`. Ensure the `TWS_PORT` in `run_paper_trade.py` matches the software you are using.
*   If testing outside regular market hours, ensure the API is configured to allow connections or use `DRY_RUN = True` for historical simulations.
