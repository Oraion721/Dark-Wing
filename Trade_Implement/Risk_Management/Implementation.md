# Risk Management Module — Implementation Plan & Documentation
## Dark Wing Pairs Trading System | Step 9

---

## Overview

After `Signals/` produces a discrete signal stream (`+1`, `-1`, `0`), the **Risk Management** layer decides:
1. **How much capital to deploy** per trade (`position_sizing.py`)
2. **When to cut losses** and protect capital (`stoploss.py`)

These two files wrap around every signal before it reaches the execution layer (`Angel_1_Implement/`, `IBKR_Implement/`).

---

## Pipeline Position

```
STEP 5 → Spreads/           (SpreadBuilder → HalfLife → ZScore)
STEP 6 → Kalman_Filter/     (KalmanFilterPairs → DynamicBeta)
STEP 7 → Kalman_Filter/     (VECMForecaster, KalmanForecaster)
STEP 8 → Signals/           (PairSignalGenerator → MultiPairSignalAggregator)
══════════════════════════════════════════════════════════════
STEP 9 → Risk_Management/   ◀  THIS MODULE
══════════════════════════════════════════════════════════════
STEP 10 → backtest/engine.py
STEP 11 → Angel_1_Implement/ | IBKR_Implement/
```

---

## File 1: `position_sizing.py` — PositionSizer

### Purpose
Translate the raw signal into a **concrete share/lot quantity** for each leg of the pair trade, respecting capital limits, volatility, and the dynamic hedge ratio $\beta_t$.

---

### Mathematical Foundation

#### Method 1 — Fixed Fractional (Simplest, Default)
Allocate a fixed fraction $f$ of capital to each trade:

$$N_Y = \left\lfloor \frac{f \cdot C}{P_Y} \right\rfloor, \quad N_X = \left\lfloor \beta_t \cdot N_Y \right\rfloor$$

where:
- $f$ = fraction of capital per leg (e.g. 0.10 = 10%)
- $C$ = total available capital
- $P_Y$, $P_X$ = current prices of Y and X
- $\beta_t$ = dynamic hedge ratio from `DynamicBeta`

**Financial meaning:** Simple, battle-tested. Keeps dollar exposure constant across trades regardless of volatility. Risk scales with capital.

---

#### Method 2 — Volatility-Scaled (Risk Parity)
Size each leg so that the **expected daily P&L volatility** is a fixed dollar amount $\sigma_{target}$:

$$N_Y = \left\lfloor \frac{\sigma_{target}}{P_Y \cdot \sigma_Y} \right\rfloor$$

$$N_X = \left\lfloor \beta_t \cdot \frac{\sigma_{target}}{P_X \cdot \sigma_X} \right\rfloor$$

where $\sigma_Y$, $\sigma_X$ are rolling daily return volatilities.

**Financial meaning:** In volatile markets, fewer shares are bought so that the daily risk stays constant. This is the foundation of **Risk Parity** funds (Bridgewater).

---

#### Method 3 — Kelly Criterion (Optimal Growth)
Kelly fraction maximises long-run compounded growth:

$$f^* = \frac{\mu_Z}{\sigma_Z^2}$$

where $\mu_Z$ = mean Z-score signal strength, $\sigma_Z^2$ = variance of Z-score over lookback.

In practice, a **fractional Kelly** is used (half-Kelly or quarter-Kelly) to reduce variance:

$$f_{Kelly} = \frac{f^*}{k}, \quad k \in \{2, 4\}$$

Then dollar exposure:

$$\text{DollarExposure} = \min(f_{Kelly} \cdot C,\ \text{MaxExposure})$$

$$N_Y = \left\lfloor \frac{\text{DollarExposure}}{P_Y} \right\rfloor$$

**Financial meaning:** Kelly sizes up when the edge is high (high $|\mu_Z|$ vs $\sigma_Z^2$) and sizes down when the edge is weak. Pairs traders use it to compound capital fastest.

---

#### Hedge Ratio Adjustment
Both legs must be sized to maintain a **dollar-neutral** portfolio:

$$\text{Dollar}_{Y} = N_Y \cdot P_Y$$
$$\text{Dollar}_{X} = N_X \cdot P_X = \beta_t \cdot \text{Dollar}_Y \cdot \frac{P_Y}{P_X} \cdot P_X = \beta_t \cdot \text{Dollar}_Y$$

Net dollar exposure (spread bet):

$$\text{Spread Exposure} = |\text{Dollar}_Y - \beta_t \cdot \text{Dollar}_X| \approx 0$$

This ensures the portfolio is **market-neutral** — profit comes only from spread convergence, not market direction.

---

#### Capital Constraint
Total capital deployed across all pairs must not exceed a limit:

$$\sum_{i=1}^{N} \left(\text{Dollar}_{Y_i} + \text{Dollar}_{X_i}\right) \leq C_{max}$$

---

### PositionSizer — Method Flowchart

```
INPUT: signal (+1/-1/0), beta_t, P_Y, P_X, Z_t, spread_series, capital
          │
          ▼
    [signal == 0?]──Yes──► N_Y = 0, N_X = 0  ──► OUTPUT
          │ No
          ▼
    [method == 'fixed_fractional']
          │──► Dollar_Y = f * capital
          │    N_Y = floor(Dollar_Y / P_Y)
          │    N_X = floor(beta_t * N_Y)
          │
    [method == 'volatility_scaled']
          │──► sigma_Y = rolling_std(returns_Y)
          │    N_Y = floor(sigma_target / (P_Y * sigma_Y))
          │    N_X = floor(beta_t * sigma_target / (P_X * sigma_X))
          │
    [method == 'kelly']
          │──► f* = mu_Z / sigma_Z^2
          │    f_kelly = f* / kelly_fraction (half/quarter Kelly)
          │    DollarExposure = min(f_kelly * capital, max_exposure)
          │    N_Y = floor(DollarExposure / P_Y)
          │    N_X = floor(beta_t * N_Y)
          │
          ▼
    [Apply capital cap: Dollar_Y + Dollar_X <= max_trade_size]
          │
          ▼
OUTPUT: {N_Y, N_X, dollar_Y, dollar_X, method_used, beta_t}
```

---

### Connection to Other Files

| Source File | What it provides | Used in PositionSizer |
|-------------|-----------------|----------------------|
| `Signals/pair_signal.py` | `signal` (+1/-1/0), `z`, `beta` | Main inputs |
| `Spreads/zscore.py` | `ZScore.entry_z`, Z-score series | Kelly mu_Z, sigma_Z computation |
| `Kalman_Filter/dynamic_beta.py` | `DynamicBeta.current_beta()` | $\beta_t$ hedge ratio |
| `Spreads/half_life.py` | `HalfLife.half_life_days` | Optional: sets sigma_target lookback |

---

### PositionSizer Class Structure

```python
class PositionSizer:

    __init__(capital, method, max_trade_fraction, max_exposure,
             sigma_target, kelly_fraction, vol_window)

    compute(signal, beta_t, price_y, price_x,
            zscore_series, returns_y, returns_x)
        → dict: {N_Y, N_X, dollar_Y, dollar_X, method, beta_t}

    _fixed_fractional(signal, beta_t, price_y, price_x)  [private]

    _volatility_scaled(signal, beta_t, price_y, price_x,
                       returns_y, returns_x)               [private]

    _kelly(signal, beta_t, price_y, price_x,
           zscore_series)                                  [private]

    _apply_capital_cap(n_y, n_x, price_y, price_x)        [private]

    summary()   → print stats for current sizing decision
    plot(zscore_series, positions_df)  → 2-panel: Z-score + position sizes
```

---

## File 2: `stoploss.py` — StopLossManager

### Purpose
Monitor all open positions in real-time (or bar-by-bar in backtest) and **trigger closes** when:
- The spread moves too far against the position (spread stop)
- The P&L drawdown exceeds a fixed threshold (dollar stop)
- Time in trade exceeds the expected holding period (time stop)
- Z-score extreme is hit (`ZScore.stop_z`)

---

### Mathematical Foundation

#### Stop 1 — Spread Stop (Primary)
Close when the spread moves $k \sigma_S$ against the entered level:

$$S_{stop}^{LONG}  = S_{entry} - k \cdot \sigma_S$$
$$S_{stop}^{SHORT} = S_{entry} + k \cdot \sigma_S$$

where $\sigma_S$ is the rolling spread standard deviation and $k$ is a multiplier (default 2.0).

**Trigger:** $S_t < S_{stop}$ (for LONG) or $S_t > S_{stop}$ (for SHORT).

---

#### Stop 2 — Dollar P&L Stop
Close when the unrealised P&L falls below a dollar threshold:

$$\text{PnL}_{t}^{LONG}  = N_Y(S_t - S_{entry})$$
$$\text{PnL}_{t}^{SHORT} = N_Y(S_{entry} - S_t)$$

Stop triggers when:

$$\text{PnL}_t < -\text{MaxLoss}$$

where `MaxLoss` = fraction of capital (e.g. 1% = 0.01 × capital).

---

#### Stop 3 — Z-score Hard Stop (From ZScore)
This is already flagged in `pair_signal.py` as signal = ±2 when $|Z_t| > \theta_{stop}$.
`StopLossManager` re-checks this independently as a **hard override**:

$$|Z_t| > \theta_{stop} \Rightarrow \text{CLOSE POSITION IMMEDIATELY}$$

This acts as a safety net even if signal generation missed it.

---

#### Stop 4 — Time Stop (Position Age)
If a position has been open longer than $\tau_{max}$ bars without reversion, exit:

$$\text{age}_t = t - t_{entry}$$
$$\text{age}_t > \tau_{max} \Rightarrow \text{EXIT}$$

Rule of thumb: $\tau_{max} = 3 \times \tau_{HL}$ where $\tau_{HL}$ is the half-life.

**Financial meaning:** If the spread hasn't reverted in 3× the expected reversion time, the pair may have broken down. Cut the position and re-evaluate.

---

#### Stop 5 — Trailing Spread Stop (Optional)
For profitable positions, lock in gains with a trailing stop:

$$S_{trail}^{LONG} = \max(S_{best} - k_{trail} \cdot \sigma_S, S_{stop})$$

where $S_{best}$ = best spread seen since entry. Ratchets up as spread improves.

---

### StopLossManager — Decision Flowchart

```
INPUT: position (+1/-1), S_t, Z_t, P&L, age, entry data
          │
          ▼
  [position == 0?]──Yes──► No action ──► OUTPUT: HOLD
          │ No
          ▼
  ┌───────────────────────────────────┐
  │  PARALLEL STOP CHECKS:            │
  │                                   │
  │  Check 1: |Z_t| > stop_z?         │──True──► STOP: "Z-score hard stop"
  │                                   │
  │  Check 2: PnL < -MaxLoss?         │──True──► STOP: "Dollar loss stop"
  │                                   │
  │  Check 3: Spread stop triggered?  │──True──► STOP: "Spread volatility stop"
  │           (S_t < S_stop for LONG) │
  │                                   │
  │  Check 4: age > tau_max?          │──True──► STOP: "Time stop"
  │                                   │
  │  Check 5: Trailing stop hit?      │──True──► STOP: "Trailing stop"
  └───────────────────────────────────┘
          │ All False
          ▼
      OUTPUT: HOLD (no stop triggered)
          │
          ▼
  [Any stop triggered?]──Yes──► signal = 0, log reason, return CLOSE
```

---

### Connection to Other Files

| Source File | What it provides | Used in StopLossManager |
|-------------|-----------------|------------------------|
| `Signals/pair_signal.py` | `signal_df['spread']`, `signal_df['z']` | $S_t$, $Z_t$ monitoring |
| `Spreads/zscore.py` | `ZScore.stop_z` | Hard Z-stop threshold |
| `Risk_Management/position_sizing.py` | `N_Y`, `dollar_Y` | Dollar P&L calculation |
| `Spreads/half_life.py` | `HalfLife.half_life_days` | `tau_max = 3 * half_life` |

---

### StopLossManager Class Structure

```python
class StopLossManager:

    __init__(zscore, half_life, capital, max_loss_pct,
             spread_stop_k, time_stop_multiplier,
             use_trailing_stop, trail_k)

    enter_position(signal, spread_entry, n_shares, timestamp)
        → stores entry state

    check_stops(S_t, Z_t, price_y, price_x, timestamp)
        → dict: {stop_triggered, reason, close_signal}

    update_trailing_stop(S_t)
        → updates self._S_trail

    close_position(reason, timestamp)
        → resets state, logs the trade

    _check_zscore_stop(Z_t)        [private]
    _check_dollar_stop(S_t)        [private]
    _check_spread_stop(S_t)        [private]
    _check_time_stop(timestamp)    [private]
    _check_trailing_stop(S_t)      [private]

    trade_log()   → pd.DataFrame of all closed trades with stop reasons
    summary()     → stats: % stopped by each type, avg loss on stops
    plot(signal_df)  → timeline with stop events marked
```

---

## Full Module Integration Flowchart

```
Financial_Mathematics/
│
├── Spreads/
│   ├── spread_builder.py ──────────────────► SpreadBuilder
│   ├── half_life.py  ──── HalfLife.tau ────► StopLossManager (time_stop_max)
│   └── zscore.py  ─────── ZScore ──────────► PairSignalGenerator + StopLossManager
│
└── Kalman_Filter/
    └── dynamic_beta.py ─── beta_t ─────────► PositionSizer (hedge ratio)

Trade_Implement/
│
├── Signals/
│   ├── pair_signal.py ───────────────────────────────────────────────┐
│   │   [signal +1/-1/0, z, spread, beta, position]                   │
│   │                                                                  ▼
│   └── multi_pair_signal.py                              ┌──────────────────────┐
│                                                         │   Risk_Management/   │
│                                              ┌──────────┤                      │
│                                              │          │  position_sizing.py  │
│                                              │          │  → N_Y, N_X, $       │
│                                              │          │                      │
│                                              │          │  stoploss.py         │
│                                              │          │  → HOLD / CLOSE      │
│                                              │          └──────────────────────┘
│                                              │                   │
│                                              ▼                   ▼
│                                    backtest/engine.py     Live Execution
│                                                          Angel_1 / IBKR
```

---

## Code Snippets

### position_sizing.py — typical usage
```python
from position_sizing import PositionSizer
from pair_signal import PairSignalGenerator

# After PairSignalGenerator.generate():
signal_df = gen.generate()
last = signal_df.iloc[-1]

sizer = PositionSizer(
    capital=500_000,
    method="kelly",
    max_trade_fraction=0.10,
    kelly_fraction=4,          # quarter-Kelly
)

sizing = sizer.compute(
    signal=int(last["signal"]),
    beta_t=float(last["beta"]),
    price_y=current_price_y,
    price_x=current_price_x,
    zscore_series=signal_df["z"],
)
# Returns: {"N_Y": 120, "N_X": 94, "dollar_Y": 48000, ...}
```

### stoploss.py — typical usage
```python
from stoploss import StopLossManager

sl = StopLossManager(
    zscore=zs,
    half_life=hl.half_life_days,
    capital=500_000,
    max_loss_pct=0.01,          # 1% of capital max loss per trade
    spread_stop_k=2.0,          # 2 sigma spread stop
    use_trailing_stop=True,
)

# On trade entry:
sl.enter_position(signal=1, spread_entry=S_entry, n_shares=N_Y, timestamp=t0)

# Each new bar:
result = sl.check_stops(S_t=S_now, Z_t=Z_now,
                        price_y=P_Y, price_x=P_X, timestamp=t_now)
if result["stop_triggered"]:
    print(f"STOP: {result['reason']}")
    # send close order to broker
```

---

## Verification Plan

1. **Position sizing unit test** — Feed synthetic Z-score series, verify Kelly and vol-scaled outputs scale correctly with signal strength.
2. **Stop loss unit test** — Simulate a spread that moves 3σ against position, confirm dollar stop triggers at correct bar.
3. **Time stop test** — Run 500 bars past a LONG entry with no reversion, confirm time stop fires at `3 * half_life`.
4. **Integration test** — Wire `PairSignalGenerator` → `PositionSizer` → `StopLossManager` on real NIFTY pair data, check no contradictory signals.
5. **Plot check** — `StopLossManager.plot()` shows stop events correctly on spread chart.
