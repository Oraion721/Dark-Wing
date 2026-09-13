# Dark Wing — Full Codebase Audit Report
### Senior Quant Developer & Researcher Perspective
**Date:** 2026-09-02 | **Auditor:** Quantitative Engineering Review

---

## Executive Summary

The codebase is **architecturally sound** — the modular pipeline design (Math → Signal → Risk → Execution → Analytics) is exactly what institutional quant desks build. However, there are **4 critical bugs**, **6 mathematical gaps**, and **3 redevelopment areas** that must be fixed before this is presentable to a recruiter. All are fixable within 3 days.

---

## 🔴 CRITICAL BUGS (Fix these TODAY — Day 1)

---

### BUG 1 — `spread_builder.py` Line 97: Wrong OLS Spread Formula
**File:** [`spread_builder.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Financial_Mathematics/Time_Series_Analysis/Spreads/spread_builder.py#L97)

**The Code:**
```python
spread_vals = (self._y.values) - (self._alpha_est - (self._beta_est * self._x.values))
```

**The Bug:** The sign on `_alpha_est` is **wrong**. Double-negative makes the formula:
$$S_t = Y_t - (\alpha - \beta X_t) = Y_t - \alpha + \beta X_t$$

**The Correct Formula** from Engle-Granger theory:
$$S_t = Y_t - \alpha - \beta X_t$$

**Fix:**
```python
spread_vals = self._y.values - self._alpha_est - (self._beta_est * self._x.values)
```

**Interview Note — Why it matters:**
> The OLS regression $Y_t = \alpha + \beta X_t + \varepsilon_t$ gives us the residuals $\varepsilon_t = Y_t - \alpha - \beta X_t$. This is the **spread** — the deviation from the long-run equilibrium. Getting the intercept sign wrong means the spread is offset by $2\alpha$, so your mean-reversion anchor $\mu$ is wrong. Your Z-score will be biased and your signals will be wrong.

---

### BUG 2 — `ibkr_executor.py` Lines 192–201: `_compute_zscore()` Wrong Attribute Access
**File:** [`ibkr_executor.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Trade_Implement/Executor/ibkr_executor.py#L192)

**The Code:**
```python
def _compute_zscore(self, S_t: float) -> Optional[float]:
    zs = self._gen._zs
    try:
        if hasattr(zs, "_mu") and hasattr(zs, "_sigma") and zs._sigma and abs(zs._sigma) > 1e-10:
            return (S_t - zs._mu) / zs._sigma
    except: pass
```

**The Bug:** `zs._mu` and `zs._sigma` are **pd.Series** (rolling window outputs), not scalars. Dividing a float by a Series returns a Series — the `abs()` and `>1e-10` comparison then act element-wise, which can silently succeed or fail arbitrarily. In live mode, `zs._sigma` at most recent index is needed.

**Fix:**
```python
def _compute_zscore(self, S_t: float) -> Optional[float]:
    zs = self._gen._zs
    try:
        if hasattr(zs, "_mu") and zs._mu is not None:
            mu = float(zs._mu.iloc[-1])
            sigma = float(zs._sigma.iloc[-1])
            if sigma > 1e-10:
                return (S_t - mu) / sigma
    except: pass
    if len(self._s_history) < 10: return None
    arr = np.array(self._s_history[-100:])
    mu, sigma = arr.mean(), arr.std(ddof=1)
    return (S_t - mu) / sigma if sigma > 1e-10 else None
```

---

### BUG 3 — `engine.py` Line 100: `bars` Count Is Wrong (Causes `Avg Duration = 0.0`)
**File:** [`engine.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Backtesting/engine.py#L100)

**The Code:**
```python
bars = len(self.signal_df.loc[self._entry_idx:idx]) if self._entry_idx is not None else 0
```

**The Bug:** `signal_df.loc[entry_idx:idx]` uses **label-based slicing** which is **inclusive** on both ends. When `entry_idx == idx` (a 1-bar trade), this returns 1 row but does NOT subtract 1. More importantly, when the index is a DatetimeIndex with daily bars, `loc[date_A:date_A]` returns 1 — which rounds to 0.0 in average when there are many 1-bar trades. Also, there is a subtle bug: **`bars` includes the entry bar itself**, inflating duration by 1.

**Fix:**
```python
# Use integer position difference for correct bar counting
try:
    i_entry = self.signal_df.index.get_loc(self._entry_idx)
    i_exit  = self.signal_df.index.get_loc(idx)
    bars    = max(i_exit - i_entry, 1)
except Exception:
    bars = 0
```

**Interview Note — Why it matters:**
> Average trade duration is a core risk metric. If your average duration is 0 bars, it means all trades open and close on the same bar — which implies you have a **look-ahead bias**: you see the exit signal (|Z| < exit_z) and close on the same bar you open. In a real system, you open on the signal bar and close at the **next** bar's open price. This is one of the most common backtesting errors interviewers test for.

---

### BUG 4 — `performance.py`: Sharpe Ratio Is Negative Because Equity Curve Is Not a DatetimeIndex
**File:** [`performance.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Backtesting/performance.py#L26)

**The Bug:** In `IBKRExecutor`, `_equity_ts` is a list of `datetime.datetime` objects, but `_equity` starts with the initial capital at index 0 (before any timestamp is appended). The final `equity_series()` call uses `self._equity[:len(idx)]` which **drops the last equity entry**, causing the final capital to mismatch. Also, `pct_change().dropna()` on a near-constant equity curve (only 10,000 → 10,172 over 744 bars) gives returns of essentially 0.0 with tiny noise. The annualised return is correct but the **Sharpe uses `sqrt(252)` annualisation on what are actually bars-replayed-in-0-seconds**, not daily observations — making it meaningless.

**Fix:** The `PerformanceAnalytics` needs to detect whether the equity index is a DatetimeIndex. If it's a sequential integer index (from bar replay), `periods_per_year` must be adjusted based on actual calendar time:
```python
# In PerformanceAnalytics.__init__(), add:
if not isinstance(equity_curve.index, pd.DatetimeIndex):
    # integer-indexed: each point = 1 bar, not 1 day
    # default ppy=252 is wrong unless bars ARE daily
    pass  # caller must pass correct periods_per_year
```
The real fix: in `run_paper_trade.py`, pass `periods_per_year=len(price_y)` for a dry-run simulation since all bars replay at once.

---

## 🟡 MATHEMATICAL GAPS (Fix on Day 2)

---

### GAP 1 — No Log-Price Transformation in `run_paper_trade.py`
**File:** [`run_paper_trade.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/run_paper_trade.py#L42)

**The Problem:**
```python
raw = yf.download(..., auto_adjust=True)["Close"]
price_y = raw[YF_Y].dropna()  # ← RAW PRICES, not log-prices
```

**Why it's wrong:** Engle-Granger cointegration and OU half-life estimation assume **log-prices**, not raw prices. Raw price series have non-constant variance (heteroskedasticity). Log returns have approximately constant variance. The OLS regression $Y_t = \alpha + \beta X_t + \varepsilon_t$ is well-specified only when $Y$ and $X$ are log-prices.

**Fix:**
```python
raw = yf.download(...)["Close"]
price_y = np.log(raw[YF_Y].dropna())
price_x = np.log(raw[YF_X].dropna())
```

**Interview Note:**
> If a recruiter asks "why do you take log prices?", the answer is: (1) Log prices are I(1) when raw prices are geometric Brownian motion — a fundamental assumption of asset pricing. (2) Log-differences $\ln(P_t) - \ln(P_{t-1}) \approx r_t$ (log-returns) have near-constant variance (needed for OLS). (3) OLS with raw prices on stocks with very different price levels ($HDFCBANK \approx 750$, $KOTAKBANK \approx 380$) distorts the regression because the variance of the left-hand series dominates.

---

### GAP 2 — No Out-of-Sample / Walk-Forward Validation
**Critical for any recruiter conversation.**

The entire pipeline fits on the **full dataset** and trades on the **same data it was fit on** — this is in-sample trading. This is the single biggest red flag for a quant researcher.

**What is missing:**
A walk-forward expanding window backtest:
- **Train window:** First $k$ observations
- **Test window:** Next $m$ observations (trade here, never look at during fit)
- **Slide:** Expand train window by $m$, repeat

**Interview Note:**
> In-sample Sharpe can be >3.0 for a random walk if you fit enough parameters. The walk-forward Sharpe is the only number that matters. A real quant fund runs **minimum 2 years of out-of-sample** returns before allocating capital to a strategy. The number a recruiter wants to hear: "My walk-forward Sharpe over 18 months of out-of-sample data is 0.8."

---

### GAP 3 — `position_sizing.py` Kelly Criterion Is Incorrectly Formulated
**File:** [`position_sizing.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Trade_Implement/Risk_Management/position_sizing.py#L95)

**The Code:**
```python
mu_z = float(z.abs().mean())  # ← uses |Z|, not expected return
sigma_z2 = float(z.var(ddof=1))
f_kelly = mu_z / sigma_z2 / self.kelly_fraction
```

**The Problem:** The Kelly Criterion for a continuous distribution is:
$$f^* = \frac{\mu}{\sigma^2}$$
where $\mu$ is the **expected return** of the trade and $\sigma^2$ is the **variance of returns**. Using `|Z|.mean()` as the numerator is not a return — it is the average magnitude of the signal. Kelly computed on Z-scores, not PnL, is dimensionally inconsistent.

**Correct implementation:**
```python
# Compute from actual trade returns in the Z-score regime
returns_in_regime = pnl_series / capital  # from historical trades
mu = returns_in_regime.mean()
sigma2 = returns_in_regime.var(ddof=1)
f_kelly = mu / sigma2 / self.kelly_fraction  # fractional Kelly
```

**Interview Note:**
> Kelly Criterion is almost always tested in quant interviews. The formula $f^* = \mu/\sigma^2$ maximises the geometric mean of wealth. The **fractional Kelly** (dividing by 4) is used in practice because full Kelly has enormous variance in outcomes (drawdown can be 50%+ even with positive EV). Thorp (2006) recommends half-Kelly as the practical optimum.

---

### GAP 4 — `half_life.py`: No Confidence Interval on τ
**File:** [`half_life.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Financial_Mathematics/Time_Series_Analysis/Spreads/half_life.py)

The half-life estimate $\hat{\tau} = \ln(2)/\hat{\kappa}$ is a **point estimate** — there is no confidence interval. The standard error `self.std_err` is computed but never used to produce a CI on $\tau$.

**What is missing:**
```python
# Delta method for CI on τ = ln(2)/κ = -ln(2)/ln(1+b)
# dτ/db = ln(2) / ((1+b) * ln(1+b)^2)
# Var(τ̂) ≈ (dτ/db)^2 * Var(b̂)
dτ_db = np.log(2) / ((1 + self.b_coef) * np.log(1 + self.b_coef)**2)
var_τ = (dτ_db**2) * (self.std_err**2)
ci_95 = 1.96 * np.sqrt(var_τ)
self.half_life_ci_95 = (self.half_life_days - ci_95, self.half_life_days + ci_95)
```

**Interview Note:**
> When asked "what's your half-life?", say "41 days ± 12 days at 95% confidence." A point estimate without uncertainty is a warning sign for a researcher. If the CI includes 500+ days, the pair is barely mean-reverting.

---

### GAP 5 — Missing VaR and CVaR in `performance.py`
**File:** [`performance.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Backtesting/performance.py)

**What is missing:** Value at Risk (VaR) and Conditional VaR (Expected Shortfall). These are the two most commonly asked risk metrics in quant finance interviews.

$$\text{VaR}_{95\%} = -\text{percentile}_{5\%}(r_t)$$
$$\text{CVaR}_{95\%} = -E[r_t | r_t < \text{VaR}_{95\%}]$$

```python
def var(self, confidence: float = 0.95) -> float:
    return float(-np.percentile(self._returns, (1 - confidence) * 100))

def cvar(self, confidence: float = 0.95) -> float:
    var = self.var(confidence)
    tail = self._returns[self._returns < -var]
    return float(-tail.mean()) if len(tail) > 0 else var
```

---

### GAP 6 — `zscore.py` Uses Wrong Z-score Window for Kalman Spread
**File:** [`run_paper_trade.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/run_paper_trade.py#L57)

```python
zs = ZScore(spread, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z)  # ← window defaults to 60
```

The `ZScore` is computed on the OLS spread, but the `DynamicBeta` Kalman filter generates a **different, adaptive spread** internally. The pipeline uses two separate spread estimates simultaneously:
1. OLS static spread → `ZScore` → static signals
2. Kalman dynamic beta → `DynamicBeta` → dynamic signals

These two paths conflict. In `PairSignalGenerator`, both are active simultaneously but generate different Z-scores. The result is that the signal is driven partly by stale OLS beta, not the Kalman beta.

**Fix:** After `DynamicBeta.fit()`, extract the Kalman spread and pass it to `ZScore`:
```python
db.fit()
kalman_spread = db._states_df["spread"]  # use Kalman spread, not OLS spread
zs = ZScore(kalman_spread, window=int(half_life_days * 2), entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z)
zs.compute()
```

---

## 🟢 REDEVELOPMENT AREAS (Day 3 — Architecture improvements)

---

### REDEVELOPMENT 1 — `run_paper_trade.py` Needs a Walk-Forward Backtest Function

Currently the script only does either (a) instant dry-run replay on full data or (b) live IBKR connection. There is no walk-forward research backtest mode.

**Add a `backtest_walkforward()` function:**
```python
def backtest_walkforward(price_y, price_x, train_size=0.6, step_months=1):
    """
    Expanding window walk-forward:
    - Train on first 60% of data
    - Test on next month's data
    - Slide train window forward by 1 month
    - Concatenate all out-of-sample results
    """
```

This is the function that produces the **credible Sharpe ratio** for your resume.

---

### REDEVELOPMENT 2 — `StopLossManager` Has a State Bug Between Trades

**File:** [`stoploss.py`](file:///c:/Users/HP/OneDrive/Documents/Stock%20Projects/Dark%20Wing/Trade_Implement/Risk_Management/stoploss.py#L113)

When `_trigger_stop()` is called, it resets `self._position = 0`. But if a new trade is opened in the very **next bar** (bar `t+1`), `enter_position()` is called before `check_stops()` has returned cleanly. In the `BacktestEngine.run()` loop, after a stop fires at bar `t`, the code checks `if not self._open and sig in (1,-1):` and immediately opens a new trade in the same bar. The `StopLossManager._current_bars` is reset to 0 — correct — but the `_spread_sigma` from the previous trade may still be stale because `spread_series` passed to `enter_position()` at bar `t+1` is `df["spread"].loc[:idx]` which includes the spread at the stopped bar.

**Fix:** Add a `cooldown_bars` parameter — after a stop, do not open a new trade for `cooldown_bars` (e.g., 5) bars.

---

### REDEVELOPMENT 3 — No Unit Tests Anywhere

**Impact:** Without `pytest` tests, you cannot prove to a recruiter that your mathematical implementations are correct.

**Minimum tests needed:**
```
tests/
├── test_spread_builder.py     # Test OLS formula: check S_t = Y - α - βX
├── test_half_life.py          # Test τ for known OU process with exact τ
├── test_zscore.py             # Test signal states: entry→hold→exit logic
├── test_performance.py        # Test Sharpe formula vs known analytical result
└── test_backtest_engine.py    # Test PnL formula for LONG and SHORT
```

**Key test — Sharpe sanity check:**
```python
def test_sharpe_iid_normal():
    """For iid N(μ, σ²) returns, Sharpe = sqrt(252) * μ/σ analytically."""
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.001, 0.02, 252))
    equity = (1 + r).cumprod() * 100_000
    pa = PerformanceAnalytics(equity, pd.DataFrame())
    assert abs(pa.sharpe() - np.sqrt(252) * r.mean() / r.std()) < 0.01
```

---

## 📋 3-Day Remediation Plan

| Day | Task | Priority | File |
|:----|:-----|:--------:|:-----|
| **Day 1** | Fix OLS spread formula (BUG 1) | 🔴 P0 | `spread_builder.py` L97 |
| **Day 1** | Fix `_compute_zscore` scalar bug (BUG 2) | 🔴 P0 | `ibkr_executor.py` L195 |
| **Day 1** | Fix `bars` duration count (BUG 3) | 🔴 P0 | `engine.py` L100 |
| **Day 1** | Add `np.log()` transform in runner (GAP 1) | 🔴 P0 | `run_paper_trade.py` L43 |
| **Day 2** | Add VaR and CVaR methods (GAP 5) | 🟡 P1 | `performance.py` |
| **Day 2** | Add half-life confidence interval (GAP 4) | 🟡 P1 | `half_life.py` |
| **Day 2** | Fix Kalman spread → ZScore pipeline (GAP 6) | 🟡 P1 | `run_paper_trade.py` |
| **Day 2** | Fix Kelly formula (GAP 3) | 🟡 P1 | `position_sizing.py` |
| **Day 3** | Write walk-forward backtest function (REDEV 1) | 🟢 P2 | `run_paper_trade.py` |
| **Day 3** | Add StopLoss cooldown period (REDEV 2) | 🟢 P2 | `stoploss.py` |
| **Day 3** | Write 5 core unit tests (REDEV 3) | 🟢 P2 | `tests/` (new dir) |

---

## 📚 Notion Notes — Mathematical Concepts for Interviews

### 1. Engle-Granger Cointegration
- **Null hypothesis:** Residuals are I(1) (non-stationary) → NOT cointegrated
- **Alternative:** Residuals are I(0) (stationary) → COINTEGRATED
- **Why log prices?** ADF critical values assume I(1) series. Log prices of equities satisfy this under GBM
- **Why OLS for β?** OLS is superconsistent for cointegrating vectors (convergence rate $T$ vs $\sqrt{T}$ for regular OLS)
- **Limitation:** OLS β is static. Real markets have time-varying β → Kalman Filter needed

### 2. Ornstein-Uhlenbeck Half-Life
- **Process:** $dS_t = \kappa(\mu - S_t)dt + \sigma dW_t$
- **Discretised:** $\Delta S_t = a + b S_{t-1} + \varepsilon_t$ where $b = e^{-\kappa\Delta t} - 1$
- **Half-life:** $\tau = \ln(2)/\kappa$ — time for deviation to decay by 50%
- **Trading rule:** Set Z-score lookback = $2\tau$; time stop = $3\tau$
- **If τ > 60 days:** Barely tradeable on daily data (too slow to revert)

### 3. Kalman Filter (State Space Model)
- **State equation:** $\beta_t = \beta_{t-1} + w_t$, $w_t \sim N(0, Q)$ (process noise)
- **Observation equation:** $P_{Y,t} = \beta_t P_{X,t} + v_t$, $v_t \sim N(0, R)$ (measurement noise)
- **Key insight:** Q/R ratio controls adaptability vs. smoothness. High Q/R → beta tracks fast (noisy). Low Q/R → beta is smooth (sluggish)
- **Interview question:** "How do you choose Q and R?" — Answer: Use EM algorithm or set Q = σ²_spread / τ (our current approach)

### 4. Z-Score Signal Generation
- **$Z_t = (S_t - \mu_t)/\sigma_t$** where $\mu_t, \sigma_t$ are rolling over $2\tau$ bars
- **Entry:** |Z| > 2.0 (2-sigma event assuming spread is N(0,1) in regime)
- **Exit:** |Z| < 0.5 (close enough to mean)
- **Stop:** |Z| > 3.5 (extreme event — spread may have regime-broken)
- **State machine:** Must prevent re-entry without crossing exit (prevents chasing)

### 5. Kelly Criterion
- **Full Kelly:** $f^* = \mu/\sigma^2$ (maximises log-wealth growth rate)
- **Fractional Kelly (1/4):** Reduces bet to 25% of full Kelly — much lower drawdown
- **Derivation:** From $G = E[\ln(1 + f \cdot r)]$ → FOC → $f^* = \mu/\sigma^2$
- **In pairs trading:** μ and σ² should be estimated from **out-of-sample trade returns**, not in-sample Z-scores

### 6. Sharpe Ratio
- **Definition:** $SR = \sqrt{T_{ann}} \cdot \frac{E[r_t - r_f]}{\text{std}(r_t - r_f)}$
- **Annualisation:** $\sqrt{252}$ for daily returns
- **Why Sharpe can be negative with 80% win rate:** Win rate does not measure magnitude. If 20% of losses are 10x larger than 80% of gains, Sharpe can be deeply negative. This is exactly what happened in our backtest
- **Sortino is better** for strategies with asymmetric returns (only penalises downside vol)
