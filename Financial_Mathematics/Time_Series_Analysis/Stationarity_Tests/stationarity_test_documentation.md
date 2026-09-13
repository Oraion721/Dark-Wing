# Stationarity Testing in Time Series Analysis
**Module:** `Financial_Mathematics/Time_Series_Analysis/Stationarity_Tests/TS_stationarity_test.py`  
**Project:** Dark Wing — Financial Mathematics Division

---

## 1. Introduction: The Concept of Stationarity

In time series analysis and quantitative trading, a process is said to be **stationary** (specifically, weak-sense stationary) if its statistical properties do not change over time. Namely:
1. **Constant Mean:** $E[X_t] = \mu \ \forall \ t$.
2. **Constant Variance:** $Var(X_t) = \sigma^2$ for all $t$.
3. **Time-Invariant Autocovariance:** $Cov(X_t, X_{t+h})$ depends only on the $lag(h)$, not on time $t$.

### Why Stationarity Matters
Most time series models (including ARMA, VAR) and statistical tests assume stationarity. If we regress one non-stationary series on another, we risk obtaining a **spurious regression** (high $R^2$ and significant t-statistics, but no economic or mathematical relationship). 

In pairs trading, confirming that individual asset price series are **non-stationary** $(I(1))$ and that their linear combination (the spread) is **stationary** $(I(0))$ is the foundational step of cointegration.

---

## 2. Mathematical Background of the Tests

The `StationarityTest` class uses a **combined testing approach** combining the **Augmented Dickey-Fuller (ADF)** test and the **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** test.

### A. Augmented Dickey-Fuller (ADF) Test
The ADF test is a unit root test. It models the series using the following regression:

$$
\Delta X_t = \alpha + \beta_t + \gamma X_{t-1} + \sum_{i=1}^p \delta_i \Delta X_{t-i} + \varepsilon_t
$$

Where:
*   $\Delta$ is the differencing operator ($\Delta X_t = X_t - X_{t-1}$).
*   $\alpha$ is a constant (drift).
*   $\beta_t$ is a deterministic time trend.
*   $\gamma X_{t-1}$ is the term tested for a unit root.
*   $\sum \limits_{i=1}^p \delta_i \Delta X_{t-i}$ are lagged difference terms added to remove serial correlation in the residuals.

#### Hypotheses:
*   **Null Hypothesis ($H_0$):** $\gamma = 0$ (The process has a unit root / is non-stationary).
*   **Alternative Hypothesis ($H_1$):** $\gamma < 0$ (The process has no unit root / is stationary).

#### Decision Rule:
If the test statistic is more negative than the critical value (or $p\text{-value} < 0.05$), we reject $H_0$ and conclude the series is stationary.

---

### B. KPSS Test
Unlike the ADF test, the KPSS test assumes stationarity as the null hypothesis. It decomposes the series into a deterministic trend, a random walk, and a stationary error:

$$
X_t = r_t + \beta_t + \varepsilon_t \\
r_t = r_{t-1} + u_t, \quad u_t \sim \text{i.i.d.}(0, \sigma_u^2)
$$

Where:
*   $r_t$ is the random walk component.
*   $\beta_t$ is the deterministic trend.
*   $\varepsilon_t$ is the stationary error.

#### Hypotheses:
*   **Null Hypothesis ($H_0$):** $\sigma_u^2 = 0$ (The random walk component has zero variance, meaning the series is stationary around a level or trend).
*   **Alternative Hypothesis ($H_1$):** $\sigma_u^2 > 0$ (Non-stationary / unit root process).

#### Decision Rule:
If the test statistic is greater than the critical value (or $p\text{-value} < 0.05$), we reject $H_0$ and conclude the series is non-stationary.

---

### C. Combined Stationarity Decision Matrix

By running both tests, we can classify the series with higher confidence:

| ADF Reject $H_0$ ($p < 0.05$) | KPSS Reject $H_0$ ($p < 0.05$) | Combined Conclusion | Interpretation / Action |
| :--- | :--- | :--- | :--- |
| **Yes** (ADF Stationary) | **No** (KPSS Stationary) | **Stationary** | High confidence in $I(0)$ behavior. |
| **No** (ADF Non-Stationary) | **Yes** (KPSS Non-Stationary) | **Non-Stationary** | High confidence in $I(1)$ unit root. |
| **Yes** (ADF Stationary) | **Yes** (KPSS Non-Stationary) | **Conflicting (Trend/Diff)** | Likely difference-stationary; may need differencing. |
| **No** (ADF Non-Stationary) | **No** (KPSS Stationary) | **Conflicting (Trend)** | Likely trend-stationary; may need detrending. |

---

## 3. Code Architecture & API Reference

### Class: `StationarityTest`

```python
class StationarityTest:
    def __init__(self, time_series, lag_max=None, regression_type="c", required_lag="AIC", regression_result=False)
```

#### Constructor Arguments:
*   `time_series` (`pd.DataFrame` or `pd.Series`): The data series to analyze.
*   `lag_max` (`int`, optional): Maximum number of lags to include. If `None`, defaults to $12 \times (T/100)^{0.25}$.
*   `regression_type` (`str`): Constant and trend order to include in regression.
    *   `"c"`: Constant only (default).
    *   `"ct"`: Constant and trend.
    *   `"ctt"`: Constant, linear, and quadratic trend.
    *   `"n"`: No constant, no trend.
*   `required_lag` (`str` or `None`): Method to automatically determine lag length:
    *   `"AIC"` (default): Minimizes Akaike Information Criterion.
    *   `"BIC"`: Minimizes Bayesian Information Criterion.
    *   `"t-stat"`: Based on significance of the last lag.
    *   `None`: Sets lags to `lag_max`.
*   `regression_result` (`bool`): If `True`, returns full statsmodels regression objects (default `False`).

---

### Methods

#### 1. `_validate_inputs(self)`
*   **Purpose:** Ensures parameters are valid, checks for empty datasets, drops any `NaN` or `Inf` values, and raises `ValueError` if observations $T < 20$.

#### 2. `difference(self, order=1)`
*   **Purpose:** Applies the differencing operator $\Delta^d X_t$ to the series.
*   **Arguments:** `order` (`int`): The number of differences (default `1`).
*   **Returns:** A new `pd.DataFrame` containing the differenced series with missing values dropped.

#### 3. `test_adf(self)`
*   **Purpose:** Computes the ADF test statistic, p-value, critical values, and optimal lags.
*   **Returns:** A dictionary:
    ```python
    {
        'test_name': 'Augmented Dicky-Fuller Test',
        'ADF_statistics': float,
        'p_value': float,
        'used_lag': int,
        'n_observation': int,
        'critical_value': {'1%': float, '5%': float, '10%': float},
        'best_ic': float or None
    }
    ```

#### 4. `test_kpss(self)`
*   **Purpose:** Computes the KPSS test statistic, p-value, and critical values.
*   **Returns:** A dictionary:
    ```python
    {
        'test_name': 'KPSS Test',
        'KPSS_statistics': float,
        'p_value': float,
        'lags': int,
        'critical_value': {'10%': float, '5%': float, '2.5%': float, '1%': float}
    }
    ```

#### 5. `check_stationarity(self)` / `run(self)`
*   **Purpose:** Runs both ADF and KPSS tests and combines their decisions using the criteria:
    *   `adf_stationary` = `p_value < 0.05`
    *   `kpss_stationary` = `p_value > 0.05`
    *   `is_stationary` = `adf_stationary and kpss_stationary`
*   **Returns:** A dictionary containing complete individual test outputs and the combined decision:
    ```python
    {
        'ADF': dict,
        'KPSS': dict,
        'is_stationary': bool
    }
    ```

---

## 4. Usage Example

```python
import numpy as np
import pandas as pd
from TS_stationarity_test import StationarityTest

# 1. Load log price series
data = pd.read_csv("market_data.csv", index_index="Date", parse_dates=True)
log_prices = np.log(data["Close"])

# 2. Initialize StationarityTest
test = StationarityTest(
    time_series=log_prices, 
    lag_max=10, 
    regression_type="c", 
    required_lag="BIC"
)

# 3. Check stationarity of raw log prices (usually non-stationary I(1))
result = test.run()
print(f"Are raw prices stationary? {result['is_stationary']}")

# 4. Difference the series to check returns (usually stationary I(0))
returns = test.difference(order=1)
ret_test = StationarityTest(returns, lag_max=10, regression_type="c")
ret_result = ret_test.run()
print(f"Are log returns stationary? {ret_result['is_stationary']}")
```
