"""
VAR Optimal Lag Selection
=========================
Module: Financial Mathematics / Vector Auto Regressive (VAR)
Project: Dark Wing

Purpose:
    Before fitting a VAR or VECM model to identify cointegrated pairs,
    we must determine the correct number of lags (p) to include.
    Using the wrong lag count leads to:
        - Under-fitting (autocorrelated residuals, invalid tests)
        - Over-fitting (spurious relationships, slow execution)
    This class wraps statsmodels VAR.select_order() into a robust,
    importable interface with validation, auto-lag, and plotting.

Usage:
    from Optimal_lag_selection import VARLagSelector
    selector = VARLagSelector(data=df, maxlags=10, criteria="aic")
    results  = selector.select_lag()
    selector.summary()
    selector.plot()
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.api import VAR


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_VALID_CRITERIA  = ("aic", "bic", "hqic", "fpe")
_VALID_TRANSFORM = ("none", "log", "returns")
_IC_COLORS       = {"aic": "steelblue", "bic": "darkorange",
                    "hqic": "seagreen",  "fpe": "mediumpurple"}


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────
class VARLagSelector:
    """
    VAR Optimal Lag Order Selector.

    Validates input data, optionally transforms it, and uses
    statsmodels VAR.select_order() to find the lag p* that minimises
    a chosen Information Criterion across [0, maxlags].

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with >= 2 columns. Each column is one time series
        (e.g. log prices of different assets).
    maxlags : int or None
        Maximum lag to test. If None, uses Ng-Perron formula:
        floor(12 * (T/100)^0.25).
    criteria : str
        IC to minimise for selecting p*. One of: 'aic', 'bic', 'hqic', 'fpe'.
        All four ICs are always computed and available in the table.
    transform : str
        How to pre-process the data before fitting:
        - 'none'    → data is already in the form you want (e.g. log prices)
        - 'log'     → apply log() to raw prices → log prices
        - 'returns' → apply log(Pt / Pt-1)     → log returns
    trend : str
        Deterministic trend for the VAR. One of: 'n', 'c', 'ct', 'ctt'.
        Default 'c' (constant only) is appropriate for log prices.
    """

    def __init__(
        self,
        data:      pd.DataFrame,
        maxlags:   int  = None,
        criteria:  str  = "aic",
        transform: str  = "none",
        trend:     str  = "c",
    ):
        # ── 1. Validate scalar arguments first ─────────────────────────────
        criteria  = criteria.lower().strip()
        transform = transform.lower().strip()
        trend     = trend.lower().strip()

        if criteria not in _VALID_CRITERIA:
            raise ValueError(
                f"criteria must be one of {_VALID_CRITERIA}, got '{criteria}'"
            )
        if transform not in _VALID_TRANSFORM:
            raise ValueError(
                f"transform must be one of {_VALID_TRANSFORM}, got '{transform}'"
            )
        if trend not in ("n", "c", "ct", "ctt"):
            raise ValueError(
                f"trend must be one of ('n','c','ct','ctt'), got '{trend}'"
            )

        self.criteria  = criteria
        self.transform = transform
        self.trend     = trend

        # ── 2. Copy & validate the DataFrame ───────────────────────────────
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame")
        if data.empty:
            raise ValueError("data is empty.")
        if data.shape[1] < 2:
            raise ValueError(
                f"VAR requires >= 2 time series columns, got {data.shape[1]}."
            )

        self.data = data.copy()

        # ── 3. Apply transformation (once, here in __init__) ───────────────
        self._prepared = self._apply_transform()   # stores the ready-to-use df

        # ── 4. Now we know T and m ──────────────────────────────────────────
        self.T = len(self._prepared)               # number of observations
        self.m = self._prepared.shape[1]           # number of series

        # ── 5. Resolve maxlags ─────────────────────────────────────────────
        if maxlags is None:
            self.maxlags = self._auto_maxlag(self.T)
        else:
            if not isinstance(maxlags, int) or maxlags < 1:
                raise ValueError("maxlags must be a positive integer.")
            # Leave 10 rows as minimum estimation buffer
            if maxlags >= self.T - 10:
                raise ValueError(
                    f"maxlags ({maxlags}) is too large for T={self.T}. "
                    f"Must be < {self.T - 10}."
                )
            self.maxlags = maxlags

        # ── 6. Result placeholders (populated by select_lag) ───────────────
        self.optimal_lag: int               = None
        self.ic_table:    pd.DataFrame      = None
        self.sigma_u:     np.ndarray        = None   # residual cov at p*
        self._fitted_model                  = None   # VAR(p*) fitted model

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_transform(self) -> pd.DataFrame:
        """Transform raw data according to self.transform and sanitise."""
        df = self.data.copy()

        if self.transform == "log":
            if (df <= 0).any().any():
                raise ValueError(
                    "transform='log' requires all values > 0. "
                    "Found non-positive values in data."
                )
            df = np.log(df)

        elif self.transform == "returns":
            if (df <= 0).any().any():
                raise ValueError(
                    "transform='returns' requires all values > 0."
                )
            df = np.log(df / df.shift(1))

        # Drop NaN rows (created by shift or missing data)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            raise ValueError(
                "After transformation and NaN removal, data is empty."
            )

        return df

    @staticmethod
    def _auto_maxlag(T: int) -> int:
        """Ng-Perron (2001) formula for automatic max lag selection."""
        return max(1, int(np.floor(12 * (T / 100) ** 0.25)))

    def _require_fitted(self, method_name: str):
        """Guard: raise if select_lag() hasn't been called yet."""
        if self.ic_table is None:
            raise RuntimeError(
                f"Call select_lag() before calling {method_name}()."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Core Method
    # ─────────────────────────────────────────────────────────────────────────
    def select_lag(self) -> dict:
        """
        Run VAR lag-order selection across lags [0, maxlags].

        Returns
        -------
        dict with keys:
            'optimal_lag'  : int   — best lag p* for chosen criteria
            'criteria'     : str   — which IC was used
            'all_criteria' : dict  — {'aic': int, 'bic': int, ...}
            'ic_table'     : pd.DataFrame — IC scores for each lag
            'sigma_u'      : np.ndarray   — residual covariance at p*
        """
        df = self._prepared   # already transformed and cleaned

        # ── Build and run statsmodels VAR ───────────────────────────────────
        model = VAR(df)
        try:
            lag_result = model.select_order(
                maxlags=self.maxlags,
                trend=self.trend,
            )
        except Exception as e:
            raise RuntimeError(f"VAR.select_order() failed: {e}") from e

        # ── Extract IC table ────────────────────────────────────────────────
        # statsmodels stores ics as dict: {'aic': array, ...}
        # index 0 = lag 0, index 1 = lag 1, ...
        n_rows = len(lag_result.ics["aic"])
        self.ic_table = pd.DataFrame(
            {crit: lag_result.ics[crit] for crit in _VALID_CRITERIA},
            index=range(0, n_rows),      # lag 0 through maxlags
        )
        self.ic_table.index.name = "lag"

        # ── Extract p* for each criterion ───────────────────────────────────
        all_optimal = {
            crit: int(lag_result.selected_orders[crit])
            for crit in _VALID_CRITERIA
        }
        self.optimal_lag = all_optimal[self.criteria]

        # ── Fit VAR(p*) to retrieve residual covariance matrix ─────────────
        if self.optimal_lag == 0:
            warnings.warn(
                f"Optimal lag = 0 selected by {self.criteria.upper()}. "
                "This means no temporal dependence was found. "
                "Setting lag = 1 for covariance estimation.",
                UserWarning,
            )
            fit_lag = 1
        else:
            fit_lag = self.optimal_lag

        self._fitted_model = model.fit(maxlags=fit_lag, trend=self.trend)
        self.sigma_u = self._fitted_model.sigma_u   # (m x m) residual cov

        return {
            "optimal_lag":  self.optimal_lag,
            "criteria":     self.criteria,
            "all_criteria": all_optimal,
            "ic_table":     self.ic_table,
            "sigma_u":      self.sigma_u,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting Methods
    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> None:
        """Print a formatted summary table of IC scores and the decision."""
        self._require_fitted("summary")

        sep = "=" * 62
        print(f"\n{sep}")
        print("  VAR LAG ORDER SELECTION SUMMARY")
        print(sep)
        print(f"  Series (m)      : {self.m}")
        print(f"  Observations (T): {self.T}")
        print(f"  Max lags tested : {self.maxlags}")
        print(f"  Trend           : {self.trend.upper()}")
        print(f"  Selection basis : {self.criteria.upper()}")
        print(f"{sep}")

        # IC table — highlight the optimal lag row
        ic_display = self.ic_table.copy()
        ic_display.columns = [c.upper() for c in ic_display.columns]

        # Mark optimal row with '*' for the chosen criterion
        marker_col = self.criteria.upper()
        opt_val    = ic_display.loc[self.optimal_lag, marker_col]
        ic_display[marker_col] = ic_display[marker_col].apply(
            lambda v: f">>> {v:.6f} <<<" if v == opt_val else f"    {v:.6f}"
        )
        # Format remaining columns
        for col in ic_display.columns:
            if col != marker_col:
                ic_display[col] = ic_display[col].apply(lambda v: f"{v:.6f}")

        print(ic_display.to_string())
        print(sep)
        print(f"  OPTIMAL LAG p* = {self.optimal_lag}  "
              f"[{self.criteria.upper()} minimised]")
        print(sep)

        # Report ALL criteria winners
        print("  All criteria decisions:")
        for crit in _VALID_CRITERIA:
            p = self.ic_table[crit].idxmin()
            flag = "  ← selected" if crit == self.criteria else ""
            print(f"    {crit.upper():5s}  →  p* = {p}{flag}")
        print(f"{sep}\n")

    def plot(self, figsize: tuple = (16, 5)) -> None:
        """
        Plot IC values vs. lag for all four criteria.
        The chosen criterion's optimal lag is highlighted with a red dashed
        vertical line and a red dot.
        """
        self._require_fitted("plot")

        fig, axes = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(
            f"VAR Lag Selection — p* = {self.optimal_lag} "
            f"[{self.criteria.upper()}]",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for ax, (crit, color) in zip(axes, _IC_COLORS.items()):
            values  = self.ic_table[crit]
            opt_lag = int(values.idxmin())

            ax.plot(
                values.index, values.values,
                marker="o", color=color, linewidth=1.8,
                markersize=4, label=crit.upper(),
            )
            # Vertical line at this criterion's own minimum
            ax.axvline(
                opt_lag, color=color, linestyle="--",
                linewidth=1.2, alpha=0.7,
            )
            # Extra emphasis if this is the selected criterion
            if crit == self.criteria:
                ax.axvline(
                    opt_lag, color="red", linestyle="--",
                    linewidth=2.0, label=f"p* = {opt_lag}",
                )
                ax.scatter(
                    [opt_lag], [values[opt_lag]],
                    color="red", zorder=5, s=100, label="optimal",
                )

            ax.set_title(crit.upper(), fontweight="bold", color=color)
            ax.set_xlabel("Lag order (p)", fontsize=9)
            ax.set_ylabel("IC value", fontsize=9)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_optimal_lag(self) -> int:
        """Return only the integer optimal lag. Useful when chaining modules."""
        self._require_fitted("get_optimal_lag")
        return self.optimal_lag

    def get_sigma_u(self) -> np.ndarray:
        """Return the (m x m) residual covariance matrix at optimal lag."""
        self._require_fitted("get_sigma_u")
        return self.sigma_u

    def __repr__(self) -> str:
        status = (
            f"optimal_lag={self.optimal_lag}"
            if self.optimal_lag is not None
            else "not fitted yet — call select_lag()"
        )
        return (
            f"VARLagSelector("
            f"m={self.m}, T={self.T}, "
            f"maxlags={self.maxlags}, "
            f"criteria={self.criteria.upper()}, "
            f"{status})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Test — Only runs when this file is executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    # Download two real equity series (log prices, 5 years daily)
    tickers = ["RELIANCE.NS", "TCS.NS"]
    raw = yf.download(tickers, start="2019-01-01", end="2024-01-01",
                      auto_adjust=True)["Close"]
    raw.dropna(inplace=True)

    print(f"Downloaded data: {raw.shape[0]} rows × {raw.shape[1]} columns")
    print(raw.head(3))

    # Instantiate — pass raw prices, let the class convert to log prices
    selector = VARLagSelector(
        data      = raw,
        maxlags   = 10,
        criteria  = "aic",
        transform = "log",    # raw prices → log prices internally
        trend     = "c",
    )
    print(selector)   # calls __repr__

    # Run lag selection
    results = selector.select_lag()

    # Print clean summary
    selector.summary()

    # Plot all IC curves
    selector.plot()

    # Use the result downstream (e.g. pass to Johansen test)
    p_star = selector.get_optimal_lag()
    print(f"\nOptimal lag to pass into Johansen/VECM: p* = {p_star}")