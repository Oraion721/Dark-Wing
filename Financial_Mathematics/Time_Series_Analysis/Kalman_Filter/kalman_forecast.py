"""
Module  : Financial Mathematics / Time Series Analysis / Kalman_Filter / kalman_forecast.py
Project : Dark Wing

Purpose:
    Use the final Kalman filter state (β̂_{T|T}, P_{T|T}) to forecast the spread
    and price h steps ahead, with growing uncertainty bands.

    Kalman Forecast Theory (from Notion notes):
    ─────────────────────────────────────────────
    For t' > t (forecasting), we run the prediction step WITHOUT correction
    (no new observations yet):
        β̂_{t+h|t} = T^h · β̂_{t|t}   →  β̂_{t+h|t} = β̂_{t|t}   (T=1, random walk)
        P_{t+h|t}  = P_{t|t} + h·Q                               (uncertainty grows linearly)

    Observation forecast:
        Ŷ_{t+h|t}  = Z_{t+h} · β̂_{t+h|t} = X_{t+h} · β̂_{t|t}

    Observation forecast variance:
        Var(Ŷ_{t+h|t}) = X_{t+h}² · P_{t+h|t} + R
                        = X_{t+h}² · (P_{t|t} + h·Q) + R

    Spread forecast:
        Ŝ_{t+h|t} = Ŷ_{t+h|t} - β̂_{t|t} · X_{t+h}  =  0  (under random walk)
    Note: Kalman spread forecast is zero by construction under a pure random walk
    β model. The spread forecast is more meaningful as the UNCERTAINTY bands:
        CI_{t+h}: ± z_α · sqrt(Var(Ŷ_{t+h|t}))

    Expected Reversion Time:
        h* = min{h : |CI_lower(h)| < μ_spread}  (when uncertainty spans zero)

Pipeline Position:
    STEP 6   → kalman_filter.py (KalmanFilterPairs) — produces final state
    STEP 7   → THIS FILE (KalmanForecaster)
    STEP 8   → signal/ — uses forecast uncertainty for position sizing

Dependencies: numpy, pandas, scipy, matplotlib
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from typing import Optional,Dict,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
for _p in [_FM_DIR,_TSA_DIR,_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

from kalman_filter import KalmanFilterPairs

# ─────────────────────────────────────────────────────────────────────────────
class KalmanForecaster:
    """
    Multi-step-ahead spread and price forecaster using the Kalman filter state.

    Uses the final filtered state β̂_{T|T} and P_{T|T} to project uncertainty
    forward. The core insight is:
        - β doesn't change in forecasting (T=1 random walk): β̂_{t+h|t} = β̂_{t|t}
        - But uncertainty grows: P_{t+h|t} = P_{t|t} + h·Q
        - This growing uncertainty produces widening confidence bands on spread

    Parameters
    ----------
    kf : KalmanFilterPairs — A fitted Kalman filter (after calling filter()).
    alpha_level : float — Confidence level for prediction intervals. Default 0.05.
    """
    def __init__(self,kf:KalmanFilterPairs,alpha_level:float=0.05):
        if not isinstance(kf,KalmanFilterPairs):
            raise TypeError("kf must be a KalmanFilterPairs instance.")
        if not kf._fitted:
            raise RuntimeError("KalmanFilterPairs must be fitted. Call filter() first.")
        self._kf=kf
        self._alpha=alpha_level
        self._z_crit=float(stats.norm.ppf(1-alpha_level/2))
        # Use final filtered state
        self._beta_T=kf.current_beta()    # β̂_{T|T}
        self._P_T=kf.current_P()          # P_{T|T}
        self._Q=kf.Q; self._R=kf.R
        # Last X from filtered states (for default future_x)
        self._last_x=float(kf.states[-1].x)
        self._last_y=float(kf.states[-1].y)
        # Historical spread statistics (for reversion baseline)
        hist_spread=np.array([s.spread for s in kf.states])
        self._spread_mu=float(np.mean(hist_spread))
        self._spread_std=float(np.std(hist_spread))
        # results (set in forecast())
        self._steps:Optional[int]=None
        self._fc_df:Optional[pd.DataFrame]=None
        self._future_x:Optional[np.ndarray]=None

    # ── beta forecast ─────────────────────────────────────────────────────────
    def forecast_beta(self,steps:int)->np.ndarray:
        """
        Project β forward h steps.
            β̂_{t+h|t} = β̂_{t|t}   (constant, random walk assumption)
            P_{t+h|t}  = P_{t|t} + h·Q  (linearly growing uncertainty)

        Returns
        -------
        np.ndarray of shape (steps,) — all equal to β̂_{T|T}.
        """
        return np.full(steps,self._beta_T)

    def forecast_P(self,steps:int)->np.ndarray:
        """
        Project covariance P forward h steps.
            P_{t+h|t} = P_{T|T} + h·Q

        Returns
        -------
        np.ndarray of shape (steps,) — growing P values.
        """
        h_arr=np.arange(1,steps+1)
        return self._P_T+h_arr*self._Q

    # ── observation (price) forecast ─────────────────────────────────────────
    def forecast_price(self,future_x:np.ndarray,steps:Optional[int]=None)->np.ndarray:
        """
        Forecast dependent price Y_{t+h} given future independent prices X_{t+h}.
            Ŷ_{t+h|t} = β̂_{t|t} · X_{t+h}

        Parameters
        ----------
        future_x : np.ndarray — Future X values [X_{t+1}, X_{t+2}, ...].
                                If shorter than steps, last value is repeated.
        steps    : int | None — If None, uses len(future_x).

        Returns
        -------
        np.ndarray — Forecast Y values.
        """
        if steps is None: steps=len(future_x)
        x_extended=self._extend_x(future_x,steps)
        return self._beta_T*x_extended

    def forecast_variance(self,future_x:np.ndarray,steps:Optional[int]=None)->np.ndarray:
        """
        Forecast variance of Ŷ_{t+h}:
            Var(Ŷ_{t+h|t}) = X_{t+h}² · P_{t+h|t} + R
                            = X_{t+h}² · (P_{T|T} + h·Q) + R

        Returns
        -------
        np.ndarray — Variance of forecast, growing with h.
        """
        if steps is None: steps=len(future_x)
        x_ext=self._extend_x(future_x,steps)
        P_h=self.forecast_P(steps)
        return x_ext**2*P_h+self._R

    def forecast_spread(self,future_x:np.ndarray,steps:Optional[int]=None)->np.ndarray:
        """
        Forecast spread h steps ahead.
            Ŝ_{t+h|t} = Ŷ_{t+h|t} - β̂_{t|t} · X_{t+h}
                       = β̂_{T|T}·X_{t+h} - β̂_{T|T}·X_{t+h} = 0

        Under the random walk model, the expected spread forecast is ZERO.
        The practical value is in the UNCERTAINTY BANDS around zero.

        Returns
        -------
        np.ndarray — All zeros (expected mean reversion to zero).
        """
        if steps is None: steps=len(future_x)
        return np.zeros(steps)

    # ── full forecast pass ───────────────────────────────────────────────────
    def forecast(self,steps:int=20,future_x:Optional[np.ndarray]=None)->pd.DataFrame:
        """
        Run a complete multi-step forecast.

        Automatically generates future_x as a flat projection (last known X)
        if not provided.

        Parameters
        ----------
        steps   : int — How many steps ahead to forecast.
        future_x : np.ndarray | None — Future X values. If None, uses last X.

        Returns
        -------
        pd.DataFrame with columns:
            [h, beta_fc, P_fc, y_forecast, y_se, y_upper, y_lower,
             spread_fc, spread_se, spread_upper, spread_lower, future_x]
        """
        self._steps=steps
        if future_x is None:
            future_x=np.full(steps,self._last_x)
        self._future_x=self._extend_x(future_x,steps)
        x_ext=self._future_x
        h_arr=np.arange(1,steps+1)
        beta_fc=self.forecast_beta(steps)
        P_fc=self.forecast_P(steps)
        y_fc=self.forecast_price(x_ext,steps)
        y_var=self.forecast_variance(x_ext,steps)
        y_se=np.sqrt(y_var)
        y_upper=y_fc+self._z_crit*y_se
        y_lower=y_fc-self._z_crit*y_se
        # Spread SE (same as Y SE divided by β, in absolute terms)
        spread_se=y_se  # Since Ŝ = Ŷ - β·X and β is constant
        spread_upper=self._z_crit*spread_se
        spread_lower=-self._z_crit*spread_se
        # Build future date index
        last_idx=self._kf._y_index[-1] if hasattr(self._kf,"_y_index") else None
        if isinstance(last_idx,pd.Timestamp):
            try:
                future_idx=pd.bdate_range(start=last_idx,periods=steps+1,freq="B")[1:]
            except Exception:
                future_idx=pd.RangeIndex(steps)
        else:
            future_idx=pd.RangeIndex(steps)
        self._fc_df=pd.DataFrame({
            "h":h_arr,"beta_fc":beta_fc,"P_fc":P_fc,
            "y_forecast":y_fc,"y_se":y_se,"y_upper":y_upper,"y_lower":y_lower,
            "spread_fc":np.zeros(steps),"spread_se":spread_se,
            "spread_upper":spread_upper,"spread_lower":spread_lower,
            "future_x":x_ext
        },index=future_idx)
        print(f"[KalmanForecaster] {steps}-step forecast complete. β̂={self._beta_T:.6f}")
        return self._fc_df

    # ── Reversion time ────────────────────────────────────────────────────────
    def expected_reversion_time(self)->Optional[int]:
        """
        Estimate h* = number of steps until the current spread (last observed)
        is within ±1σ of the historical mean, based on β stability.

        For random walk β: the spread expected value never deviates, but
        this method checks when the CURRENT SPREAD magnitude (at T) will
        decay to ±1σ of historical mean under OU dynamics.

        Returns
        -------
        int | None — estimated number of steps for reversion, or None.
        """
        if self._fc_df is None:
            raise RuntimeError("Call forecast() first.")
        # current spread magnitude
        last_spread=float(self._kf.states[-1].spread)
        target=self._spread_mu+self._spread_std  # mean + 1σ threshold
        # Spread reverts as: |S_{T+h}| ≈ |S_T| · exp(-κ·h)
        # Estimate κ from historical innovations
        innovations=np.array([s.innovation for s in self._kf.states])
        # rough AR(1): κ ≈ -ln(autocorr(innovations))
        if len(innovations)>2:
            acf1=float(np.corrcoef(innovations[:-1],innovations[1:])[0,1])
            if -1<acf1<1 and acf1<0:
                kappa=-np.log(abs(acf1))
                if kappa>0:
                    h_star=int(np.ceil(np.log(abs(last_spread)/max(abs(target),1e-8))/kappa))
                    h_star=max(1,min(h_star,self._steps))
                    print(f"[KalmanForecaster] Estimated reversion time: {h_star} steps (κ≈{kappa:.4f})")
                    return h_star
        print("[KalmanForecaster] Cannot estimate reversion time from current data.")
        return None

    def confidence_interval(self,steps:Optional[int]=None,
                            alpha:Optional[float]=None,
                            future_x:Optional[np.ndarray]=None)->Dict[str,np.ndarray]:
        """
        Return confidence interval arrays for the spread forecast.
        Convenience wrapper around forecast().
        """
        if steps is not None or self._fc_df is None:
            self.forecast(steps or 20,future_x)
        a=alpha or self._alpha
        z=float(stats.norm.ppf(1-a/2))
        se=self._fc_df["spread_se"].values
        return {"upper":z*se,"lower":-z*se,"se":se,"h":self._fc_df["h"].values}

    # ── Helper ────────────────────────────────────────────────────────────────
    def _extend_x(self,future_x:np.ndarray,steps:int)->np.ndarray:
        """Pad or truncate future_x to exactly `steps` length using last value."""
        if len(future_x)>=steps:
            return np.asarray(future_x[:steps],dtype=float)
        pad=np.full(steps-len(future_x),future_x[-1] if len(future_x)>0 else self._last_x)
        return np.concatenate([np.asarray(future_x,dtype=float),pad])

    # ── Summary & Plot ────────────────────────────────────────────────────────
    def summary(self):
        if self._fc_df is None:
            raise RuntimeError("Call forecast() first.")
        sep="="*58
        print(f"\n{sep}")
        print("  KALMAN FORECASTER — SUMMARY")
        print(sep)
        print(f"  Forecast steps        : {self._steps}")
        print(f"  β̂ (T|T)              : {self._beta_T:.6f}")
        print(f"  P (T|T)               : {self._P_T:.4e}")
        print(f"  P at horizon {self._steps}       : {self._fc_df['P_fc'].iloc[-1]:.4e}")
        print(f"  Q (process noise)     : {self._Q:.4e}")
        print(f"  R (meas. noise)       : {self._R:.4e}")
        print(f"  Hist. spread μ / σ   : {self._spread_mu:.4f} / {self._spread_std:.4f}")
        se_final=self._fc_df["spread_se"].iloc[-1]
        z=self._z_crit
        print(f"  CI at horizon {self._steps}      : ±{z*se_final:.4f}")
        h=self.expected_reversion_time()
        if h: print(f"  Est. reversion time   : {h} steps")
        print(sep+"\n")

    def plot_forecast(self,figsize:tuple=(14,9)):
        """
        Two-panel forecast dashboard:
        (top) Y price forecast with growing uncertainty bands.
        (bot) Spread uncertainty bands (expected mean = 0, CI grows).
        """
        if self._fc_df is None:
            raise RuntimeError("Call forecast() first.")
        df=self._fc_df; idx=df.index
        # Append historical spread for context
        hist_states=self._kf.states
        hist_idx=self._kf._y_index if hasattr(self._kf,"_y_index") else np.arange(len(hist_states))
        hist_y=np.array([s.y for s in hist_states])
        hist_spread=np.array([s.spread for s in hist_states])
        fig,axes=plt.subplots(2,1,figsize=figsize)
        # Price forecast
        ax=axes[0]
        ax.plot(hist_idx[-60:],hist_y[-60:],color="steelblue",linewidth=1.0,label="Historical Y_t")
        ax.plot(idx,df["y_forecast"],color="darkorange",linewidth=1.5,marker="o",markersize=3,label="Forecast Ŷ_{t+h|t}")
        ax.fill_between(idx,df["y_lower"],df["y_upper"],alpha=0.25,color="darkorange",
                        label=f"{int((1-self._alpha)*100)}% CI (growing)")
        ax.axvline(hist_idx[-1] if len(hist_idx)>0 else 0,color="gray",linestyle=":",linewidth=1.0,label="Forecast start")
        ax.set_title(f"Price Forecast  (β̂={self._beta_T:.4f}  |  {self._steps}-step ahead)",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # Spread uncertainty
        ax=axes[1]
        ax.plot(hist_idx[-60:],hist_spread[-60:],color="mediumpurple",linewidth=0.9,label="Historical Spread")
        ax.axhline(self._spread_mu,color="black",linestyle="--",linewidth=0.9,label=f"μ={self._spread_mu:.4f}")
        ax.fill_between(idx,df["spread_lower"],df["spread_upper"],alpha=0.30,color="lightcoral",label=f"{int((1-self._alpha)*100)}% Spread CI")
        ax.axvline(hist_idx[-1] if len(hist_idx)>0 else 0,color="gray",linestyle=":",linewidth=1.0)
        h_rev=self.expected_reversion_time()
        if h_rev and h_rev<=self._steps:
            ax.axvline(idx[h_rev-1],color="green",linestyle="-.",linewidth=1.2,label=f"Est. reversion @ h={h_rev}")
        ax.set_title(f"Spread Forecast Uncertainty (±{self._z_crit:.2f}σ CI)",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        if isinstance(idx,pd.DatetimeIndex):
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                plt.setp(ax.xaxis.get_majorticklabels(),rotation=25)
        fig.suptitle("Kalman Filter — Multi-Step Forecast Dashboard",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        status="forecasted" if self._fc_df is not None else "not forecasted — call forecast()"
        return (f"KalmanForecaster(β̂={self._beta_T:.6f}, P={self._P_T:.2e}, "
                f"steps={self._steps}, {status})")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import yfinance as yf
    raw=yf.download(["RELIANCE.NS","TCS.NS"],start="2021-01-01",end="2024-01-01",auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["RELIANCE.NS"],log_px["TCS.NS"]
    kf=KalmanFilterPairs(Q=1e-5,R=1e-3); kf.filter(y,x)
    fcaster=KalmanForecaster(kf,alpha_level=0.05)
    fc_df=fcaster.forecast(steps=30)
    fcaster.summary(); fcaster.plot_forecast()
