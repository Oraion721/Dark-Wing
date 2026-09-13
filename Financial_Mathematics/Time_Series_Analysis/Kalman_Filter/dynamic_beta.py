"""
Module  : Financial Mathematics / Time Series Analysis / Kalman_Filter / dynamic_beta.py
Project : Dark Wing

Purpose:
    High-level pipeline orchestrator that chains:
        SpreadBuilder (OLS initial β)
        → KalmanFilterPairs (dynamic β_{t|t})
        → KalmanSmoother (β_{t|T} for backtesting)
        → HalfLife (τ from dynamic spread)
        → ZScore (normalised signal)

    This is the PRIMARY interface used by signal/ and backtest/ modules.
    It produces a time-varying hedge ratio β_t and a Kalman-filtered spread
    that is more statistically robust than a fixed OLS spread.

    Key addition: Regime Detection
        A structural break in β_t is detected when the smoothed β_t shows
        a significant level shift. This signals the pair has broken down
        and positions should be reduced.

    Q/R Heuristic (from Notion notes + OU connection):
        Q ≈ σ²_ε / τ   (process noise scales inversely with half-life)
        R ≈ var(OLS_residuals)  (measurement noise = spread variance)
        These can be overridden by MLE optimisation (optimize=True).

Pipeline Position:
    STEP 6.2 → THIS FILE (DynamicBeta)
    STEP 7   → vecm_forecast.py, kalman_forecast.py
    STEP 8   → signal/pair_signal.py

Dependencies: numpy, pandas, scipy, matplotlib
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from typing import Optional,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

from spread_builder import SpreadBuilder
from half_life import HalfLife
from zscore import ZScore
from kalman_filter import KalmanFilterPairs
from smoother import KalmanSmoother

# ─────────────────────────────────────────────────────────────────────────────
class DynamicBeta:
    """
    Full pairs-trading pipeline with dynamic hedge ratio.

    Parameters
    ----------
    y          : pd.Series — Dependent (Y) log-price series.
    x          : pd.Series — Independent (X) log-price series.
    Q          : float | None — Process noise. If None, estimated from half-life.
    R          : float | None — Measurement noise. If None, from OLS residuals.
    P_init     : float — Initial state covariance. Default 1000.
    beta_init  : float | None — Initial β. If None, uses OLS β.
    optimize   : bool — If True, run MLE to optimise Q and R.
    half_life_window : int — Window for rolling half-life (obs). Default 252.
    entry_z    : float — Z-score entry threshold.
    exit_z     : float — Z-score exit threshold.
    stop_z     : float — Z-score stop-loss threshold.
    run_smoother : bool — If True, also run RTS backward smoother.
    """
    def __init__(self,y:pd.Series,x:pd.Series,Q:Optional[float]=None,
                 R:Optional[float]=None,P_init:float=1000.0,
                 beta_init:Optional[float]=None,optimize:bool=False,
                 half_life_window:int=252,entry_z:float=2.0,
                 exit_z:float=0.5,stop_z:float=3.5,run_smoother:bool=True):
        if not isinstance(y,pd.Series) or not isinstance(x,pd.Series):
            raise TypeError("y and x must be pd.Series.")
        aligned=pd.concat([y.rename("y"),x.rename("x")],axis=1).dropna()
        self._y=aligned["y"]; self._x=aligned["x"]
        self.P_init=P_init
        self.optimize=optimize
        self.half_life_window=half_life_window
        self.entry_z=entry_z; self.exit_z=exit_z; self.stop_z=stop_z
        self.run_smoother_flag=run_smoother
        # Step 1: OLS spread (initial β estimate)
        self._sb=SpreadBuilder(y=self._y,x=self._x,mode="ols")
        self._sb.build()
        ols_beta=self._sb.get_beta()
        self._ols_spread=self._sb.spread_series()
        # Step 2: half-life → Q heuristic
        _hl=HalfLife(self._ols_spread)
        _hl.estimate()
        self._tau=_hl.half_life_days
        _ou_resid_var=float(np.var(_hl.get_ou_residuals())) if _hl.get_ou_residuals() is not None else 1e-4
        # Q/R defaults from OU heuristic
        self._Q=Q if Q is not None else max(1e-8,_ou_resid_var/max(self._tau,1.0))
        self._R=R if R is not None else max(1e-6,float(np.var(self._ols_spread.values)))
        self._beta_init=beta_init if beta_init is not None else ols_beta
        # Internal objects (set in fit())
        self._kf:Optional[KalmanFilterPairs]=None
        self._ks:Optional[KalmanSmoother]=None
        self._states_df:Optional[pd.DataFrame]=None
        self._zscore:Optional[ZScore]=None
        self._fitted:bool=False

    # ── Main fit ─────────────────────────────────────────────────────────────
    def fit(self)->pd.DataFrame:
        """
        Run the full pipeline:
            OLS β → Q/R heuristic → [MLE optimisation] → Kalman filter
            → [RTS smoother] → half-life of Kalman spread → Z-score

        Returns
        -------
        pd.DataFrame — All Kalman states plus zscore column.
        """
        print(f"\n{'='*55}")
        print(f"  DynamicBeta Pipeline")
        print(f"{'='*55}")
        print(f"  OLS β init   : {self._beta_init:.6f}")
        print(f"  Half-life τ  : {self._tau:.2f} obs")
        print(f"  Q (initial)  : {self._Q:.4e}")
        print(f"  R (initial)  : {self._R:.4e}")
        # Kalman filter
        self._kf=KalmanFilterPairs(Q=self._Q,R=self._R,
                                    beta_init=self._beta_init,P_init=self.P_init)
        if self.optimize:
            print("  Running MLE optimisation for Q, R...")
            res=self._kf.optimize_QR(self._y,self._x)
            print(f"  Optimised Q={res['Q_opt']:.4e}, R={res['R_opt']:.4e}")
        self._states_df=self._kf.filter(self._y,self._x)
        # Smoother
        if self.run_smoother_flag:
            self._ks=KalmanSmoother(self._kf)
            self._ks.smooth()
            self._states_df["smoothed_beta"]=self._ks._smoothed_beta
            self._states_df["smoothed_P"]=self._ks._smoothed_P
            self._states_df["smoothed_spread"]=self._ks.smoothed_spread().values
        # Z-score on Kalman spread
        kalman_spread=self._kf.spread_series()
        # Cap window: 2*τ but never exceed half the dataset (avoids ZScore ValueError when pair barely mean-reverts)
        zs_window=max(10,min(int(self._tau*2),len(kalman_spread)//2))
        self._zscore=ZScore(kalman_spread,window=zs_window,mode="rolling",
                            entry_z=self.entry_z,exit_z=self.exit_z,stop_z=self.stop_z)
        self._zscore.compute()
        self._zscore.generate_signals()
        self._states_df["zscore"]=self._zscore.zscore_series().values
        self._states_df["signal"]=self._zscore.signal_series().values
        self._fitted=True
        print(f"\n  Pipeline complete. Z-score window: {zs_window} obs")
        print(f"  Final beta (T|T): {self._kf.current_beta():.6f}")
        return self._states_df

    # ── Regime detection ─────────────────────────────────────────────────────
    def regime_detect(self,window:int=126,threshold_sigma:float=2.0)->pd.Series:
        """
        Detect structural breaks in β_t using a rolling CUSUM test.

        Method: For each point t, compare the mean of β in [t-W, t] to the
        overall mean. A break is flagged when the difference exceeds
        `threshold_sigma` rolling standard deviations.

        Returns
        -------
        pd.Series of bool — True where a regime break is detected.
        """
        self._require_fitted("regime_detect")
        beta=self._states_df["beta"]
        rolling_mean=beta.rolling(window=window,min_periods=window//2).mean()
        rolling_std=beta.rolling(window=window,min_periods=window//2).std()
        global_mean=beta.mean()
        deviation=(rolling_mean-global_mean).abs()
        threshold=threshold_sigma*rolling_std
        breaks=(deviation>threshold)
        n_breaks=breaks.sum()
        print(f"[DynamicBeta] Regime breaks detected: {n_breaks} ({n_breaks/len(beta)*100:.1f}%)")
        return breaks

    # ── Getters ───────────────────────────────────────────────────────────────
    def _require_fitted(self,method:str):
        if not self._fitted:
            raise RuntimeError(f"Call fit() before calling {method}().")

    def current_beta(self)->float:
        """Latest β_{T|T} — use for live position sizing."""
        self._require_fitted("current_beta"); return self._kf.current_beta()

    def current_P(self)->float:
        """Latest P_{T|T} — uncertainty in current β."""
        self._require_fitted("current_P"); return self._kf.current_P()

    def dynamic_spread_series(self)->pd.Series:
        """Kalman-filtered spread S_t = Y_t - β̂_{t|t}·X_t."""
        self._require_fitted("dynamic_spread_series")
        return self._states_df["spread"].rename("dynamic_spread")

    def zscore_series(self)->pd.Series:
        """Rolling Z-score of Kalman spread."""
        self._require_fitted("zscore_series"); return self._states_df["zscore"].rename("zscore")

    def signal_series(self)->pd.Series:
        """Discrete trading signals {-2,-1,0,1,2}."""
        self._require_fitted("signal_series"); return self._states_df["signal"].rename("signal")

    def get_filter(self)->KalmanFilterPairs:
        """Return the underlying fitted KalmanFilterPairs object."""
        self._require_fitted("get_filter"); return self._kf

    def get_smoother(self)->Optional[KalmanSmoother]:
        """Return the underlying KalmanSmoother (if run_smoother=True)."""
        self._require_fitted("get_smoother"); return self._ks

    # ── Summary & Dashboard ────────────────────────────────────────────────
    def summary(self):
        self._require_fitted("summary")
        sep="="*58
        df=self._states_df
        sig=df["signal"]
        print(f"\n{sep}")
        print("  DYNAMIC BETA — PIPELINE SUMMARY")
        print(sep)
        print(f"  Observations          : {len(df)}")
        print(f"  OLS β (static)        : {self._sb.get_beta():.6f}")
        print(f"  Kalman β range        : [{df['beta'].min():.4f}, {df['beta'].max():.4f}]")
        print(f"  Final β̂ (T|T)        : {self.current_beta():.6f}")
        print(f"  Half-life τ           : {self._tau:.2f} obs")
        print(f"  Q / R                 : {self._kf.Q:.4e} / {self._kf.R:.4e}")
        print(f"  Kalman spread std     : {df['spread'].std():.6f}")
        print(f"  Z-score window        : {self._zscore.window} obs")
        print(f"  LONG  %               : {(sig==1).mean()*100:.1f}%")
        print(f"  SHORT %               : {(sig==-1).mean()*100:.1f}%")
        print(f"  FLAT  %               : {(sig==0).mean()*100:.1f}%")
        print(sep+"\n")

    def plot_dashboard(self,figsize:tuple=(14,12)):
        """Five-panel dashboard: OLS vs Kalman β, spread, Z-score, signal, P."""
        self._require_fitted("plot_dashboard")
        df=self._states_df; idx=df.index
        ols_beta_const=self._sb.get_beta()
        fig,axes=plt.subplots(4,1,figsize=figsize,sharex=True)
        # β panel
        ax=axes[0]
        ax.axhline(ols_beta_const,color="gray",linewidth=1.2,linestyle="--",label=f"OLS β={ols_beta_const:.3f}")
        ax.plot(idx,df["beta"],color="steelblue",linewidth=1.0,label="Kalman β̂_{t|t}")
        if "smoothed_beta" in df.columns:
            ax.plot(idx,df["smoothed_beta"],color="crimson",linewidth=1.1,linestyle="-.",label="Smoothed β̂_{t|T}")
        ax.fill_between(idx,df["beta"]-2*np.sqrt(df["P"]),df["beta"]+2*np.sqrt(df["P"]),alpha=0.12,color="steelblue")
        ax.set_title("Dynamic Hedge Ratio β_t",fontweight="bold"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # Spread panel
        ax=axes[1]
        s=df["spread"]
        ax.plot(idx,s,color="darkorange",linewidth=0.9,label="Kalman Spread")
        ax.axhline(s.mean(),color="black",linestyle="--",linewidth=0.9)
        ax.fill_between(idx,s.mean()-2*s.std(),s.mean()+2*s.std(),alpha=0.07,color="darkorange")
        ax.set_title("Dynamic Spread S_t = Y_t − β̂_t·X_t",fontweight="bold"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # Z-score panel
        ax=axes[2]
        z=df["zscore"]
        ax.plot(idx,z,color="mediumpurple",linewidth=0.9,label="Z-score")
        for lv,col,ls in [(self.entry_z,"green","--"),(-self.entry_z,"green","--"),
                           (self.stop_z,"red","-."),(- self.stop_z,"red","-.")]:
            ax.axhline(lv,color=col,linewidth=0.9,linestyle=ls)
        ax.fill_between(idx,self.entry_z,z,where=z>self.entry_z,alpha=0.15,color="tomato")
        ax.fill_between(idx,-self.entry_z,z,where=z<-self.entry_z,alpha=0.15,color="limegreen")
        ax.set_title(f"Z-Score (window={self._zscore.window})",fontweight="bold"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # Signal panel
        ax=axes[3]
        sig=df["signal"]
        colors_map={1:"limegreen",-1:"tomato",0:"lightgray",2:"orange",-2:"orange"}
        for val,col in colors_map.items():
            mask=sig==val
            if mask.any(): ax.fill_between(idx,0,sig.where(mask,0),color=col,alpha=0.7)
        ax.set_title("Trading Signal  (1=Long, −1=Short, 0=Flat, ±2=Stop)",fontweight="bold")
        ax.set_ylim(-1.5,1.5); ax.grid(True,alpha=0.3)
        if isinstance(idx,pd.DatetimeIndex):
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=25)
        # fig.suptitle("DynamicBeta — Full Pipeline Dashboard",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        status="fitted" if self._fitted else "not fitted — call fit()"
        return (f"DynamicBeta(Q={self._Q:.2e}, R={self._R:.2e}, "
                f"τ={self._tau:.1f} obs, {status})")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import yfinance as yf
    raw=yf.download(["DAL","UAL"],period='3y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["DAL"],log_px["UAL"]
    db=DynamicBeta(y=y,x=x,optimize=False,run_smoother=True)
    states=db.fit()
    db.summary(); db.plot_dashboard()
    breaks=db.regime_detect()
    print(f"Regime break dates:\n{breaks[breaks].index.tolist()[:5]}")
