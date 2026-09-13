"""
Module  : Financial Mathematics / Time Series Analysis / Kalman_Filter / smoother.py
Project : Dark Wing

Purpose:
    RTS (Rauch-Tung-Striebel) Backward Smoother.

    While the Kalman filter is a *forward* pass — estimating β_t using data
    up to time t — the smoother is a *backward* pass that refines ALL past
    estimates using ALL available data.

    Key insight:
        β̂_{t|T}  = E[β_t | Y_1, Y_2, ..., Y_T]   (smoother)
        β̂_{t|t}  = E[β_t | Y_1, Y_2, ..., Y_t]   (filter)

    The smoother gives a *lower* variance estimate because it uses future
    data to refine the past. This is valuable for:
        - Post-hoc analysis and backtesting
        - Parameter estimation (Q, R tuning)
        - Computing the smoothed spread for offline signal research

    RTS Backward Recursion (from Notion notes):
        Smoothing Gain:  Ξ_t  = P_{t|t} · T' / P_{t+1|t}   (scalar: Ξ_t = P_{t|t}/P_{t+1|t})
        Smoothed State:  β̂_{t|T} = β̂_{t|t} + Ξ_t · (β̂_{t+1|T} - β̂_{t+1|t})
        Smoothed Cov.:   P_{t|T} = P_{t|t} - Ξ_t·(P_{t+1|t} - P_{t+1|T})·Ξ_t

    Important:
        - The smoother CANNOT be run in real-time (needs future data).
        - Use filter for live trading; use smoother for backtesting/research.

Pipeline Position:
    STEP 6   → kalman_filter.py  (KalmanFilterPairs)
    STEP 6.1 → THIS FILE (KalmanSmoother)
    STEP 6.2 → dynamic_beta.py  (DynamicBeta)

Dependencies: numpy, pandas, matplotlib
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Tuple,List

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
for _p in [_FM_DIR,_TSA_DIR,_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

from kalman_filter import KalmanFilterPairs,KalmanState

# ─────────────────────────────────────────────────────────────────────────────
class KalmanSmoother:
    """
    RTS Backward Smoother for dynamic β in pairs trading.

    Takes a *fitted* KalmanFilterPairs object (after calling filter()) and runs
    the RTS backward pass to produce smoothed β̂_{t|T} and P_{t|T} for all t.

    Parameters
    ----------
    kf : KalmanFilterPairs — A fitted Kalman filter. Must have called filter().
    """
    def __init__(self,kf:KalmanFilterPairs):
        if not isinstance(kf,KalmanFilterPairs):
            raise TypeError("kf must be a KalmanFilterPairs instance.")
        if not kf._fitted:
            raise RuntimeError("KalmanFilterPairs must be fitted (call filter()) before smoothing.")
        self._kf=kf
        self._states=kf.states           # list of KalmanState
        self.T=len(self._states)
        # outputs
        self._smoothed_beta:Optional[np.ndarray]=None
        self._smoothed_P:Optional[np.ndarray]=None
        self._smoothing_gain:Optional[np.ndarray]=None
        self._done:bool=False

    # ── Core: RTS backward pass ────────────────────────────────────────────
    def smooth(self)->Tuple[np.ndarray,np.ndarray]:
        """
        Run the RTS backward smoother over all T stored filter states.

        Algorithm (backward from t = T-1 to t = 0):
            For t = T-1, T-2, ..., 0:
                Ξ_t  = P_{t|t} / P_{t+1|t}         (smoothing gain)
                β̂_{t|T} = β̂_{t|t} + Ξ_t · (β̂_{t+1|T} − β̂_{t+1|t})
                P_{t|T} = P_{t|t} − Ξ_t · (P_{t+1|t} − P_{t+1|T}) · Ξ_t

        Returns
        -------
        (smoothed_beta, smoothed_P) — numpy arrays of length T.
        """
        T=self.T
        states=self._states
        smoothed_beta=np.zeros(T)
        smoothed_P=np.zeros(T)
        smoothing_gain=np.zeros(T-1)
        # Initialise: at t=T, smoothed = filtered
        smoothed_beta[T-1]=states[T-1].beta
        smoothed_P[T-1]=states[T-1].P
        # Backward pass: t = T-2 down to 0
        for t in range(T-2,-1,-1):
            P_t_t=states[t].P          # P_{t|t}
            P_t1_t=states[t+1].P_pred  # P_{t+1|t} (predicted cov at t+1)
            if P_t1_t<=0:
                warnings.warn(f"P_{{t+1|t}}={P_t1_t:.2e} at t={t}. Setting Ξ_t=0.",RuntimeWarning)
                xi=0.0
            else:
                # Ξ_t = P_{t|t} · T' / P_{t+1|t}  →  scalar (T=1): Ξ = P_{t|t}/P_{t+1|t}
                xi=P_t_t/P_t1_t
            smoothing_gain[t]=xi
            beta_t_t=states[t].beta            # β̂_{t|t}
            beta_t1_T=smoothed_beta[t+1]       # β̂_{t+1|T} (already computed)
            beta_t1_t=states[t+1].beta_pred    # β̂_{t+1|t}
            # Smoothed state
            smoothed_beta[t]=beta_t_t+xi*(beta_t1_T-beta_t1_t)
            # Smoothed covariance
            P_t1_T=smoothed_P[t+1]             # P_{t+1|T}
            smoothed_P[t]=P_t_t-xi*(P_t1_t-P_t1_T)*xi
            if smoothed_P[t]<0:
                smoothed_P[t]=P_t_t  # safety floor
        self._smoothed_beta=smoothed_beta
        self._smoothed_P=smoothed_P
        self._smoothing_gain=smoothing_gain
        self._done=True
        print(f"[KalmanSmoother] Backward pass complete. T={T}")
        return smoothed_beta,smoothed_P

    # ── Derived outputs ───────────────────────────────────────────────────────
    def smoothed_spread(self)->pd.Series:
        """
        Compute the smoothed spread using smoothed β̂_{t|T}.
            S_t^{smooth} = Y_t - β̂_{t|T} · X_t

        Returns pd.Series with same index as the filter.
        """
        self._require_done("smoothed_spread")
        x_arr=np.array([s.x for s in self._states])
        y_arr=np.array([s.y for s in self._states])
        spread=y_arr-self._smoothed_beta*x_arr
        idx=self._kf._y_index if hasattr(self._kf,"_y_index") else np.arange(self.T)
        return pd.Series(spread,index=idx,name="smoothed_spread")

    def get_smoothed_states(self)->pd.DataFrame:
        """
        Return smoothed β and P as a tidy DataFrame.

        Columns: [smoothed_beta, smoothed_P, filtered_beta, filtered_P,
                  smoothed_spread, gain]
        """
        self._require_done("get_smoothed_states")
        filt_beta=np.array([s.beta for s in self._states])
        filt_P=np.array([s.P for s in self._states])
        x_arr=np.array([s.x for s in self._states])
        y_arr=np.array([s.y for s in self._states])
        smth_spread=y_arr-self._smoothed_beta*x_arr
        filt_spread=y_arr-filt_beta*x_arr
        gain_full=np.append(self._smoothing_gain,np.nan)  # T-1 gains + NaN at end
        idx=self._kf._y_index if hasattr(self._kf,"_y_index") else np.arange(self.T)
        df=pd.DataFrame({"smoothed_beta":self._smoothed_beta,
                          "smoothed_P":self._smoothed_P,
                          "filtered_beta":filt_beta,
                          "filtered_P":filt_P,
                          "smoothed_spread":smth_spread,
                          "filtered_spread":filt_spread,
                          "smoothing_gain":gain_full},index=idx)
        return df

    def variance_reduction(self)->pd.Series:
        """
        Variance reduction from smoothing: R_t = 1 - P_{t|T}/P_{t|t}.
        Values in [0,1]. High R_t means smoothing gave big uncertainty reduction.
        """
        self._require_done("variance_reduction")
        filt_P=np.array([s.P for s in self._states])
        reduction=1.0-self._smoothed_P/np.maximum(filt_P,1e-20)
        idx=self._kf._y_index if hasattr(self._kf,"_y_index") else np.arange(self.T)
        return pd.Series(np.clip(reduction,0,1),index=idx,name="variance_reduction")

    # ── summary & plot ────────────────────────────────────────────────────────
    def _require_done(self,method:str):
        if not self._done:
            raise RuntimeError(f"Call smooth() before calling {method}().")

    def summary(self):
        self._require_done("summary")
        sep="="*60
        df=self.get_smoothed_states()
        vr=self.variance_reduction()
        print(f"\n{sep}")
        print("  RTS SMOOTHER — SUMMARY")
        print(sep)
        print(f"  Observations          : {self.T}")
        print(f"  Smoothed β range      : [{df['smoothed_beta'].min():.4f}, {df['smoothed_beta'].max():.4f}]")
        print(f"  Filtered β range      : [{df['filtered_beta'].min():.4f}, {df['filtered_beta'].max():.4f}]")
        print(f"  β std (smoothed)      : {df['smoothed_beta'].std():.6f}")
        print(f"  β std (filtered)      : {df['filtered_beta'].std():.6f}")
        print(f"  Avg variance reduction: {vr.mean()*100:.2f}%")
        print(f"  Smoothed spread std   : {df['smoothed_spread'].std():.6f}")
        print(f"  Filtered spread std   : {df['filtered_spread'].std():.6f}")
        print(sep+"\n")

    def plot_comparison(self,figsize:tuple=(14,10)):
        """
        Four-panel comparison: β, covariance P, spreads, variance reduction.
        """
        self._require_done("plot_comparison")
        df=self.get_smoothed_states()
        vr=self.variance_reduction()
        idx=df.index
        fig,axes=plt.subplots(4,1,figsize=figsize,sharex=True)
        # β comparison
        ax=axes[0]
        ax.plot(idx,df["filtered_beta"],color="steelblue",linewidth=0.9,alpha=0.7,label="Filtered β̂_{t|t}")
        ax.plot(idx,df["smoothed_beta"],color="crimson",linewidth=1.2,label="Smoothed β̂_{t|T}")
        ax.fill_between(idx,df["smoothed_beta"]-2*np.sqrt(df["smoothed_P"]),
                        df["smoothed_beta"]+2*np.sqrt(df["smoothed_P"]),alpha=0.12,color="crimson",label="±2σ (smoothed)")
        ax.set_title("Hedge Ratio β: Filtered vs. Smoothed",fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # P comparison
        ax=axes[1]
        ax.plot(idx,df["filtered_P"],color="steelblue",linewidth=0.9,label="P_{t|t} (filtered)")
        ax.plot(idx,df["smoothed_P"],color="crimson",linewidth=1.1,label="P_{t|T} (smoothed)")
        ax.set_title("Covariance P: Filtered vs. Smoothed",fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # Spread comparison
        ax=axes[2]
        ax.plot(idx,df["filtered_spread"],color="steelblue",linewidth=0.9,alpha=0.7,label="Filtered Spread")
        ax.plot(idx,df["smoothed_spread"],color="crimson",linewidth=1.1,label="Smoothed Spread")
        ax.axhline(0,color="black",linewidth=0.8,linestyle="--")
        ax.set_title("Spread: Filtered vs. Smoothed",fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        # Variance reduction
        ax=axes[3]
        ax.fill_between(idx,0,vr*100,alpha=0.6,color="mediumseagreen",label="Var. Reduction %")
        ax.set_title("Variance Reduction from Smoothing (%)",fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_ylabel("%")
        if isinstance(idx,pd.DatetimeIndex):
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=25)
        fig.suptitle("RTS Smoother vs. Kalman Filter — Pairs Trading",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        status="smoothed" if self._done else "not smoothed — call smooth()"
        return f"KalmanSmoother(T={self.T}, {status})"

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import yfinance as yf
    raw=yf.download(["RELIANCE.NS","TCS.NS"],start="2021-01-01",end="2024-01-01",auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["RELIANCE.NS"],log_px["TCS.NS"]
    kf=KalmanFilterPairs(Q=1e-5,R=1e-3); kf.filter(y,x)
    ks=KalmanSmoother(kf); ks.smooth()
    ks.summary(); ks.plot_comparison()
