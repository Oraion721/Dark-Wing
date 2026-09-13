""" Purpose:
        Cointegration relationships are NOT static. Market regimes shift. 
        This module slides a fixed window across time and re-runs either Engle-Granger or Johansen at each step to:
            1.  Detect WHEN cointegration breaks down (regime detection).
            2.  Track how the hedge ratio β evolves over time.
            3.  Generate a "stability score" to decide whether to trade the pair.    """

import sys; import os
import warnings
import yfinance as yf
import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Literal
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from engle_granger import EngleGrangerTest
from johansen import JohansenTest
_TSA_DIR = os.path.dirname(_THIS_DIR)
if _TSA_DIR not in sys.path:
    sys.path.insert(0, _TSA_DIR)

class RollingCointegration:
    """ Args: 
            data:pd.DataFrame: Price series (log prices recommended). Minimum 2 columns.
            window:int: Rolling window size in observations. Default 252 (≈ 1 trading year).
            step:int: How many observations to advance the window each iteration.
                step=1 → fully rolling (expensive); step=5 → weekly step.
            method:str: 'engle_granger' (pairs only, fast) or 'johansen' (multi-asset).
            sig_level:float: p-value threshold to classify a window as cointegrated. Default 0.05.
            min_obs:int: Minimum observations required per window. Default same as window.
        O/P: A time-indexed record of:
            - p-value of cointegration at each window
            - Rolling hedge ratio β
            - Whether the pair is cointegrated at each point
            - Cointegration stability score (fraction of windows that are cointegrated)  """

    def __init__(self,data:pd.DataFrame,window:int=252,step:int= 1,method:Literal["engle_granger","johansen"]="engle_granger",sig_level:float=0.05, min_obs:Optional[int]=None):
        if not isinstance(data,pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame.")
        if data.shape[1]<2:
            raise ValueError("data must have at least 2 columns.")
        if method=="engle_granger" and data.shape[1]!=2:
            raise ValueError(f"engle_granger method requires exactly 2 columns. Got {data.shape[1]}.\nUse method='johansen' for 3+ series.")
        if method not in ("engle_granger","johansen"):
            raise ValueError(f"method must be 'engle_granger' or 'johansen', got '{method}'.")
        if window<50:
            raise ValueError("window must be >= 50 observations.")
        if step<1:
            raise ValueError("step must be >= 1.")
        if not (0<sig_level<1):
            raise ValueError("sig_level must be between 0 and 1.")

        self.data=data.copy().dropna()
        self.T=len(self.data); self.m=self.data.shape[1]
        self.cols=list(self.data.columns)
        self.window=window; self.step=step
        self.method=method; self.sig_level=sig_level
        self.min_obs   = min_obs or window
        if self.T<self.window+10:
            raise ValueError(f"Not enough data: T={self.T}, window={self.window}.\nNeed at least {self.window + 10} observations.")
        self.results_df:Optional[pd.DataFrame]=None  # full rolling output
        self.stability_score:Optional[float]=None  # [0, 1]

    def _require_fitted(self,method:str):
        if self.results_df is None:
            raise RuntimeError(f"Call run() before calling {method}().")

    def _run_eg_window(self,window_data:pd.DataFrame):
        try:
            eg=EngleGrangerTest(y=window_data.iloc[:,0],x=window_data.iloc[:,1],sig_level=self.sig_level,verify_i1=False)
            result=eg.run()
            return { "p_value":result["adf_result"]["p_value"],
                "beta":result["beta"],"alpha":result["alpha"],
                "is_cointegrated":result["is_cointegrated"],
                "spread_std":result["spread_std"]  }
        except Exception as e:
            warnings.warn(f"EG failed on window: {e}",RuntimeWarning)
            return {"p_value":np.nan,"beta":np.nan,
                "alpha":np.nan,"is_cointegrated":False,"spread_std":np.nan }

    def _run_johansen_window(self,window_data:pd.DataFrame):
        try:
            jt=JohansenTest(data=window_data,det_order="restricted_constant",k_ar_diff=1,sig_level=1,verify_i1=False)
            result=jt.run()
            rank=result["rank"]
            beta_0= (float(result["beta"][1, 0])   # 2nd element of 1st coint vector
                if rank>0 and result["beta"] is not None else np.nan)
            # Use trace stat for r=0 as the "p-like" measure
            trace_r0=float(jt.trace_stats[0])
            cv_r0=float(jt.trace_cvs[0,1])   # 5% CV
            pseudo_p=0.01 if trace_r0>cv_r0 else 0.50
            return {"p_value":pseudo_p,"beta":beta_0,"alpha":np.nan,"is_cointegrated":rank>0,"spread_std":np.nan,"rank":rank}
        except Exception as e:
            warnings.warn(f"Johansen failed on window: {e}", RuntimeWarning)
            return {"p_value":np.nan,"beta":np.nan,"alpha":np.nan,"is_cointegrated":False,"spread_std":np.nan,"rank":0}

    def run(self):
        records=[]; indices=[]
        # Iterate windows
        start_positions=range(0,self.T-self.window+1,self.step)
        n_windows=len(list(start_positions))
        print(f"[RollingCointegration] Running {n_windows} windows\n(method='{self.method}', window={self.window}, step={self.step})...")
        for i, start in enumerate(range(0,self.T-self.window+1,self.step)):
            end=start+self.window
            window_data=self.data.iloc[start:end]
            if len(window_data)<self.min_obs:
                continue
            end_date=self.data.index[end-1]
            indices.append(end_date)
            if self.method=="engle_granger":
                record=self._run_eg_window(window_data)
            else:
                record=self._run_johansen_window(window_data)
            records.append(record)

            # Progress log every 50 windows
            if (i+1)%50==0 or i==0:
                pct = (i+1)/n_windows*100
                print(f"  ...{pct:.0f}% complete ({i+1}/{n_windows})",end="\r")
        print(f"\n  Done. {len(records)} windows processed.")
        self.results_df=pd.DataFrame(records,index=indices)
        self.results_df.index.name="window_end"
        # Stability score: fraction of valid windows where pair was cointegrated
        valid_mask=self.results_df["p_value"].notna()
        if valid_mask.sum()>0:
            self.stability_score=float(self.results_df.loc[valid_mask,"is_cointegrated"].mean())
        else:
            self.stability_score=0.0
        print(f"  Cointegration Stability Score: {self.stability_score:.2%}")
        return self.results_df
    
    def current_status(self):
        """Return the most recent window's cointegration result."""
        self._require_fitted("current_status")
        last = self.results_df.iloc[-1]
        return {"date":self.results_df.index[-1], "is_cointegrated":bool(last.get("is_cointegrated", False)),
            "p_value":float(last.get("p_value",np.nan)),
            "beta":float(last.get("beta",np.nan)), "stability_score": self.stability_score }

    def cointegration_windows(self):        # Return only the windows where the pair WAS cointegrated. 
        self._require_fitted("cointegration_windows")
        return self.results_df[self.results_df["is_cointegrated"]==True]

    def rolling_beta(self):
        self._require_fitted("rolling_beta")
        return self.results_df["beta"].dropna()

    def summary(self):
        self._require_fitted("summary")
        sep = "=" * 62
        status = self.current_status()
        print(f"\n{sep}")
        print("  ROLLING COINTEGRATION — SUMMARY")
        print(sep)
        print(f"  Method         : {self.method}")
        print(f"  Series         : {self.cols}")
        print(f"  Total obs (T)  : {self.T}")
        print(f"  Window size    : {self.window}")
        print(f"  Step size      : {self.step}")
        print(f"  Windows run    : {len(self.results_df)}")
        print(f"  Significance   : {self.sig_level*100:.0f}%")
        print(sep)
        print(f"  CURRENT STATUS ({status['date'].date()}):")
        flag = "✅ COINTEGRATED" if status["is_cointegrated"] else "❌ NOT COINTEGRATED"
        print(f"    → {flag}")
        print(f"    p-value    : {status['p_value']:.4f}")
        print(f"    Hedge β    : {status['beta']:.4f}")
        print(sep)
        print(f"  STABILITY SCORE: {self.stability_score:.2%}")
        print(f"  ({self.stability_score:.2%} of windows were cointegrated)")
        if self.stability_score>=0.75:
            print("  ✅ HIGHLY STABLE — good candidate for pairs trading")
        elif self.stability_score>=0.50:
            print("  ⚠️  MODERATELY STABLE — trade with caution, monitor closely")
        else:
            print("  ❌ UNSTABLE — do not trade this pair algorithmically")
        beta_series=self.rolling_beta()
        print(f"\n  ROLLING BETA β STATISTICS:")
        print(f"    Mean   : {beta_series.mean():.4f}")
        print(f"    Std Dev: {beta_series.std():.4f}")
        print(f"    Min    : {beta_series.min():.4f}")
        print(f"    Max    : {beta_series.max():.4f}")
        print(f"{sep}\n")

    def plot(self, figsize: tuple = (14, 10)):
        self._require_fitted("plot")
        df=self.results_df
        fig,axes=plt.subplots(3,1,figsize=figsize,sharex=True)
        # ============== Panel 1: Rolling p-value ===================
        axes[0].plot(df.index, df["p_value"],color="steelblue",linewidth=1.2,label="p-value")
        axes[0].axhline(self.sig_level,color="red",linestyle="--",linewidth=1.5,label=f"α = {self.sig_level}")
        axes[0].fill_between(df.index,0,self.sig_level,color="green",alpha=0.07,label="Cointegrated zone" )
        axes[0].set_ylabel("p-value")
        axes[0].set_title("Rolling Cointegration p-value",fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        # ================ Panel 2: Rolling β =========================
        axes[1].plot(df.index, df["beta"],color="darkorange",linewidth=1.2,label="Hedge ratio β")
        axes[1].axhline(df["beta"].mean(),color="black",linestyle="--",linewidth=1.0,label=f"Mean β = {df['beta'].mean():.4f}")
        axes[1].set_ylabel("Hedge ratio β")
        axes[1].set_title("Rolling Hedge Ratio (β)",fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        # ================== Panel 3: Cointegration regime (binary) ====================
        coint_flag = df["is_cointegrated"].astype(float)
        axes[2].fill_between(df.index,0,coint_flag,step="post",color="seagreen",alpha=0.6,label="Cointegrated")
        axes[2].fill_between(df.index,coint_flag,1,step="post",color="salmon",alpha=0.4,label="Not cointegrated")
        axes[2].set_ylabel("Regime")
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["No","Yes"])
        axes[2].set_title(f"Cointegration Regime  |  Stability: {self.stability_score:.1%}",fontweight="bold")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        pair_label = f"{self.cols[0]} / {self.cols[1]}"
        fig.suptitle(f"Rolling Cointegration ({self.method})|{pair_label} Window = {self.window} obs | Step = {self.step}",fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        status = (f"stability={self.stability_score:.2%}" if self.stability_score is not None else "not fitted — call run()")
        return (f"RollingCointegration(m={self.m}, T={self.T}, "
            f"method='{self.method}', window={self.window}, "
            f"step={self.step}, {status})")

if __name__ == "__main__":
    tickers=["DAL","UAL"]
    raw=yf.download(tickers,period='3y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_prices = np.log(raw)
    print(f"Data: {log_prices.shape}")
    rc=RollingCointegration(data=log_prices,window=252,step=5,method="engle_granger",sig_level = 0.05)
    results_df=rc.run()
    rc.summary()
    rc.plot()
    current = rc.current_status()
    print(f"\nCurrent status: {current}")