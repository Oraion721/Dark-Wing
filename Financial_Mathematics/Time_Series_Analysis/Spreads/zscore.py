"""
Module  : Financial Mathematics/Time_Series_Analysis/Spreads/zscore.py
Project : Dark Wing
Purpose:
    Normalise a stationary spread S_t into a dimensionless Z-score signal that drives pair-trading entry, exit, and position-sizing decisions. Raw spreads are not directly tradeable because their scale (mean and std) changes across time. The Z-score maps the spread onto a universal scale:
        Z_t = (S_t - μ_t) / σ_t
    where μ_t and σ_t are rolling (or expanding) statistics.

    Two normalisation modes:
    'rolling'  : μ_t = rolling mean over last `window` obs (adaptive). Recommended when half-life < 60 days.
    'expanding': μ_t = expanding mean from inception (full history). Recommended for very slow mean-reverting pairs.

    Signal thresholds (default):
        |Z_t| > entry_z  → OPEN position (sell high Z, buy low Z)
        |Z_t| < exit_z   → CLOSE position
        |Z_t| > stop_z   → STOP LOSS
    Downstream:
        signal/pair_signal.py   → consumes zscore_series() and generate_signals()
        Kalman_Filter/          → uses Z_t as an optional normalised observation            """
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
for _p in [_FM_DIR,_TSA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

class ZScore:       # Compute rolling or expanding Z-score of a stationary spread and generate entry/exit/stop signals.
    """ Args:
            ~ spread:pd.Series —> stationary spread S_t from SpreadBuilder.
            ~ window:int —> lookback window in observations. Rule of thumb: set to int(half_life * 2).
            ~ mode:str —> 'rolling' (default) | 'expanding'.
            ~ min_periods:int —> minimum obs before first valid Z-score. Default = window // 2.
            ~ entry_z:float —> |Z| threshold to open a position.  Default 2.0.
            ~ exit_z:float —> |Z| threshold to close a position. Default 0.5.
            ~ stop_z:float —> |Z| threshold for stop-loss.       Default 3.5.     """
    _VALID_MODES=("rolling","expanding")

    def __init__(self,spread:pd.Series,window:int=60,mode:str="rolling",min_periods:Optional[int]=None,entry_z:float=2.0,exit_z:float=0.5,stop_z:float=3.5):
        if not isinstance(spread,pd.Series):
            raise TypeError("spread must be a pd.Series.")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}.")
        if not (exit_z < entry_z < stop_z):
            raise ValueError(f"Must satisfy: exit_z < entry_z < stop_z. Got {exit_z},{entry_z},{stop_z}.")
        self._spread=spread.dropna()
        if len(self._spread)<max(10,window):
            raise ValueError(f"Spread has {len(self._spread)} obs but window={window}.")
        self.window=int(window)
        self.mode=mode
        self.min_periods=min_periods if min_periods is not None else max(self.window//2,5)
        self.entry_z=entry_z
        self.exit_z=exit_z
        self.stop_z=stop_z
        # computed
        self._zscore:Optional[pd.Series]=None
        self._mu:Optional[pd.Series]=None
        self._sigma:Optional[pd.Series]=None
        self._signals:Optional[pd.Series]=None

    def compute(self)->pd.Series:
        """
        O/P:
            pd.Series — Z_t = (S_t - μ_t) / σ_t, same index as spread.      """
        s=self._spread
        if self.mode=="rolling":
            roller=s.rolling(window=self.window,min_periods=self.min_periods)
            self._mu=roller.mean()
            self._sigma=roller.std(ddof=1)
        else:  # expanding
            expander=s.expanding(min_periods=self.min_periods)
            self._mu=expander.mean()
            self._sigma=expander.std(ddof=1)
        # Guard: zero std → set NaN to avoid ÷0
        sigma_safe=self._sigma.replace(0.0,np.nan)
        self._zscore=((s-self._mu)/sigma_safe).rename("zscore")
        n_nan=self._zscore.isna().sum()
        n_valid=self._zscore.notna().sum()
        print(f"[ZScore] mode={self.mode}  window={self.window}  valid={n_valid}  NaN={n_nan}")
        return self._zscore

    def generate_signals(self)->pd.Series:
        """ Signal encoding:
                 1  → LONG spread  (Z < -entry_z  → spread cheap, buy Y / sell X)
                -1  → SHORT spread (Z >  entry_z  → spread expensive, sell Y / buy X)
                 0  → FLAT         (|Z| < exit_z  → close / no position)
                 2  → STOP LOSS    (|Z| > stop_z  → forced exit)
        Rules:
            State machine prevents re-entering without crossing exit first.
            STOP LOSS overrides all other states.
        O/P:
            pd.Series of integers {-2, -1, 0, 1, 2} (STOP is ±2).       """
        self._require_computed("generate_signals")
        z=self._zscore.values
        n=len(z)
        sig=np.zeros(n,dtype=int)
        position=0   # current state: -1, 0, +1
        for t in range(n):
            zt=z[t]
            if np.isnan(zt):
                sig[t]=0; continue
            # stop-loss check (overrides everything)
            if abs(zt)>self.stop_z:
                sig[t]=2*int(np.sign(zt)); position=0; continue
            # Exit: close position when |Z| drops below exit_z
            if position!=0 and abs(zt)<self.exit_z:
                position=0
            # Entry: only if currently flat
            if position==0:
                if zt<-self.entry_z:
                    position=1
                elif zt>self.entry_z:
                    position=-1
            sig[t]=position
        self._signals=pd.Series(sig,index=self._zscore.index,name="signal")
        long_pct =(self._signals==1).mean()*100
        short_pct=(self._signals==-1).mean()*100
        flat_pct =(self._signals==0).mean()*100
        stop_pct =(self._signals.abs()==2).mean()*100
        print(f"[ZScore.signals] LONG={long_pct:.1f}%  SHORT={short_pct:.1f}%  "
              f"FLAT={flat_pct:.1f}%  STOP={stop_pct:.1f}%")
        return self._signals

    def zscore_series(self)->pd.Series:
        """Z-score time series — can be passed to Kalman Filter as an observation."""
        self._require_computed("zscore_series"); return self._zscore

    def signal_series(self)->pd.Series:
        """Discrete {-2,-1,0,1,2} trading signal series."""
        if self._signals is None:
            raise RuntimeError("Call generate_signals() first.")
        return self._signals

    def rolling_mu(self)->pd.Series:
        self._require_computed("rolling_mu"); return self._mu

    def rolling_sigma(self)->pd.Series:
        self._require_computed("rolling_sigma"); return self._sigma

    def current_zscore(self)->float:
        """Most recent Z-score value — used for live signal generation."""
        self._require_computed("current_zscore"); return float(self._zscore.iloc[-1])

    def signal_stats(self)->dict:
        """Summary statistics of signal distribution."""
        if self._signals is None: raise RuntimeError("Call generate_signals() first.")
        return {"long_pct":(self._signals==1).mean()*100,
                "short_pct":(self._signals==-1).mean()*100,
                "flat_pct":(self._signals==0).mean()*100,
                "stop_pct":(self._signals.abs()==2).mean()*100,
                "n_trades":int((self._signals.diff().abs()>0).sum())}

    def _require_computed(self,method:str):
        if self._zscore is None:
            raise RuntimeError(f"Call compute() before calling {method}().")

    def summary(self):
        self._require_computed("summary")
        sep="="*58
        print(f"\n{sep}")
        print("  ZSCORE — SUMMARY")
        print(sep)
        print(f"  Mode              : {self.mode}")
        print(f"  Window            : {self.window} obs")
        print(f"  Observations      : {len(self._spread)}")
        print(f"  Valid Z-scores    : {self._zscore.notna().sum()}")
        print(f"  Z-score mean      : {self._zscore.mean():.6f}")
        print(f"  Z-score std       : {self._zscore.std():.6f}")
        print(f"  Z-score min/max   : {self._zscore.min():.4f} / {self._zscore.max():.4f}")
        print(f"  Entry threshold   : ±{self.entry_z}")
        print(f"  Exit threshold    : ±{self.exit_z}")
        print(f"  Stop threshold    : ±{self.stop_z}")
        if self._signals is not None:
            st=self.signal_stats()
            print(f"  LONG  exposure    : {st['long_pct']:.1f}%")
            print(f"  SHORT exposure    : {st['short_pct']:.1f}%")
            print(f"  FLAT  exposure    : {st['flat_pct']:.1f}%")
            print(f"  # Signal changes  : {st['n_trades']}")
        print(sep+"\n")

    def plot(self,figsize:tuple=(13,7),show_signals:bool=True):
        """
        Two-panel dashboard:
        (top) Z-score with entry/exit/stop lines and shaded regions.
        (bot) Discrete trading signal series.
        """
        self._require_computed("plot")
        rows=2 if (show_signals and self._signals is not None) else 1
        fig,axes=plt.subplots(rows,1,figsize=figsize,sharex=True)
        if rows==1: axes=[axes]
        # ========== Top: Z-score ==================
        ax=axes[0]
        z=self._zscore
        ax.plot(z.index,z.values,color="steelblue",linewidth=0.9,label="Z-score")
        ax.axhline(0,color="black",linewidth=0.8)
        for lv,col,ls in [(self.entry_z,"green","--"),(-self.entry_z,"green","--"),
                          (self.exit_z,"gray",":"),(- self.exit_z,"gray",":"),
                          (self.stop_z,"red","-."),(- self.stop_z,"red","-.")]:
            ax.axhline(lv,color=col,linewidth=0.9,linestyle=ls)
        ax.fill_between(z.index, self.entry_z,z.values,where=z.values>self.entry_z,
                        alpha=0.15,color="tomato",label=f"Short zone (Z>{self.entry_z})")
        ax.fill_between(z.index,-self.entry_z,z.values,where=z.values<-self.entry_z,
                        alpha=0.15,color="limegreen",label=f"Long zone (Z<-{self.entry_z})")
        ax.set_title(f"Z-Score  [{self.mode}, window={self.window}]",fontweight="bold")
        ax.set_ylabel("Z-score"); ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.3)
        # ======================== Bottom: Signals ==========================
        if rows==2:
            ax=axes[1]
            sig=self._signals
            colors={1:"limegreen",-1:"tomato",0:"lightgray",2:"orange",-2:"orange"}
            for val,col in colors.items():
                mask=sig==val
                if mask.any():
                    ax.fill_between(sig.index,0,sig.where(mask,0),color=col,alpha=0.7)
            ax.set_title("Trading Signal  (1=Long, -1=Short, 0=Flat, ±2=Stop)",fontweight="bold")
            ax.set_ylabel("Signal"); ax.set_ylim(-1.5,1.5); ax.grid(True,alpha=0.3)
        if isinstance(z.index,pd.DatetimeIndex):
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=25)
        fig.suptitle("Z-Score Signal Dashboard",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        if self._zscore is not None:
            return (f"ZScore(mode='{self.mode}', window={self.window}, "
                    f"entry±{self.entry_z}, exit±{self.exit_z}, stop±{self.stop_z}, "
                    f"current_z={self.current_zscore():.4f})")
        return f"ZScore(mode='{self.mode}', window={self.window}, not computed — call compute())"

if __name__=="__main__":
    import yfinance as yf
    from spread_builder import SpreadBuilder
    from half_life import HalfLife
    raw=yf.download(["KOTAKBANK.NS","HDFCBANK.NS"],period='9y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["KOTAKBANK.NS"],log_px["HDFCBANK.NS"]
    sb=SpreadBuilder(y=y,x=x,mode="ols"); sb.build()
    hl=HalfLife(sb.spread_series()); hl.estimate()
    win=max(10,int(hl.half_life_days*2))
    zs=ZScore(sb.spread_series(),window=win,mode="rolling",entry_z=2.0,exit_z=0.5,stop_z=3.5)
    zs.compute(); zs.generate_signals(); zs.summary(); zs.plot()