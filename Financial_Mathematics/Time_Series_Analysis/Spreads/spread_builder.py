import sys,os,warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Union
_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
for _p in [_FM_DIR,_TSA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

class SpreadBuilder:        # Build a stationary spread series from cointegrated price data.
    """ Supports three modes (set via `mode` parameter):
    Mode 'ols': Two-asset pair. Estimate beta via OLS then
                S_t = Y_t - alpha - beta * X_t.     Use when coming directly from Engle-Granger.
    Mode 'fixed_beta': Two-asset pair with a user-supplied beta (e.g. from JohansenTest.beta or KalmanFilter.beta).
                         S_t = Y_t - beta * X_t.
    Mode 'johansen_ect': Multi-asset (m >= 2). Supply the full price DataFrame and the Johansen beta matrix (m x r*).
                         S_t = data @ beta  →  returns r* ECT columns.

    Args:
        ~ y:pd.Series: Dependent asset log-price series (used in 'ols' and 'fixed_beta').
        ~ x:pd.Series: Independent asset log-price series (used in 'ols' and 'fixed_beta').
        ~ data:pd.DataFrame: Full price matrix (used in 'johansen_ect' mode).
        ~ beta:float or np.ndarray or None: Hedge ratio. float for 'fixed_beta'; (m x r*) array for 'johansen_ect'.
        ~ mode:str: 'ols' | 'fixed_beta' | 'johansen_ect'.
        ~ include_intercept:bool: If True (default), include intercept in OLS spread (mode='ols').        """
    _VALID_MODES=("ols","fixed_beta","johansen_ect")

    def __init__(self,y:Optional[pd.Series]=None,x:Optional[pd.Series]=None,data:Optional[pd.DataFrame]=None,beta:Optional[Union[float,np.ndarray]]=None,mode:str="ols",include_intercept:bool=True):
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got '{mode}'.")
        self.mode=mode
        self.include_intercept=include_intercept
        self._beta_est=None          # scalar or vector, set in build()
        self._alpha_est=None         # intercept if OLS mode
        self._spread:Optional[pd.Series]=None
        self._ect_df:Optional[pd.DataFrame]=None   # for johansen_ect

        if mode in ("ols","fixed_beta"):
            if y is None or x is None:
                raise ValueError("mode='ols'/'fixed_beta' requires both y and x series.")
            if not isinstance(y,pd.Series) or not isinstance(x,pd.Series):
                raise TypeError("y and x must be pd.Series.")
            # Align on common index, drop NaN
            aligned=pd.concat([y.rename("y"),x.rename("x")],axis=1).dropna()
            if len(aligned)<10:
                raise ValueError(f"Insufficient aligned observations: {len(aligned)}. Min 10 required.")
            self._y=aligned["y"]
            self._x=aligned["x"]
            self._data=None
            if mode=="fixed_beta":
                if beta is None:
                    raise ValueError("mode='fixed_beta' requires beta argument.")
                self._beta_input=float(beta)
            else:
                self._beta_input=None

        elif mode=="johansen_ect":
            if data is None or beta is None:
                raise ValueError("mode='johansen_ect' requires data (DataFrame) and beta (ndarray).")
            if not isinstance(data,pd.DataFrame):
                raise TypeError("data must be a pd.DataFrame.")
            if not isinstance(beta,np.ndarray):
                raise TypeError("beta must be a np.ndarray of shape (m, r*).")
            self._data=data.dropna()
            self._beta_input=beta
            self._y=None; self._x=None

        self.T=len(self._data) if self._data is not None else len(self._y)

    def _require_built(self,method:str):
        if self._spread is None and self._ect_df is None:
            raise RuntimeError(f"Call build() before calling {method}().")

    def _ols_beta(self):        # Estimate beta (and alpha if include_intercept) via OLS.
        y=self._y.values; x=self._x.values
        if self.include_intercept:
            X=np.column_stack([np.ones(len(x)),x])
            coef=np.linalg.lstsq(X,y,rcond=None)[0]
            alpha,beta=float(coef[0]),float(coef[1])
        else:
            X=x.reshape(-1,1)
            coef=np.linalg.lstsq(X,y,rcond=None)[0]
            alpha,beta=0.0,float(coef[0])
        return alpha,beta

    def build(self):            # O/P: Union[pd.Series,pd.DataFrame]
        """O/P:
            ~ pd.Series  (modes 'ols', 'fixed_beta') — one spread column.
            ~ pd.DataFrame (mode 'johansen_ect')      — r* ECT columns.         """
        if self.mode=="ols":
            self._alpha_est,self._beta_est=self._ols_beta()
            spread_vals=(self._y.values)-self._alpha_est-(self._beta_est*self._x.values)
            self._spread=pd.Series(spread_vals,index=self._y.index,name="spread")
            print(f"[SpreadBuilder] OLS  |  a={self._alpha_est:.6f}  b={self._beta_est:.6f}")

        elif self.mode=="fixed_beta":
            self._beta_est=self._beta_input
            self._alpha_est=0.0
            spread_vals=(self._y.values)-(self._beta_est*self._x.values)
            self._spread=pd.Series(spread_vals,index=self._y.index,name="spread")
            print(f"[SpreadBuilder] fixed_beta  |  β={self._beta_est:.6f}")

        elif self.mode=="johansen_ect":
            ect_vals=self._data.values @ self._beta_input   # (T × r*)
            r=self._beta_input.shape[1]
            self._ect_df=pd.DataFrame(ect_vals,index=self._data.index,columns=[f"ECT_{j+1}" for j in range(r)])
            # Set primary spread as first ECT for half_life / zscore compatibility
            self._spread=self._ect_df["ECT_1"].rename("spread")
            print(f"[SpreadBuilder] johansen_ect  |  r*={r} ECT columns built.")

        return self._spread if self.mode!="johansen_ect" else self._ect_df

    def spread_series(self):        # Getters (Kalman_Filter interface)
        """Primary stationary spread S_t — direct input to Kalman Filter."""
        self._require_built("spread_series")
        return self._spread

    def y_series(self)->pd.Series:
        """Dependent (Y) leg prices — maps to Kalman's y argument."""
        if self._y is None:
            raise ValueError("y series not available in 'johansen_ect' mode.")
        return self._y

    def x_series(self):
        """Independent (X) leg prices — maps to Kalman's x argument."""
        if self._x is None:
            raise ValueError("x series not available in 'johansen_ect' mode.")
        return self._x

    def ect_dataframe(self):
        """All r* ECT columns (johansen_ect mode only)."""
        self._require_built("ect_dataframe")
        if self._ect_df is None:
            raise ValueError("ect_dataframe() is only available in 'johansen_ect' mode.")
        return self._ect_df

    def get_beta(self):
        """Estimated or supplied hedge ratio beta."""
        self._require_built("get_beta")
        return self._beta_est

    def get_alpha(self):
        """Estimated intercept (zero if fixed_beta or johansen_ect)."""
        self._require_built("get_alpha")
        return self._alpha_est if self._alpha_est is not None else 0.0

    def summary(self):
        self._require_built("summary")
        sep="="*58
        print(f"\n{sep}")
        print("  SPREAD BUILDER — SUMMARY")
        print(sep)
        print(f"  Mode              : {self.mode}")
        print(f"  Observations (T)  : {self.T}")
        if self.mode in ("ols","fixed_beta"):
            print(f"  Y series          : {self._y.name}")
            print(f"  X series          : {self._x.name}")
            print(f"  Beta (hedge ratio): {self._beta_est:.6f}")
            if self.mode=="ols":
                print(f"  Alpha (intercept) : {self._alpha_est:.6f}")
            s=self._spread
            print(f"  Spread mean       : {s.mean():.6f}")
            print(f"  Spread std        : {s.std():.6f}")
            print(f"  Spread min/max    : {s.min():.4f} / {s.max():.4f}")
        else:
            print(f"  ECT columns       : {list(self._ect_df.columns)}")
        print(sep+"\n")

    def plot(self,figsize:tuple=(13,4)):
        self._require_built("plot")
        if self.mode=="johansen_ect" and self._ect_df is not None:
            r=self._ect_df.shape[1]
            fig,axes=plt.subplots(1,r,figsize=(figsize[0],figsize[1]))
            if r==1: axes=[axes]
            for j,col in enumerate(self._ect_df.columns):
                s=self._ect_df[col]
                axes[j].plot(s.index,s.values,linewidth=1.0,color="steelblue")
                axes[j].axhline(s.mean(),color="black",linestyle="--",linewidth=0.9)
                axes[j].set_title(col,fontweight="bold"); axes[j].grid(True,alpha=0.3)
        else:
            fig,ax=plt.subplots(figsize=figsize)
            s=self._spread
            ax.plot(s.index,s.values,color="steelblue",linewidth=1.0,label="Spread")
            ax.axhline(s.mean(),color="black",linestyle="--",linewidth=0.9,label=f"μ={s.mean():.4f}")
            ax.axhline(s.mean()+2*s.std(),color="red",linestyle=":",linewidth=0.9,label="+2σ")
            ax.axhline(s.mean()-2*s.std(),color="red",linestyle=":",linewidth=0.9,label="-2σ")
            ax.fill_between(s.index,s.mean()-2*s.std(),s.mean()+2*s.std(),alpha=0.07,color="red")
            ax.set_title(f"Spread: S_t = Y - {self._beta_est:.4f}·X",fontweight="bold")
            ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
            if isinstance(s.index,pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                plt.xticks(rotation=30)
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        built="built" if self._spread is not None else "not built — call build()"
        return f"SpreadBuilder(mode='{self.mode}', T={self.T}, {built})"

if __name__=="__main__":
    companies=['RELIANCE.NS','TCS.NS']
    raw=yf.download(tickers=companies,period='5y',auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px[companies[0]],log_px[companies[1]]
    sb=SpreadBuilder(y=y,x=x,mode="ols")
    sb.build(); sb.summary(); sb.plot()
