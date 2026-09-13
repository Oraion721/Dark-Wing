""" Mathematical Reference: See: Coint Mathematics/Johansen_Test.tex
Code reference: https://www.interactivebrokers.com/campus/ibkr-quant-news/johansen-cointegration-test-learn-how-to-implement-it-in-python/  """

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Optional
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))         # .../Cointegration Tests/
_TSA_DIR=os.path.dirname(_THIS_DIR)                          # .../Time Series Analysis/
_VAR_DIR=os.path.join(os.path.dirname(_TSA_DIR),"Vector_Auto_Regressive")        # .../VAR/
_PROJECT_DIR=os.path.dirname(_TSA_DIR)
for _path in [_TSA_DIR,_PROJECT_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from Stationarity_Tests.TS_stationarity_test import StationarityTest
from Vector_Auto_Regressive.VAR_Lag_Select import VAROptimalLagSelect

# ==================== Constants ===========================
# det_order maps to the deterministic term in the VECM
# -1 → no intercept, no trend
#  0 → restricted constant  (most common for log prices)
#  1 → restricted trend
_DET_ORDER_MAP = {"none":-1,"restricted_constant":0,"restricted_trend":1}
_SIG_LABELS = {0:"10%",1:"5%",2:"1%"}      # Johansen critical value significance levels

class JohansenTest:
    def __init__(self,data:pd.DataFrame,det_order:str="restricted_constant",k_ar_diff:Optional[int]=None,sig_level:int=1,verify_i1:bool=True):
        """ Args: 
            data:pd.DataFrame: DataFrame of I(1) level series (e.g. log prices). Must have >= 2 cols.
            det_order:str: Deterministic term specification. One of:
                - 'none'                → no intercept in CE or VAR  (det_order = -1)
                - 'restricted_constant' → intercept only in CE       (det_order =  0)  ← default
                - 'restricted_trend'    → linear trend in CE         (det_order =  1)
            k_ar_diff:int or None: Number of lagged differences in VECM (= VAR lag p - 1). If None, auto-selected via VAROptimalLagSelect (AIC).
            sig_level:int: Index into Johansen critical value table: 0=10%, 1=5%, 2=1%. Default 1 (5%).
            verify_i1:bool: If True (default), run ADF+KPSS on every column before the test and warn if any series is NOT I(1). """
        if not isinstance(data,pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame.")
        if data.shape[1]<2:
            raise ValueError(f"Johansen test requires >= 2 series, got {data.shape[1]}.")
        if det_order not in _DET_ORDER_MAP:
            raise ValueError(f"det_order must be one of {list(_DET_ORDER_MAP)}, got '{det_order}'.")
        if sig_level not in (0,1,2):
            raise ValueError("sig_level must be 0 (10%), 1 (5%), or 2 (1%).")
        
        self.data=data.copy().dropna()
        self.det_order=det_order
        self._det_int=_DET_ORDER_MAP[det_order]
        self.sig_level=sig_level
        self.verify_i1=verify_i1
        self.T=len(self.data)
        self.m=self.data.shape[1]
        self.cols=list(self.data.columns)

        # ================ Resolve lag (k_ar_diff = p - 1) =====================
        if k_ar_diff is not None:
            if not isinstance(k_ar_diff,int) or k_ar_diff<1:
                raise ValueError("k_ar_diff must be a positive integer.")
            self.k_ar_diff=k_ar_diff
        else:
            self.k_ar_diff=self._auto_lag()

        # =============== Result placeholders ====================
        self.rank:Optional[int]=None   # r* cointegrating rank
        self.trace_stats:Optional[np.ndarray]=None
        self.max_eig_stats:Optional[np.ndarray]=None
        self.trace_cvs:Optional[np.ndarray]=None
        self.max_eig_cvs:Optional[np.ndarray]=None 
        self.eigenvalues:Optional[np.ndarray]=None
        self.beta:Optional[np.ndarray]=None   # (m x r*) coint vectors
        self.alpha:Optional[np.ndarray]=None   # (m x r*) adjustment speeds
        self._raw_result=None

    def _auto_lag(self):    # Use VAROptimalLagSelect (AIC) to find optimal VAR lag p, then return p-1
        selector=VAROptimalLagSelect(data=self.data,criteria="aic",transform="none",trend="c")
        selector.select_lag()
        p=selector.get_optimal_lag()
        k=max(1,p-1)   # k_ar_diff = p - 1; minimum 1
        print(f"[JohansenTest] Auto lag:VAR p={p} → k_ar_diff={k}")
        return k

    def _verify_integration_order(self):    # Run ADF+KPSS on each column to confirm all series are I(1). Warns but does NOT raise — the user may have already verified.
        for col in self.cols:
            series=self.data[[col]]
            tester=StationarityTest(time_series=series,lag_max=None,regression_type="c",required_lag="AIC")
            result=tester.check_stationarity()
            if result["is_stationary"]:
                warnings.warn(f"Series '{col}' appears STATIONARY (I(0))—""Johansen test assumes I(1) inputs.""Results may be unreliable.",UserWarning)
            else:
                print(f"'{col}' confirmed I(1).")

    def _require_fitted(self, method: str):
        if self._raw_result is None:
            raise RuntimeError(f"Call run() before calling {method}().")
    
    # ====================== Core Method =============================
    def run(self):
        """ Algorithm:
                1.  (Optional) verify all series are I(1).
                2.  Call statsmodels coint_johansen().
                3.  Determine rank r* using Trace statistic (sequential test):
                    - Start at H0: r = 0
                    - If trace_stat > critical_value → reject, test r = 1
                    - Continue until we fail to reject
                    - r* = first r where we fail to reject
                4.  Extract β (cointegrating vectors) and α (adjustment speeds).    """

        if self.verify_i1:
            self._verify_integration_order()
        try:
            result=coint_johansen(endog=self.data.values,det_order=self._det_int,k_ar_diff=self.k_ar_diff)
        except Exception as e:
            raise RuntimeError(f"coint_johansen() failed: {e}") from e

        self._raw_result    = result
        self.eigenvalues    = result.eig
        self.trace_stats    = result.lr1        # shape (m,)
        self.max_eig_stats  = result.lr2        # shape (m,)
        self.trace_cvs      = result.cvt        # shape (m, 3)  [10%, 5%, 1%]
        self.max_eig_cvs    = result.cvm        # shape (m, 3)

        # ================= Determine rank r* via sequential trace test ===================
        self.rank=0
        for r in range(self.m):
            if (self.trace_stats[r])>(self.trace_cvs[r, self.sig_level]):
                self.rank=r+1   # reject H0: rank <= r → at least r+1
            else:
                break
        # ============== Extract β and α ==============================  
        # result.evec columns are the cointegrating vectors β (m × m matrix)
        # We take only the first r* columns (strongest relationships)
        if self.rank>0:
            self.beta=result.evec[:,:self.rank]    # (m × r*)
            # Normalise: divide each vector by its first element → β[0,j] = 1
            for j in range(self.rank):
                if self.beta[0,j]!= 0:
                    self.beta[:,j]/=self.beta[0,j]
            # α not directly available from coint_johansen; computed via VECM
            # Store as None here; fit_vecm.py will compute it properly
            self.alpha=None
        else:
            self.beta=None
            self.alpha=None

        # ============== Build result tables =====================
        trace_table=pd.DataFrame({"H0: rank ≤ r":[f"r = {r}" for r in range(self.m)], 
                                  "Eigenvalue":np.round(self.eigenvalues,6),
                                  "Trace Stat":np.round(self.trace_stats,4), 
                                  f"CV {_SIG_LABELS[self.sig_level]}":np.round(self.trace_cvs[:,self.sig_level], 4),
                                  "Reject H0":["Yes ✓" if (self.trace_stats[r])>(self.trace_cvs[r,self.sig_level]) else "No" for r in range(self.m)]})
        
        max_eig_table=pd.DataFrame({"H0: rank = r":[f"r = {r}" for r in range(self.m)],
                                    "Max-Eig Stat":np.round(self.max_eig_stats,4),
                                    f"CV {_SIG_LABELS[self.sig_level]}":np.round(self.max_eig_cvs[:,self.sig_level],4),
                                    "Reject H0":["Yes ✓" if (self.max_eig_stats[r])>(self.max_eig_cvs[r, self.sig_level]) else "No" for r in range(self.m)]})

        return {
            "rank":          self.rank,
            "eigenvalues":   self.eigenvalues,
            "trace_table":   trace_table,
            "max_eig_table": max_eig_table,
            "beta":          self.beta,
            "alpha":         self.alpha,
            "k_ar_diff":     self.k_ar_diff,
            "det_order":     self.det_order }

    def summary(self):
        self._require_fitted("summary")
        sep="="*68
        sig_label=_SIG_LABELS[self.sig_level]
        print(f"\n{sep}")
        print("  JOHANSEN COINTEGRATION TEST — SUMMARY")
        print(sep)
        print(f"  Series (m)        : {self.m}  {self.cols}")
        print(f"  Observations (T)  : {self.T}")
        print(f"  Lag differences   : k_ar_diff = {self.k_ar_diff}")
        print(f"  Det. order        : {self.det_order}")
        print(f"  Significance      : {sig_label}")
        print(f"  Cointegrating rank: r* = {self.rank}")
        print(sep)
        # Trace table
        print("\n  TRACE STATISTIC  (H0: at most r cointegrating vectors)")
        trace_table=pd.DataFrame({"H0 (rank ≤ r)":[f"r = {r}" for r in range(self.m)], "Trace Stat":np.round(self.trace_stats,4),
            f"CV ({sig_label})":np.round(self.trace_cvs[:,self.sig_level],4),
            "Reject?": ["Yes ✓" if (self.trace_stats[r])>(self.trace_cvs[r,self.sig_level]) else "No" for r in range(self.m)]})
        print(trace_table.to_string(index=False))
        # Max-Eigenvalue table
        print(f"\n  MAX-EIGENVALUE STATISTIC  (H0: exactly r vectors)")
        me_table=pd.DataFrame({"H0 (rank = r)":[f"r = {r}" for r in range(self.m)], "Max-Eig Stat":np.round(self.max_eig_stats,4),
            f"CV ({sig_label})": np.round(self.max_eig_cvs[:, self.sig_level],4),
            "Reject?": ["Yes ✓" if (self.max_eig_stats[r])>(self.max_eig_cvs[r,self.sig_level]) else "No" for r in range(self.m)]})
        print(me_table.to_string(index=False))
        # Cointegrating vectors
        if self.rank>0 and self.beta is not None:
            print(f"\n  COINTEGRATING VECTORS β  (m × r* = {self.m} × {self.rank})")
            beta_df=pd.DataFrame(self.beta,index=self.cols,columns=[f"β{j+1}" for j in range(self.rank)])
            print(beta_df.round(6).to_string())
        else:
            print("\n  No cointegrating vectors found (r* = 0).")
        print(f"\n{sep}\n")

    def plot(self,figsize:tuple=(14,5)):
        self._require_fitted("plot")
        sig_label=_SIG_LABELS[self.sig_level]
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=figsize)
        r_vals=range(self.m)
        ax1.bar(r_vals,self.trace_stats,color="steelblue",alpha=0.7,label="Trace Stat")
        ax1.plot(r_vals,self.trace_cvs[:,self.sig_level],"r--o",linewidth=2,markersize=6,label=f"CV ({sig_label})")
        ax1.set_xticks(list(r_vals))
        ax1.set_xticklabels([f"r ≤ {r}" for r in r_vals])
        ax1.set_title("Trace Statistic",fontweight="bold")
        ax1.set_xlabel("H0: rank ≤ r")
        ax1.set_ylabel("Statistic value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Max-Eigenvalue
        ax2.bar(r_vals, self.max_eig_stats, color="darkorange", alpha=0.7, label="Max-Eig Stat")
        ax2.plot(r_vals, self.max_eig_cvs[:, self.sig_level], "r--o", linewidth=2, markersize=6, label=f"CV ({sig_label})")
        ax2.set_xticks(list(r_vals))
        ax2.set_xticklabels([f"r = {r}" for r in r_vals])
        ax2.set_title("Max-Eigenvalue Statistic", fontweight="bold")
        ax2.set_xlabel("H0: rank = r")
        ax2.set_ylabel("Statistic value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Johansen Cointegration — r* = {self.rank} | Significance: {sig_label}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def get_rank(self):
        self._require_fitted("get_rank")
        return self.rank

    def get_beta(self):
        self._require_fitted("get_beta")
        if self.beta is None:
            raise ValueError("rank = 0, no cointegrating vectors exist.")
        return self.beta

    def __repr__(self):
        status = (f"rank={self.rank}" if self.rank is not None else "not fitted — call run()")
        return (f"JohansenTest(m={self.m}, T={self.T}, " 
                f"k_ar_diff={self.k_ar_diff}, "
                f"det_order='{self.det_order}', {status})" )
    
# ============== Standalone Test ========================
if __name__ == "__main__":
    tickers = ["RELIANCE.NS","TCS.NS"]
    raw = yf.download(tickers, start="2020-01-01", end="2024-01-01",auto_adjust=True)["Close"].dropna()
    log_prices = np.log(raw)
    print(f"Data: {log_prices.shape[0]} rows × {log_prices.shape[1]} cols")
    jt = JohansenTest(data=log_prices,det_order="restricted_constant",k_ar_diff=None,sig_level=1,verify_i1=True,)
    results = jt.run()
    jt.summary(); jt.plot()
    print(f"\nPass to VECM:  rank r* = {jt.get_rank()}, "
          f"k_ar_diff = {jt.k_ar_diff}")