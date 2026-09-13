""" When to use Engle-Granger vs. Johansen:
        - Use Engle-Granger for PAIRS TRADING (exactly 2 assets).
        - Use Johansen (johansen.py) for BASKETS (3+ assets).   """
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm
from typing import Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TSA_DIR  = os.path.dirname(_THIS_DIR)
if _TSA_DIR not in sys.path:
    sys.path.insert(0, _TSA_DIR)
from Stationarity_Tests.TS_stationarity_test import StationarityTest

class EngleGrangerTest:
    """
    Step 1 — OLS Regression:
        Y_t = α + β · X_t + ε_t;    Estimate α (intercept) and β (hedge ratio / cointegrating coefficient).
    Step 2 — ADF on Residuals:
        Test H0: ε_t ~ I(1)  (non-stationary → NOT cointegrated)
        vs  H1: ε_t ~ I(0)  (stationary     → COINTEGRATED)
    Args: 
        y : pd.Series or pd.DataFrame (single column)
            Dependent variable — the series you are LONG.
        x : pd.Series or pd.DataFrame (single column)
            Independent variable — the series you are SHORT/HEDGE.
        max_lag : int or None
            Max lags for ADF on residuals. None → Ng-Perron auto formula.
        regression_type : str
            ADF regression specification. 'c' (default) or 'ct' or 'n'.
        sig_level : float
            Significance threshold for the ADF p-value. Default 0.05.
        verify_i1 : bool
            If True (default), run ADF+KPSS on y and x to confirm I(1).    """

    def __init__(self,y:pd.Series,x:pd.Series,max_lag:Optional[int]=None,regression_type:str= "c",sig_level:float=0.05,verify_i1:bool=True):
        if not isinstance(y,(pd.Series,pd.DataFrame)):
            raise TypeError("y must be a pandas.Series or single-column pandas.DataFrame.")
        if not isinstance(x,(pd.Series,pd.DataFrame)):
            raise TypeError("x must be a pandas.Series or single-column pandas.DataFrame.")
        self.y=y.squeeze() if isinstance(y,pd.DataFrame) else y         # convert single dataframe into array
        self.x=x.squeeze() if isinstance(x,pd.DataFrame) else x
        if not isinstance(self.y,pd.Series) or not isinstance(self.x,pd.Series):
            raise ValueError("y and x must be single-column series.")
        self.y_name=self.y.name or "Y"
        self.x_name=self.x.name or "X"
        combined=pd.concat([self.y,self.x],axis=1).dropna()
        if len(combined)<30:
            raise ValueError(f"After alignment, only {len(combined)} observations remain. Need at least 30 for a reliable test.")
        self.y=combined.iloc[:,0]
        self.x=combined.iloc[:,1]
        self.T=len(combined)

        if regression_type not in ("c","ct","n"):
            raise ValueError(f"regression_type must be 'c', 'ct', or 'n', got '{regression_type}'.")
        if not (0<sig_level<1):
            raise ValueError("sig_level must be between 0 and 1.")

        self.max_lag         = max_lag
        self.regression_type = regression_type
        self.sig_level       = sig_level
        self.verify_i1       = verify_i1

        # ================ Result placeholders ============================= 
        self.alpha:           Optional[float]      = None   # OLS intercept
        self.beta:            Optional[float]      = None   # OLS slope (hedge ratio)
        self.residuals:       Optional[pd.Series]  = None   # OLS residuals ε_t
        self.adf_result:      Optional[dict]       = None   # ADF on residuals
        self.is_cointegrated: Optional[bool]       = None
        self._ols_result      = None

    def _require_fitted(self, method: str):
        if self.residuals is None:
            raise RuntimeError(f"Call run() before calling {method}().")

    def _verify_integration_order(self):
        for name,series in [(self.y_name,self.y),(self.x_name,self.x)]:
            tester=StationarityTest(time_series=series.to_frame(),lag_max=None,regression_type="c",required_lag="AIC")
            result=tester.check_stationarity()
            if result["is_stationary"]:
                warnings.warn(f"Series '{name}' appears STATIONARY (I(0)).\nEngle-Granger assumes both series are I(1). Results may be spurious", UserWarning)
            else:
                print(f"  ✓  '{name}' confirmed I(1).")

    def _run_ols(self):
        X_const=add_constant(self.x.values)
        ols=OLS(self.y.values,X_const).fit()
        self._ols_result=ols
        self.alpha=float(ols.params[0])    # intercept
        self.beta=float(ols.params[1])    # slope = hedge ratio
        self.residuals=pd.Series(ols.resid,index=self.y.index,name=f"residuals({self.y_name}−β·{self.x_name})")

    def _run_adf_on_residuals(self):        # Step 2: ADF test on OLS residuals.
        """ NOTE: Standard ADF critical values are WRONG for residuals because residuals are estimated, not observed.
        We use MacKinnon (2010) response surface critical values which statsmodels applies automatically via adfuller(). """
        adf = adfuller(self.residuals,maxlag=self.max_lag,regression=self.regression_type,autolag="AIC")
        return {"test_name":"ADF on EG Residuals",
                "adf_statistic":round(float(adf[0]),6),
                "p_value":round(float(adf[1]),6),
                "used_lags":int(adf[2]),"n_obs":int(adf[3]),
                "critical_values":{k: round(float(v),4) for k,v in adf[4].items()}}
    
    def run(self) -> dict:
        if self.verify_i1:
            self._verify_integration_order()
        self._run_ols()
        self.adf_result=self._run_adf_on_residuals()
        self.is_cointegrated=(self.adf_result["p_value"]<self.sig_level)
        return {"alpha":self.alpha,"beta":self.beta,"residuals":self.residuals,
                "adf_result":self.adf_result,"is_cointegrated":self.is_cointegrated,
                "spread_mean":float(self.residuals.mean()),
                "spread_std":float(self.residuals.std())}

    def summary(self):
        self._require_fitted("summary")
        sep="="*62
        decision="✅ COINTEGRATED" if self.is_cointegrated else "❌ NOT COINTEGRATED"
        adf=self.adf_result
        print(f"\n{sep}")
        print("  ENGLE-GRANGER COINTEGRATION TEST — SUMMARY")
        print(sep)
        print(f"  Dependent   Y : {self.y_name}")
        print(f"  Independent X : {self.x_name}")
        print(f"  Observations  : {self.T}")
        print(f"  Significance  : {self.sig_level*100:.0f}%")
        print(sep)
        print("  STEP 1 — OLS REGRESSION:  Y = α + β·X + ε")
        print(f"    Intercept α  : {self.alpha:.6f}")
        print(f"    Hedge Ratio β: {self.beta:.6f}")
        print(f"    Interpretation: Short {self.beta:.4f} units of {self.x_name}\nper 1 unit of {self.y_name}")
        print(sep)
        print("  STEP 2 — ADF TEST ON RESIDUALS (ε_t):")
        print(f"    ADF statistic : {adf['adf_statistic']:.6f}")
        print(f"    p-value       : {adf['p_value']:.6f}")
        print(f"    Used lags     : {adf['used_lags']}")
        print(f"    Critical values:")
        for level,cv in adf["critical_values"].items():
            marker=" ←" if float(adf["adf_statistic"])<cv else ""
            print(f"      {level:5s}: {cv:.4f}{marker}")
        print(sep)
        print(f"  SPREAD STATISTICS:")
        print(f"    Mean          : {self.residuals.mean():.6f}")
        print(f"    Std Dev       : {self.residuals.std():.6f}")
        print(f"    Skewness      : {self.residuals.skew():.4f}")
        print(sep)
        print(f"  DECISION: {decision}")
        print(f"{sep}\n")

    def plot(self, figsize: tuple = (14, 10)):
        """
        3-panel plot:
        1. Y and scaled X prices over time (visual alignment)
        2. Spread (residuals ε_t) over time with ±2σ bands
        3. Histogram of spread with normal overlay      """
        self._require_fitted("plot")
        fig,axes=plt.subplots(3,1,figsize=figsize)
        y_norm=(self.y-self.y.mean())/self.y.std()
        x_norm=(self.x-self.x.mean())/self.x.std()
        axes[0].plot(y_norm.index,y_norm.values,label=self.y_name,color="steelblue",linewidth=1.2)
        axes[0].plot(x_norm.index,x_norm.values,label=self.x_name,color="darkorange",linewidth=1.2,alpha=0.8)
        axes[0].set_title("Normalised Price Series",fontweight="bold")
        axes[0].legend()
        axes[0].grid(True,alpha=0.3)
        mu=self.residuals.mean()
        sig=self.residuals.std()
        axes[1].plot(self.residuals.index,self.residuals.values,color="seagreen",linewidth=1.2,label="Spread ε_t")
        axes[1].axhline(mu,color="black",linestyle="-",linewidth=1.0,label=f"Mean ({mu:.4f})")
        axes[1].axhline(mu+2*sig,color="red",linestyle="--",linewidth=1.0,label=f"+2σ ({mu+2*sig:.4f})")
        axes[1].axhline(mu-2*sig,color="red",linestyle="--",linewidth=1.0,label=f"-2σ ({mu-2*sig:.4f})")
        axes[1].fill_between(self.residuals.index,mu-2*sig,mu+2*sig,color="red",alpha=0.05)
        status="Cointegrated ✅" if self.is_cointegrated else "Not Cointegrated ❌"
        axes[1].set_title(f"Spread  (ADF p={self.adf_result['p_value']:.4f})  |  {status}",fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        # =================== Spread histogram =========================
        axes[2].hist(self.residuals.values,bins=50,color="steelblue",edgecolor="white",alpha=0.7,density=True)
        xr = np.linspace(self.residuals.min(), self.residuals.max(), 200)
        axes[2].plot(xr,sp_norm.pdf(xr, mu, sig),"r-",linewidth=2,label="Normal fit")
        axes[2].set_title("Spread Distribution",fontweight="bold")
        axes[2].legend()
        axes[2].grid(True,alpha=0.3)
        fig.suptitle(f"Engle-Granger: {self.y_name} / {self.x_name}  |  β = {self.beta:.4f}",fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.show()

    def get_spread(self) -> pd.Series:
        self._require_fitted("get_spread")
        return self.residuals

    def get_hedge_ratio(self) -> float:
        self._require_fitted("get_hedge_ratio")
        return self.beta

    def __repr__(self):
        status = (f"cointegrated={self.is_cointegrated}, β={self.beta:.4f}" if self.beta is not None else "not fitted — call run()")
        return (f"EngleGrangerTest("
                f"Y='{self.y_name}', X='{self.x_name}', "
                f"T={self.T}, {status})")

if __name__ == "__main__":
    import yfinance as yf
    tickers = ["RELIANCE.NS", "TCS.NS"]
    raw = yf.download(tickers,period='5y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_prices = np.log(raw)
    eg = EngleGrangerTest(y=log_prices["RELIANCE.NS"],x=log_prices["TCS.NS"],sig_level=0.05,verify_i1=True,)
    results = eg.run()
    eg.summary()
    eg.plot()
    if results["is_cointegrated"]:
        spread = eg.get_spread()
        print(f"\nSpread ready for Z-score / Kalman:  {spread.shape}")