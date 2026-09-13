"""
Module  : Financial Mathematics / Time Series Analysis / Kalman_Filter / vecm_forecast.py
Project : Dark Wing

Purpose:
    Use a fitted VECMModel to produce h-step-ahead forecasts of cointegrated
    price levels and derive spread forecasts with confidence bands.

    VECM Forecast Theory:
    ─────────────────────
    The VECM in companion VAR form:
        ΔX_t = Π·X_{t-1} + Γ_1·ΔX_{t-1} + ... + Γ_{k-1}·ΔX_{t-k+1} + ε_t
    where Π = α·β' (rank r* matrix).

    For h-step-ahead forecasting, convert VECM to levels VAR(k) form:
        X_{t+h} = Σ_{j=1}^{k} A_j · X_{t+h-j} + ε_{t+h}

    statsmodels VECMResults.predict(steps=h) handles this internally.

    Spread Forecast from VECM:
        Ŝ_{t+h|t} = β' · X̂_{t+h|t}
    where β is the Johansen cointegrating vector and X̂ is the forecasted
    price level vector.

    Forecast Variance:
        Var(Ŝ_{t+h|t}) = β' · MSE(X̂_{t+h|t}) · β
    where MSE grows with h (h-step mean squared error matrix).

Pipeline Position:
    STEP 4 → fit_vecm.py (VECMModel) — produces β matrix, fitted residuals
    STEP 7 → THIS FILE (VECMForecaster) — price-level + spread forecasts
    STEP 8 → signal/ — uses spread forecasts for position management

Dependencies: numpy, pandas, scipy, matplotlib, statsmodels
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Tuple,Dict,List
from scipy import stats

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
_VECM_DIR=os.path.join(_FM_DIR,"VECM")
for _p in [_FM_DIR,_TSA_DIR,_THIS_DIR,_VECM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

try:
    from fit_vecm import VECMModel
    _VECM_AVAILABLE=True
except ImportError:
    warnings.warn("VECMModel not found. VECMForecaster requires fit_vecm.py in VECM/.",ImportWarning)
    _VECM_AVAILABLE=False

# ─────────────────────────────────────────────────────────────────────────────
class VECMForecaster:
    """
    h-step-ahead forecaster built on top of a fitted VECMModel.

    Parameters
    ----------
    vecm_model : VECMModel — A fitted VECMModel instance (after calling fit()).
    alpha_level : float — Confidence level for prediction intervals. Default 0.05 (95%).
    """
    def __init__(self,vecm_model,alpha_level:float=0.05):
        if _VECM_AVAILABLE and not isinstance(vecm_model,VECMModel):
            raise TypeError("vecm_model must be a fitted VECMModel instance.")
        if not hasattr(vecm_model,"_fitted_result") or vecm_model._fitted_result is None:
            raise RuntimeError("VECMModel must be fitted. Call vecm_model.fit() first.")
        self._vm=vecm_model
        self._result=vecm_model._fitted_result       # statsmodels VECMResults
        self._beta=vecm_model.get_beta()             # (m × r*) cointegrating matrix
        self._alpha_conf=alpha_level
        self._z_crit=float(stats.norm.ppf(1-alpha_level/2))  # e.g. 1.96 for 95%
        # results (set in forecast())
        self._steps:Optional[int]=None
        self._price_fc:Optional[pd.DataFrame]=None  # (steps × m)
        self._spread_fc:Optional[pd.DataFrame]=None  # (steps × r*)
        self._spread_var:Optional[np.ndarray]=None
        self._done:bool=False

    # ── Core forecast ─────────────────────────────────────────────────────────
    def forecast(self,steps:int=20)->pd.DataFrame:
        """
        Run h-step-ahead forecast using VECMResults.predict().

        VECM is converted internally to companion VAR form for the recursion:
            X̂_{t+h|t} = A_1·X̂_{t+h-1|t} + ... + A_k·X̂_{t+h-k|t}

        Parameters
        ----------
        steps : int — Number of steps ahead to forecast.

        Returns
        -------
        pd.DataFrame — Price-level forecasts, shape (steps × m).
        """
        self._steps=steps
        result=self._result
        # statsmodels .predict(steps) returns (steps × m) ndarray of LEVELS
        try:
            fc_array=result.predict(steps=steps)
        except Exception as e:
            raise RuntimeError(f"VECM forecast failed: {e}") from e
        cols=result.model.endog_names if hasattr(result.model,"endog_names") else [f"X{i+1}" for i in range(fc_array.shape[1])]
        # Build future index
        last_date=result.model.endog_lagged.index[-1] if hasattr(result.model.endog_lagged,"index") else None
        if isinstance(last_date,pd.Timestamp):
            try:
                future_idx=pd.bdate_range(start=last_date,periods=steps+1,freq="B")[1:]
            except Exception:
                future_idx=pd.RangeIndex(steps)
        else:
            future_idx=pd.RangeIndex(steps)
        self._price_fc=pd.DataFrame(fc_array[:steps],index=future_idx,columns=cols)
        print(f"[VECMForecaster] {steps}-step forecast complete. Series: {cols}")
        return self._price_fc

    # ── Spread forecast ───────────────────────────────────────────────────────
    def spread_forecast(self,steps:Optional[int]=None)->pd.DataFrame:
        """
        Derive spread forecasts from price-level forecasts.
            Ŝ_{t+h|t} = X̂_{t+h|t} @ β
        where β is the (m × r*) Johansen cointegrating matrix.

        Returns
        -------
        pd.DataFrame — Spread (ECT) forecasts, shape (steps × r*).
        """
        if steps is not None:
            self.forecast(steps)
        if self._price_fc is None:
            raise RuntimeError("Call forecast(steps) first.")
        fc_values=self._price_fc.values          # (steps × m)
        beta=self._beta                           # (m × r*)
        spread_fc=fc_values @ beta               # (steps × r*)
        r=beta.shape[1]
        self._spread_fc=pd.DataFrame(spread_fc,
                                      index=self._price_fc.index,
                                      columns=[f"ECT_{j+1}" for j in range(r)])
        return self._spread_fc

    # ── Confidence bands ──────────────────────────────────────────────────────
    def confidence_bands(self,steps:Optional[int]=None,
                         alpha:Optional[float]=None)->Dict[str,pd.DataFrame]:
        """
        Compute forecast confidence intervals.

        For h-step VECM forecasts, the MSE matrix grows with h:
            MSE(h) = Σ_{j=0}^{h-1} Φ_j · Σ_ε · Φ_j'
        where Φ_j are the MA coefficient matrices.

        Spread variance: Var(Ŝ) = β' · MSE(h) · β

        Approximation used here: the diagonal of MSE from statsmodels
        VECMResults.ma_rep() and sigma_u.

        Returns
        -------
        dict with keys: 'upper', 'lower', 'se' — each a DataFrame (steps × r*).
        """
        if steps is not None:
            self.forecast(steps)
        if self._spread_fc is None:
            self.spread_forecast()
        alpha_use=alpha if alpha is not None else self._alpha_conf
        z=float(stats.norm.ppf(1-alpha_use/2))
        h=len(self._price_fc)
        # Get MA representation (companion form): shape (h, m, m)
        try:
            ma_coef=self._result.ma_rep(maxn=h)  # (h × m × m)
            sigma_u=self._result.sigma_u          # (m × m)
            # MSE(h) = Σ_{j=0}^{h-1} Φ_j Σ Φ_j'
            m=sigma_u.shape[0]
            mse=np.zeros((h,m,m))
            cumulative=np.zeros((m,m))
            for step in range(h):
                Phi=ma_coef[step]  # (m × m)
                cumulative+=Phi@sigma_u@Phi.T
                mse[step]=cumulative.copy()
            # Spread variance: β' · MSE(h) · β  → (h × r* × r*)
            beta=self._beta  # (m × r*)
            r=beta.shape[1]
            spread_se=np.zeros((h,r))
            for step in range(h):
                var_mat=beta.T@mse[step]@beta  # (r* × r*)
                spread_se[step]=np.sqrt(np.maximum(np.diag(var_mat),0))
        except Exception as e:
            warnings.warn(f"MSE computation failed ({e}). Using flat SE=0.",RuntimeWarning)
            spread_se=np.zeros((h,self._beta.shape[1]))
        se_df=pd.DataFrame(spread_se,index=self._spread_fc.index,
                           columns=self._spread_fc.columns)
        upper=self._spread_fc+z*se_df
        lower=self._spread_fc-z*se_df
        self._done=True
        return {"upper":upper,"lower":lower,"se":se_df,"point":self._spread_fc}

    # ── Reversion horizon ────────────────────────────────────────────────────
    def reversion_horizon(self,ect_col:str="ECT_1")->Optional[int]:
        """
        Estimate the number of steps for spread forecast to cross zero.
        Returns None if spread doesn't cross zero within the forecast window.
        """
        if self._spread_fc is None:
            raise RuntimeError("Call spread_forecast() first.")
        s=self._spread_fc[ect_col].values
        crossings=np.where(np.diff(np.sign(s)))[0]
        if len(crossings)>0:
            h=int(crossings[0]+1)
            print(f"[VECMForecaster] Expected reversion in {h} step(s).")
            return h
        print("[VECMForecaster] Spread does not cross zero within forecast horizon.")
        return None

    # ── Summary & Plot ────────────────────────────────────────────────────────
    def summary(self):
        if self._price_fc is None:
            raise RuntimeError("Call forecast() first.")
        sep="="*58
        print(f"\n{sep}")
        print("  VECM FORECASTER — SUMMARY")
        print(sep)
        print(f"  Forecast steps        : {self._steps}")
        print(f"  Asset series          : {list(self._price_fc.columns)}")
        print(f"  Cointegrating rank r* : {self._beta.shape[1]}")
        if self._spread_fc is not None:
            for col in self._spread_fc.columns:
                s=self._spread_fc[col]
                print(f"  {col} forecast range: [{s.min():.4f}, {s.max():.4f}]")
                h=self.reversion_horizon(col)
                if h: print(f"  {col} reversion horizon: {h} steps")
        print(sep+"\n")

    def plot_forecast(self,figsize:tuple=(14,9)):
        """Two-panel: price level forecasts and spread (ECT) forecasts with bands."""
        if self._price_fc is None:
            raise RuntimeError("Call forecast() first.")
        if self._spread_fc is None:
            self.spread_forecast()
        bands=self.confidence_bands()
        m=len(self._price_fc.columns)
        r=len(self._spread_fc.columns)
        fig,axes=plt.subplots(1+r,1,figsize=figsize,sharex=False)
        if 1+r==1: axes=[axes]
        # Panel 1: Price levels
        ax=axes[0]
        colors=plt.cm.tab10(np.linspace(0,0.6,m))
        for i,col in enumerate(self._price_fc.columns):
            ax.plot(self._price_fc.index,self._price_fc[col],color=colors[i],linewidth=1.5,
                    label=f"{col} forecast",marker="o",markersize=3)
        ax.set_title(f"VECM {self._steps}-Step Ahead Price Level Forecast",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # Panels: spread ECT forecasts
        for j in range(r):
            ax=axes[j+1]
            col=f"ECT_{j+1}"
            fc=self._spread_fc[col]; up=bands["upper"][col]; lo=bands["lower"][col]
            ax.plot(self._spread_fc.index,fc.values,color="steelblue",linewidth=1.5,label=f"{col} forecast",marker="o",markersize=3)
            ax.fill_between(self._spread_fc.index,lo.values,up.values,alpha=0.2,color="steelblue",label=f"{int((1-self._alpha_conf)*100)}% CI")
            ax.axhline(0,color="black",linestyle="--",linewidth=0.9)
            ax.set_title(f"{col} — Spread Forecast with Confidence Bands",fontweight="bold")
            ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        fig.suptitle("VECM Forecast — Price Levels and Spread ECTs",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        status="forecasted" if self._price_fc is not None else "not forecasted"
        return f"VECMForecaster(steps={self._steps}, {status})"

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    print("VECMForecaster demo requires a fitted VECMModel from VECM/fit_vecm.py")
    print("Usage:")
    print("  from fit_vecm import VECMModel")
    print("  from vecm_forecast import VECMForecaster")
    print("  vm = VECMModel(data, rank=1, lag_order=2); vm.fit()")
    print("  vf = VECMForecaster(vm)")
    print("  vf.forecast(steps=20)")
    print("  vf.spread_forecast()")
    print("  vf.summary(); vf.plot_forecast()")
