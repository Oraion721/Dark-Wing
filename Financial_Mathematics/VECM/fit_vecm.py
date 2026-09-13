"""
Module  : Financial Mathematics/VECM/fit_vecm.py
Project : Dark Wing
Purpose:
    After the Johansen test confirms r* cointegrating relationships among a system of I(1) series, this module fits the full Vector Error Correction Model (VECM). It extracts:
        α  — adjustment speeds (how fast each series corrects back to equilibrium)
        β  — cointegrating vectors (the long-run equilibrium equations)
        Γ  — short-run dynamics matrices (lagged difference effects)
        Σ  — residual covariance matrix
        ECT — error correction terms (the stationary spreads β'Xₜ)
    The ECT is the key output: it becomes the input to the Kalman filter and the spread/signal modules for live trading.
Pipeline Position:
    STEP 1 → Stationarity  (Stationarity_Tests.TS_stationarity_test)
    STEP 2 → Lag Selection  (Vector_Auto_Regressive.VAR_Lag_Select)
    STEP 3 → Johansen Test  (Time_Series_Analysis.Cointegration_Tests.johansen)
    STEP 4 → THIS FILE  (VECMModel)
    STEP 5 → Spread/Kalman Filter
Design decision:
    - Uses statsmodels VECM (method='ml') which solves the same Johansen MLE eigenvalue problem as the test, ensuring consistency.
    - det_order string is mapped to statsmodels 'deterministic' parameter strings.  """

import sys; import os; import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, TYPE_CHECKING
from statsmodels.tsa.vector_ar.vecm import VECM
if TYPE_CHECKING:
    from Time_Series_Analysis.Cointegration_Tests.johansen import JohansenTest
_THIS_DIR=os.path.dirname(os.path.abspath(__file__))          # .../VECM/
_FM_DIR=os.path.dirname(_THIS_DIR)                          # .../Financial_Mathematics/
_TSA_DIR=os.path.join(_FM_DIR,"Time_Series_Analysis")       # .../Time_Series_Analysis/
_COINT_DIR=os.path.join(_TSA_DIR,"Cointegration_Tests")       # .../Cointegration_Tests/
for _p in [_FM_DIR,_TSA_DIR,_COINT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)
from Time_Series_Analysis.Stationarity_Tests.TS_stationarity_test import StationarityTest
_DET_ORDER_MAP_VECM={"none":"nc","restricted_constant": "ci","restricted_trend":"li","unrestricted_constant":"co"}
_VALID_DET_ORDERS=list(_DET_ORDER_MAP_VECM.keys())
def _max_canonical_angle(A:np.ndarray,B:np.ndarray):
    """
    Measure the maximum canonical angle between the column spaces of A and B. Returns a value in [0, 90] degrees.
    0° → identical subspaces. 90° → orthogonal (completely different) subspaces.
    Used to reconcile the Johansen β with the VECM β (normalisations differ).   """
    Qa, _=np.linalg.qr(A)     # QR decomposition to get orthonormal bases
    Qb, _=np.linalg.qr(B)
    sv=np.linalg.svd(Qa.T@Qb,compute_uv=False)     # Singular values of Qa'Qb are cosines of canonical angles
    sv=np.clip(sv,-1.0,1.0)
    angles_deg=np.degrees(np.arccos(sv))
    return float(np.max(angles_deg))

class VECMModel:        # Fit a Vector Error Correction Model (VECM) using MLE.
    """
    The VECM equation is:
        ΔXₜ = C + α·β'·Xₜ₋₁  +  Γ₁·ΔXₜ₋₁ + ... + Γₖ·ΔXₜ₋ₖ  + ηₜ
    where:
        α     (m × r*)  adjustment speeds — how fast each series error-corrects
        β     (m × r*)  cointegrating vectors — long-run equilibrium weights
        Γᵢ   (m × m)   short-run lag matrices (k = k_ar_diff of them)
        ηₜ  ~ N(0, Σ)  white noise residuals
    Args: 
        ~ data:pd.DataFrame: DataFrame of I(1) log-price series. Min 2 columns, no NaNs.
        ~ k_ar_diff:int: Number of lagged differences (= VAR lag p − 1). Min 1.
        ~ coint_rank:int: Cointegrating rank r* from the Johansen test. Must satisfy 0 < r* < m.
        ~ det_order:str: Deterministic term. One of: 'none', 'restricted_constant' (default), 'restricted_trend', 'unrestricted_constant'.
        ~ verify_i1:bool: If True, run ADF+KPSS on each column to confirm I(1) before fitting.
        ~ johansen_beta : np.ndarray or None: Optional β from a prior JohansenTest. Used only for a reconciliation check — does NOT constrain the VECM fit. """

    def __init__(self,data:pd.DataFrame,k_ar_diff:int,coint_rank:int,det_order:str="restricted_constant",verify_i1:bool=False,johansen_beta:Optional[np.ndarray]=None):
        if not isinstance(data,pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame.")
        if data.shape[1]<2:
            raise ValueError(f"VECM requires >= 2 series, got {data.shape[1]}.")
        self.data=data.copy().dropna()
        self.T=len(self.data); self.m=self.data.shape[1]
        self.cols=list(self.data.columns)
        if not isinstance(k_ar_diff,int) or k_ar_diff<1:
            raise ValueError("k_ar_diff must be a positive integer >= 1.")
        if not isinstance(coint_rank,int) or not (0<coint_rank<self.m):
            raise ValueError(f"coint_rank must satisfy 0 < r < m={self.m}, got {coint_rank}.")
        if det_order not in _VALID_DET_ORDERS:
            raise ValueError(f"det_order must be one of {_VALID_DET_ORDERS}, got '{det_order}'.")
        if self.T<(k_ar_diff+coint_rank+self.m+10):
            raise ValueError(f"Not enough observations (T={self.T}) for k_ar_diff={k_ar_diff},coint_rank={coint_rank}, m={self.m}.")

        self.k_ar_diff=k_ar_diff
        self.coint_rank=coint_rank
        self.det_order=det_order
        self._det_str=_DET_ORDER_MAP_VECM[det_order]
        self.verify_i1=verify_i1
        self._johansen_beta=johansen_beta                               # for reconciliation check only
        self.alpha:Optional[np.ndarray]=None                            # (m × r*)
        self.beta:Optional[np.ndarray]=None                             # (m × r*)
        self.gamma:Optional[List[np.ndarray]]=None                      # list of k (m × m)
        self.sigma_u:Optional[np.ndarray]=None                          # (m × m) residual cov
        self.ect:Optional[pd.DataFrame]=None                            # (T_eff × r*) ECT series
        self.resid:Optional[np.ndarray]=None                            # (T_eff × m)
        self.fittedvalues:Optional[np.ndarray]=None                     # (T_eff × m)
        self._vecm_result=None                                          # raw statsmodels result
        self._is_stable:Optional[bool]=None                             # stability verdict
        self._recon_angle:Optional[float]=None                          # β reconciliation angle (°)

    @classmethod
    def from_johansen(cls,johansen_test,verify_i1:bool=False):
        """
        This is the recommended way to chain the pipeline:
            jt = JohansenTest(...); jt.run()
            vm = VECMModel.from_johansen(jt)
            vm.fit()
        Args: 
            ~ johansen_test:JohansenTest: A JohansenTest instance that has already had .run() called.
            ~ verify_i1:bool: Pass-through to VECMModel.__init__. Default False (already verified in JohansenTest).
        Raises:
            ValueError  if johansen_test.rank == 0 (nothing to fit).
            RuntimeError if johansen_test has not been run yet.     """
        if johansen_test._raw_result is None:
            raise RuntimeError("JohansenTest has not been run. Call johansen_test.run() first.")
        if johansen_test.rank==0:
            raise ValueError("Johansen test found rank = 0 (no cointegration).\nVECM cannot be fitted. Use VAR in differences instead.")
        return cls(data=johansen_test.data,k_ar_diff=johansen_test.k_ar_diff,coint_rank=johansen_test.rank,det_order= johansen_test.det_order,verify_i1=verify_i1,johansen_beta=johansen_test.beta)

    def _require_fitted(self,method:str):
        if self._vecm_result is None:
            raise RuntimeError(f"Call fit() before calling {method}().")

    def _verify_integration_order(self):        # Run ADF+KPSS on each column to confirm I(1). Warns, does not raise
        for col in self.cols:
            tester=StationarityTest(time_series=self.data[[col]],lag_max=None,regression_type="c",required_lag="AIC")
            result=tester.check_stationarity()
            if result["is_stationary"]:
                warnings.warn(f"Series '{col}' appears STATIONARY (I(0)).\nVECM assumes I(1) inputs. Results may be unreliable.",UserWarning)
            else:
                print(f"  ✓  '{col}' confirmed I(1).")

    def _extract_gamma(self,raw_gamma:np.ndarray):
        if raw_gamma is None or raw_gamma.size==0:
            return []
        k=self.k_ar_diff      # raw_gamma shape: (m, m * k_ar_diff)
        m=self.m
        gamma_list=[]
        for i in range(k):
            gamma_list.append(raw_gamma[:,i*m:(i+1)*m])
        return gamma_list

    def _check_stability(self):
        """ Check VECM stability via companion-form eigenvalues.A correctly fitted VECM(r*) should have exactly r* unit roots
        (eigenvalues with |λ| = 1) and all remaining roots inside the unit circle (|λ| < 1). Warn if extra near-unit roots exist. """
        try:
            roots=self._vecm_result.roots
            n_unit=int(np.sum(np.abs(roots)>=0.99))
            if n_unit>self.coint_rank:
                warnings.warn(f"Stability check: found {n_unit} roots ≥ 0.99, but expected {self.coint_rank} (= coint_rank).\nThe system may be near-explosive or have additional unit roots.",UserWarning)
                return False
            return True
        except Exception:
            warnings.warn("Could not compute stability check (roots not available).",UserWarning)
            return True

    def _reconcile_beta(self):
        """ Compare the column space of the VECM β against the Johansen β. Uses the maximum canonical angle (see _max_canonical_angle()). A large angle (> 15°) suggests a potential discrepancy.   """
        if self._johansen_beta is None:
            return
        try:
            angle=_max_canonical_angle(self.beta,self._johansen_beta)
            self._recon_angle=angle
            if angle>15.0:
                warnings.warn(f"β reconciliation: maximum canonical angle = {angle:.2f}°. ""The VECM β column space differs substantially from the Johansen β. ""This can occur with small samples or near-collinear series. ""Inspect both manually.",UserWarning)
            else:
                print(f"  ✓  β reconciliation OK (max canonical angle = {angle:.2f}°).")
        except Exception as e:
            warnings.warn(f"β reconciliation failed: {e}", RuntimeWarning)

    def fit(self):      # Fit the VECM model via Maximum Likelihood (Johansen MLE).
        """ Steps performed internally:
            1.  (Optional) verify I(1) order for all series.
            2.  Build statsmodels VECM with the stored parameters.
            3.  Call model.fit(method='ml').
            4.  Extract α, β, Γ list, Σ, residuals, fitted values.
            5.  Compute Error Correction Terms: ECTₜ = Xₜ · β  (T × r*).
            6.  Reconcile VECM β against the Johansen β (if provided).
            7.  Run stability check on companion-form eigenvalues.
        O/P: dict with keys:{alpha, beta, gamma, sigma_u, ect, resid,fittedvalues, k_ar_diff, coint_rank, det_order, is_stable, recon_angle_deg}      """
        if self.verify_i1:
            self._verify_integration_order()
        try:
            model=VECM(endog=self.data,k_ar_diff=self.k_ar_diff,coint_rank=self.coint_rank,deterministic=self._det_str)
            vecm_res=model.fit(method="ml")
        except Exception as e:
            raise RuntimeError(f"VECM.fit() failed: {e}") from e
        self._vecm_result=vecm_res
        self.alpha=vecm_res.alpha                               # (m × r*)
        self.beta=vecm_res.beta                                 # (m × r*) — statsmodels normalisation
        self.gamma=self._extract_gamma(vecm_res.gamma)          # list of k (m × m)
        self.sigma_u=vecm_res.sigma_u                           # (m × m)
        self.resid=vecm_res.resid                               # (T_eff × m) — T_eff = T - k_ar_diff - 1
        self.fittedvalues=vecm_res.fittedvalues                 # (T_eff × m)
        ect_values=self.data.values @ self.beta                 # (T × r*)
        ect_index=self.data.index                               # full length index
        self.ect=pd.DataFrame(ect_values,index=ect_index,columns=[f"ECT_{j+1}" for j in range(self.coint_rank)])
        self._reconcile_beta()
        self._is_stable=self._check_stability()
        return {"alpha":self.alpha,
            "beta":self.beta,
            "gamma":self.gamma,
            "sigma_u":self.sigma_u,
            "ect":self.ect,
            "resid":self.resid,
            "fittedvalues":self.fittedvalues,
            "k_ar_diff":self.k_ar_diff,
            "coint_rank":self.coint_rank,
            "det_order":self.det_order,
            "is_stable":self._is_stable,
            "recon_angle_deg":self._recon_angle    }

    def get_alpha(self):        # Return α (m × r*) — adjustment speed matrix
        self._require_fitted("get_alpha")
        return self.alpha

    def get_beta(self):     # Return β (m × r*) — cointegrating vectors
        self._require_fitted("get_beta")
        return self.beta

    def get_gamma(self):        # Return list of k Γ matrices (each m × m) — short-run dynamics
        self._require_fitted("get_gamma")
        return self.gamma

    def get_ect(self):      # Return the Error Correction Term(s) as a DataFrame.
        # Shape: (T × r*). This is the stationary spread, ready for z-scoring and signal generation in spread/ and kalman/ modules. """
        self._require_fitted("get_ect")
        return self.ect

    def get_residuals(self):        # Return model residuals (T_eff × m)
        self._require_fitted("get_residuals")
        return self.resid

    def summary(self):
        self._require_fitted("summary")
        sep="="*70
        print(f"\n{sep}")
        print("  VECM MODEL — FIT SUMMARY")
        print(sep)
        print(f"  Series (m)           : {self.m}   {self.cols}")
        print(f"  Observations (T)     : {self.T}")
        print(f"  Lag differences (k)  : {self.k_ar_diff}")
        print(f"  Cointegrating rank r*: {self.coint_rank}")
        print(f"  Deterministic term   : {self.det_order} → '{self._det_str}'")
        print(sep)
        print("\n  ADJUSTMENT SPEED MATRIX  α  (m × r*)")
        print("  (Significant α → series IS error-correcting)")
        print("  (Insignificant α → series is WEAKLY EXOGENOUS — it leads, not follows)")
        try:
            tvals=self._vecm_result.tvalues_alpha   # (m × r*)
            pvals=self._vecm_result.pvalues_alpha   # (m × r*)
            alpha_df = pd.DataFrame(self.alpha,index=self.cols,columns=[f"ECR_{j+1}" for j in range(self.coint_rank)])
            for j in range(self.coint_rank):
                col_name = f"ECR_{j+1}"
                print(f"\n  Equation {j+1}:")
                for i, series_name in enumerate(self.cols):
                    a=self.alpha[i,j]
                    t=tvals[i,j]
                    p=pvals[i,j]
                    sig="***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
                    weak="  ← WEAKLY EXOGENOUS" if p>=0.10 else ""
                    print(f"    {series_name:20s}  α = {a:+.6f}   t = {t:+.4f}   p = {p:.4f} {sig}{weak}")
        except AttributeError:
            alpha_df=pd.DataFrame(self.alpha,index=self.cols,columns=[f"ECR_{j+1}" for j in range(self.coint_rank)])
            print(alpha_df.round(6).to_string())
        print(f"\n{sep}")
        print(f"  COINTEGRATING VECTORS  β  (m × r*)")
        beta_df=pd.DataFrame(self.beta,index=self.cols,columns=[f"β_{j+1}" for j in range(self.coint_rank)])
        print(beta_df.round(6).to_string())
        if self.gamma:
            print(f"\n{sep}")
            print(f"  SHORT-RUN DYNAMICS  Γ  ({len(self.gamma)} lag matrices, each {self.m}×{self.m})")
            for i,G in enumerate(self.gamma):
                print(f"\n  Γ_{i+1}:")
                gamma_df=pd.DataFrame(G,index=self.cols,columns=self.cols)
                print(gamma_df.round(6).to_string())
        # ================= Stability & reconciliation ====================== 
        print(f"\n{sep}")
        stable_str="✅ STABLE" if self._is_stable else "⚠️  POTENTIALLY UNSTABLE"
        print(f"  System stability : {stable_str}")
        if self._recon_angle is not None:
            ok_str="✅ OK" if self._recon_angle<=15.0 else "⚠️  DIVERGES"
            print(f"  β reconciliation : {ok_str}  (max canonical angle = {self._recon_angle:.2f}°)")
        print(sep)

    def forecast(self,steps:int=10):
        """ Generate h-step-ahead point forecasts for ΔXₜ using the fitted VECM.
        Args:
            ~ steps:int: Number of steps ahead to forecast.
        O/P: pd.DataFrame of shape (steps × m) with level forecasts (X̂ₜ₊ₕ).     """
        self._require_fitted("forecast")
        if steps<1:
            raise ValueError("steps must be >= 1.")
        try:
            # statsmodels forecast() returns ΔX forecasts; we convert to levels
            delta_forecast = self._vecm_result.predict(steps=steps)  # (steps × m)
            # Reconstruct levels by cumulative sum from last observed level
            last_level = self.data.values[-1]   # (m,)
            level_forecast = np.cumsum(delta_forecast, axis=0) + last_level  # (steps × m)
            if isinstance(self.data.index,pd.DatetimeIndex):
                freq=pd.infer_freq(self.data.index)
                future_index=pd.date_range(start=self.data.index[-1],periods=steps+1,freq=freq or "B")[1:]
            else:
                future_index=range(self.data.index[-1]+1,self.data.index[-1]+1+steps)
            return pd.DataFrame(level_forecast,index=future_index,columns=self.cols)
        except Exception as e:
            raise RuntimeError(f"VECM forecast failed: {e}") from e

    def impulse_response(self,periods:int=20,orth:bool=True,cumulative:bool=False):
        """ Compute Impulse Response Functions (IRFs).
        Shows how a one-standard-deviation shock to one variable propagates through the entire system over time.
        Args:
            periods:int: Number of periods to trace the impulse response.
            orth:bool: If True, use orthogonalised shocks (Cholesky decomposition). This attributes instantaneous effects to a causal ordering.
            cumulative:bool: If True, return cumulative IRFs instead of period-by-period.
        O/P: 
            pd.DataFrame with MultiIndex columns (shock_variable, response_variable).       """
        self._require_fitted("impulse_response")
        try:
            irf_result=self._vecm_result.irf(periods=periods)
            if cumulative:
                irf_values=irf_result.cum_effects if orth else irf_result.cum_effects
            else:
                irf_values=irf_result.orth_irfs if orth else irf_result.irfs        # irf_values shape: (periods+1, m, m) — [period, response, shock]
            col_tuples=[(f"shock:{self.cols[shock]}",f"resp:{self.cols[resp]}")for shock in range(self.m)for resp  in range(self.m)]
            flat=irf_values.reshape(periods+1,self.m*self.m)
            return pd.DataFrame(flat,index=range(periods+1),columns=pd.MultiIndex.from_tuples(col_tuples))
        except Exception as e:
            raise RuntimeError(f"Impulse response failed: {e}") from e

    def plot(self,figsize:tuple=(14,4)):
        """
        Each ECT = β'ⱼ · Xₜ is the j-th cointegrating spread. If the VECM is correctly specified, each ECT should be stationary (fluctuating around a constant mean).       """
        self._require_fitted("plot")
        r=self.coint_rank
        fig,axes=plt.subplots(r,1,figsize=(figsize[0],figsize[1]*r),sharex=True,squeeze=False)
        for j in range(r):
            ax =axes[j, 0]
            ect=self.ect[f"ECT_{j+1}"]
            mu =ect.mean()
            sig=ect.std()
            ax.plot(ect.index,ect.values,color="steelblue",linewidth=1.2,label=f"ECT_{j+1}")
            ax.axhline(mu,color="black",linestyle="-",linewidth=1.0,label=f"μ = {mu:.4f}")
            ax.axhline(mu+2*sig,color="red",linestyle="--",linewidth=0.9,label=f"+2σ")
            ax.axhline(mu-2*sig,color="red",linestyle="--",linewidth=0.9,label=f"-2σ")
            ax.fill_between(ect.index,mu-2*sig,mu+2*sig,color="red",alpha=0.05)
            ax.set_title(f"Error Correction Term {j+1}  (β{j+1}' · Xₜ)",fontweight="bold")
            ax.set_ylabel("ECT value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if isinstance(self.data.index, pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

        fig.suptitle(f"VECM Error Correction Terms  |  r* = {self.coint_rank}  | {self.cols}",fontsize=12,fontweight="bold")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        status="fitted" if self._vecm_result is not None else "not fitted — call fit()"
        stable = (f", stable={self._is_stable}" if self._is_stable is not None else "")
        return (f"VECMModel("
            f"m={self.m}, T={self.T}, "
            f"k_ar_diff={self.k_ar_diff}, "
            f"coint_rank={self.coint_rank}, "
            f"det_order='{self.det_order}', "
            f"{status}{stable})")

if __name__ == "__main__":
    tickers=['JIOFIN.NS','RELIANCE.NS']
    raw=yf.download(tickers,period='5y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_prices=np.log(raw)
    from Time_Series_Analysis.Cointegration_Tests.johansen import JohansenTest
    jt=JohansenTest(data=log_prices,det_order="restricted_constant",k_ar_diff=None,sig_level=1,verify_i1=False)
    jt.run()
    jt.summary()
    print(f"\nJohansen → rank r* = {jt.rank}, k_ar_diff = {jt.k_ar_diff}")
    vm=VECMModel.from_johansen(jt,verify_i1=False)
    result=vm.fit()
    print(vm); vm.summary()
    ect = vm.get_ect()
    print(f"\nECT shape: {ect.shape}")
    print(ect.tail(3))
    fc=vm.forecast(steps=10)
    print(f"\nForecast (10 steps):"); print(fc.round(4))
    vm.plot()