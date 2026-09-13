"""
Module  : Financial Mathematics/VECM/diagnostics.py
Project : Dark Wing
Purpose : Post-estimation validity checks on a fitted VECMModel.
          Answers: "is this VECM fit trustworthy enough to build a spread on?"
Tests run:
    1. Whiteness (Portmanteau)      —  no residual autocorrelation
    2. Normality (Doornik-Hansen)   —  Gaussian errors assumption
    3. ARCH                         —  no conditional heteroskedasticity
    4. ECT Stationarity             —  spreads are confirmed I(0)
    5. Split-sample stability       —  parameters stable across sub-samples
Health Score = fraction of {whiteness+ECT_stationarity+split-stability} passed. Normality and ARCH are informational only (common failures in finance)"""

import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2
from statsmodels.stats.diagnostic import het_arch
from typing import Optional,Dict

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_FM_DIR=os.path.dirname(_THIS_DIR)
_TSA_DIR=os.path.join(_FM_DIR,"Time_Series_Analysis")
_COINT_DIR=os.path.join(_TSA_DIR,"Cointegration_Tests")
for _p in [_FM_DIR,_TSA_DIR,_COINT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)
from fit_vecm import VECMModel,_max_canonical_angle
from Time_Series_Analysis.Stationarity_Tests.TS_stationarity_test import StationarityTest

class VECMDiagnostics:
    """
    Args:
        vecm_model   : VECMModel  — must have .fit() already called.
        whiteness_lags: int       — lags for Portmanteau whiteness test.  Default 10.
        arch_lags    : int        — lags for ARCH-LM test per residual.   Default 5.
        sig_level    : float      — p-value threshold for pass/fail.       Default 0.05.     """
    def __init__(self,vecm_model:VECMModel,whiteness_lags:int=10,arch_lags:int=5,sig_level:float=0.05):
        if vecm_model._vecm_result is None:
            raise RuntimeError("VECMModel has not been fitted. Call vecm_model.fit() first.")
        if not (0<sig_level<1):
            raise ValueError("sig_level must be between 0 and 1.")
        self.model=vecm_model
        self.resid=vecm_model.resid             # (T_eff × m)
        self.ect=vecm_model.ect                 # (T × r*)
        self.cols=vecm_model.cols
        self.m=vecm_model.m; self.r=vecm_model.coint_rank; self.T=vecm_model.T
        self.k_ar_diff=vecm_model.k_ar_diff
        self.det_order=vecm_model.det_order
        self.whiteness_lags=whiteness_lags
        self.arch_lags=arch_lags
        self.sig_level=sig_level
        # result placeholders
        self.whiteness:Optional[dict]=None
        self.normality:Optional[dict]=None
        self.arch:Optional[Dict[str,dict]]=None
        self.ect_stationarity:Optional[Dict[str,dict]]=None
        self.split_stability:Optional[dict]=None
        self.health_score:Optional[float]=None

    @classmethod
    def from_vecm(cls,vecm_model:VECMModel,**kwargs)->"VECMDiagnostics":
        """Convenience constructor: VECMDiagnostics.from_vecm(vm, whiteness_lags=12)"""
        return cls(vecm_model=vecm_model,**kwargs)

    def _require_run(self,method:str):
        if self.health_score is None:
            raise RuntimeError(f"Call run() before calling {method}().")

    def _test_whiteness(self):
        """Portmanteau whiteness test on multivariate residuals via statsmodels."""
        try:
            res=self.model._vecm_result.test_whiteness(nlags=self.whiteness_lags,adjusted=True)
            stat=float(res.test_statistic)
            df=int(res.df)
            p=float(res.pvalue)
            return {"statistic":stat,"df":df,"p_value":p,"passed":p>self.sig_level}
        except Exception as e:
            warnings.warn(f"Whiteness test failed: {e}",RuntimeWarning)
            return {"statistic":np.nan,"df":np.nan,"p_value":np.nan,"passed":False}

    def _test_normality(self)->dict:
        """Doornik-Hansen multivariate normality test via statsmodels."""
        try:
            res=self.model._vecm_result.test_normality()
            stat=float(res.test_statistic)
            df=int(res.df)
            p=float(res.pvalue)
            return {"statistic":stat,"df":df,"p_value":p,"passed":p>self.sig_level}
        except Exception as e:
            warnings.warn(f"Normality test failed: {e}",RuntimeWarning)
            return {"statistic":np.nan,"df":np.nan,"p_value":np.nan,"passed":False}

    def _test_arch(self):
        """Per-equation ARCH-LM test (conditional heteroskedasticity)."""
        results={}
        for i,col in enumerate(self.cols):
            try:
                e=self.resid[:,i]
                lm,lm_p,_,_=het_arch(e,nlags=self.arch_lags)
                results[col]={"lm_stat":float(lm),"p_value":float(lm_p),"passed":float(lm_p)>self.sig_level}
            except Exception as ex:
                warnings.warn(f"ARCH test failed for '{col}': {ex}",RuntimeWarning)
                results[col]={"lm_stat":np.nan,"p_value":np.nan,"passed":False}
        return results

    def _test_ect_stationarity(self):
        """ADF+KPSS stationarity check on each ECT series — they MUST be I(0)."""
        results={}
        for col in self.ect.columns:
            try:
                tester=StationarityTest(time_series=self.ect[[col]],lag_max=None,regression_type="c",required_lag="AIC")
                r=tester.check_stationarity()
                results[col]={"adf_p":r.get("adf_p_value",np.nan),"kpss_p":r.get("kpss_p_value",np.nan),"is_stationary":r["is_stationary"],"passed":bool(r["is_stationary"])}
            except Exception as e:
                warnings.warn(f"ECT stationarity failed for '{col}': {e}",RuntimeWarning)
                results[col]={"adf_p":np.nan,"kpss_p":np.nan,"is_stationary":False,"passed":False}
        return results

    def _test_split_stability(self):
        """
        Split sample in half; refit VECM on each half; compare beta column spaces
        via canonical angle and compute a Likelihood Ratio test for parameter stability.
        """
        try:
            T_half=self.T//2
            data1=self.model.data.iloc[:T_half]
            data2=self.model.data.iloc[T_half:]
            k=self.k_ar_diff; r=self.r; det=self.det_order
            # guard: halves must be large enough
            min_obs=k+r+self.m+10
            if len(data1)<min_obs or len(data2)<min_obs:
                warnings.warn("Split-sample stability: insufficient obs in one half.",UserWarning)
                return {"beta_angle_deg":np.nan,"LR_stat":np.nan,"df":np.nan,"p_value":np.nan,"stable":False}
            vm1=VECMModel(data=data1,k_ar_diff=k,coint_rank=r,det_order=det,verify_i1=False);vm1.fit()
            vm2=VECMModel(data=data2,k_ar_diff=k,coint_rank=r,det_order=det,verify_i1=False);vm2.fit()
            angle=_max_canonical_angle(vm1.beta,vm2.beta)
            # LR test: 2*(llf_1 + llf_2 - llf_pooled)
            llf1=self.model._vecm_result.llf if hasattr(self.model._vecm_result,"llf") else np.nan
            llf_1=vm1._vecm_result.llf if hasattr(vm1._vecm_result,"llf") else np.nan
            llf_2=vm2._vecm_result.llf if hasattr(vm2._vecm_result,"llf") else np.nan
            if not any(np.isnan([llf1,llf_1,llf_2])):
                LR=2*((llf_1+llf_2)-llf1)
                # df = number of free parameters in split − pooled
                n_params_per_eq=self.m*k+r+1      # rough count
                df=self.m*n_params_per_eq
                df=max(df,1)
                pval=float(1-chi2.cdf(LR,df))
            else:
                LR=np.nan; df=np.nan; pval=np.nan
            stable=(angle<=15.0) and (np.isnan(pval) or pval>self.sig_level)
            return {"beta_angle_deg":float(angle),"LR_stat":float(LR) if not np.isnan(LR) else np.nan,"df":df,"p_value":pval,"stable":stable}
        except Exception as e:
            warnings.warn(f"Split-stability test failed: {e}",RuntimeWarning)
            return {"beta_angle_deg":np.nan,"LR_stat":np.nan,"df":np.nan,"p_value":np.nan,"stable":False}

    def run(self):
        """O/P:
            dict with keys: whiteness, normality, arch, ect_stationarity,split_stability, health_score       """
        self.whiteness=self._test_whiteness()
        self.normality=self._test_normality()
        self.arch=self._test_arch()
        self.ect_stationarity=self._test_ect_stationarity()
        self.split_stability=self._test_split_stability()
        # Health score: fraction of HARD tests passed
        # whiteness (1), each ECT (r checks), split_stability (1)
        hard_results=[self.whiteness["passed"]]
        hard_results+=[v["passed"] for v in self.ect_stationarity.values()]
        hard_results.append(self.split_stability["stable"])
        self.health_score=float(sum(hard_results)/len(hard_results)) if hard_results else 0.0
        return {"whiteness":self.whiteness,"normality":self.normality,"arch":self.arch,"ect_stationarity":self.ect_stationarity,"split_stability":self.split_stability,"health_score":self.health_score}

    def summary(self):
        """Structured pass/fail table with econometric interpretation per test."""
        self._require_run("summary")
        sep="="*70
        print(f"\n{sep}")
        print("  VECM DIAGNOSTICS — SUMMARY")
        print(sep)
        print(f"  Series (m)  : {self.m}  {self.cols}")
        print(f"  Rank (r*)   : {self.r}   |  k_ar_diff: {self.k_ar_diff}   |  det: {self.det_order}")
        print(f"  sig_level   : {self.sig_level*100:.0f}%")
        print(sep)
        def _flag(passed): return "✅ PASS" if passed else "❌ FAIL"
        # 1. Whiteness
        w=self.whiteness
        print(f"\n  [1] RESIDUAL WHITENESS (Portmanteau, lags={self.whiteness_lags})")
        print(f"      Stat={w['statistic']:.4f}  df={w['df']}  p={w['p_value']:.4f}  →  {_flag(w['passed'])}")
        if not w["passed"]:
            print("      ⚠️  Residual autocorrelation detected. Consider raising k_ar_diff.")
        # 2. Normality
        n=self.normality
        print(f"\n  [2] RESIDUAL NORMALITY (Doornik-Hansen)  [informational]")
        print(f"      Stat={n['statistic']:.4f}  df={n['df']}  p={n['p_value']:.4f}  →  {_flag(n['passed'])}")
        if not n["passed"]:
            print("      ℹ️  Non-normal residuals. α/β t-stats are asymptotically valid but finite-sample inference is approximate.")
        # 3. ARCH
        print(f"\n  [3] ARCH-LM TEST (lags={self.arch_lags})  [informational]")
        for col,res in self.arch.items():
            print(f"      {col:20s}  LM={res['lm_stat']:.4f}  p={res['p_value']:.4f}  →  {_flag(res['passed'])}")
            if not res["passed"]:
                print(f"           ⚠️  Volatility clustering in '{col}'. Motivates GARCH spread variance or Kalman Q-tuning.")
        # 4. ECT Stationarity
        print(f"\n  [4] ECT STATIONARITY (ADF + KPSS per ECT)")
        for col,res in self.ect_stationarity.items():
            print(f"      {col:10s}  ADF_p={res['adf_p']:.4f}  KPSS_p={res['kpss_p']:.4f}  I(0)={res['is_stationary']}  →  {_flag(res['passed'])}")
            if not res["passed"]:
                print(f"           🚨 {col} NOT confirmed stationary. Re-check rank/det_order. DO NOT trade this pair.")
        # 5. Split-sample stability
        ss=self.split_stability
        print(f"\n  [5] SPLIT-SAMPLE STABILITY  (T//2 split)")
        print(f"      β canonical angle: {ss['beta_angle_deg']:.2f}°   LR stat: {ss['LR_stat']:.4f}   p: {ss['p_value']:.4f}  →  {_flag(ss['stable'])}")
        if not ss["stable"]:
            print("      ⚠️  Parameters unstable across sample halves. Prefer rolling refit or shorter in-sample window.")
        # Overall verdict
        print(f"\n{sep}")
        print(f"  HEALTH SCORE: {self.health_score:.2%}  ({self.health_score*100:.0f}% of hard checks passed)")
        if self.health_score>=0.80:
            verdict="✅ APPROVED — VECM is trustworthy. Proceed to Spread / Kalman."
        elif self.health_score>=0.60:
            verdict="⚠️  CAUTION — Some issues found. Monitor closely before deploying."
        else:
            verdict="❌ REJECTED — VECM diagnostics failed. Do NOT use for live trading."
        print(f"  VERDICT: {verdict}")
        print(f"{sep}\n")

    def plot(self,figsize:tuple=(14,10)):
        """ 2×2 diagnostic dashboard:
        (1,1) ACF of each residual column           →  whiteness check
        (1,2) Q-Q plot of standardised residuals    →  normality check
        (2,1) Squared residuals over time           →  ARCH / volatility clustering
        (2,2) ECT panels with T//2 split line       →  stability visual     """
        self._require_run("plot")
        fig,axes=plt.subplots(2,2,figsize=figsize)
        from statsmodels.graphics.tsaplots import plot_acf as sm_acf
        ax=axes[0,0]
        colors=plt.cm.Set2(np.linspace(0,0.8,self.m))
        for i,col in enumerate(self.cols):
            sm_acf(self.resid[:,i],lags=min(self.whiteness_lags,len(self.resid)//5),
                   ax=ax,alpha=0.05,color=colors[i],label=col,zero=False)
        ax.set_title("ACF of Residuals  (Whiteness)",fontweight="bold")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        p_w=self.whiteness["p_value"]
        ax.set_title(f"ACF of Residuals  (Whiteness p={p_w:.4f}  {'✅' if self.whiteness['passed'] else '❌'})",fontweight="bold")
        # ============= Panel (0,1): Q-Q plot =======================
        ax=axes[0,1]
        from scipy import stats
        for i,col in enumerate(self.cols):
            std=self.resid[:,i].std()
            e_std=self.resid[:,i]/std if std>0 else self.resid[:,i]
            (osm,osr),_=stats.probplot(e_std,dist="norm",fit=True)
            ax.scatter(osm,osr,s=12,alpha=0.6,color=colors[i],label=col)
        lim=ax.get_xlim(); ax.plot(lim,lim,color="red",linewidth=1.2,linestyle="--",label="45° line")
        ax.set_title(f"Q-Q Plot (Normality p={self.normality['p_value']:.4f}  {'✅' if self.normality['passed'] else '❌'})",fontweight="bold")
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # ============= Panel (1,0): Squared residuals ========================
        ax=axes[1,0]
        idx=range(len(self.resid))
        for i,col in enumerate(self.cols):
            ax.plot(idx,self.resid[:,i]**2,linewidth=0.7,alpha=0.7,color=colors[i],label=col)
        ax.set_title("Squared Residuals  (ARCH / Vol. Clustering)",fontweight="bold")
        ax.set_xlabel("Observation"); ax.set_ylabel("ε²"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # ============== Panel (1,1): ECT with split line =====================
        ax=axes[1,1]
        T_half=self.T//2
        ect_colors=plt.cm.tab10(np.linspace(0,0.5,self.r))
        for j in range(self.r):
            col=f"ECT_{j+1}"
            s=self.ect[col]
            ax.plot(s.index,s.values,linewidth=1.0,color=ect_colors[j],label=col)
        if isinstance(self.ect.index,pd.DatetimeIndex):
            split_date=self.ect.index[T_half]
            ax.axvline(split_date,color="red",linestyle="--",linewidth=1.5,label=f"T/2 split ({split_date.date()})")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.xaxis.get_majorticklabels(),rotation=30)
        else:
            ax.axvline(T_half,color="red",linestyle="--",linewidth=1.5,label="T/2 split")
        ax.axhline(0,color="black",linewidth=0.8,linestyle="-")
        angle=self.split_stability["beta_angle_deg"]
        ax.set_title(f"ECT Panels  (β angle={angle:.1f}°  {'✅' if self.split_stability['stable'] else '❌'})",fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("ECT value"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        fig.suptitle(f"VECM Diagnostics  |  Health Score: {self.health_score:.1%}  |  {self.cols}",fontsize=13,fontweight="bold")
        plt.tight_layout()
        plt.show()

    def get_health_score(self)->float:
        """Quick numeric filter for automated pair screening in backtest/engine.py."""
        self._require_run("get_health_score")
        return self.health_score

    def __repr__(self)->str:
        hs=f"health_score={self.health_score:.2%}" if self.health_score is not None else "not run — call run()"
        return (f"VECMDiagnostics(m={self.m}, r={self.r}, T={self.T}, "
                f"sig_level={self.sig_level}, {hs})")

if __name__=="__main__":
    import yfinance as yf
    from Time_Series_Analysis.Cointegration_Tests.johansen import JohansenTest
    tickers=["RELIANCE.NS","TCS.NS","INFY.NS"]
    raw=yf.download(tickers,period='6y',interval='1d',auto_adjust=True)["Close"].dropna()
    log_prices=np.log(raw)
    print(f"Data: {log_prices.shape}")
    jt=JohansenTest(data=log_prices,det_order="restricted_constant",k_ar_diff=None,sig_level=1,verify_i1=False)
    jt.run()
    vm=VECMModel.from_johansen(jt,verify_i1=False)
    vm.fit()
    diag=VECMDiagnostics.from_vecm(vm,whiteness_lags=10,arch_lags=5,sig_level=0.05)
    diag.run()
    diag.summary()
    print(f"\nHealth Score: {diag.get_health_score():.2%}")
    diag.plot()