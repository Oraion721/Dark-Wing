"""
Method: Ornstein-Uhlenbeck (OU) regression
    ΔS_t = κ(μ - S_{t-1}) + σ·dW_t
    Discretised for OLS:
        ΔS_t = a + b·S_{t-1} + ε_t
    where b = e^{-κΔt} - 1  ≈ -κΔt  for small Δt.
    Half-life: τ = -ln(2) / ln(1 + b) = ln(2) / κ
Output: half_life_days : float  — passed directly to zscore.py (window) and Kalman_Filter (governs Q = σ²/τ heuristic)        """
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from typing import Optional

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
for _p in [_FM_DIR,_TSA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

class HalfLife:
    """
    Args:
        ~ spread:pd.Series: Stationary spread S_t from SpreadBuilder.spread_series().
        ~ freq:str: 'daily' (default) | 'weekly' | 'monthly'. Used only for labelling.
        ~ clip_hl:tuple (lo, hi): Clip half-life to this range in observations. Default (1, 500). Prevents unreasonable values from bad data. """
    def __init__(self,spread:pd.Series,freq:str="daily",clip_hl:tuple=(1,500)):
        if not isinstance(spread,pd.Series):
            raise TypeError("spread must be a pd.Series.")
        self._spread=spread.dropna()
        if len(self._spread)<20:
            raise ValueError(f"Spread too short ({len(self._spread)} obs). Need ≥ 20.")
        self.freq=freq
        self.clip_lo,self.clip_hi=clip_hl
        # results (set in estimate())
        self.half_life_days:Optional[float]=None
        self.half_life_ci_95:Optional[tuple]=None  # (lo, hi) 95% CI via delta method
        self.kappa:Optional[float]=None          # mean-reversion speed
        self.mu_ou:Optional[float]=None          # long-run mean
        self.b_coef:Optional[float]=None         # OLS slope ≈ -κΔt
        self.a_coef:Optional[float]=None         # OLS intercept = κ·μ·Δt
        self.r_squared:Optional[float]=None
        self.p_value:Optional[float]=None        # p-val on b_coef (t-test)
        self.std_err:Optional[float]=None
        self._ou_resid:Optional[np.ndarray]=None

    def estimate(self)->float:
        """Run OU OLS regression and compute half-life.
        Regression:
            ΔS_t = a + b·S_{t-1} + ε_t
        OLS via scipy.stats.linregress (robust, fast).
        O/P: half_life_days : float
        Also sets self.half_life_ci_95 = (lo, hi) — 95% delta-method confidence interval on τ.

        Delta-method derivation:
            τ = -ln(2) / ln(1 + b)   [exact formula]
            dτ/db = ln(2) / ((1+b) · ln(1+b)²)
            Var(τ̂) ≈ (dτ/db)² · Var(b̂)   [where Var(b̂) = std_err²]
            CI = τ̂ ± 1.96 · sqrt(Var(τ̂))
        """
        s=self._spread.values
        S_lag=s[:-1]; dS=np.diff(s)
        result=stats.linregress(S_lag,dS)
        self.b_coef=float(result.slope)
        self.a_coef=float(result.intercept)
        self.r_squared=float(result.rvalue**2)
        self.p_value=float(result.pvalue)
        self.std_err=float(result.stderr)
        if self.b_coef>=0:
            warnings.warn(f"OLS slope b={self.b_coef:.6f} >= 0. Spread may NOT be mean-reverting.\nHalf-life estimate is unreliable.",UserWarning)
            self.kappa=0.0; self.mu_ou=0.0
            self.half_life_days=float(self.clip_hi)
            self.half_life_ci_95=None
        else:
            self.kappa=-np.log(1.0+self.b_coef)
            self.mu_ou=-self.a_coef/self.b_coef
            raw_hl=np.log(2.0)/self.kappa
            self.half_life_days=float(np.clip(raw_hl,self.clip_lo,self.clip_hi))
            # Delta-method 95% CI on τ:
            #   dτ/db = ln(2) / ((1+b) * ln(1+b)^2)
            #   Var(τ) ≈ (dτ/db)^2 * std_err^2
            ln1b=np.log(1.0+self.b_coef)  # negative since b < 0
            dtau_db=np.log(2.0)/((1.0+self.b_coef)*ln1b**2)
            var_tau=(dtau_db**2)*(self.std_err**2)
            ci_half=1.96*np.sqrt(var_tau)
            ci_lo=max(float(raw_hl-ci_half),self.clip_lo)
            ci_hi=min(float(raw_hl+ci_half),self.clip_hi)
            self.half_life_ci_95=(ci_lo,ci_hi)
        self._ou_resid=dS-(self.a_coef+self.b_coef*S_lag)
        ci_str=(f"  [{self.half_life_ci_95[0]:.1f}, {self.half_life_ci_95[1]:.1f}] 95% CI"
                if self.half_life_ci_95 else "  [CI unavailable — b >= 0]")
        print(f"[HalfLife] b={self.b_coef:.6f}  kappa={self.kappa:.6f}  "
              f"mu_OU={self.mu_ou:.6f}  tau={self.half_life_days:.2f} obs{ci_str}")
        return self.half_life_days


    def rolling(self,window:int=252)->pd.Series:
        """ Args: 
                ~ window:int — look-back observations (default 252 trading days).
         O/P:
            pd.Series of rolling half-life estimates.       """
        s=self._spread
        hl_list=[]
        idx_list=[]
        for end in range(window,len(s)+1):
            sub=s.iloc[end-window:end]
            try:
                hl_sub=HalfLife(sub,clip_hl=(self.clip_lo,self.clip_hi))
                hl_list.append(hl_sub.estimate())
            except Exception:
                hl_list.append(np.nan)
            idx_list.append(s.index[end-1])
        return pd.Series(hl_list,index=idx_list,name="rolling_half_life")

    def _require_estimated(self,method:str):
        if self.half_life_days is None:
            raise RuntimeError(f"Call estimate() before calling {method}().")

    def get_half_life(self)->float:
        """Return the scalar half-life estimate (passed to zscore.py as window)."""
        self._require_estimated("get_half_life"); return self.half_life_days

    def get_kappa(self)->float:
        """Return the mean-reversion speed κ."""
        self._require_estimated("get_kappa"); return self.kappa

    def get_mu_ou(self)->float:
        """Return the long-run mean μ of the OU process."""
        self._require_estimated("get_mu_ou"); return self.mu_ou

    def get_ou_residuals(self)->np.ndarray:
        """Return OLS residuals from ΔS = a + b·S_{t-1} + ε."""
        self._require_estimated("get_ou_residuals"); return self._ou_resid

    def summary(self):
        self._require_estimated("summary")
        sep="="*58
        sig="***" if self.p_value<0.01 else ("**" if self.p_value<0.05 else ("*" if self.p_value<0.10 else ""))
        print(f"\n{sep}")
        print("  HALF-LIFE ESTIMATOR — SUMMARY")
        print(sep)
        print(f"  Observations          : {len(self._spread)}")
        print(f"  OLS slope  (b)        : {self.b_coef:.6f}  {sig}")
        print(f"  OLS intercept (a)     : {self.a_coef:.6f}")
        print(f"  p-value (b ≠ 0)       : {self.p_value:.6f}")
        print(f"  R-squared             : {self.r_squared:.6f}")
        print(f"  Mean-rev. speed  (κ)  : {self.kappa:.6f} per {self.freq} obs")
        print(f"  Long-run mean   (μ_OU): {self.mu_ou:.6f}")
        print(f"  HALF-LIFE  (τ)        : {self.half_life_days:.2f} {self.freq} obs")
        if self.half_life_ci_95 is not None:
            print(f"  95% CI on τ           : [{self.half_life_ci_95[0]:.1f},  {self.half_life_ci_95[1]:.1f}] obs  (delta method)")
        else:
            print("  95% CI on τ           : unavailable (b >= 0, non-mean-reverting)")
        if self.half_life_days>=self.clip_hi*0.9:
            print("  ⚠️  Half-life near upper clip. Spread barely mean-reverts.")
        elif self.half_life_days<5:
            print("  ⚠️  Half-life < 5 obs. Spread reverts very fast — check data.")
        print(sep+"\n")

    def plot(self,figsize:tuple=(13,5)):
        """Two-panel plot: spread with OU bands, and ΔS vs S_{t-1} regression."""
        self._require_estimated("plot")
        fig,axes=plt.subplots(1,2,figsize=figsize)
        # Panel 1: spread with mean and ±1σ_OU bands
        ax=axes[0]
        s=self._spread
        mu=self.mu_ou; sig=np.std(self._ou_resid) if self._ou_resid is not None else s.std()
        ax.plot(s.index,s.values,color="steelblue",linewidth=0.9,label="Spread S_t")
        ax.axhline(mu,color="black",linestyle="--",linewidth=1.0,label=f"μ_OU={mu:.4f}")
        ax.axhline(mu+sig,color="tomato",linestyle=":",linewidth=0.9,label="+1σ")
        ax.axhline(mu-sig,color="tomato",linestyle=":",linewidth=0.9,label="-1σ")
        ax.fill_between(s.index,mu-sig,mu+sig,alpha=0.08,color="tomato")
        ax.set_title(f"Spread Series  (τ = {self.half_life_days:.1f} obs)",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        if isinstance(s.index,pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); plt.setp(ax.xaxis.get_majorticklabels(),rotation=25)
        # Panel 2: ΔS vs S_{t-1} scatter + OLS line
        ax=axes[1]
        S_lag=s.values[:-1]; dS=np.diff(s.values)
        ax.scatter(S_lag,dS,s=6,alpha=0.4,color="steelblue")
        xs=np.linspace(S_lag.min(),S_lag.max(),200)
        ax.plot(xs,self.a_coef+self.b_coef*xs,color="crimson",linewidth=1.5,label=f"ΔS={self.a_coef:.4f}+{self.b_coef:.4f}·S_{{t-1}}")
        ax.set_xlabel("$S_{t-1}$"); ax.set_ylabel("$\\Delta S_t$")
        ax.set_title(f"OU Regression  (R²={self.r_squared:.4f})",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        fig.suptitle("Half-Life Estimation — Ornstein-Uhlenbeck OLS",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def __repr__(self)->str:
        if self.half_life_days is not None:
            return f"HalfLife(τ={self.half_life_days:.2f} obs, κ={self.kappa:.6f}, p={self.p_value:.4f})"
        return "HalfLife(not estimated — call estimate())"

if __name__=="__main__":
    import yfinance as yf
    from spread_builder import SpreadBuilder
    raw=yf.download(["RELIANCE.NS","TCS.NS"],start="2021-01-01",end="2024-01-01",auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["RELIANCE.NS"],log_px["TCS.NS"]
    sb=SpreadBuilder(y=y,x=x,mode="ols"); sb.build()
    hl=HalfLife(sb.spread_series())
    hl.estimate(); hl.summary(); hl.plot()