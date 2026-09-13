"""
Purpose:
    Implements the full general-form state-space model from Notion notes:
    State-Space Model (General Form):
        State Equation:       S_t = T_t · S_{t-1} + R_t · η_t,   η_t ~ N(0, Q_t)
        Observation Equation: Y_t = Z_t · S_t   + ε_t,            ε_t ~ N(0, H_t)
    For pairs trading (univariate β):
        S_t  = β_t        (hidden state = hedge ratio)
        T_t  = 1          (random walk state transition)
        Z_t  = X_t        (observation matrix = independent price)
        Y_t  = price_Y_t  (observation = dependent price)
        Q    = process noise variance   (how fast β evolves)
        R/H  = measurement noise variance (spread/price noise)

    Predict Step:
        β̂_{t|t-1}  = T · β̂_{t-1|t-1}
        P_{t|t-1}   = T · P_{t-1|t-1} · T' + R·Q·R'

    Update (Correction) Step:
        ε_t  = Y_t - Z_t · β̂_{t|t-1}               (innovation)
        F_t  = Z_t · P_{t|t-1} · Z_t' + H           (innovation covariance)
        G_t  = P_{t|t-1} · Z_t' · F_t^{-1}          (Kalman Gain)
        β̂_{t|t} = β̂_{t|t-1} + G_t · ε_t            (state update)
        P_{t|t} = (I - G_t · Z_t) · P_{t|t-1}       (covariance update)

    Log-likelihood per step:
        ℓ_t = -½ [ log(2π·F_t) + ε_t²/F_t ]

Pipeline Position:
    STEP 5 → Spreads/ (SpreadBuilder, HalfLife, ZScore)
    STEP 6 → THIS FILE  (KalmanFilterPairs)
    STEP 6.1 → smoother.py (KalmanSmoother)
    STEP 6.2 → dynamic_beta.py (DynamicBeta)
    STEP 7  → Forecasting (VECMForecaster, KalmanForecaster)
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from dataclasses import dataclass,field
from typing import Optional,List,Tuple,Dict

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TSA_DIR=os.path.dirname(_THIS_DIR)
_FM_DIR=os.path.dirname(_TSA_DIR)
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KalmanState:
    """Immutable record of all Kalman quantities at a single timestep."""
    t:int
    # Prediction quantities
    beta_pred:float        # β̂_{t|t-1}
    P_pred:float           # P_{t|t-1}
    # Correction quantities
    innovation:float       # ε_t = Y_t - Z_t·β̂_{t|t-1}
    F:float                # F_t = innovation variance
    kalman_gain:float      # G_t = Kalman gain
    # Updated quantities
    beta:float             # β̂_{t|t}
    P:float                # P_{t|t}
    # Derived
    spread:float           # S_t = Y_t - β̂_{t|t}·X_t
    log_lik:float          # ℓ_t = log-likelihood contribution
    x:float                # X_t (independent price)
    y:float                # Y_t (dependent price)

# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilterPairs:
    """
    Kalman Filter for dynamic hedge ratio estimation in pairs trading.

    State-space formulation (univariate β):
        β_t = β_{t-1} + η_t,     η_t ~ N(0, Q)   [random walk state]
        Y_t = β_t · X_t + ε_t,   ε_t ~ N(0, R)   [observation]

    Parameters
    ----------
    Q          : float — Process noise variance. Controls β evolution speed.
                         Small Q → stable β; Large Q → fast-adapting β.
    R          : float — Measurement noise variance. Spread/price noise.
    beta_init  : float — Initial hedge ratio (prior mean). Default 1.0.
    P_init     : float — Initial state covariance (prior uncertainty).
                         Large P_init → initially trust data more. Default 1000.
    """
    def __init__(self,Q:float=1e-5,R:float=1e-3,beta_init:float=1.0,P_init:float=1000.0):
        if Q<=0 or R<=0:
            raise ValueError(f"Q and R must be positive. Got Q={Q}, R={R}.")
        if P_init<=0:
            raise ValueError(f"P_init must be positive. Got {P_init}.")
        self.Q=float(Q); self.R=float(R)
        self.beta_init=float(beta_init); self.P_init=float(P_init)
        # mutable state
        self._beta=float(beta_init); self._P=float(P_init)
        self.states:List[KalmanState]=[]
        self._log_lik_sum:float=0.0
        self._fitted:bool=False

    # ── predict step ─────────────────────────────────────────────────────────
    def predict(self)->Tuple[float,float]:
        """
        Prior step: project state forward by one timestep.
            β̂_{t|t-1}  = β̂_{t-1|t-1}         (T = 1, random walk)
            P_{t|t-1}   = P_{t-1|t-1} + Q      (uncertainty grows by Q)

        Returns
        -------
        (beta_pred, P_pred)
        """
        beta_pred=self._beta            # T=1: no state decay
        P_pred=self._P+self.Q           # Uncertainty grows each step
        return beta_pred,P_pred

    # ── update step ──────────────────────────────────────────────────────────
    def update(self,y:float,x:float,t:int=0)->KalmanState:
        """
        Posterior step: incorporate new observation (Y_t, X_t).

        Equations:
            ε_t  = Y_t - x · β̂_{t|t-1}          [innovation / prediction error]
            F_t  = x² · P_{t|t-1} + R             [innovation variance]
            G_t  = P_{t|t-1} · x / F_t            [Kalman Gain]
            β̂_{t|t}  = β̂_{t|t-1} + G_t · ε_t   [state update]
            P_{t|t}   = (1 - G_t · x) · P_{t|t-1} [covariance update]
            S_t  = Y_t - β̂_{t|t} · X_t           [spread]
            ℓ_t  = -½[log(2π·F_t) + ε_t²/F_t]   [log-likelihood]

        Parameters
        ----------
        y : float — Dependent price Y_t.
        x : float — Independent price X_t.
        t : int   — Timestep index (for record).

        Returns
        -------
        KalmanState — complete record of this timestep.
        """
        if x==0.0:
            raise ValueError(f"X_t cannot be 0 at timestep {t}. Division by zero in Kalman gain.")
        beta_pred,P_pred=self.predict()
        # Innovation
        innovation=y-beta_pred*x
        # Innovation covariance (scalar for univariate case: F = x²P + R)
        F=(x**2)*P_pred+self.R
        if F<=0:
            raise ValueError(f"Innovation variance F={F:.6e} is non-positive at t={t}. Check Q,R values.")
        # Kalman Gain
        G=P_pred*x/F
        # State update
        self._beta=beta_pred+G*innovation
        # Covariance update (Joseph form for numerical stability)
        self._P=(1.0-G*x)*P_pred
        # Guard: covariance must stay positive
        if self._P<=0:
            warnings.warn(f"Covariance P={self._P:.6e} collapsed at t={t}. Resetting to P_init.",RuntimeWarning)
            self._P=self.P_init
        # Spread using updated β
        spread=y-self._beta*x
        # Log-likelihood contribution: ℓ_t = -½[log(2πF) + ε²/F]
        log_lik=-0.5*(np.log(2*np.pi*F)+(innovation**2)/F)
        self._log_lik_sum+=log_lik
        state=KalmanState(t=t,beta_pred=beta_pred,P_pred=P_pred,
                          innovation=innovation,F=F,kalman_gain=G,
                          beta=self._beta,P=self._P,spread=spread,
                          log_lik=log_lik,x=x,y=y)
        self.states.append(state)
        return state

    # ── full filter pass ─────────────────────────────────────────────────────
    def filter(self,y_series:pd.Series,x_series:pd.Series)->pd.DataFrame:
        """
        Run the full Kalman filter over historical data.

        Parameters
        ----------
        y_series : pd.Series — Dependent (Y) log-price series.
        x_series : pd.Series — Independent (X) log-price series.

        Returns
        -------
        pd.DataFrame with columns: [beta, P, beta_pred, P_pred, innovation,
                                    F, kalman_gain, spread, log_lik, x, y]
        indexed by the shared DatetimeIndex.
        """
        if not isinstance(y_series,pd.Series) or not isinstance(x_series,pd.Series):
            raise TypeError("y_series and x_series must be pd.Series.")
        aligned=pd.concat([y_series.rename("y"),x_series.rename("x")],axis=1).dropna()
        if len(aligned)<5:
            raise ValueError(f"Insufficient aligned data: {len(aligned)} rows. Need ≥ 5.")
        self.reset()
        for t,(idx,row) in enumerate(aligned.iterrows()):
            self.update(y=float(row["y"]),x=float(row["x"]),t=t)
        self._fitted=True
        self._y_index=aligned.index
        return self.state_dataframe()

    # ── outputs ───────────────────────────────────────────────────────────────
    def state_dataframe(self)->pd.DataFrame:
        """Convert stored KalmanState list to a tidy DataFrame."""
        self._require_fitted("state_dataframe")
        rows=[{"t":s.t,"beta":s.beta,"P":s.P,"beta_pred":s.beta_pred,
               "P_pred":s.P_pred,"innovation":s.innovation,"F":s.F,
               "kalman_gain":s.kalman_gain,"spread":s.spread,
               "log_lik":s.log_lik,"x":s.x,"y":s.y}
              for s in self.states]
        df=pd.DataFrame(rows)
        if hasattr(self,"_y_index"):
            df.index=self._y_index
        return df

    def log_likelihood(self)->float:
        """Total log-likelihood: ℓ = Σ_t ℓ_t (for Q/R optimisation)."""
        return self._log_lik_sum

    def current_beta(self)->float:
        """Latest filtered β_{T|T} — use for live position sizing."""
        self._require_fitted("current_beta"); return self._beta

    def current_P(self)->float:
        """Latest filtered P_{T|T} — uncertainty in current β."""
        self._require_fitted("current_P"); return self._P

    def spread_series(self)->pd.Series:
        """Dynamic spread S_t = Y_t - β̂_{t|t}·X_t as pd.Series."""
        df=self.state_dataframe()
        return df["spread"].rename("kalman_spread")

    # ── Q/R optimisation ──────────────────────────────────────────────────────
    def optimize_QR(self,y_series:pd.Series,x_series:pd.Series,
                    Q_bounds:Tuple[float,float]=(1e-8,1e-1),
                    R_bounds:Tuple[float,float]=(1e-6,1.0),
                    method:str="L-BFGS-B")->dict:
        """
        Find optimal Q and R via Maximum Likelihood Estimation.
        Minimises -ℓ(Q,R) = -Σ_t log p(Y_t | Y_{1:t-1}).

        Parameters
        ----------
        y_series, x_series : pd.Series — historical data.
        Q_bounds, R_bounds : (lo, hi) — search bounds for L-BFGS-B.
        method : str — scipy.optimize method.

        Returns
        -------
        dict with keys: Q_opt, R_opt, log_likelihood, converged, n_iter.
        """
        y=y_series.values; x=x_series.values
        aligned=pd.concat([y_series.rename("y"),x_series.rename("x")],axis=1).dropna()
        y_arr=aligned["y"].values; x_arr=aligned["x"].values
        def neg_loglik(params):
            Q,R=params
            if Q<=0 or R<=0: return 1e12
            tmp=KalmanFilterPairs(Q=Q,R=R,beta_init=self.beta_init,P_init=self.P_init)
            try:
                for yi,xi in zip(y_arr,x_arr):
                    tmp.update(y=float(yi),x=float(xi))
                return -tmp.log_likelihood()
            except Exception:
                return 1e12
        x0=[self.Q,self.R]
        bounds=[Q_bounds,R_bounds]
        res=minimize(neg_loglik,x0,method=method,bounds=bounds,
                     options={"maxiter":200,"ftol":1e-10})
        if res.success:
            self.Q,self.R=float(res.x[0]),float(res.x[1])
            print(f"[KalmanFilterPairs] Optimised  Q={self.Q:.2e}  R={self.R:.2e}  ℓ={-res.fun:.4f}")
        else:
            warnings.warn(f"Q/R optimisation did not converge: {res.message}",RuntimeWarning)
        return {"Q_opt":res.x[0],"R_opt":res.x[1],"log_likelihood":-res.fun,
                "converged":res.success,"n_iter":res.nit}

    # ── summary & plot ────────────────────────────────────────────────────────
    def _require_fitted(self,method:str):
        if not self._fitted:
            raise RuntimeError(f"Call filter() before calling {method}().")

    def summary(self):
        self._require_fitted("summary")
        df=self.state_dataframe()
        sep="="*60
        print(f"\n{sep}")
        print("  KALMAN FILTER — SUMMARY")
        print(sep)
        print(f"  Observations      : {len(df)}")
        print(f"  Q (process noise) : {self.Q:.4e}")
        print(f"  R (meas. noise)   : {self.R:.4e}")
        print(f"  β_init / P_init   : {self.beta_init} / {self.P_init}")
        print(f"  Final β̂ (T|T)    : {df['beta'].iloc[-1]:.6f}")
        print(f"  Final P (T|T)     : {df['P'].iloc[-1]:.6e}")
        print(f"  β range           : [{df['beta'].min():.4f}, {df['beta'].max():.4f}]")
        print(f"  Spread mean / std : {df['spread'].mean():.6f} / {df['spread'].std():.6f}")
        print(f"  Total log-lik ℓ   : {self._log_lik_sum:.4f}")
        print(sep+"\n")

    def plot(self,figsize:tuple=(14,9)):
        """Three-panel plot: β over time, spread, and Kalman gain."""
        self._require_fitted("plot")
        df=self.state_dataframe()
        fig,axes=plt.subplots(3,1,figsize=figsize,sharex=True)
        idx=df.index if hasattr(self,"_y_index") else df["t"]
        # Panel 1: Dynamic β
        ax=axes[0]
        ax.plot(idx,df["beta"],color="steelblue",linewidth=1.1,label="β̂_{t|t}")
        ax.fill_between(idx,df["beta"]-2*np.sqrt(df["P"]),df["beta"]+2*np.sqrt(df["P"]),
                        alpha=0.18,color="steelblue",label="±2σ band")
        ax.set_title("Dynamic Hedge Ratio β̂_t",fontweight="bold")
        ax.set_ylabel("β"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # Panel 2: Kalman Spread
        ax=axes[1]
        s=df["spread"]
        ax.plot(idx,s,color="darkorange",linewidth=0.9,label="Kalman Spread")
        ax.axhline(s.mean(),color="black",linestyle="--",linewidth=0.9)
        ax.axhline(s.mean()+2*s.std(),color="tomato",linestyle=":",linewidth=0.9)
        ax.axhline(s.mean()-2*s.std(),color="tomato",linestyle=":",linewidth=0.9)
        ax.fill_between(idx,s.mean()-2*s.std(),s.mean()+2*s.std(),alpha=0.07,color="tomato")
        ax.set_title("Kalman Spread S_t = Y_t − β̂_t·X_t",fontweight="bold")
        ax.set_ylabel("Spread"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        # Panel 3: Kalman Gain
        ax=axes[2]
        ax.plot(idx,df["kalman_gain"],color="mediumseagreen",linewidth=0.9,label="G_t")
        ax.set_title("Kalman Gain G_t (Trust in New Data)",fontweight="bold")
        ax.set_ylabel("G_t"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        if hasattr(self,"_y_index") and isinstance(self._y_index,pd.DatetimeIndex):
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(axes[-1].xaxis.get_majorticklabels(),rotation=25)
        fig.suptitle("Kalman Filter — Dynamic Pairs Trading",fontsize=12,fontweight="bold")
        plt.tight_layout(); plt.show()

    def reset(self):
        """Reset filter to initial state (for new pair or re-optimisation)."""
        self._beta=self.beta_init; self._P=self.P_init
        self.states=[]; self._log_lik_sum=0.0; self._fitted=False

    def __repr__(self)->str:
        status="fitted" if self._fitted else "not fitted"
        return (f"KalmanFilterPairs(Q={self.Q:.2e}, R={self.R:.2e}, "
                f"β_init={self.beta_init}, P_init={self.P_init}, {status})")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import yfinance as yf
    raw=yf.download(["RELIANCE.NS","TCS.NS"],start="2021-01-01",end="2024-01-01",auto_adjust=True)["Close"].dropna()
    log_px=np.log(raw)
    y,x=log_px["RELIANCE.NS"],log_px["TCS.NS"]
    kf=KalmanFilterPairs(Q=1e-5,R=1e-3)
    print("Optimising Q and R via MLE...")
    kf.optimize_QR(y,x)
    states_df=kf.filter(y,x)
    kf.summary()
    print(f"\nFirst 5 states:\n{states_df[['beta','P','spread','kalman_gain']].head()}")
    kf.plot()
