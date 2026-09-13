"""
Methods :
    fixed_fractional  : Dollar_Y = f*C,  N_Y = floor(Dollar_Y/P_Y)
    volatility_scaled : N_Y = floor(sigma_target / (P_Y*sigma_Y))  [risk parity]
    kelly             : f* = mu_Z/sigma_Z^2,  f_kelly = f*/kelly_fraction
    (all methods)     : N_X = round(|beta_t|*N_Y)  [dollar-neutral hedge]       """
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional,Dict,Tuple

_THIS_DIR   =os.path.dirname(os.path.abspath(__file__))
_TRADE_DIR  =os.path.dirname(_THIS_DIR)
_ROOT_DIR   =os.path.dirname(_TRADE_DIR)
_FM_DIR     =os.path.join(_ROOT_DIR,"Financial_Mathematics")
_TSA_DIR    =os.path.join(_FM_DIR,"Time_Series_Analysis")
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
_KALMAN_DIR =os.path.join(_TSA_DIR,"Kalman_Filter")
_SIGNALS_DIR=os.path.join(_TRADE_DIR,"Signals")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_KALMAN_DIR,_SIGNALS_DIR,_THIS_DIR]:
    if _p not in sys.path: sys.path.insert(0,_p)

# ─────────────────────────────────────────────────────────────────────────────
class PositionSizer:
    _VALID_METHODS=("fixed_fractional","volatility_scaled","kelly")

    def __init__(self,capital:float=1_000_000.0,method:str="fixed_fractional",
                 max_trade_fraction:float=0.10,max_exposure:float=0.25,
                 fixed_fraction:float=0.05,sigma_target:float=5_000.0,
                 vol_window:int=20,kelly_fraction:float=4.0,kelly_window:int=60,min_shares:int=1):
        if method not in self._VALID_METHODS: raise ValueError(f"method must be one of {self._VALID_METHODS}.")
        if capital<=0: raise ValueError("capital must be > 0.")
        if not 0<max_trade_fraction<=1: raise ValueError("max_trade_fraction in (0,1].")
        if not 0<max_exposure<=2: raise ValueError("max_exposure in (0,2].")
        if kelly_fraction<1: raise ValueError("kelly_fraction >= 1.")
        self.capital=float(capital); self.method=method
        self.max_trade_fraction=float(max_trade_fraction); self.max_exposure=float(max_exposure)
        self.fixed_fraction=float(fixed_fraction); self.sigma_target=float(sigma_target)
        self.vol_window=int(vol_window); self.kelly_fraction=float(kelly_fraction)
        self.kelly_window=int(kelly_window); self.min_shares=int(min_shares)
        self._history:list=[]

    def compute(self,signal:int,beta_t:float,price_y:float,price_x:float,
                zscore_series:Optional[pd.Series]=None,returns_y:Optional[pd.Series]=None,
                returns_x:Optional[pd.Series]=None)->Dict:
        """ Compute N_Y, N_X for both legs. signal: +1=LONG, -1=SHORT, 0/±2=FLAT. """
        if price_y<=0 or price_x<=0: raise ValueError(f"Prices must be positive. Got P_Y={price_y}, P_X={price_x}.")
        if abs(beta_t)<1e-6: warnings.warn("beta_t near zero. Using beta_t=1.0."); beta_t=1.0
        if signal==0 or abs(signal)==2:
            result=self._zero_sizing(signal,beta_t); self._history.append(result); return result
        if self.method=="fixed_fractional":
            n_y=self._fixed_fractional(price_y)
        elif self.method=="volatility_scaled":
            if returns_y is None or returns_x is None:
                warnings.warn("volatility_scaled needs returns_y/returns_x. Falling back to fixed_fractional.")
                n_y=self._fixed_fractional(price_y)
            else: n_y=self._volatility_scaled(price_y,returns_y)
        elif self.method=="kelly":
            if zscore_series is None:
                warnings.warn("kelly needs zscore_series. Falling back to fixed_fractional.")
                n_y=self._fixed_fractional(price_y)
            else: n_y=self._kelly(price_y,zscore_series)
        n_x=max(int(round(abs(beta_t)*n_y)),self.min_shares if n_y>0 else 0)
        n_y,n_x=self._apply_capital_cap(n_y,n_x,price_y,price_x)
        dollar_y=n_y*price_y; dollar_x=n_x*price_x; total=dollar_y+dollar_x
        dir_y,dir_x=("BUY","SELL") if signal==1 else ("SELL","BUY")
        result={"N_Y":n_y,"N_X":n_x,"dollar_Y":round(dollar_y,2),"dollar_X":round(dollar_x,2),
                "total_dollar":round(total,2),"signal":signal,"direction_Y":dir_y,"direction_X":dir_x,
                "method_used":self.method,"beta_t":round(float(beta_t),6),"leverage":round(total/self.capital,4)}
        self._history.append(result)
        return result

    def _fixed_fractional(self,price_y:float)->int:
        dollar_y=min(self.fixed_fraction*self.capital,self.max_trade_fraction*self.capital)
        return max(int(dollar_y/price_y),self.min_shares)

    def _volatility_scaled(self,price_y:float,returns_y:pd.Series)->int:
        recent=returns_y.dropna().iloc[-self.vol_window:]
        if len(recent)<5:
            warnings.warn("Not enough return history for vol-scaled sizing."); return self._fixed_fractional(price_y)
        sigma_y=float(recent.std(ddof=1))
        if sigma_y<1e-8:
            warnings.warn("sigma_Y near zero. Falling back to fixed_fractional."); return self._fixed_fractional(price_y)
        n_y=max(int(self.sigma_target/(price_y*sigma_y)),self.min_shares)
        return min(n_y,int(self.max_trade_fraction*self.capital/price_y))

    def _kelly(self,price_y:float,zscore_series:pd.Series)->int:
        """Kelly Criterion for pairs trading.

        Full Kelly: f* = μ / σ²   (maximises geometric mean of wealth)
        Fractional Kelly: f_kelly = f* / kelly_fraction  (reduces variance in outcomes)

        For a mean-reverting Z-score strategy, a bar's realised return is approximately:
            r_t ≈ direction_t * |ΔZ_t| * (spread_tick_value / capital)
        However, since we don't have realised PnL series here, we approximate using the
        Z-score series as a clean signal-strength proxy:
            μ_edge  = mean of |Z_t| for |Z_t| > entry threshold  [average edge at entry]
            σ²_edge = variance of (Z_t differences) at signal bars  [outcome uncertainty]
        This is a conservative Kelly estimate — it sizes proportionally to edge strength.

        Note: If you have historical trade returns, pass them as zscore_series for exact Kelly.
        """
        z=zscore_series.dropna().iloc[-self.kelly_window:]
        if len(z)<10:
            warnings.warn("Not enough Z history for Kelly sizing."); return self._fixed_fractional(price_y)
        # Compute per-step Z-score changes — proxy for per-bar P&L distribution
        dz=z.diff().dropna()
        if len(dz)<5:
            return self._fixed_fractional(price_y)
        mu_edge=float(dz.abs().mean())   # average magnitude of spread move per bar
        sigma2_edge=float(dz.var(ddof=1))  # variance of spread moves
        if sigma2_edge<1e-8:
            warnings.warn("Z-score variance near zero."); return self._fixed_fractional(price_y)
        # Kelly fraction: f* = μ/σ² — divide by kelly_fraction for fractional Kelly
        f_kelly=float(np.clip(mu_edge/sigma2_edge/self.kelly_fraction,0.0,self.max_trade_fraction))
        return max(int(f_kelly*self.capital/price_y),self.min_shares)

    def _apply_capital_cap(self,n_y:int,n_x:int,price_y:float,price_x:float)->Tuple[int,int]:
        total=n_y*price_y+n_x*price_x; cap=self.max_exposure*self.capital
        if total>cap and total>0:
            scale=cap/total
            n_y=max(int(n_y*scale),self.min_shares); n_x=max(int(n_x*scale),self.min_shares)
        return n_y,n_x

    def _zero_sizing(self,signal:int,beta_t:float)->Dict:
        return {"N_Y":0,"N_X":0,"dollar_Y":0.0,"dollar_X":0.0,"total_dollar":0.0,
                "signal":signal,"direction_Y":"FLAT","direction_X":"FLAT",
                "method_used":self.method,"beta_t":round(float(beta_t),6),"leverage":0.0}

    def history_df(self)->pd.DataFrame:
        return pd.DataFrame(self._history) if self._history else pd.DataFrame()

    def plot(self,zscore_series:Optional[pd.Series]=None,figsize:Tuple[int,int]=(14,7),save_path:Optional[str]=None):
        df=self.history_df()
        if df.empty: print("[PositionSizer] No history. Call compute() first."); return
        n_panels=2 if zscore_series is not None else 1
        fig,axes=plt.subplots(n_panels,1,figsize=figsize,sharex=False)
        if n_panels==1: axes=[axes]
        fig.patch.set_facecolor("#12121e")
        if zscore_series is not None:
            ax1=axes[0]; ax1.set_facecolor("#1e1e2e")
            ax1.plot(zscore_series.index,zscore_series.values,color="#4a9eda",linewidth=0.9)
            ax1.axhline(0,color="white",linewidth=0.5,linestyle="--",alpha=0.4)
            ax1.set_title("Z-score Signal",fontsize=10,color="white"); ax1.set_ylabel("Z_t",color="white",fontsize=9)
            ax1.tick_params(colors="gray")
            for sp in ax1.spines.values(): sp.set_color("#444")
        ax2=axes[-1]; ax2.set_facecolor("#1e1e2e")
        colors=["#2ecc71" if s==1 else "#e74c3c" if s==-1 else "#888888" for s in df["signal"].values]
        ax2.bar(range(len(df)),df["N_Y"].values,color=colors,width=0.8,alpha=0.85)
        ax2.set_title(f"Position Size N_Y  [{self.method}]",fontsize=10,color="white")
        ax2.set_ylabel("Shares (Y leg)",color="white",fontsize=9); ax2.set_xlabel("Decision #",color="white",fontsize=9)
        ax2.tick_params(colors="gray")
        for sp in ax2.spines.values(): sp.set_color("#444")
        fig.suptitle(f"PositionSizer — Dark Wing  |  Capital: ₹{self.capital:,.0f}",fontsize=12,fontweight="bold",color="white")
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
        plt.show()

    def summary(self)->dict:
        df=self.history_df()
        if df.empty: print("[PositionSizer] No history yet."); return {}
        active=df[df["N_Y"]>0]
        result={"method":self.method,"capital":self.capital,"n_decisions":len(df),"n_active":len(active),
                "avg_N_Y":round(active["N_Y"].mean(),1) if len(active) else 0,
                "avg_N_X":round(active["N_X"].mean(),1) if len(active) else 0,
                "avg_dollar_Y":round(active["dollar_Y"].mean(),0) if len(active) else 0,
                "avg_dollar_X":round(active["dollar_X"].mean(),0) if len(active) else 0,
                "avg_leverage":round(active["leverage"].mean(),4) if len(active) else 0,
                "max_leverage":round(df["leverage"].max(),4)}
        sep="="*50
        print(f"\n{sep}\n  PositionSizer Summary  [{self.method}]\n{sep}")
        print(f"  Capital         : ₹{self.capital:>15,.0f}")
        print(f"  Decisions       : {result['n_decisions']:>6d}    Active: {result['n_active']:>6d}")
        print(f"  Avg N_Y / N_X   : {result['avg_N_Y']:>8.1f} / {result['avg_N_X']:.1f} shares")
        print(f"  Avg Dollar Y/X  : ₹{result['avg_dollar_Y']:>10,.0f} / ₹{result['avg_dollar_X']:,.0f}")
        print(f"  Avg / Max Lev   : {result['avg_leverage']:.4f}x / {result['max_leverage']:.4f}x\n{sep}\n")
        return result

    def __repr__(self)->str:
        return (f"PositionSizer(capital={self.capital:,.0f}, method='{self.method}', "
                f"max_trade_fraction={self.max_trade_fraction}, max_exposure={self.max_exposure})")

# ── Standalone ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    sizer=PositionSizer(capital=500_000,method="kelly",max_trade_fraction=0.10,kelly_fraction=4)
    sizing=sizer.compute(signal=1,beta_t=0.85,price_y=2500.0,price_x=3200.0)
    print(sizing); sizer.summary()
