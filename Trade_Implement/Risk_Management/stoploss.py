"""
Module  : Trade_Implement/Risk_Management/stoploss.py
Project : Dark Wing — Quantitative Statistical Arbitrage Engine
Purpose : Dynamic Multi-Barrier Risk Management and Stop-Loss Controller
Style   : Institutional Quant (Compact, PEP8, Cointegration Risk Architecture)
"""
import sys,os,warnings,datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Dict,List,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TRADE_DIR=os.path.dirname(_THIS_DIR)
_ROOT_DIR=os.path.dirname(_TRADE_DIR)
_FM_DIR=os.path.join(_ROOT_DIR,"Financial_Mathematics")
_TSA_DIR=os.path.join(_FM_DIR,"Time_Series_Analysis")
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
_SIGNALS_DIR=os.path.join(_TRADE_DIR,"Signals")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_SIGNALS_DIR,_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

from zscore import ZScore
from half_life import HalfLife

# ==================== Stop-Loss Manager ===========================
class StopLossManager:
    """
    Multi-barrier risk controller evaluating positions against:
      1. Structural breakdown: |Z_t| > stop_z
      2. Spread volatility expansion: S_t exceeds k-sigma threshold
      3. Capital preservation: Dollar loss > max_loss_pct * capital
      4. Ornstein-Uhlenbeck time decay: Holding age > multiplier * tau_half_life
      5. Post-stop volatility cooldown: Blocks immediate re-entry for N bars
    """
    _REASONS={
        "zscore_hard":"Z-score hard stop (|Z| > stop_z)",
        "dollar_pnl":"Dollar P&L stop (loss > max_loss)",
        "spread_vol":"Spread volatility stop (k-sigma widening)",
        "time_stop":"Time stop (age > tau_max)",
        "trailing":"Trailing spread stop",
        "none":"No stop triggered"
    }

    def __init__(self,zscore:ZScore,half_life:float,capital:float=1_000_000.0,
                 max_loss_pct:float=0.03,spread_stop_k:float=2.5,
                 spread_stop_window:int=30,time_stop_multiplier:float=3.0,
                 use_trailing_stop:bool=False,trail_k:float=1.5,
                 cooldown_bars:int=5,currency:str="USD",bar_duration_min:float=390.0):
        if not isinstance(zscore,ZScore):
            raise TypeError("zscore must be a ZScore instance.")
        if half_life<=0:
            raise ValueError("half_life must be > 0.")
        if capital<=0:
            raise ValueError("capital must be > 0.")
        if not 0<max_loss_pct<1:
            raise ValueError("max_loss_pct must be in (0, 1).")
        if spread_stop_k<=0:
            raise ValueError("spread_stop_k must be > 0.")
        if time_stop_multiplier<1:
            raise ValueError("time_stop_multiplier must be >= 1.")

        self._zs=zscore
        self.half_life=float(half_life)
        self.capital=float(capital)
        self.currency=str(currency)
        self.max_loss=max_loss_pct*capital
        self.spread_stop_k=float(spread_stop_k)
        self.spread_stop_window=int(spread_stop_window)
        self.tau_max=float(time_stop_multiplier*half_life)
        self.use_trailing_stop=bool(use_trailing_stop)
        self.trail_k=float(trail_k)
        self._cooldown_bars_param=max(0,int(cooldown_bars))
        self.bar_duration_min=float(bar_duration_min)

        # Active trade state
        self._position:int=0
        self._n_shares:int=0
        self._n_shares_x:int=0
        self._S_entry:float=0.0
        self._P_Y_entry:float=0.0
        self._P_X_entry:float=0.0
        self._t_entry:Optional[object]=None
        self._S_best:float=0.0
        self._S_stop_level:float=0.0
        self._spread_sigma:float=0.0
        self._current_bars:int=0
        self._cooldown_remaining:int=0
        self._trade_log:List[Dict]=[]

    def enter_position(self,signal:int,spread_entry:float,n_shares:int,timestamp=None,
                       spread_series:Optional[pd.Series]=None,
                       price_y_entry:Optional[float]=None,
                       price_x_entry:Optional[float]=None,
                       n_shares_x:Optional[int]=None)->None:
        """
        Registers new position entry and sets dynamic volatility stop levels.
        """
        if signal not in (1,-1):
            raise ValueError(f"signal must be +1 or -1, got {signal}.")
        if n_shares<0:
            raise ValueError("n_shares must be >= 0.")

        self._position=signal
        self._n_shares=int(n_shares)
        self._n_shares_x=int(n_shares_x) if n_shares_x is not None else 0
        self._S_entry=float(spread_entry)
        self._P_Y_entry=float(price_y_entry) if price_y_entry is not None else 0.0
        self._P_X_entry=float(price_x_entry) if price_x_entry is not None else 0.0
        self._t_entry=timestamp if timestamp is not None else datetime.datetime.now()
        self._S_best=float(spread_entry)
        self._current_bars=0

        # Spread sigma anchored to calibrated training distribution
        sigma_cand=None
        if hasattr(self._zs,"_spread") and self._zs._spread is not None:
            sp=self._zs._spread.dropna()
            if len(sp)>=self.spread_stop_window:
                s=float(sp.iloc[-self.spread_stop_window:].std(ddof=1))
                if s>1e-6: sigma_cand=s
        if sigma_cand is None and spread_series is not None and len(spread_series.dropna())>=5:
            recent=spread_series.dropna().iloc[-self.spread_stop_window:]
            s=float(recent.std(ddof=1))
            if s>1e-6: sigma_cand=s
        self._spread_sigma=sigma_cand if sigma_cand is not None else 0.01

        self._S_stop_level=(self._S_entry-self.spread_stop_k*self._spread_sigma if signal==1
                            else self._S_entry+self.spread_stop_k*self._spread_sigma)

    @property
    def in_cooldown(self)->bool:
        """True when re-entry is suppressed after a stop-loss event."""
        return self._cooldown_remaining>0

    def check_stops(self,S_t:float,Z_t:float,timestamp=None,price_y:Optional[float]=None,price_x:Optional[float]=None,is_bar_close:bool=True)->Dict:
        """
        Evaluates active position against all risk barriers.
        """
        if self._cooldown_remaining>0:
            if is_bar_close: self._cooldown_remaining-=1
            return self._no_stop_result(S_t)
        if self._position==0:
            return self._no_stop_result(S_t)
        if is_bar_close:
            self._current_bars+=1

        pnl=self._compute_pnl(S_t,price_y=price_y,price_x=price_x)
        if self.use_trailing_stop:
            self._update_trailing_stop(S_t)

        for check_fn in [lambda:self._check_zscore_stop(Z_t),
                         lambda:self._check_dollar_stop(pnl),
                         lambda:self._check_spread_stop(S_t),
                         lambda:self._check_time_stop(timestamp)]:
            stop,reason=check_fn()
            if stop:
                return self._trigger_stop(reason,S_t,pnl)

        if self.use_trailing_stop:
            stop,reason=self._check_trailing_stop(S_t)
            if stop:
                return self._trigger_stop(reason,S_t,pnl)

        return self._no_stop_result(S_t,pnl)

    def _check_zscore_stop(self,Z_t:float)->Tuple[bool,str]:
        return (abs(Z_t)>self._zs.stop_z,"zscore_hard") if abs(Z_t)>self._zs.stop_z else (False,"")

    def _check_dollar_stop(self,pnl:float)->Tuple[bool,str]:
        return (True,"dollar_pnl") if pnl<-self.max_loss else (False,"")

    def _check_spread_stop(self,S_t:float)->Tuple[bool,str]:
        if self._position==1 and S_t<self._S_stop_level: return True,"spread_vol"
        if self._position==-1 and S_t>self._S_stop_level: return True,"spread_vol"
        return False,""

    def _check_time_stop(self,timestamp=None)->Tuple[bool,str]:
        if self._current_bars>self.tau_max:
            return True,"time_stop"
        return False,""

    def _check_trailing_stop(self,S_t:float)->Tuple[bool,str]:
        td=self.trail_k*self._spread_sigma
        if self._position==1 and S_t<self._S_best-td: return True,"trailing"
        if self._position==-1 and S_t>self._S_best+td: return True,"trailing"
        return False,""

    def _compute_pnl(self,S_t:float,price_y:Optional[float]=None,price_x:Optional[float]=None)->float:
        if price_y is not None and price_x is not None and self._P_Y_entry>0 and self._P_X_entry>0:
            if self._position==1:
                return self._n_shares*(price_y-self._P_Y_entry)+self._n_shares_x*(self._P_X_entry-price_x)
            elif self._position==-1:
                return self._n_shares*(self._P_Y_entry-price_y)+self._n_shares_x*(price_x-self._P_X_entry)
        scale=self._P_Y_entry if self._P_Y_entry>0 else (self.capital/max(self._n_shares,1) if self._n_shares>0 else 1.0)
        if self._position==1: return self._n_shares*scale*(S_t-self._S_entry)
        if self._position==-1: return self._n_shares*scale*(self._S_entry-S_t)
        return 0.0

    def _update_trailing_stop(self,S_t:float)->None:
        if self._position==1: self._S_best=max(self._S_best,S_t)
        elif self._position==-1: self._S_best=min(self._S_best,S_t)

    def _trigger_stop(self,reason:str,S_t:float,pnl:float)->Dict:
        self._trade_log.append({
            "direction":"LONG" if self._position==1 else "SHORT",
            "S_entry":round(self._S_entry,6),"S_exit":round(S_t,6),
            "pnl":round(pnl,2),"bars_in_trade":self._current_bars,
            "stop_reason":self._REASONS.get(reason,reason),"entry_time":self._t_entry
        })
        bars=self._current_bars
        self._position=0
        self._n_shares=0
        self._n_shares_x=0
        self._S_entry=0.0
        self._P_Y_entry=0.0
        self._P_X_entry=0.0
        self._t_entry=None
        self._S_best=0.0
        self._S_stop_level=0.0
        self._spread_sigma=0.0
        self._current_bars=0
        self._cooldown_remaining=self._cooldown_bars_param

        return {"stop_triggered":True,"reason":self._REASONS.get(reason,reason),
                "close_signal":0,"unrealised_pnl":round(pnl,2),"bars_in_trade":bars}

    def _no_stop_result(self,S_t:float,pnl:float=0.0)->Dict:
        return {"stop_triggered":False,"reason":"none","close_signal":-99,
                "unrealised_pnl":round(pnl,2),"bars_in_trade":self._current_bars}

    def close_position(self,S_t:float,reason:str="manual",timestamp=None,price_y:Optional[float]=None,price_x:Optional[float]=None)->Dict:
        if self._position==0: return {"message":"No open position to close."}
        return self._trigger_stop(reason,S_t,self._compute_pnl(S_t,price_y=price_y,price_x=price_x))

    def reset_position(self)->None:
        """Resets active position state without triggering cooldown."""
        self._position=0
        self._n_shares=0
        self._n_shares_x=0
        self._S_entry=0.0
        self._P_Y_entry=0.0
        self._P_X_entry=0.0
        self._t_entry=None
        self._S_best=0.0
        self._S_stop_level=0.0
        self._spread_sigma=0.0
        self._current_bars=0

    def trade_log(self)->pd.DataFrame:
        return pd.DataFrame(self._trade_log) if self._trade_log else pd.DataFrame()

    def plot(self,signal_df:Optional[pd.DataFrame]=None,figsize:Tuple[int,int]=(16,8),save_path:Optional[str]=None):
        df_log=self.trade_log()
        n_panels=3 if (signal_df is not None and not df_log.empty) else 1
        fig,axes=plt.subplots(n_panels,1,figsize=figsize,sharex=False)
        if n_panels==1: axes=[axes]
        fig.patch.set_facecolor("#12121e")

        if signal_df is not None and "spread" in signal_df.columns:
            ax1=axes[0]; ax1.set_facecolor("#1e1e2e")
            sp=signal_df["spread"]
            ax1.plot(sp.index,sp.values,color="#7ecfff",linewidth=0.8,label="Spread $S_t$")
            ax1.axhline(0,color="white",linewidth=0.5,linestyle="--",alpha=0.4)
            ax1.set_title("Cointegrated Spread Series",fontsize=10,color="white")
            ax1.tick_params(colors="gray")
            ax1.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white")
            for sp_ in ax1.spines.values(): sp_.set_color("#444")

        if signal_df is not None and "z" in signal_df.columns and n_panels>1:
            ax2=axes[1]; ax2.set_facecolor("#1e1e2e")
            z=signal_df["z"]
            ax2.plot(z.index,z.values,color="#4a9eda",linewidth=0.8)
            ax2.axhline(self._zs.stop_z,color="#8e44ad",linewidth=1,linestyle="-.",label=f"stop_z=±{self._zs.stop_z}")
            ax2.axhline(-self._zs.stop_z,color="#8e44ad",linewidth=1,linestyle="-.")
            ax2.axhline(self._zs.entry_z,color="#e74c3c",linewidth=0.8,linestyle="--",label=f"entry_z=±{self._zs.entry_z}")
            ax2.axhline(-self._zs.entry_z,color="#e74c3c",linewidth=0.8,linestyle="--")
            ax2.axhline(0,color="white",linewidth=0.4,alpha=0.4)
            ax2.set_title("Rolling Normalised Z-Score",fontsize=10,color="white")
            ax2.tick_params(colors="gray")
            ax2.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white")
            for sp_ in ax2.spines.values(): sp_.set_color("#444")

        if not df_log.empty and n_panels>=3:
            ax3=axes[2]; ax3.set_facecolor("#1e1e2e")
            cum=df_log["pnl"].cumsum().values
            idx=range(len(cum))
            ax3.step(idx,cum,color="#f39c12",linewidth=1.2,where="post")
            ax3.fill_between(idx,0,cum,where=(cum>=0),color="#2ecc71",alpha=0.25,step="post")
            ax3.fill_between(idx,0,cum,where=(cum<0),color="#e74c3c",alpha=0.25,step="post")
            ax3.axhline(0,color="white",linewidth=0.5,linestyle="--",alpha=0.5)
            ax3.set_title(f"Cumulative P&L [{self.currency}]",fontsize=10,color="white")
            ax3.set_ylabel(self.currency,color="white",fontsize=9)
            ax3.set_xlabel("Trade #",color="white",fontsize=9)
            ax3.tick_params(colors="gray")
            for sp_ in ax3.spines.values(): sp_.set_color("#444")

        fig.suptitle("StopLossManager — Risk Monitor",fontsize=12,fontweight="bold",color="white")
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
        plt.show()

    def summary(self)->dict:
        df=self.trade_log()
        if df.empty:
            print("[StopLossManager] No closed trades recorded.")
            return {}
        n=len(df)
        total_pnl=df["pnl"].sum()
        win_rate=(df["pnl"]>0).mean()*100
        avg_pnl=df["pnl"].mean()
        avg_dur=df["bars_in_trade"].mean()
        stop_pct=(df["stop_reason"].value_counts()/n*100).round(1).to_dict()
        result={"n_trades":n,"total_pnl":round(total_pnl,2),"win_rate_pct":round(win_rate,2),
                "avg_pnl":round(avg_pnl,2),"avg_duration_bars":round(avg_dur,1),"stop_breakdown":stop_pct}
        sep="="*55
        print(f"\n{sep}\n  StopLossManager Quantitative Audit\n{sep}")
        print(f"  Trades         : {n}    Win rate: {win_rate:.1f}%")
        print(f"  Total P&L      : {self.currency} {total_pnl:>12,.2f}")
        print(f"  Avg P&L/trade  : {self.currency} {avg_pnl:>10,.2f}    Avg duration: {avg_dur:.1f} bars")
        print(f"\n  Stop Barrier Breakdown:")
        for reason,pct in stop_pct.items():
            print(f"    {reason:<45s}  {pct:.1f}%")
        print(f"{sep}\n")
        return result

    @property
    def is_flat(self)->bool: return self._position==0

    @property
    def current_position(self)->int: return self._position

    @property
    def bars_in_trade(self)->int: return self._current_bars

    def __repr__(self)->str:
        pos={1:"LONG",-1:"SHORT",0:"FLAT"}.get(self._position,"?")
        return (f"StopLossManager(pos={pos}, cap={self.currency} {self.capital:,.0f}, "
                f"k={self.spread_stop_k}, tau_max={self.tau_max:.0f} bars)")

# =====================================================================
if __name__=="__main__":
    print("StopLossManager — Multi-barrier statistical arbitrage risk controller.")