import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Dict,List,Tuple

_THIS_DIR   =os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   =os.path.dirname(_THIS_DIR)
_FM_DIR     =os.path.join(_ROOT_DIR,"Financial_Mathematics")
_TSA_DIR    =os.path.join(_FM_DIR,"Time_Series_Analysis")
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
_KALMAN_DIR =os.path.join(_TSA_DIR,"Kalman_Filter")
_TRADE_DIR  =os.path.join(_ROOT_DIR,"Trade_Implement")
_SIGNALS_DIR=os.path.join(_TRADE_DIR,"Signals")
_RISK_DIR   =os.path.join(_TRADE_DIR,"Risk_Management")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_KALMAN_DIR,_SIGNALS_DIR,_RISK_DIR,_THIS_DIR]:
    if _p not in sys.path: 
        sys.path.insert(0,_p)

from zscore import ZScore
from half_life import HalfLife
from pair_signal import PairSignalGenerator
from position_sizing import PositionSizer
from stoploss import StopLossManager

# ─────────────────────────────────────────────────────────────────────────────
class BacktestEngine:
    # PnL = N_Y*(S_exit - S_entry) [LONG] | N_Y*(S_entry - S_exit) [SHORT] minus TC.
    def __init__(self,signal_df:pd.DataFrame,price_y:pd.Series,price_x:pd.Series,zscore:ZScore,half_life:float,capital:float=1_000_000.0,commission_pct:float=0.0003,slippage_pct:float=0.0005,sizing_method:str="fixed_fractional",fixed_fraction:Optional[float]=None,max_trade_fraction:float=0.10,max_exposure:float=0.25,kelly_fraction:float=4.0,sigma_target:float=5_000.0,spread_stop_k:float=2.0,max_loss_pct:float=0.01,time_stop_mult:float=3.0,use_trailing_stop:bool=False,pair_name:str="Pair"):
        required={"z","spread","signal","position","beta"}
        if not required.issubset(signal_df.columns): 
            raise ValueError(f"signal_df must contain {required}.")
        if not isinstance(zscore,ZScore): 
            raise TypeError("zscore must be ZScore instance.")
        if capital<=0: 
            raise ValueError("capital > 0.")
        self.signal_df=signal_df.copy(); self.price_y=price_y.reindex(signal_df.index).ffill()
        self.price_x=price_x.reindex(signal_df.index).ffill()
        self.capital=float(capital); self.commission_pct=float(commission_pct)
        self.slippage_pct=float(slippage_pct); self.pair_name=str(pair_name)
        ff=float(fixed_fraction) if fixed_fraction is not None else float(max_trade_fraction)
        self._sizer=PositionSizer(capital=capital,method=sizing_method,fixed_fraction=ff,max_trade_fraction=max_trade_fraction,max_exposure=max_exposure,kelly_fraction=kelly_fraction,sigma_target=sigma_target)
        self._sl=StopLossManager(zscore=zscore,half_life=half_life,capital=capital,max_loss_pct=max_loss_pct,spread_stop_k=spread_stop_k,time_stop_multiplier=time_stop_mult,use_trailing_stop=use_trailing_stop)
        self._trade_log:List[Dict]=[]
        self._equity:List[float]=[]
        self._equity_index:List=[]
        self._is_run=False
        # internal open-trade state
        self._open:bool=False
        self._direction:int=0
        self._S_entry:float=0.0
        self._N_Y:int=0; self._N_X:int=0
        self._P_Y_entry:float=0.0; self._P_X_entry:float=0.0
        self._entry_idx=None
        self._tc_entry:float=0.0

    def run(self)->pd.DataFrame:
        capital=self.capital; df=self.signal_df
        for idx,row in df.iterrows():
            S_t=float(row["spread"]); Z_t=float(row["z"])
            sig=int(row["signal"]); pos=int(row["position"])
            P_Y=float(self.price_y.get(idx,np.nan)); P_X=float(self.price_x.get(idx,np.nan))
            beta_t=float(row["beta"])
            if np.isnan(P_Y) or np.isnan(P_X): self._equity.append(capital); self._equity_index.append(idx); continue
            # check stops if in trade
            if self._open:
                stop_res=self._sl.check_stops(S_t=S_t,Z_t=Z_t,timestamp=idx,price_y=P_Y,price_x=P_X)
                if stop_res["stop_triggered"]:
                    capital+=self._close_trade(S_t,P_Y,P_X,idx,stop_res["reason"]); self._open=False
                elif sig==0 or abs(sig)==2:  # normal exit signal
                    capital+=self._close_trade(S_t,P_Y,P_X,idx,"exit_signal"); self._open=False
            # open new trade
            if not self._open and not self._sl.in_cooldown and sig in (1,-1):
                sz=self._sizer.compute(signal=sig,beta_t=beta_t,price_y=P_Y,price_x=P_X,
                                       zscore_series=df["z"].loc[:idx])
                N_Y=sz["N_Y"]; N_X=sz["N_X"]
                if N_Y>0 and N_X>0:
                    tc=self._apply_costs(N_Y,N_X,P_Y,P_X); capital-=tc
                    self._sl.enter_position(signal=sig,spread_entry=S_t,n_shares=N_Y,timestamp=idx,
                                            spread_series=df["spread"].loc[:idx],
                                            price_y_entry=P_Y,price_x_entry=P_X,n_shares_x=N_X)
                    self._open=True; self._direction=sig; self._S_entry=S_t
                    self._N_Y=N_Y; self._N_X=N_X; self._entry_idx=idx; self._tc_entry=tc
                    self._P_Y_entry=P_Y; self._P_X_entry=P_X
            self._equity.append(capital); self._equity_index.append(idx)
        # close any open trade at end
        if self._open:
            last=df.iloc[-1]; S_last=float(last["spread"])
            P_Y_last=float(self.price_y.iloc[-1]); P_X_last=float(self.price_x.iloc[-1])
            capital+=self._close_trade(S_last,P_Y_last,P_X_last,df.index[-1],"end_of_data")
            self._equity[-1]=capital
        self._is_run=True
        return self.trade_log()

    def _close_trade(self,S_exit:float,P_Y:float,P_X:float,idx,reason:str)->float:
        if self._P_Y_entry>0 and self._P_X_entry>0:
            if self._direction==1:
                pnl=self._N_Y*(P_Y-self._P_Y_entry)+self._N_X*(self._P_X_entry-P_X)
            else:
                pnl=self._N_Y*(self._P_Y_entry-P_Y)+self._N_X*(P_X-self._P_X_entry)
        else:
            pnl=(self._N_Y*(S_exit-self._S_entry) if self._direction==1
                 else self._N_Y*(self._S_entry-S_exit))*P_Y
        tc_exit=self._apply_costs(self._N_Y,self._N_X,P_Y,P_X)
        net_pnl=pnl-tc_exit
        try:
             i_entry=self.signal_df.index.get_loc(self._entry_idx)
             i_exit=self.signal_df.index.get_loc(idx)
             bars=max(i_exit-i_entry, 1)
        except Exception:
            bars=0
        self._trade_log.append({"direction":"LONG" if self._direction==1 else "SHORT",
                                "entry_idx":self._entry_idx,"exit_idx":idx,
                                "P_Y_entry":round(self._P_Y_entry,4),"P_X_entry":round(self._P_X_entry,4),
                                "P_Y_exit":round(P_Y,4),"P_X_exit":round(P_X,4),
                                "S_entry":round(self._S_entry,6),"S_exit":round(S_exit,6),
                                "N_Y":self._N_Y,"N_X":self._N_X,"gross_pnl":round(pnl,2),
                                "tc":round(tc_exit+self._tc_entry,2),"net_pnl":round(net_pnl,2),
                                "bars":bars,"close_reason":reason})
        return net_pnl

    def _apply_costs(self,N_Y:int,N_X:int,P_Y:float,P_X:float)->float:
        notional=N_Y*P_Y+N_X*P_X
        return notional*(self.commission_pct+self.slippage_pct)

    def trade_log(self)->pd.DataFrame:
        self._require_run("trade_log")
        return pd.DataFrame(self._trade_log) if self._trade_log else pd.DataFrame()

    def equity_curve(self)->pd.Series:
        self._require_run("equity_curve")
        return pd.Series(self._equity,index=self._equity_index,name="equity")

    def drawdown_series(self)->pd.Series:
        eq=self.equity_curve(); peak=eq.cummax()
        return ((peak-eq)/peak.replace(0,np.nan)).fillna(0).rename("drawdown")

    def summary(self)->dict:
        self._require_run("summary")
        df=self.trade_log(); eq=self.equity_curve()
        if df.empty: print("[BacktestEngine] No trades executed."); return {}
        n=len(df); n_win=int((df["net_pnl"]>0).sum()); total_pnl=df["net_pnl"].sum()
        total_tc=df["tc"].sum(); dd=self.drawdown_series(); mdd=float(dd.max())
        final_cap=eq.iloc[-1]; ret_pct=(final_cap-self.capital)/self.capital*100
        result={"pair":self.pair_name,"n_trades":n,"win_rate":round(n_win/n*100,2),"total_net_pnl":round(total_pnl,2),
                "total_tc":round(total_tc,2),"max_drawdown_pct":round(mdd*100,2),"final_capital":round(final_cap,2),
                "return_pct":round(ret_pct,2),"avg_net_pnl":round(df["net_pnl"].mean(),2),
                "avg_bars":round(df["bars"].mean(),1)}
        sep="="*55
        print(f"\n{sep}\n  BacktestEngine Summary — {self.pair_name}\n{sep}")
        print(f"  Trades      : {n}    Win rate : {result['win_rate']:.1f}%")
        print(f"  Total PnL   : ₹{total_pnl:>12,.2f}    TC: ₹{total_tc:,.2f}")
        print(f"  Return      : {ret_pct:.2f}%    Max DD: {mdd*100:.2f}%")
        print(f"  Final cap   : ₹{final_cap:>12,.2f}")
        print(f"  Avg PnL/tr  : ₹{result['avg_net_pnl']:>8,.2f}    Avg dur: {result['avg_bars']:.1f} bars\n{sep}\n")
        return result

    def plot(self,figsize:Tuple[int,int]=(16,10),save_path:Optional[str]=None):
        self._require_run("plot")
        eq=self.equity_curve(); dd=self.drawdown_series()
        df=self.signal_df; sp=df["spread"]; z=df["z"]
        fig,axes=plt.subplots(3,1,figsize=figsize,sharex=False,gridspec_kw={"height_ratios":[2,1,2]})
        fig.patch.set_facecolor("#12121e")
        ax1=axes[0]; ax1.set_facecolor("#1e1e2e")
        ax1.plot(eq.index,eq.values,color="#4a9eda",linewidth=1.1,label="Equity")
        ax1.axhline(self.capital,color="white",linewidth=0.6,linestyle="--",alpha=0.5)
        ax1.set_title(f"Equity Curve — {self.pair_name}",fontsize=11,fontweight="bold",color="white")
        ax1.set_ylabel("Capital ₹",color="white",fontsize=9); ax1.tick_params(colors="gray")
        ax1.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white")
        for sp_ in ax1.spines.values(): sp_.set_color("#444")
        ax2=axes[1]; ax2.set_facecolor("#1e1e2e")
        ax2.fill_between(dd.index,0,dd.values*100,color="#e74c3c",alpha=0.7)
        ax2.set_title("Drawdown %",fontsize=10,color="white"); ax2.set_ylabel("DD %",color="white",fontsize=9)
        ax2.tick_params(colors="gray")
        for sp_ in ax2.spines.values(): sp_.set_color("#444")
        ax3=axes[2]; ax3.set_facecolor("#1e1e2e")
        ax3.plot(sp.index,sp.values,color="#7ecfff",linewidth=0.8,label="Spread")
        tl=self.trade_log()
        if not tl.empty:
            for _,tr in tl.iterrows():
                col="#2ecc71" if tr["direction"]=="LONG" else "#e74c3c"
                try:
                    ei=tr["entry_idx"]; xi=tr["exit_idx"]
                    ax3.axvspan(ei,xi,alpha=0.15,color=col)
                    ax3.scatter([ei],[sp.get(ei,np.nan)],marker="^" if tr["direction"]=="LONG" else "v",
                                color=col,s=30,zorder=5)
                except: pass
        ax3.set_title("Spread + Trade Regions",fontsize=10,color="white")
        ax3.set_ylabel("Spread",color="white",fontsize=9); ax3.set_xlabel("Date",color="white",fontsize=9)
        ax3.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white"); ax3.tick_params(colors="gray")
        for sp_ in ax3.spines.values(): sp_.set_color("#444")
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
        plt.show()

    def _require_run(self,caller:str):
        if not self._is_run: raise RuntimeError(f"{caller}() requires run() first.")

    def __repr__(self)->str:
        return (f"BacktestEngine(pair='{self.pair_name}', capital={self.capital:,.0f}, "
                f"trades={len(self._trade_log)}, run={self._is_run})")

if __name__=="__main__":
    print("BacktestEngine — wire PairSignalGenerator output then call run().")
