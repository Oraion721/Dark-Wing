"""
Module  : Backtesting / walkforward.py
Project : Dark Wing

Purpose:
    Walk-forward expanding-window backtest for pairs trading.

    Method:
        1. Train on first `train_frac` of data (fit DynamicBeta + ZScore parameters)
        2. Test on next `step_bars` bars (frozen parameters - NO refitting)
        3. Expand train window by `step_bars`, repeat
        4. Concatenate all out-of-sample (OOS) trade logs and equity curves

    This is the ONLY correct way to get an unbiased Sharpe ratio.
    In-sample Sharpe is always inflated - walk-forward Sharpe is what matters.

Usage:
    from walkforward import WalkForwardBacktest
    wf = WalkForwardBacktest(price_y, price_x, capital=10_000)
    results = wf.run()
    results['summary']   # dict of aggregate OOS metrics
    results['equity']    # pd.Series - concatenated OOS equity curve
    results['trades']    # pd.DataFrame - all OOS trades
"""
import sys,os,warnings
import numpy as np
import pandas as pd
from typing import Optional,Dict,List

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_ROOT=os.path.dirname(_THIS_DIR)
_TSA=os.path.join(_ROOT,"Financial_Mathematics","Time_Series_Analysis")
for _p in [os.path.join(_TSA,"Spreads"),os.path.join(_TSA,"Kalman_Filter"),
           os.path.join(_ROOT,"Trade_Implement","Signals"),
           os.path.join(_ROOT,"Trade_Implement","Risk_Management"),_THIS_DIR]:
    if _p not in sys.path: sys.path.insert(0,_p)

from dynamic_beta import DynamicBeta
from half_life import HalfLife
from zscore import ZScore
from pair_signal import PairSignalGenerator
from kalman_forecast import KalmanForecaster
from engine import BacktestEngine
from performance import PerformanceAnalytics


class WalkForwardBacktest:      # Expanding-window walk-forward backtest for pairs trading.
    """ Args:
            ~ price_y: pd.Series - Log-price of dependent leg (already log-transformed).
            ~ price_x: pd.Series - Log-price of independent leg (already log-transformed).
            ~ capital: float - Starting capital in Rs.
            ~ train_frac: float - Fraction of full dataset used as initial training window (default 0.60).
            ~ step_bars: int | None - Bars in each OOS test window. If None, defaults to int(0.10 * T).
            ~ entry_z: float - Z-score entry threshold (default 2.0).
            ~ exit_z: float - Z-score exit threshold (default 0.5).
            ~ stop_z: float - Z-score stop threshold (default 3.0).
            ~ commission: float - Commission + slippage fraction (default 0.0005).
            ~ max_trade_f: float - Max capital fraction per trade (default 0.10).
            ~ verbose: bool - Print window-level summaries (default True).  """

    def __init__(self,price_y:pd.Series,price_x:pd.Series,capital:float=10_000.0,train_frac:float=0.60,step_bars:Optional[int]=None,entry_z:float=2.0,exit_z:float=0.5,stop_z:float=3.0,commission:float=0.0005,max_trade_f:float=0.10,fixed_fraction:Optional[float]=None,max_exposure:float=0.60,sizing_method:str="fixed_fractional",verbose:bool=True):
        aligned=pd.concat([price_y.rename("y"),price_x.rename("x")],axis=1).dropna()
        self._y=aligned["y"]; self._x=aligned["x"]
        self.T=len(aligned)
        self.capital=float(capital)
        self.train_frac=float(train_frac)
        self.step_bars=int(step_bars) if step_bars else max(75,int(0.10*self.T))
        self.entry_z=entry_z; self.exit_z=exit_z; self.stop_z=stop_z
        self.commission=commission; self.max_trade_f=max_trade_f
        self.fixed_fraction=float(fixed_fraction) if fixed_fraction is not None else float(max_trade_f)
        self.max_exposure=float(max_exposure)
        self.sizing_method=str(sizing_method)
        self.verbose=verbose
        self._train_start=max(20,int(train_frac*self.T))
        self._window_results:List[Dict]=[]
        self._all_trades:List[pd.DataFrame]=[]
        self._all_equity:List[pd.Series]=[]
        self._run_complete=False

    def run(self)->Dict:        # Run the full walk-forward backtest.
        """
        Returns dict: summary, equity, trades, window_results.  """
        print(f"\n{'='*60}")
        print(f"  Walk-Forward Backtest")
        print(f"  T={self.T} bars  train_start={self._train_start}  step={self.step_bars}")
        print(f"  Windows: {self._count_windows()}")
        print(f"{'='*60}")

        window_id=0
        train_end=self._train_start
        capital_running=self.capital

        while train_end<self.T:
            test_start=train_end
            test_end=min(train_end+self.step_bars,self.T)
            if test_end-test_start<5:
                break

            y_train=self._y.iloc[:train_end]; x_train=self._x.iloc[:train_end]
            if self.verbose:
                print(f"\n[Window {window_id+1}] Train=[0,{train_end})  Test=[{test_start},{test_end})")

            try:
                db=DynamicBeta(y=y_train,x=x_train,entry_z=self.entry_z,
                               exit_z=self.exit_z,stop_z=self.stop_z)
                db.fit()
                hl=HalfLife(db.dynamic_spread_series())
                hl.estimate()
                half_life=hl.half_life_days
                zs_win=max(10,min(int(half_life*2),len(y_train)//2))
                zs_train=ZScore(db.dynamic_spread_series(),window=zs_win,mode="rolling",entry_z=self.entry_z,exit_z=self.exit_z,stop_z=self.stop_z)
                zs_train.compute()
            except Exception as e:
                warnings.warn(f"[Window {window_id+1}] Train fit failed: {e}. Skipping.")
                train_end+=self.step_bars; window_id+=1; continue

            try:
                y_test=self._y.iloc[test_start:test_end]
                x_test=self._x.iloc[test_start:test_end]
                oos_signal_df=self._build_oos_signals(db,hl,zs_win,y_test,x_test)
                if oos_signal_df.empty:
                    train_end+=self.step_bars; window_id+=1; continue
            except Exception as e:
                warnings.warn(f"[Window {window_id+1}] OOS signal build failed: {e}. Skipping.")
                train_end+=self.step_bars; window_id+=1; continue

            try:
                engine=BacktestEngine(
                    signal_df=oos_signal_df,
                    price_y=np.exp(y_test),
                    price_x=np.exp(x_test),
                    zscore=zs_train,half_life=half_life,
                    capital=capital_running,
                    commission_pct=self.commission,slippage_pct=0.0002,
                    sizing_method=self.sizing_method,
                    fixed_fraction=self.fixed_fraction,
                    max_trade_fraction=self.max_trade_f,max_exposure=self.max_exposure,
                    pair_name=f"Window-{window_id+1}"
                )
                engine.run()
                eq=engine.equity_curve(); tlog=engine.trade_log()
                capital_running=float(eq.iloc[-1])

                w_result={"window":window_id+1,"train_bars":train_end,
                           "test_start":test_start,"test_end":test_end,
                           "n_trades":len(tlog),"final_capital":round(capital_running,2),
                           "return_pct":round((capital_running-self.capital)/self.capital*100,2)}
                if not tlog.empty:
                    w_result["win_rate"]=round((tlog["net_pnl"]>0).mean()*100,2)
                    w_result["net_pnl"]=round(tlog["net_pnl"].sum(),2)
                else:
                    w_result["win_rate"]=0.0; w_result["net_pnl"]=0.0

                self._window_results.append(w_result)
                if not tlog.empty: self._all_trades.append(tlog)
                self._all_equity.append(eq)
                if self.verbose: print(f"  Trades={len(tlog)}  Win={w_result.get('win_rate',0):.1f}%  Capital=Rs{capital_running:,.0f}")
            except Exception as e:
                warnings.warn(f"[Window {window_id+1}] Backtest failed: {e}.")

            train_end+=self.step_bars; window_id+=1

        self._run_complete=True
        return self._compile_results()

    def _build_oos_signals(self,db:DynamicBeta,hl:HalfLife,
                            zs_window:int,y_test:pd.Series,x_test:pd.Series)->pd.DataFrame:
        """
        Apply trained parameters to OOS bars bar-by-bar with online recursive Kalman updates
        and rolling causal Z-score normalisation (strictly zero look-ahead bias).
        """
        kf=db.get_filter()
        train_spread=db.dynamic_spread_series()
        win=min(zs_window,len(train_spread))
        spread_history=list(train_spread.iloc[-win:].values)

        rows=[]
        for ts,ly,lx in zip(y_test.index,y_test.values,x_test.values):
            # Online Kalman Bayesian update with new observation
            try:
                state=kf.update(float(ly),float(lx))
                beta_t=float(state.beta)
            except Exception:
                beta_t=float(kf.current_beta())

            spread=float(ly)-beta_t*float(lx)
            spread_history.append(spread)

            # Causal rolling Z-score over recent window
            recent=spread_history[-win:]
            mu_t=float(np.mean(recent))
            sigma_t=float(np.std(recent,ddof=1))
            if sigma_t<1e-8: sigma_t=1.0
            z=(spread-mu_t)/sigma_t

            signal=0
            if   z> db.entry_z: signal=-1
            elif z<-db.entry_z: signal= 1
            rows.append({"spread":spread,"z":z,"signal":signal,
                         "position":signal,"beta":beta_t,"regime_flag":False})

        if not rows: return pd.DataFrame()
        df=pd.DataFrame(rows,index=y_test.index)
        curr_sig=0
        signals=[]
        positions=[]
        for z_val in df["z"]:
            if curr_sig==0:
                if z_val<-db.entry_z: curr_sig=1
                elif z_val>db.entry_z: curr_sig=-1
            else:
                if abs(z_val)<db.exit_z: curr_sig=0
                elif abs(z_val)>db.stop_z: curr_sig=2 if curr_sig==1 else -2
            signals.append(curr_sig)
            positions.append(1 if curr_sig==1 else (-1 if curr_sig==-1 else 0))
            if abs(curr_sig)==2: curr_sig=0

        df["signal"]=signals
        df["position"]=positions
        return df

    def _compile_results(self)->Dict:
        all_trades=(pd.concat(self._all_trades,ignore_index=True)
                    if self._all_trades else pd.DataFrame())
        all_equity=(pd.concat(self._all_equity).sort_index()
                    if self._all_equity else pd.Series(dtype=float))

        total_windows=len(self._window_results)
        total_trades=len(all_trades)
        final_cap=float(all_equity.iloc[-1]) if not all_equity.empty else self.capital
        total_return_pct=(final_cap-self.capital)/self.capital*100

        summary={"total_windows":total_windows,"total_oos_trades":total_trades,
                  "initial_capital":self.capital,"final_capital":round(final_cap,2),
                  "total_return_pct":round(total_return_pct,2)}

        if not all_trades.empty and "net_pnl" in all_trades.columns:
            wins=all_trades[all_trades["net_pnl"]>0]["net_pnl"].sum()
            losses=all_trades[all_trades["net_pnl"]<0]["net_pnl"].abs().sum()
            summary["oos_win_rate_pct"]=round((all_trades["net_pnl"]>0).mean()*100,2)
            summary["oos_profit_factor"]=round(wins/max(losses,1e-6),4)

        if len(all_equity)>2:
            pa=PerformanceAnalytics(all_equity,all_trades if not all_trades.empty else pd.DataFrame())
            summary["oos_sharpe"]=round(pa.sharpe(),4)
            summary["oos_sortino"]=round(pa.sortino(),4)
            summary["oos_max_dd_pct"]=round(pa.max_drawdown()*100,2)
            summary["oos_var_95_pct"]=round(pa.var(0.95)*100,4)
            summary["oos_cvar_95_pct"]=round(pa.cvar(0.95)*100,4)

        sep="="*60
        print(f"\n{sep}"); print(f"  Walk-Forward OOS Results"); print(f"{sep}")
        print(f"  Windows         : {total_windows}")
        print(f"  OOS Trades      : {total_trades}")
        print(f"  Total Return    : {total_return_pct:.2f}%")
        print(f"  Final Capital   : Rs{final_cap:,.0f}")
        if "oos_sharpe" in summary: print(f"  OOS Sharpe      : {summary['oos_sharpe']:.4f}")
        if "oos_win_rate_pct" in summary: print(f"  OOS Win Rate    : {summary['oos_win_rate_pct']:.1f}%")
        if "oos_profit_factor" in summary: print(f"  OOS Prof.Factor : {summary['oos_profit_factor']:.4f}")
        if "oos_var_95_pct" in summary: print(f"  OOS VaR 95%     : {summary['oos_var_95_pct']:.4f}%")
        print(f"{sep}\n")
        return {"summary":summary,"equity":all_equity,"trades":all_trades,
                "window_results":self._window_results}

    def _count_windows(self)->int:
        n=0; te=self._train_start
        while te<self.T:
            if self.T-te>=5: n+=1
            te+=self.step_bars
        return n

if __name__=="__main__":
    print("WalkForwardBacktest - import and instantiate with log-price pd.Series, then call run().")
