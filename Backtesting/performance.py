"""
Module  : Backtesting/performance.py
Project : Dark Wing — Quantitative Statistical Arbitrage Engine
Purpose : Quantitative Performance Analytics, Tail-Risk Metrics, and Audit Reporting
Style   : Institutional Quant (Compact, PEP8, Mathematical Finance Architecture)
"""
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Dict,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0,_THIS_DIR)

# ==================== Performance Analytics ===========================
class PerformanceAnalytics:
    """
    Computes institutional systematic-fund performance and tail-risk metrics:
      - Sharpe Ratio  : sqrt(252) * mean(r_excess) / std(r_excess)
      - Sortino Ratio : sqrt(252) * mean(r_excess) / std(r_excess[r < 0])
      - Calmar Ratio  : Ann.Return / MaxDrawdown
      - VaR 95%       : -percentile_5(r_t) [1-period 95% Value-at-Risk]
      - CVaR 95%      : -E[r_t | r_t < -VaR_95] [Expected Shortfall / Coherent Tail Risk]
      - Omega Ratio   : E[max(r - L, 0)] / E[max(L - r, 0)]
    """
    def __init__(self,equity_curve:pd.Series,trade_log:pd.DataFrame,
                 capital:float=1_000_000.0,risk_free_rate:float=0.045,
                 periods_per_year:Optional[int]=None,currency:str="USD"):
        if not isinstance(equity_curve,pd.Series):
            raise TypeError("equity_curve must be a pd.Series.")
        if len(equity_curve)<2:
            raise ValueError("equity_curve requires at least 2 points.")

        self.equity=equity_curve.copy()
        self.tlog=trade_log.copy() if not trade_log.empty else pd.DataFrame()
        self.capital=float(capital)
        self.rfr=float(risk_free_rate)
        self.currency=str(currency)

        # Standard daily trading periods per year = 252
        if periods_per_year is not None:
            self.ppy=max(int(periods_per_year),1)
        elif isinstance(equity_curve.index,pd.DatetimeIndex):
            self.ppy=252
        else:
            self.ppy=252

        self._returns:pd.Series=self.equity.pct_change().dropna()
        self._dd:pd.Series=self._calc_drawdown()

    def _calc_drawdown(self)->pd.Series:
        peak=self.equity.cummax()
        return ((peak-self.equity)/peak.replace(0,np.nan)).fillna(0.0)

    def sharpe(self)->float:
        if len(self._returns)<5: return 0.0
        r=self._returns
        rfr_period=self.rfr/self.ppy
        excess=r-rfr_period
        std=float(excess.std(ddof=1))
        if std<1e-5: return 0.0
        return float(np.sqrt(self.ppy)*excess.mean()/std)

    def sortino(self)->float:
        if len(self._returns)<5: return 0.0
        r=self._returns
        rfr_period=self.rfr/self.ppy
        excess=r-rfr_period
        downside=excess[excess<0]
        sigma_d=float(np.sqrt((downside**2).mean())) if len(downside)>0 else 0.0
        if sigma_d<1e-5: return 0.0
        return float(np.sqrt(self.ppy)*excess.mean()/sigma_d)

    def max_drawdown(self)->float:
        return float(self._dd.max())

    def calmar(self)->float:
        mdd=self.max_drawdown()
        return float(self.annualised_return()/mdd) if mdd>1e-10 else 0.0

    def annualised_return(self)->float:
        total_ret=(self.equity.iloc[-1]-self.equity.iloc[0])/self.equity.iloc[0]
        n_years=len(self.equity)/self.ppy
        return float((1+total_ret)**(1/max(n_years,1e-6))-1)

    def var(self,confidence:float=0.95)->float:
        """
        1-period Value-at-Risk at specified confidence level.
        VaR_α = -percentile_{(1 - α)*100}(r_t)
        Positive number denoting loss magnitude.
        """
        if len(self._returns)<5: return 0.0
        return float(-np.percentile(self._returns,(1-confidence)*100))

    def cvar(self,confidence:float=0.95)->float:
        """
        Conditional VaR (Expected Shortfall) at specified confidence level.
        CVaR_α = -E[r_t | r_t < -VaR_α].
        Guaranteed CVaR >= VaR (coherent risk metric).
        """
        if len(self._returns)<5: return 0.0
        var_val=self.var(confidence)
        tail=self._returns[self._returns<-var_val]
        return float(-tail.mean()) if len(tail)>0 else var_val

    def omega_ratio(self,threshold:float=0.0)->float:
        """
        Omega Ratio = E[max(r - L, 0)] / E[max(L - r, 0)]
        Incorporates higher moments without normality assumption.
        """
        r=self._returns
        gains=r[r>threshold]-threshold
        losses=threshold-r[r<=threshold]
        gain_sum=gains.sum() if len(gains)>0 else 0.0
        loss_sum=losses.sum() if len(losses)>0 else 1e-10
        return float(gain_sum/loss_sum) if loss_sum>1e-10 else float("inf")

    def win_rate(self)->float:
        if self.tlog.empty or "net_pnl" not in self.tlog.columns: return 0.0
        return float((self.tlog["net_pnl"]>0).mean())

    def profit_factor(self)->float:
        if self.tlog.empty or "net_pnl" not in self.tlog.columns: return 0.0
        wins=self.tlog["net_pnl"][self.tlog["net_pnl"]>0].sum()
        losses=self.tlog["net_pnl"][self.tlog["net_pnl"]<0].abs().sum()
        return float(wins/losses) if losses>1e-10 else float("inf")

    def avg_trade_duration(self)->float:
        if self.tlog.empty or "bars_in_trade" not in self.tlog.columns: return 0.0
        return float(self.tlog["bars_in_trade"].mean())

    def total_trades(self)->int:
        return len(self.tlog) if not self.tlog.empty else 0

    def all_metrics(self)->Dict:
        return {
            "sharpe":round(self.sharpe(),4),
            "sortino":round(self.sortino(),4),
            "calmar":round(self.calmar(),4),
            "max_drawdown_pct":round(self.max_drawdown()*100,2),
            "annualised_return_pct":round(self.annualised_return()*100,2),
            "var_95_pct":round(self.var(0.95)*100,4),
            "cvar_95_pct":round(self.cvar(0.95)*100,4),
            "omega_ratio":round(self.omega_ratio(),4),
            "win_rate_pct":round(self.win_rate()*100,2),
            "profit_factor":round(self.profit_factor(),4),
            "avg_trade_duration_bars":round(self.avg_trade_duration(),1),
            "total_trades":self.total_trades(),
            "initial_capital":self.capital,
            "final_capital":round(float(self.equity.iloc[-1]),2),
            "periods_per_year":self.ppy
        }

    def summary(self)->Dict:
        m=self.all_metrics()
        sep="="*55
        print(f"\n{sep}\n  Quantitative Performance & Risk Analytics\n{sep}")
        print(f"  Sharpe Ratio        : {m['sharpe']:>8.4f}")
        print(f"  Sortino Ratio       : {m['sortino']:>8.4f}")
        print(f"  Calmar Ratio        : {m['calmar']:>8.4f}")
        print(f"  Omega Ratio         : {m['omega_ratio']:>8.4f}")
        print(f"  Ann. Return         : {m['annualised_return_pct']:>7.2f}%")
        print(f"  Max Drawdown        : {m['max_drawdown_pct']:>7.2f}%")
        print(f"  VaR  95% (1-period) : {m['var_95_pct']:>7.4f}%")
        print(f"  CVaR 95% (Exp.Short): {m['cvar_95_pct']:>7.4f}%")
        print(f"  Win Rate            : {m['win_rate_pct']:>7.2f}%")
        print(f"  Profit Factor       : {m['profit_factor']:>8.4f}")
        print(f"  Total Trades        : {m['total_trades']:>6d}")
        print(f"  Avg Duration (bars) : {m['avg_trade_duration_bars']:>6.1f} days")
        print(f"  Final Capital       : {self.currency} {m['final_capital']:>12,.2f}")
        print(f"  Periods / Year (ppy): {m['periods_per_year']:>6d}  [daily swing]")
        print(f"{sep}\n")
        return m

    def plot(self,figsize:Tuple[int,int]=(16,12),save_path:Optional[str]=None):
        fig,axes=plt.subplots(2,2,figsize=figsize)
        fig.patch.set_facecolor("#12121e")
        colors_bg="#1e1e2e"
        color_line="#4a9eda"
        color_dd="#e74c3c"

        # Panel 1: Equity curve
        ax1=axes[0][0]; ax1.set_facecolor(colors_bg)
        ax1.plot(self.equity.index,self.equity.values,color=color_line,linewidth=1.2)
        ax1.axhline(self.capital,color="white",linewidth=0.6,linestyle="--",alpha=0.5)
        ax1.set_title("Strategy Equity Trajectory",fontsize=10,color="white",fontweight="bold")
        ax1.set_ylabel(self.currency,color="white",fontsize=9)
        ax1.tick_params(colors="gray")
        if isinstance(self.equity.index,pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
            plt.setp(ax1.xaxis.get_majorticklabels(),rotation=25)
        for sp in ax1.spines.values(): sp.set_color("#444")

        # Panel 2: Drawdown profile
        ax2=axes[0][1]; ax2.set_facecolor(colors_bg)
        ax2.fill_between(self._dd.index,0,self._dd.values*100,color=color_dd,alpha=0.75)
        ax2.set_title("Drawdown Depth (%)",fontsize=10,color="white",fontweight="bold")
        ax2.set_ylabel("DD %",color="white",fontsize=9)
        ax2.tick_params(colors="gray")
        if isinstance(self._dd.index,pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
            plt.setp(ax2.xaxis.get_majorticklabels(),rotation=25)
        for sp in ax2.spines.values(): sp.set_color("#444")

        # Panel 3: Trade PnL distribution
        ax3=axes[1][0]; ax3.set_facecolor(colors_bg)
        if not self.tlog.empty and "net_pnl" in self.tlog.columns:
            pnl=self.tlog["net_pnl"]
            cols=["#2ecc71" if v>0 else "#e74c3c" for v in pnl.values]
            ax3.bar(range(len(pnl)),pnl.values,color=cols,alpha=0.85,width=0.7)
            ax3.axhline(0,color="white",linewidth=0.6,linestyle="--",alpha=0.5)
        ax3.set_title("Trade Realised Net PnL",fontsize=10,color="white",fontweight="bold")
        ax3.set_xlabel("Trade #",color="white",fontsize=9)
        ax3.set_ylabel(self.currency,color="white",fontsize=9)
        ax3.tick_params(colors="gray")
        for sp in ax3.spines.values(): sp.set_color("#444")

        # Panel 4: Daily returns distribution with VaR & CVaR overlay
        ax4=axes[1][1]; ax4.set_facecolor(colors_bg)
        r=self._returns*100
        var95=self.var(0.95)*100
        cvar95=self.cvar(0.95)*100
        n_r,_,_=ax4.hist(r.values,bins=40,color="#4a9eda",alpha=0.65,density=True,label="Return dist.")
        ax4.axvline(-var95,color="#f39c12",linewidth=1.5,linestyle="--",label=f"VaR 95%: {var95:.3f}%")
        ax4.axvline(-cvar95,color="#e74c3c",linewidth=1.5,linestyle="-.",label=f"CVaR 95%: {cvar95:.3f}%")
        ax4.axvline(0,color="white",linewidth=0.6,alpha=0.5)
        ymax=n_r.max()*1.1 if len(n_r)>0 else 1.0
        ax4.fill_betweenx([0,ymax],r.min(),-var95,alpha=0.12,color="#e74c3c")
        ax4.set_ylim(0,ymax)
        ax4.set_title("Return Distribution & Tail Risk (VaR / CVaR)",fontsize=10,color="white",fontweight="bold")
        ax4.set_xlabel("Return %",color="white",fontsize=9)
        ax4.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white",loc="upper left")
        ax4.tick_params(colors="gray")
        for sp in ax4.spines.values(): sp.set_color("#444")

        fig.suptitle("Dark Wing — Quantitative Performance Audit",fontsize=13,fontweight="bold",color="white")
        if save_path:
            plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
            plt.close(fig)
        else:
            try: plt.show()
            except Exception: plt.close(fig)

    def __repr__(self)->str:
        return (f"PerformanceAnalytics(trades={self.total_trades()}, "
                f"sharpe={self.sharpe():.3f}, mdd={self.max_drawdown()*100:.1f}%, "
                f"var95={self.var(0.95)*100:.3f}%, cvar95={self.cvar(0.95)*100:.3f}%)")

# =====================================================================
if __name__=="__main__":
    print("PerformanceAnalytics — Institutional performance and tail-risk reporting.")
