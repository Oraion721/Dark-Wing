"""
Module  : Trade_Implement/Signals/multi_pair_signal.py
Project : Dark Wing
Purpose : Portfolio-level aggregator — runs PairSignalGenerator across N pairs, weights by
          inverse-variance, caps by correlation and max concurrent pairs.
Math:
    w_i = (1/σ²_i) / Σ_j(1/σ²_j)            (inv-variance weight)
    Consensus_t = sign(Σ_i w_i·s_{i,t})       (weighted vote)
    Exposure_adj = sqrt(wᵀΣw) ≤ MaxExposure   (correlation-adjusted cap)
Pipeline:
    STEP 8.1 → pair_signal.py (PairSignalGenerator)
    STEP 8.2 → THIS FILE (MultiPairSignalAggregator)
    STEP 9   → Risk_Management/ """
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict,List,Optional,Tuple

_THIS_DIR   =os.path.dirname(os.path.abspath(__file__))
_TRADE_DIR  =os.path.dirname(_THIS_DIR)
_ROOT_DIR   =os.path.dirname(_TRADE_DIR)
_FM_DIR     =os.path.join(_ROOT_DIR,"Financial_Mathematics")
_TSA_DIR    =os.path.join(_FM_DIR,"Time_Series_Analysis")
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
_KALMAN_DIR =os.path.join(_TSA_DIR,"Kalman_Filter")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_KALMAN_DIR,_THIS_DIR]:
    if _p not in sys.path: sys.path.insert(0,_p)

from zscore import ZScore
from dynamic_beta import DynamicBeta
from kalman_forecast import KalmanForecaster
from pair_signal import PairSignalGenerator

# ─────────────────────────────────────────────────────────────────────────────
class MultiPairSignalAggregator:
    def __init__(self,capital:float=1_000_000.0,max_pairs_open:int=5,correlation_cap:float=0.80,
                 max_exposure:float=0.30,use_forecast_filter:bool=True,use_regime_filter:bool=True,
                 forecast_steps:int=5):
        if capital<=0: raise ValueError("capital must be positive.")
        if max_pairs_open<1: raise ValueError("max_pairs_open >= 1.")
        if not 0<correlation_cap<=1: raise ValueError("correlation_cap in (0,1].")
        self.capital=float(capital); self.max_pairs_open=int(max_pairs_open)
        self.correlation_cap=float(correlation_cap); self.max_exposure=float(max_exposure)
        self.use_forecast_filter=use_forecast_filter; self.use_regime_filter=use_regime_filter
        self.forecast_steps=int(forecast_steps)
        self._pairs:Dict[str,Dict]={}
        self._signal_frames:Dict[str,pd.DataFrame]={}
        self._weights:Optional[pd.Series]=None
        self._portfolio_signals:Optional[pd.DataFrame]=None
        self._aggregate:Optional[pd.Series]=None
        self._is_run=False

    def add_pair(self,name:str,zscore:ZScore,dynamic_beta:DynamicBeta,
                 kalman_forecaster:Optional[KalmanForecaster]=None)->"MultiPairSignalAggregator":
        if not isinstance(zscore,ZScore): raise TypeError(f"'{name}': zscore must be ZScore.")
        if not isinstance(dynamic_beta,DynamicBeta): raise TypeError(f"'{name}': dynamic_beta must be DynamicBeta.")
        if kalman_forecaster is not None and not isinstance(kalman_forecaster,KalmanForecaster):
            raise TypeError(f"'{name}': kalman_forecaster must be KalmanForecaster or None.")
        if name in self._pairs: warnings.warn(f"Pair '{name}' already registered. Overwriting.")
        self._pairs[name]={"zscore":zscore,"dynamic_beta":dynamic_beta,"kalman_forecaster":kalman_forecaster}
        self._is_run=False
        return self

    def run_all(self)->Dict[str,pd.DataFrame]:
        if not self._pairs: raise RuntimeError("No pairs registered. Use add_pair() first.")
        self._signal_frames={}
        for name,cfg in self._pairs.items():
            try:
                gen=PairSignalGenerator(zscore=cfg["zscore"],dynamic_beta=cfg["dynamic_beta"],
                                        kalman_forecaster=cfg["kalman_forecaster"],
                                        forecast_steps=self.forecast_steps,
                                        use_forecast_filter=self.use_forecast_filter,
                                        use_regime_filter=self.use_regime_filter)
                self._signal_frames[name]=gen.generate()
            except Exception as exc:
                warnings.warn(f"[{name}] Signal generation failed: {exc}")
                self._signal_frames[name]=pd.DataFrame()
        self._portfolio_signals=self._build_signal_matrix()
        self._weights=self._compute_weights()
        self._aggregate=self._compute_aggregate_signal()
        self._is_run=True
        return self._signal_frames

    def _build_signal_matrix(self)->pd.DataFrame:
        frames={n:df["signal"] for n,df in self._signal_frames.items() if not df.empty and "signal" in df.columns}
        if not frames: raise RuntimeError("No valid signal frames generated.")
        return pd.DataFrame(frames).fillna(0).astype(int).sort_index()

    def _compute_weights(self)->pd.Series:
        variances={}
        for name,cfg in self._pairs.items():
            try:
                z=cfg["zscore"].zscore_series(); n=min(252,len(z))
                variances[name]=max(float(np.var(z.iloc[-n:],ddof=1)),1e-8)
            except: variances[name]=1.0
        inv_v={k:1.0/v for k,v in variances.items()}; total=sum(inv_v.values())
        return pd.Series({k:v/total for k,v in inv_v.items()},dtype=float)

    def _compute_aggregate_signal(self)->pd.Series:
        self._require_fitted("_compute_aggregate_signal")
        mat=self._portfolio_signals.copy().astype(float)
        z_frames={n:cfg["zscore"].zscore_series().reindex(mat.index).fillna(0).abs()
                  for n,cfg in self._pairs.items() if True}
        z_mat=pd.DataFrame({n: (cfg["zscore"].zscore_series().reindex(mat.index).fillna(0).abs()
                                if True else pd.Series(0.0,index=mat.index))
                            for n,cfg in self._pairs.items()}).fillna(0)
        # max_pairs_open constraint
        mat_capped=mat.copy()
        for idx in mat.index:
            active=mat.loc[idx]!=0
            if active.sum()>self.max_pairs_open:
                ranked=z_mat.loc[idx][active].nlargest(self.max_pairs_open).index
                mat_capped.loc[idx,active[active].index.difference(ranked)]=0
        # weighted vote
        w=self._weights.reindex(mat_capped.columns).fillna(0)
        if w.sum()>0: w=w/w.sum()
        consensus=np.sign(mat_capped.mul(w,axis=1).sum(axis=1)).astype(int)
        # correlation-adjusted exposure cap
        if z_mat.shape[1]>1:
            corr=z_mat.corr().fillna(0); w_arr=w.values
            exp_adj=float(np.sqrt(w_arr@corr.values@w_arr))
            if exp_adj>self.max_exposure and exp_adj>0:
                scale=self.max_exposure/exp_adj
                consensus=(consensus*scale).apply(lambda v: int(np.sign(v)) if abs(v)>=0.5 else 0)
        return pd.Series(consensus,index=mat.index,name="aggregate_signal")

    def portfolio_signals(self)->pd.DataFrame:
        self._require_fitted("portfolio_signals"); return self._portfolio_signals

    def aggregate_signal(self)->pd.Series:
        self._require_fitted("aggregate_signal"); return self._aggregate

    def capital_weights(self)->pd.Series:
        self._require_fitted("capital_weights"); return self._weights

    def active_pairs(self)->List[str]:
        self._require_fitted("active_pairs")
        latest=self._portfolio_signals.iloc[-1]
        return list(latest[latest!=0].index)

    def plot_all(self,figsize:Tuple[int,int]=(18,4),save_path:Optional[str]=None):
        self._require_fitted("plot_all")
        n_pairs=len(self._signal_frames)
        if n_pairs==0: warnings.warn("No pairs to plot."); return
        n_cols=min(n_pairs,3); n_rows=(n_pairs+n_cols-1)//n_cols
        fig,axes=plt.subplots(n_rows,n_cols,figsize=(figsize[0],figsize[1]*n_rows),squeeze=False)
        axes_flat=axes.flatten(); fig.patch.set_facecolor("#12121e")
        for i,(name,df) in enumerate(self._signal_frames.items()):
            ax=axes_flat[i]; ax.set_facecolor("#1e1e2e")
            if df.empty or "z" not in df.columns: ax.set_title(f"{name}\n(no data)"); ax.axis("off"); continue
            z=df["z"]; zs=self._pairs[name]["zscore"]
            ez,xz,sz=zs.entry_z,zs.exit_z,zs.stop_z
            ax.plot(z.index,z.values,color="#4a9eda",linewidth=0.9)
            for lvl,col,ls in [(ez,"#e74c3c","--"),(-ez,"#e74c3c","--"),(xz,"#2ecc71",":"),(-xz,"#2ecc71",":"),(sz,"#8e44ad","-."),(-sz,"#8e44ad","-.")]:
                ax.axhline(lvl,color=col,linewidth=0.8,linestyle=ls,alpha=0.8)
            sig=df["signal"]
            ax.fill_between(z.index,z.min(),z.max(),where=(sig==1),color="#2ecc71",alpha=0.10)
            ax.fill_between(z.index,z.min(),z.max(),where=(sig==-1),color="#e74c3c",alpha=0.10)
            ax.set_title(f"{name}",fontsize=10,fontweight="bold",color="white")
            ax.tick_params(colors="gray")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
            plt.setp(ax.xaxis.get_majorticklabels(),rotation=30)
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")
        for i in range(n_pairs,len(axes_flat)): axes_flat[i].set_visible(False)
        fig.suptitle("Multi-Pair Signal Dashboard — Dark Wing",fontsize=14,fontweight="bold",color="white",y=1.01)
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
        plt.show()

    def plot_aggregate(self,figsize:Tuple[int,int]=(16,6),save_path:Optional[str]=None):
        self._require_fitted("plot_aggregate")
        fig,axes=plt.subplots(2,1,figsize=figsize,gridspec_kw={"height_ratios":[2,1]})
        fig.patch.set_facecolor("#12121e")
        ax1=axes[0]; mat=self._portfolio_signals
        im=ax1.imshow(mat.T.values,aspect="auto",cmap="RdYlGn",vmin=-1,vmax=1,interpolation="none")
        ax1.set_yticks(range(len(mat.columns))); ax1.set_yticklabels(mat.columns,fontsize=9,color="white")
        ax1.set_xticks([]); ax1.set_title("Per-Pair Signal Heatmap  (Green=LONG, Red=SHORT)",fontsize=10,color="white")
        ax1.set_facecolor("#1e1e2e"); plt.colorbar(im,ax=ax1,orientation="vertical",fraction=0.02)
        ax2=axes[1]; agg=self._aggregate; ax2.set_facecolor("#1e1e2e")
        ax2.step(agg.index,agg.values,color="#4a9eda",linewidth=1.2,where="post")
        ax2.fill_between(agg.index,0,agg.values,where=(agg.values>0),color="#2ecc71",alpha=0.35,step="post")
        ax2.fill_between(agg.index,0,agg.values,where=(agg.values<0),color="#e74c3c",alpha=0.35,step="post")
        ax2.axhline(0,color="white",linewidth=0.7,linestyle="--")
        ax2.set_yticks([-1,0,1]); ax2.set_yticklabels(["SHORT","FLAT","LONG"],fontsize=8,color="white")
        ax2.set_title("Portfolio Consensus Signal",fontsize=10,color="white"); ax2.tick_params(colors="gray")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y")); plt.setp(ax2.xaxis.get_majorticklabels(),rotation=30)
        fig.suptitle("Multi-Pair Portfolio Signal Overview — Dark Wing",fontsize=13,fontweight="bold",color="white")
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e")
        plt.show()

    def summary(self)->dict:
        self._require_fitted("summary")
        z_frames={n:cfg["zscore"].zscore_series() for n,cfg in self._pairs.items()}
        z_mat=pd.DataFrame(z_frames).fillna(0); corr=z_mat.corr().fillna(0)
        agg=self._aggregate; counts={"LONG":int((agg==1).sum()),"SHORT":int((agg==-1).sum()),"FLAT":int((agg==0).sum())}
        avg_z={n:round(float(z.abs().iloc[-60:].mean()),4) for n,z in z_frames.items()}
        result={"n_pairs":len(self._pairs),"n_active_now":len(self.active_pairs()),
                "active_pairs_now":self.active_pairs(),"weights":self._weights.to_dict(),
                "corr_matrix":corr,"aggregate_counts":counts,"avg_z_by_pair":avg_z}
        print(f"\n{'='*55}\n  MultiPairSignalAggregator — Summary\n{'='*55}")
        print(f"  Pairs registered : {result['n_pairs']}")
        print(f"  Pairs active now : {result['n_active_now']}  →  {result['active_pairs_now']}")
        print(f"\n  Capital weights (inv-variance):")
        for n,w in result["weights"].items(): print(f"    {n:<30s}  {w:.4f}")
        print(f"\n  Aggregate signal:  LONG={counts['LONG']}  SHORT={counts['SHORT']}  FLAT={counts['FLAT']}")
        print(f"\n  Avg |Z| (last 60 bars):")
        for n,z in avg_z.items(): print(f"    {n:<30s}  {z:.4f}")
        print(f"{'='*55}\n")
        return result

    def _require_fitted(self,caller:str):
        if not self._is_run:
            raise RuntimeError(f"{caller}() requires run_all() first. Register pairs with add_pair().")

    def __repr__(self)->str:
        return (f"MultiPairSignalAggregator(pairs={list(self._pairs.keys())}, "
                f"capital={self.capital:,.0f}, max_pairs_open={self.max_pairs_open}, "
                f"status={'fitted' if self._is_run else 'not fitted'})")

# ── Standalone ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    print("MultiPairSignalAggregator — add_pair() then run_all() to generate portfolio signals.")
