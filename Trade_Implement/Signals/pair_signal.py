""" Signal Rules:
        Z_t >  stop_z  → SHORT stop (-2)    Z_t < -stop_z → LONG stop (+2)
        Z_t >  entry_z → SHORT (-1)         Z_t < -entry_z → LONG (+1)
        |Z_t| < exit_z → EXIT (0)           else → carry previous (ffill)
Regime CUSUM filter: CUSUM_t = rolling_sum(|Δβ_t|, window) → suppress if top 10%
Forecast filter:  suppress LONG if Ŝ_{t+h} < S_t, suppress SHORT if Ŝ_{t+h} > S_t   """
import sys,os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional,Tuple

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
LONG=1; SHORT=-1; FLAT=0; STOP_LONG=2; STOP_SHORT=-2

class PairSignalGenerator:
    def __init__(self,zscore:ZScore,dynamic_beta:DynamicBeta,kalman_forecaster:Optional[KalmanForecaster]=None,forecast_steps:int=5,use_forecast_filter:bool=True,use_regime_filter:bool=True,regime_window:int=20,pair_name:str="Pair",max_z_velocity:Optional[float]=None):
        if not isinstance(zscore,ZScore): raise TypeError("zscore must be ZScore.")
        if not isinstance(dynamic_beta,DynamicBeta): raise TypeError("dynamic_beta must be DynamicBeta.")
        if kalman_forecaster is not None and not isinstance(kalman_forecaster,KalmanForecaster):
            raise TypeError("kalman_forecaster must be KalmanForecaster or None.")
        if forecast_steps<1: raise ValueError("forecast_steps >= 1.")
        if regime_window<5: raise ValueError("regime_window >= 5.")
        self._zs=zscore; self._db=dynamic_beta; self._kf=kalman_forecaster
        self.forecast_steps=int(forecast_steps)
        self.use_forecast_filter=use_forecast_filter and (kalman_forecaster is not None)
        self.use_regime_filter=use_regime_filter
        self.regime_window=int(regime_window); self.pair_name=str(pair_name)
        self.max_z_velocity=float(max_z_velocity) if max_z_velocity is not None else None
        self._signals_df:Optional[pd.DataFrame]=None
        self._regime_flag:Optional[pd.Series]=None
        self._is_fitted=False
        self._n_regime_suppressed=0; self._n_forecast_filtered=0

    def generate(self)->pd.DataFrame:
        z_series=self._zs.zscore_series()
        if z_series is None or len(z_series)==0:
            raise RuntimeError("ZScore has no series. Call zscore_series() first.")
        spread_series=self._zs._spread.reindex(z_series.index)
        raw_signal=self._apply_thresholds(z_series)
        if self.max_z_velocity is not None:
            # Suppress opening fresh positions on bars with violent spread velocity (momentum shocks)
            dz=z_series.diff().abs()
            violent_shock=dz>self.max_z_velocity
            raw_signal=raw_signal.copy()
            raw_signal[violent_shock]=0
        if self.use_regime_filter:
            filtered_signal=self._apply_regime_filter(raw_signal,z_series.index)
        else:
            filtered_signal=raw_signal.copy(); self._regime_flag=pd.Series(False,index=z_series.index)
        if self.use_forecast_filter and self._kf is not None:
            final_signal=self._apply_forecast_filter(filtered_signal)
        else:
            final_signal=filtered_signal.copy(); self._n_forecast_filtered=0
        position_series=self._build_position_series(final_signal)
        beta_series=self._get_beta_series(z_series.index)
        df=pd.DataFrame({"z":z_series,"spread":spread_series,"signal":final_signal,"position":position_series,"beta":beta_series,"regime_flag":self._regime_flag.reindex(z_series.index).fillna(False)},index=z_series.index)
        df["signal"]=df["signal"].fillna(0).astype(int)
        df["position"]=df["position"].fillna(0).astype(int)
        df["regime_flag"]=df["regime_flag"].astype(bool)
        self._signals_df=df; self._is_fitted=True
        return df

    def signal_series(self)->pd.Series:
        self._require_fitted("signal_series"); return self._signals_df["signal"]

    def position_series(self)->pd.Series:
        self._require_fitted("position_series"); return self._signals_df["position"]

    def current_signal(self)->dict:
        self._require_fitted("current_signal")
        last=self._signals_df.iloc[-1]
        return {"signal":int(last["signal"]),"position":int(last["position"]),"z_score":float(last["z"]),"spread":float(last["spread"]),"beta":float(last["beta"]),"regime_flag":bool(last["regime_flag"]),"timestamp":self._signals_df.index[-1]}

    def _apply_thresholds(self,z:pd.Series)->pd.Series:
        ez=self._zs.entry_z; xz=self._zs.exit_z; sz=self._zs.stop_z
        sig=pd.Series(np.nan,index=z.index,dtype=float)
        sig[z>sz]=float(STOP_SHORT); sig[z<-sz]=float(STOP_LONG)
        sig[(z>ez)&(z<=sz)]=float(SHORT); sig[(z<-ez)&(z>=-sz)]=float(LONG)
        sig[z.abs()<xz]=float(FLAT)
        return sig.ffill().fillna(0.0).astype(int)

    def _apply_regime_filter(self,signal:pd.Series,idx:pd.Index)->pd.Series:
        beta=self._get_beta_series(idx)
        cusum=beta.diff().abs().rolling(window=self.regime_window,min_periods=self.regime_window//2).sum()
        regime_flag=cusum>cusum.quantile(0.90)
        self._regime_flag=regime_flag.reindex(idx).fillna(False)
        filtered=signal.copy().astype(float)
        n_before=int((filtered!=0).sum())
        filtered[self._regime_flag]=0.0
        self._n_regime_suppressed=n_before-int((filtered!=0).sum())
        return filtered.astype(int)

    def _apply_forecast_filter(self,signal:pd.Series)->pd.Series:
        filtered=signal.copy()
        last_signal=int(signal.iloc[-1])
        if last_signal==FLAT: return filtered
        try:
            beta_series=self._get_beta_series(signal.index)
            last_x=float(self._db._x.iloc[-1]) if (hasattr(self._db,"_x") and self._db._x is not None) else float(beta_series.iloc[-1])
            future_x=np.array([last_x]*self.forecast_steps,dtype=float)
            fc=self._kf.forecast_spread(future_x=future_x,steps=self.forecast_steps)
            last_fc=float(fc.iloc[-1] if isinstance(fc,pd.Series) else (fc[-1] if isinstance(fc,np.ndarray) else fc))
            if (last_signal==LONG and last_fc<0) or (last_signal==SHORT and last_fc>0):
                filtered.iloc[-1]=FLAT; self._n_forecast_filtered+=1
        except Exception as exc:
            warnings.warn(f"[PairSignalGenerator] Forecast filter failed: {exc}. Skipping.")
        return filtered

    def _build_position_series(self,signal:pd.Series)->pd.Series:
        sig_arr=signal.values.astype(int); pos_arr=np.zeros(len(sig_arr),dtype=int); state=0
        for i,s in enumerate(sig_arr):
            if abs(s)==2: state=0
            elif s==FLAT: state=0
            elif s==LONG: state=LONG
            elif s==SHORT: state=SHORT
            pos_arr[i]=state
        return pd.Series(pos_arr,index=signal.index,name="position")

    def _get_beta_series(self,idx:pd.Index)->pd.Series:
        beta=None
        if hasattr(self._db,"_states_df") and self._db._states_df is not None and "beta" in self._db._states_df.columns:
            beta=self._db._states_df["beta"]
        if beta is None and hasattr(self._db,"filter_result") and self._db.filter_result is not None:
            fr=self._db.filter_result
            beta=fr["beta"] if isinstance(fr,pd.DataFrame) and "beta" in fr.columns else (fr if isinstance(fr,pd.Series) else None)
        if beta is None:
            try: beta=pd.Series(float(self._db.current_beta()),index=idx)
            except: beta=pd.Series(1.0,index=idx)
        return beta.reindex(idx).ffill().bfill()

    def plot(self,figsize:Tuple[int,int]=(16,10),save_path:Optional[str]=None):
        self._require_fitted("plot")
        df=self._signals_df; z=df["z"]; spread=df["spread"]; sig=df["signal"]; pos=df["position"]; regime=df["regime_flag"]
        ez=self._zs.entry_z; xz=self._zs.exit_z; sz=self._zs.stop_z
        fig,axes=plt.subplots(3,1,figsize=figsize,sharex=True,gridspec_kw={"height_ratios":[2,2,1]})
        fig.patch.set_facecolor("#12121e")
        ax1=axes[0]; ax1.set_facecolor("#1e1e2e")
        ax1.plot(spread.index,spread.values,color="#7ecfff",linewidth=0.9,label="Spread $S_t$")
        ax1.axhline(0,color="white",linewidth=0.6,linestyle="--",alpha=0.5)
        le=sig[sig==LONG].index; se=sig[sig==SHORT].index
        if len(le): ax1.scatter(le,spread.reindex(le),marker="^",color="#2ecc71",s=30,zorder=5,label="LONG entry")
        if len(se): ax1.scatter(se,spread.reindex(se),marker="v",color="#e74c3c",s=30,zorder=5,label="SHORT entry")
        ax1.set_ylabel("Spread $S_t$",color="white",fontsize=9); ax1.set_title(f"Pair Signal — {self.pair_name}",fontsize=12,fontweight="bold",color="white")
        ax1.tick_params(colors="gray"); ax1.legend(fontsize=8,facecolor="#2a2a3e",labelcolor="white")
        for sp in ax1.spines.values(): sp.set_color("#444")
        ax2=axes[1]; ax2.set_facecolor("#1e1e2e")
        ax2.plot(z.index,z.values,color="#4a9eda",linewidth=0.9,label="Z-score $Z_t$")
        for lvl,col,lbl in [(ez,"#e74c3c",f"+entry"),(- ez,"#e74c3c",f"-entry"),(xz,"#2ecc71","+exit"),(-xz,"#2ecc71","-exit"),(sz,"#8e44ad","+stop"),(-sz,"#8e44ad","-stop")]:
            ax2.axhline(lvl,color=col,linewidth=0.8,linestyle="--",alpha=0.8,label=lbl)
        ax2.fill_between(z.index,z.min(),z.max(),where=(pos==1),color="#2ecc71",alpha=0.08,step="mid")
        ax2.fill_between(z.index,z.min(),z.max(),where=(pos==-1),color="#e74c3c",alpha=0.08,step="mid")
        if regime.any(): ax2.fill_between(z.index,z.min(),z.max(),where=regime.values,color="#888888",alpha=0.12,step="mid",label="Regime suppressed")
        ax2.set_ylabel("Z-score $Z_t$",color="white",fontsize=9); ax2.axhline(0,color="white",linewidth=0.4,alpha=0.4)
        ax2.tick_params(colors="gray"); ax2.legend(fontsize=7,facecolor="#2a2a3e",labelcolor="white",ncol=2)
        for sp in ax2.spines.values(): sp.set_color("#444")
        ax3=axes[2]; ax3.set_facecolor("#1e1e2e")
        ax3.step(pos.index,pos.values,color="#f39c12",linewidth=1.2,where="post")
        ax3.fill_between(pos.index,0,pos.values,where=(pos.values>0),color="#2ecc71",alpha=0.3,step="post")
        ax3.fill_between(pos.index,0,pos.values,where=(pos.values<0),color="#e74c3c",alpha=0.3,step="post")
        ax3.axhline(0,color="white",linewidth=0.5,linestyle="--",alpha=0.5)
        ax3.set_yticks([-1,0,1]); ax3.set_yticklabels(["SHORT","FLAT","LONG"],fontsize=8,color="white")
        ax3.set_ylabel("Position",color="white",fontsize=9); ax3.set_xlabel("Date",color="white",fontsize=9)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y")); plt.setp(ax3.xaxis.get_majorticklabels(),rotation=30)
        ax3.tick_params(colors="gray")
        for sp in ax3.spines.values(): sp.set_color("#444")
        plt.tight_layout()
        if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight",facecolor="#12121e"); print(f"[PairSignalGenerator] Saved → {save_path}")
        plt.show()

    def summary(self)->dict:
        self._require_fitted("summary")
        df=self._signals_df; sig=df["signal"]; pos=df["position"]; n=len(df)
        n_long=int((sig==1).sum()); n_short=int((sig==-1).sum())
        n_stop=int((sig.abs()==2).sum()); n_flat=int((sig==0).sum())
        n_lp=int((pos==1).sum()); n_sp=int((pos==-1).sum())
        avg_z=float(df["z"].abs().mean())
        n_trades=int(pos.diff().ne(0).sum()); avg_dur=round(n/max(n_trades,1),2)
        result={"total_bars":n,"long_signals":n_long,"short_signals":n_short,"stop_signals":n_stop,
                "flat_signals":n_flat,"long_positions":n_lp,"short_positions":n_sp,
                "regime_suppressed":self._n_regime_suppressed,"regime_suppressed_pct":round(100*self._n_regime_suppressed/max(n,1),2),
                "forecast_filtered":self._n_forecast_filtered,"avg_abs_z":round(avg_z,4),"avg_signal_duration":avg_dur}
        sep="="*52
        print(f"\n{sep}\n  PairSignalGenerator Summary — {self.pair_name}\n{sep}")
        print(f"  Total bars          : {n}")
        print(f"  LONG/SHORT signals  : {n_long} / {n_short}")
        print(f"  STOP signals        : {n_stop}    FLAT: {n_flat}")
        print(f"  Regime suppressed   : {self._n_regime_suppressed}  ({result['regime_suppressed_pct']:.1f}%)")
        print(f"  Forecast filtered   : {self._n_forecast_filtered}")
        print(f"  Avg |Z_t|           : {avg_z:.4f}    Avg duration: {avg_dur} bars\n{sep}\n")
        return result

    def _require_fitted(self,caller:str):
        if not self._is_fitted: raise RuntimeError(f"{caller}() requires generate() first.")

    def __repr__(self)->str:
        return f"PairSignalGenerator(pair='{self.pair_name}', status={'fitted' if self._is_fitted else 'not fitted'})"

# ── Standalone ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    print("PairSignalGenerator — run generate() after wiring ZScore, DynamicBeta, KalmanForecaster.")
