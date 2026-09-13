import sys,os,warnings,time,csv,datetime
import numpy as np
import pandas as pd
from typing import Optional,Dict,List,Tuple

_THIS_DIR   =os.path.dirname(os.path.abspath(__file__))
_TRADE_DIR  =os.path.dirname(_THIS_DIR)
_ROOT_DIR   =os.path.dirname(_TRADE_DIR)
_FM_DIR     =os.path.join(_ROOT_DIR,"Financial_Mathematics")
_TSA_DIR    =os.path.join(_FM_DIR,"Time_Series_Analysis")
_SPREADS_DIR=os.path.join(_TSA_DIR,"Spreads")
_KALMAN_DIR =os.path.join(_TSA_DIR,"Kalman_Filter")
_SIGNALS_DIR=os.path.join(_TRADE_DIR,"Signals")
_RISK_DIR   =os.path.join(_TRADE_DIR,"Risk_Management")
_BACKTEST_DIR=os.path.join(_ROOT_DIR,"Backtesting")
for _p in [_FM_DIR,_TSA_DIR,_SPREADS_DIR,_KALMAN_DIR,_SIGNALS_DIR,_RISK_DIR,_BACKTEST_DIR,_THIS_DIR]:
    if _p not in sys.path: sys.path.insert(0,_p)

try:
    from SmartApi import SmartConnect
    import pyotp
    _ANGEL_AVAILABLE=True
except ImportError:
    warnings.warn("[AngelExecutor] smartapi-python not installed. Run: pip install smartapi-python pyotp. Running dry-run.")
    _ANGEL_AVAILABLE=False

from zscore import ZScore
from half_life import HalfLife
from pair_signal import PairSignalGenerator
from position_sizing import PositionSizer
from stoploss import StopLossManager
from performance import PerformanceAnalytics

# ─────────────────────────────────────────────────────────────────────────────
_LOG_FILE=os.path.join(_THIS_DIR,"live_trade_log_angel.csv")
_LOG_FIELDS=["timestamp","direction","symbol_y","symbol_x","N_Y","N_X","S_entry","S_exit","Z_entry",
             "gross_pnl","tc","net_pnl","close_reason","capital_after"]

# ─────────────────────────────────────────────────────────────────────────────
class AngelExecutor:
    # Live/paper execution engine for NSE equity pairs via Angel One SmartAPI.
    # Polls LTP at interval_sec cadence, computes Z-score, generates signal,
    # places MARKET orders via SmartConnect.place_order().
    # Product type: 'MIS' (intraday) or 'CNC' (delivery/positional).
    # paper_trading=True → dry-run mode (no real orders placed).

    def __init__(self,signal_gen:PairSignalGenerator,position_sizer:PositionSizer,
                 stop_loss_mgr:StopLossManager,
                 symbol_y:str,token_y:str,symbol_x:str,token_x:str,
                 api_key:str,client_id:str,totp_secret:str,
                 capital:float=1_000_000.0,exchange:str="NSE",
                 product_type:str="MIS",commission_pct:float=0.0003,
                 pair_name:str="Pair",paper_trading:bool=True):
        self._gen=signal_gen; self._sizer=position_sizer; self._sl=stop_loss_mgr
        self.symbol_y=symbol_y; self.token_y=token_y
        self.symbol_x=symbol_x; self.token_x=token_x
        self.api_key=api_key; self.client_id=client_id; self.totp_secret=totp_secret
        self.capital=float(capital); self.exchange=exchange; self.product_type=product_type
        self.commission_pct=float(commission_pct); self.pair_name=pair_name
        self.paper_trading=paper_trading or not _ANGEL_AVAILABLE
        self._client:Optional["SmartConnect"]=None
        self._position:int=0; self._N_Y:int=0; self._N_X:int=0
        self._S_entry:float=0.0; self._Z_entry:float=0.0
        self._equity:List[float]=[capital]; self._equity_ts:List=[]
        self._closed_trades:List[Dict]=[]
        self._running=False; self._s_history:List[float]=[]
        os.makedirs(_THIS_DIR,exist_ok=True)
        if not os.path.exists(_LOG_FILE):
            with open(_LOG_FILE,"w",newline="") as f: csv.DictWriter(f,fieldnames=_LOG_FIELDS).writeheader()

    def connect(self)->bool:
        if self.paper_trading: print("[AngelExecutor] paper_trading=True. Skipping SmartAPI login."); return True
        if not _ANGEL_AVAILABLE: return False
        try:
            totp=pyotp.TOTP(self.totp_secret).now()
            self._client=SmartConnect(api_key=self.api_key)
            data=self._client.generateSession(self.client_id,password="",totp=totp)
            if data.get("status"):
                print(f"[AngelExecutor] Connected as {self.client_id}"); return True
            warnings.warn(f"[AngelExecutor] Login failed: {data.get('message','')}"); return False
        except Exception as e:
            warnings.warn(f"[AngelExecutor] Connection error: {e}"); return False

    def _get_prices(self)->Tuple[Optional[float],Optional[float]]:
        if self.paper_trading or self._client is None: return None,None
        try:
            r_y=self._client.ltpData(self.exchange,self.symbol_y,self.token_y)
            r_x=self._client.ltpData(self.exchange,self.symbol_x,self.token_x)
            P_Y=r_y["data"]["ltp"] if r_y.get("status") else None
            P_X=r_x["data"]["ltp"] if r_x.get("status") else None
            return float(P_Y) if P_Y else None,float(P_X) if P_X else None
        except Exception as e:
            warnings.warn(f"[AngelExecutor] Price fetch error: {e}"); return None,None

    def _get_beta(self)->float:
        try:
            db=self._gen._db
            if hasattr(db,"_states_df") and db._states_df is not None and "beta" in db._states_df.columns:
                return float(db._states_df["beta"].iloc[-1])
            return float(db.current_beta())
        except: return 1.0

    def _compute_zscore(self,S_t:float)->Optional[float]:
        zs=self._gen._zs
        try:
            if hasattr(zs,"_mu") and hasattr(zs,"_sigma") and zs._sigma and abs(zs._sigma)>1e-10:
                return (S_t-zs._mu)/zs._sigma
        except: pass
        if len(self._s_history)<10: return None
        arr=np.array(self._s_history[-100:])
        mu,sigma=arr.mean(),arr.std(ddof=1)
        return (S_t-mu)/sigma if sigma>1e-10 else None

    def _get_signal(self,Z_t:float)->int:
        zs=self._gen._zs; ez=zs.entry_z; xz=zs.exit_z; sz=zs.stop_z
        if Z_t>sz: return -2
        if Z_t<-sz: return 2
        if Z_t>ez: return -1
        if Z_t<-ez: return 1
        if abs(Z_t)<xz: return 0
        return 0

    def _place_order(self,symbol:str,token:str,action:str,qty:int,price:float)->bool:
        if self.paper_trading:
            print(f"  [DRY-RUN] {action} {qty}x{symbol}@{price:.2f}"); return True
        if self._client is None: return False
        try:
            params={"variety":"NORMAL","tradingsymbol":symbol,"symboltoken":token,
                    "transactiontype":action.upper(),"exchange":self.exchange,
                    "ordertype":"MARKET","producttype":self.product_type,
                    "duration":"DAY","quantity":str(qty),"price":"0","squareoff":"0","stoploss":"0"}
            resp=self._client.placeOrder(params)
            print(f"  [ORDER] {action} {qty}x{symbol} | orderId={resp}"); return True
        except Exception as e:
            warnings.warn(f"[AngelExecutor] Order error: {e}"); return False

    def _place_pair_order(self,signal:int,N_Y:int,N_X:int,P_Y:float,P_X:float)->bool:
        ay,ax=("BUY","SELL") if signal==1 else ("SELL","BUY")
        ok_y=self._place_order(self.symbol_y,self.token_y,ay,N_Y,P_Y)
        ok_x=self._place_order(self.symbol_x,self.token_x,ax,N_X,P_X)
        return ok_y and ok_x

    def _close_position(self,S_exit:float,P_Y:float,P_X:float,reason:str,ts):
        pnl=(self._N_Y*(S_exit-self._S_entry) if self._position==1 else self._N_Y*(self._S_entry-S_exit))
        tc=(self._N_Y*P_Y+self._N_X*P_X)*self.commission_pct; net=pnl-tc
        self._equity[-1]+=net
        cy,cx=("SELL","BUY") if self._position==1 else ("BUY","SELL")
        self._place_order(self.symbol_y,self.token_y,cy,self._N_Y,P_Y)
        self._place_order(self.symbol_x,self.token_x,cx,self._N_X,P_X)
        print(f"  [CLOSE] reason={reason} | PnL=₹{net:.2f} | Cap=₹{self._equity[-1]:,.0f}")
        row={"timestamp":str(ts),"direction":"LONG" if self._position==1 else "SHORT",
             "symbol_y":self.symbol_y,"symbol_x":self.symbol_x,"N_Y":self._N_Y,"N_X":self._N_X,
             "S_entry":round(self._S_entry,6),"S_exit":round(S_exit,6),"Z_entry":round(self._Z_entry,4),
             "gross_pnl":round(pnl,2),"tc":round(tc,2),"net_pnl":round(net,2),
             "close_reason":reason,"capital_after":round(self._equity[-1],2)}
        self._closed_trades.append(row)
        with open(_LOG_FILE,"a",newline="") as f: csv.DictWriter(f,fieldnames=_LOG_FIELDS).writerow(row)
        self._N_Y=0; self._N_X=0; self._S_entry=0.0; self._Z_entry=0.0

    def run_loop(self,interval_sec:float=60.0,max_bars:Optional[int]=None):
        self._running=True; bar=0
        print(f"[AngelExecutor] Starting loop — pair={self.pair_name}, interval={interval_sec}s, paper={self.paper_trading}")
        try:
            while self._running:
                if max_bars and bar>=max_bars: break
                P_Y,P_X=self._get_prices()
                if P_Y is None or P_X is None: print(f"[Bar {bar}] Price unavailable."); time.sleep(interval_sec); bar+=1; continue
                beta_t=self._get_beta(); S_t=P_Y-beta_t*P_X; self._s_history.append(S_t)
                Z_t=self._compute_zscore(S_t)
                if Z_t is None: time.sleep(interval_sec); bar+=1; continue
                ts=datetime.datetime.now()
                print(f"[Bar {bar}] {ts.strftime('%H:%M:%S')} | S={S_t:.4f} Z={Z_t:.3f} Pos={self._position} Cap=₹{self._equity[-1]:,.0f}")
                if self._position!=0:
                    stop_res=self._sl.check_stops(S_t=S_t,Z_t=Z_t,timestamp=ts)
                    if stop_res["stop_triggered"]:
                        self._close_position(S_t,P_Y,P_X,stop_res["reason"],ts); self._position=0
                    elif self._get_signal(Z_t) in (0,2,-2):
                        self._close_position(S_t,P_Y,P_X,"exit_signal",ts); self._position=0
                if self._position==0:
                    sig=self._get_signal(Z_t)
                    if sig in (1,-1):
                        sz=self._sizer.compute(signal=sig,beta_t=beta_t,price_y=P_Y,price_x=P_X)
                        N_Y,N_X=sz["N_Y"],sz["N_X"]
                        if N_Y>0 and N_X>0 and self._place_pair_order(sig,N_Y,N_X,P_Y,P_X):
                            tc=(N_Y*P_Y+N_X*P_X)*self.commission_pct; self._equity[-1]-=tc
                            self._sl.enter_position(signal=sig,spread_entry=S_t,n_shares=N_Y,timestamp=ts,
                                                    spread_series=pd.Series(self._s_history))
                            self._position=sig; self._N_Y=N_Y; self._N_X=N_X; self._S_entry=S_t; self._Z_entry=Z_t
                self._equity.append(self._equity[-1]); self._equity_ts.append(ts)
                time.sleep(interval_sec); bar+=1
        except KeyboardInterrupt:
            print("\n[AngelExecutor] KeyboardInterrupt — closing positions.")
        finally:
            if self._position!=0:
                P_Y,P_X=self._get_prices()
                if P_Y and P_X: self._close_position(P_Y-self._get_beta()*P_X,P_Y,P_X,"shutdown",datetime.datetime.now())
            self._running=False; print("[AngelExecutor] Loop ended.")

    def close_all(self):
        if self._position!=0:
            P_Y,P_X=self._get_prices()
            if P_Y and P_X: self._close_position(P_Y-self._get_beta()*P_X,P_Y,P_X,"manual_close",datetime.datetime.now())
        self._position=0; print("[AngelExecutor] All positions closed.")

    def equity_series(self)->pd.Series:
        idx=self._equity_ts if self._equity_ts else range(len(self._equity))
        return pd.Series(self._equity[:len(idx)],index=idx,name="equity")

    def closed_trades_df(self)->pd.DataFrame:
        return pd.DataFrame(self._closed_trades) if self._closed_trades else pd.DataFrame(columns=_LOG_FIELDS)

    def live_pnl(self)->float:
        if self._position==0: return 0.0
        P_Y,P_X=self._get_prices()
        if P_Y is None: return 0.0
        S=P_Y-self._get_beta()*P_X
        return self._N_Y*(S-self._S_entry) if self._position==1 else self._N_Y*(self._S_entry-S)

    def performance(self)->Optional[PerformanceAnalytics]:
        eq=self.equity_series(); tl=self.closed_trades_df()
        if len(eq)<2: print("[AngelExecutor] Not enough data for analytics."); return None
        return PerformanceAnalytics(equity_curve=eq,trade_log=tl,capital=self.capital)

    def disconnect(self):
        self.close_all()
        if not self.paper_trading and self._client:
            try: self._client.terminateSession(self.client_id)
            except: pass
        print("[AngelExecutor] Disconnected.")

    def __repr__(self)->str:
        return (f"AngelExecutor(pair='{self.pair_name}', pos={self._position}, "
                f"cap=₹{self._equity[-1]:,.0f}, paper={self.paper_trading})")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    print("AngelExecutor — connect() → run_loop(interval_sec=60)")
