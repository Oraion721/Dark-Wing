"""
Module  : Trade_Implement/Executor/ibkr_executor.py
Project : Dark Wing — Quantitative Statistical Arbitrage Engine
Purpose : Automated Execution and Live Replay Engine via Interactive Brokers API (TWS / Gateway)
Style   : Institutional Quant (Compact, PEP8, State-Space & Dynamic Execution Architecture)
"""
import sys,os,time,datetime,threading,warnings,csv
import numpy as np
import pandas as pd
from typing import Optional,Dict,List,Tuple

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
_TRADE_DIR=os.path.dirname(_THIS_DIR)
_ROOT_DIR=os.path.dirname(_TRADE_DIR)
_SIGNALS_DIR=os.path.join(_TRADE_DIR,"Signals")
_RISK_DIR=os.path.join(_TRADE_DIR,"Risk_Management")
_BACKTEST_DIR=os.path.join(_ROOT_DIR,"Backtesting")
for _p in [_THIS_DIR,_TRADE_DIR,_SIGNALS_DIR,_RISK_DIR,_BACKTEST_DIR]:
    if _p not in sys.path:
        sys.path.insert(0,_p)

from pair_signal import PairSignalGenerator
from position_sizing import PositionSizer
from stoploss import StopLossManager
from performance import PerformanceAnalytics

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    _IBKR_AVAILABLE=True
except ImportError:
    _IBKR_AVAILABLE=False

_LOG_FILE=os.path.join(_THIS_DIR,"live_trade_log_ibkr.csv")
_LOG_FIELDS=["timestamp","direction","symbol_y","symbol_x","N_Y","N_X","S_entry","S_exit","Z_entry","gross_pnl","tc","net_pnl","bars_in_trade","close_reason","capital_after"]

# ==================== Helper Functions ===========================
def _make_contract(symbol:str,exchange:str="SMART",currency:str="USD",sec_type:str="STK")->object:
    if not _IBKR_AVAILABLE: return None
    c=Contract()
    c.symbol=symbol
    c.secType=sec_type
    c.exchange=exchange
    c.currency=currency
    return c

def _mkt_order(action:str,quantity:int)->object:
    if not _IBKR_AVAILABLE: return None
    o=Order()
    o.action=action
    o.totalQuantity=quantity
    o.orderType="MKT"
    o.transmit=True
    return o

def _stop_order(action:str,quantity:int,stop_price:float,tif:str="GTC")->object:
    if not _IBKR_AVAILABLE: return None
    o=Order()
    o.action=action
    o.totalQuantity=quantity
    o.orderType="STP"
    o.auxPrice=float(round(stop_price,2))
    o.tif=tif
    o.transmit=True
    return o

# ==================== Market Data Handler ===========================
if _IBKR_AVAILABLE:
    class MarketDataHandler(EWrapper,EClient):
        def __init__(self):
            EClient.__init__(self,self)
            self.market_data:Dict[int,Dict]={}
            self.order_status:Dict[int,Dict]={}
            self.next_valid_order_ID:Optional[int]=None
            self._ready=threading.Event()
            self._api_thread:Optional[threading.Thread]=None
            self._req_counter:int=1000      # Sequential req ID counter; avoids timestamp collisions

        def nextValidId(self,orderId:int):
            self.next_valid_order_ID=orderId
            # Anchor req ID counter above the order ID block to prevent overlap
            self._req_counter=max(self._req_counter,orderId+10_000)
            self._ready.set()

        def orderStatus(self,orderId:int,status:str,filled:float,remaining:float,avgFillPrice:float,permId:int,parentId:int,lastFillPrice:float,clientId:int,whyHeld:str,mktCapPrice:float):
            self.order_status[orderId]={"status":status,"filled":filled,"remaining":remaining,"avgFillPrice":avgFillPrice}

        def next_orderID(self)->int:
            oid=self.next_valid_order_ID
            if oid is not None:
                self.next_valid_order_ID+=1
                return oid
            return int(time.time()*1000)%1_000_000

        def _next_req_id(self)->int:
            rid=self._req_counter
            self._req_counter+=1
            return rid

        def tickPrice(self,reqId:int,tickType:int,price:float,attrib):
            if reqId not in self.market_data: self.market_data[reqId]={}
            _TICK_MAP={1:"bid",2:"ask",4:"last",9:"close",66:"delayed_bid",67:"delayed_ask",68:"delayed_last",75:"delayed_close"}
            if tickType in _TICK_MAP and price>0:
                self.market_data[reqId][_TICK_MAP[tickType]]=price

        def error(self,reqId:int,errorTime:int,errorCode:int,errorString:str,advancedOrderRejectJson:str=""):
            # Newer ibapi (ProtoBuf, Python 3.13+) adds errorTime as 2nd positional arg.
            # Suppress common benign info codes: 2104=mkt data ok, 2106=hist data ok, 2158=delayed.
            _BENIGN={2100,2103,2104,2105,2106,2107,2108,2119,2157,2158,10167}
            if errorCode not in _BENIGN:
                warnings.warn(f"[IBKR] reqId={reqId} code={errorCode}: {errorString}")

        def wait_for_ready(self,timeout:float=15.0)->bool:
            return self._ready.wait(timeout=timeout)

        def start_api_thread(self):
            """
            Launches EClient.run() in a background daemon thread.
            REQUIRED: Without this thread, no EWrapper callbacks (tickPrice, nextValidId,
            orderStatus, etc.) ever fire because the socket message queue is never drained.
            """
            self._api_thread=threading.Thread(target=self.run,name="ibkr-api",daemon=True)
            self._api_thread.start()

        def qualify_contract(self,contract:Contract)->Contract:
            return contract

        def request_market_data(self,contract:Contract,params:Dict)->int:
            req_id=self._next_req_id()
            self.market_data[req_id]={}
            self.reqMktData(req_id,contract,"",params.get("snapshot",False),False,[])
            return req_id
else:
    class MarketDataHandler:
        pass

# ==================== Core Execution Engine ===========================
class IBKRExecutor:
    def __init__(self,signal_gen:PairSignalGenerator,position_sizer:PositionSizer,
                 stop_loss_mgr:StopLossManager,
                 symbol_y:str,symbol_x:str,exchange:str="SMART",currency:str="USD",
                 sec_type:str="STK",capital:float=1_000_000.0,
                 host:str="127.0.0.1",port:int=4002,client_id:int=1,
                 commission_pct:float=0.0003,pair_name:str="Pair",dry_run:bool=False,
                 sim_price_y:Optional[pd.Series]=None,sim_price_x:Optional[pd.Series]=None,
                 bar_duration_sec:float=86400.0,manual_exit_mode:bool=False,
                 enable_server_stops:bool=True,server_stop_pct:Optional[float]=None,
                 intraday_check_interval_sec:float=30.0,
                 yf_symbol_y:Optional[str]=None,yf_symbol_x:Optional[str]=None):
        """
        Statistical Arbitrage Execution Engine for Daily Swing Trading.
        
        Args:
            signal_gen: Calibrated PairSignalGenerator pipeline.
            position_sizer: PositionSizer with fractional/Kelly exposure constraints.
            stop_loss_mgr: Multi-barrier StopLossManager with post-stop cooldown protection.
            symbol_y: Ticker symbol for dependent leg Y.
            symbol_x: Ticker symbol for independent leg X.
            exchange: Primary execution routing venue (e.g. 'SMART', 'NSE', 'TSE').
            currency: Quote currency denomination (e.g. 'USD', 'EUR', 'JPY', 'INR').
            capital: Total allocated strategy equity.
            dry_run: True = historical replay simulation; False = live socket execution.
            sim_price_y: Raw unadjusted price series for Y replay.
            sim_price_x: Raw unadjusted price series for X replay.
            manual_exit_mode: True = manual discretionary exit alerts; False = algorithmic closing.
            enable_server_stops: True = place resting GTC STP orders on IBKR upon entry.
            server_stop_pct: Fractional leg stop distance (e.g. 0.05). If None, auto-calibrated from spread sigma.
            intraday_check_interval_sec: Polling interval in seconds for active intraday risk monitoring.
        """
        self._gen=signal_gen
        self._sizer=position_sizer
        self._sl=stop_loss_mgr
        self.symbol_y=symbol_y
        self.symbol_x=symbol_x
        self.exchange=exchange
        self.currency=currency
        self.sec_type=sec_type
        self.capital=float(capital)
        self.host=host
        self.port=port
        self.client_id=client_id
        self.commission_pct=float(commission_pct)
        self.pair_name=pair_name
        self.dry_run=dry_run or not _IBKR_AVAILABLE
        self.bar_duration_sec=float(bar_duration_sec)
        self.manual_exit_mode=bool(manual_exit_mode)
        self.enable_server_stops=bool(enable_server_stops)
        self.server_stop_pct=float(server_stop_pct) if server_stop_pct is not None else None
        self.intraday_check_interval_sec=max(5.0,float(intraday_check_interval_sec))
        self.yf_symbol_y=yf_symbol_y
        self.yf_symbol_x=yf_symbol_x
        self._warned_yf_fallback=False

        self._client:Optional[MarketDataHandler]=None
        self._contract_y=None
        self._contract_x=None
        self._reqid_y:Optional[int]=None
        self._reqid_x:Optional[int]=None
        self._stop_oid_y:Optional[int]=None
        self._stop_oid_x:Optional[int]=None

        # Execution and state machine variables
        self._position:int=0
        self._N_Y:int=0
        self._N_X:int=0
        self._S_entry:float=0.0
        self._Z_entry:float=0.0
        self._P_Y_entry:float=0.0
        self._P_X_entry:float=0.0
        self._equity:List[float]=[capital]
        self._equity_ts:List=[]
        self._closed_trades:List[Dict]=[]
        self._bars_in_trade:int=0
        self._running=False
        self._lock=threading.Lock()

        self._z_history:List[float]=[]
        self._s_history:List[float]=[]
        self._p_y_history:List[float]=[]
        self._p_x_history:List[float]=[]

        # Dry-run historical bar simulation buffers
        self._sim_y:Optional[List[float]]=list(sim_price_y.values) if sim_price_y is not None else None
        self._sim_x:Optional[List[float]]=list(sim_price_x.values) if sim_price_x is not None else None
        self._sim_bar:int=0

        # Live swing trading: gate that enforces exactly one bar per calendar day.
        # Prevents the continuous price-poll loop from treating each 5-second tick
        # as a new daily bar and flooding the state machine with spurious signals.
        self._last_bar_date:Optional[datetime.date]=None
        self._last_intraday_heartbeat:float=0.0

        os.makedirs(_THIS_DIR,exist_ok=True)
        if not os.path.exists(_LOG_FILE):
            with open(_LOG_FILE,"w",newline="",encoding="utf-8") as f:
                csv.DictWriter(f,fieldnames=_LOG_FIELDS).writeheader()

    def connect(self)->bool:
        if self.dry_run:
            print("[IBKRExecutor] dry_run=True. Initialising local simulation engine.")
            return True
        self._client=MarketDataHandler()
        try:
            self._client.connect(host=self.host,port=self.port,clientId=self.client_id)
            # EClient.run() must start in a background thread BEFORE wait_for_ready().
            # It drains the socket receive queue; without it, no EWrapper callbacks ever fire.
            self._client.start_api_thread()
            if not self._client.wait_for_ready(timeout=15):
                warnings.warn("[IBKRExecutor] Timed out waiting for nextValidId. Is IB Gateway / TWS running?")
                return False
            # Allow the API thread 1 s to drain the burst of post-connect info messages
            # (codes 2104, 2106, etc.) before any reqMktData calls are issued.
            time.sleep(1)
            print(f"[IBKRExecutor] Connected to {self.host}:{self.port} | NextOrderID={self._client.next_valid_order_ID}")
            return True
        except Exception as e:
            warnings.warn(f"[IBKRExecutor] Connection failed: {e}")
            return False

    def subscribe_prices(self)->bool:
        if self.dry_run:
            print("[IBKRExecutor] dry_run — Historical feed mounted successfully.")
            return True
        if self._client is None:
            raise RuntimeError("Call connect() before subscribing to price feeds.")
        try:
            # MarketDataType 3 = Delayed | 4 = Delayed + Frozen (last available snapshot).
            # Type 4 is more reliable for non-US exchanges (XETRA, TSE, etc.) because it falls
            # back to the last frozen close when real-time delayed ticks are momentarily absent.
            try:
                self._client.reqMarketDataType(4)
                print("[IBKRExecutor] Set MarketDataType = 4 (Delayed + Frozen)")
            except Exception as e_md:
                print(f"[IBKRExecutor] reqMarketDataType note: {e_md}")

            self._contract_y=self._client.qualify_contract(_make_contract(self.symbol_y,self.exchange,self.currency,self.sec_type))
            self._contract_x=self._client.qualify_contract(_make_contract(self.symbol_x,self.exchange,self.currency,self.sec_type))
            # generic_tick "" requests standard tick types; avoid non-subscribed generic ticks (100,101)
            # which can cause error 10092 ("no subscription") for delayed non-US data.
            self._reqid_y=self._client.request_market_data(contract=self._contract_y,params={"snapshot":False})
            self._reqid_x=self._client.request_market_data(contract=self._contract_x,params={"snapshot":False})
            print(f"[IBKRExecutor] Subscribed: Y={self.symbol_y} (reqId={self._reqid_y}) | X={self.symbol_x} (reqId={self._reqid_x})")

            # Allow time for initial tick burst from IBKR before entering the loop
            print("[IBKRExecutor] Waiting 5s for initial market data ticks...")
            time.sleep(5)

            # Diagnostic: validate that at least one price arrived for each leg
            P_Y,P_X=self._fetch_prices()
            if P_Y is None or P_X is None:
                print(f"[IBKRExecutor] WARNING: No price ticks received yet.")
                print(f"  Y ({self.symbol_y}) ticks : {self._client.market_data.get(self._reqid_y,{})}")
                print(f"  X ({self.symbol_x}) ticks : {self._client.market_data.get(self._reqid_x,{})}")
                print("  Possible causes: market closed, no market data subscription, wrong exchange.")
                print("  The loop will keep retrying. Use Ctrl+C to abort if prices never arrive.")
            else:
                print(f"[IBKRExecutor] Prices validated: {self.symbol_y}={P_Y:.2f} | {self.symbol_x}={P_X:.2f}")
            return True
        except Exception as e:
            warnings.warn(f"[IBKRExecutor] Price subscription failed: {e}")
            return False

    def run_loop(self,interval_sec:float=0.0,max_bars:Optional[int]=None):
        """
        Main execution loop for Daily Swing Statistical Arbitrage.
        In dry_run mode, steps sequentially through historical daily bars.
        In live mode, queries market data on configured interval.
        """
        self._running=True
        bar=0
        print(f"[IBKRExecutor] Starting swing loop — pair={self.pair_name}, dry_run={self.dry_run}, manual_exit={self.manual_exit_mode}")
        try:
            while self._running:
                if max_bars and bar>=max_bars:
                    break
                raw_P_Y,raw_P_X=self._fetch_prices()
                if raw_P_Y is None or raw_P_X is None:
                    if self.dry_run: break
                    # Do NOT advance bar counter while prices are absent — it is still the same bar
                    print(f"[Day {bar:>3d}] Price stream pending. Retrying..."); time.sleep(max(interval_sec,2.0)); continue

                # ---- Daily bar gate (live swing mode only) ----
                # In live mode the loop polls every interval_sec seconds.
                # For daily swing trading we process exactly one bar per calendar day for signals.
                # During the trading day between EOD bars, an ACTIVE INTRADAY RISK MONITOR continually
                # polls prices, tracks live Z-score, and immediately executes stops on regime shock.
                if not self.dry_run:
                    today=datetime.date.today()
                    if self._last_bar_date is not None and today==self._last_bar_date:
                        if self._position!=0:
                            # Active Intraday Risk & Stop Monitor (evaluates stops between daily bars)
                            time.sleep(self.intraday_check_interval_sec)
                            cur_p_y,cur_p_x=self._fetch_prices()
                            if cur_p_y is not None and cur_p_x is not None and cur_p_y>0 and cur_p_x>0:
                                cur_s=np.log(cur_p_y)-self._get_beta()*np.log(cur_p_x)
                                cur_z=self._compute_zscore(cur_s)
                                ts_now=datetime.datetime.now()

                                # 1. Check if IBKR native server stop triggered
                                if self._check_server_stops_triggered():
                                    print(f"\n  [SERVER STOP TRIGGERED] IBKR native stop filled! Closing remaining leg...")
                                    self._close_position(cur_s,cur_p_y,cur_p_x,"server_stop_executed",ts_now)
                                    continue

                                # 2. Check algorithmic risk barriers (Z-score hard stop, dollar loss stop, spread vol stop)
                                if cur_z is not None:
                                    stop_res=self._sl.check_stops(S_t=cur_s,Z_t=cur_z,timestamp=ts_now,price_y=cur_p_y,price_x=cur_p_x,is_bar_close=False)
                                    if stop_res["stop_triggered"]:
                                        reason=stop_res["reason"]
                                        print(f"\n  [INTRADAY RISK ALERT] Stop triggered: {reason} | S={cur_s:+.4f} | Z={cur_z:+.3f}")
                                        if self.manual_exit_mode:
                                            print(f"  [MANUAL ALERT] Regime shock detected! Recommended action: CLOSE position immediately.")
                                        self._close_position(cur_s,cur_p_y,cur_p_x,f"intraday_{reason}",ts_now)
                                        continue

                                # 3. Periodic telemetry heartbeat (every 300 s)
                                now_sec=time.time()
                                if now_sec-self._last_intraday_heartbeat>=300.0:
                                    self._last_intraday_heartbeat=now_sec
                                    z_str=f"{cur_z:+.3f}" if cur_z is not None else "N/A"
                                    print(f"  [Intraday Monitor {ts_now.strftime('%H:%M:%S')}] S={cur_s:+.4f} | Z={z_str} | Pos={self._position:>2d} | Risk: ACTIVE")
                        else:
                            time.sleep(60)
                        continue
                    self._last_bar_date=today

                # Cointegration math strictly operates on log prices
                log_P_Y=np.log(raw_P_Y)
                log_P_X=np.log(raw_P_X)
                beta_t=self._get_beta()
                S_t=log_P_Y-beta_t*log_P_X

                self._s_history.append(S_t)
                self._p_y_history.append(raw_P_Y)
                self._p_x_history.append(raw_P_X)

                Z_t=self._compute_zscore(S_t)
                if Z_t is None:
                    if not self.dry_run: time.sleep(interval_sec)
                    bar+=1; continue
                self._z_history.append(Z_t)
                ts=datetime.datetime.now()

                # Console progress telemetry
                print(f"[Day {bar:>3d}] S={S_t:+.4f} | β={beta_t:.3f} | Z={Z_t:+.3f} | Pos={self._position:>2d} | Cap={self.currency} {self._equity[-1]:>12,.2f}")

                # Check StopLoss and Exit conditions
                if self._position!=0:
                    self._bars_in_trade+=1
                    stop_res=self._sl.check_stops(S_t=S_t,Z_t=Z_t,timestamp=ts,price_y=raw_P_Y,price_x=raw_P_X,is_bar_close=True)
                    if stop_res["stop_triggered"]:
                        if self.manual_exit_mode:
                            print(f"  [MANUAL ALERT] Stop triggered: {stop_res['reason']} | Unrealised PnL={stop_res.get('unrealised_pnl',0.0):.2f}")
                        self._close_position(S_t,raw_P_Y,raw_P_X,stop_res["reason"],ts)
                    else:
                        sig=self._get_signal(Z_t)
                        if sig==0 or abs(sig)==2:
                            if self.manual_exit_mode:
                                print(f"  [MANUAL ALERT] Mean-reversion target reached (|Z| < exit_z). Recommended action: CLOSE position.")
                            self._close_position(S_t,raw_P_Y,raw_P_X,"exit_signal",ts)
                elif self._sl.in_cooldown:
                    # Advance cooldown counter on each completed bar when flat
                    self._sl.check_stops(S_t=S_t,Z_t=Z_t,timestamp=ts,price_y=raw_P_Y,price_x=raw_P_X,is_bar_close=True)

                # Check Entry conditions (guarded against post-stop cooldown)
                if self._position==0 and not self._sl.in_cooldown:
                    sig=self._get_signal(Z_t)
                    if sig in (1,-1):
                        # Position sizing requires real currency asset prices
                        sz=self._sizer.compute(signal=sig,beta_t=beta_t,price_y=raw_P_Y,price_x=raw_P_X)
                        N_Y,N_X=sz["N_Y"],sz["N_X"]
                        if N_Y>0 and N_X>0:
                            if self._place_pair_order(sig,N_Y,N_X,raw_P_Y,raw_P_X):
                                tc=(N_Y*raw_P_Y+N_X*raw_P_X)*self.commission_pct
                                self._equity[-1]-=tc
                                self._sl.enter_position(signal=sig,spread_entry=S_t,n_shares=N_Y,timestamp=ts,price_y_entry=raw_P_Y,price_x_entry=raw_P_X,n_shares_x=N_X)
                                self._position=sig
                                self._N_Y=N_Y
                                self._N_X=N_X
                                self._S_entry=S_t
                                self._Z_entry=Z_t
                                self._P_Y_entry=raw_P_Y
                                self._P_X_entry=raw_P_X

                self._equity.append(self._equity[-1])
                self._equity_ts.append(ts)
                bar+=1
                if interval_sec>0:
                    time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("\n[IBKRExecutor] KeyboardInterrupt received — initiating safe shutdown.")
        finally:
            if self._position!=0:
                r_Y,r_X=self._fetch_prices()
                if r_Y and r_X:
                    S_final=np.log(r_Y)-self._get_beta()*np.log(r_X)
                    self._close_position(S_final,r_Y,r_X,"shutdown",datetime.datetime.now())
            self._running=False
            print("[IBKRExecutor] Execution loop terminated.")

    def _fetch_prices(self)->Tuple[Optional[float],Optional[float]]:
        if self.dry_run:
            if self._sim_y is None or self._sim_x is None: return None,None
            if self._sim_bar>=len(self._sim_y) or self._sim_bar>=len(self._sim_x): return None,None
            P_Y=self._sim_y[self._sim_bar]
            P_X=self._sim_x[self._sim_bar]
            self._sim_bar+=1
            return float(P_Y),float(P_X)

        if self._client is None or self._reqid_y is None: return None,None
        def _mid(reqid):
            d=self._client.market_data.get(reqid,{})
            bid=d.get("bid") or d.get("delayed_bid")
            ask=d.get("ask") or d.get("delayed_ask")
            if bid is not None and ask is not None:
                return (float(bid)+float(ask))/2.0
            p=d.get("last") or d.get("delayed_last") or d.get("close") or d.get("delayed_close")
            return float(p) if p is not None else None

        py,px=_mid(self._reqid_y),_mid(self._reqid_x)
        if py is not None and px is not None:
            return py,px

        # Resilient fallback: If IBKR market data subscription is absent (e.g. NSE India on global IBKR accounts),
        # query real-time market quote via yfinance fast_info so execution loop is never stalled.
        if self.yf_symbol_y and self.yf_symbol_x:
            try:
                import yfinance as yf
                tickers=yf.Tickers(f"{self.yf_symbol_y} {self.yf_symbol_x}")
                y_info=tickers.tickers.get(self.yf_symbol_y)
                x_info=tickers.tickers.get(self.yf_symbol_x)
                if y_info and x_info:
                    p_y=getattr(y_info,"fast_info",{}).get("lastPrice")
                    p_x=getattr(x_info,"fast_info",{}).get("lastPrice")
                    if p_y and p_x and p_y>0 and p_x>0:
                        if not self._warned_yf_fallback:
                            print(f"[IBKRExecutor] Market data fallback active: Y={self.yf_symbol_y} ({p_y:.2f}) | X={self.yf_symbol_x} ({p_x:.2f})")
                            self._warned_yf_fallback=True
                        return float(p_y),float(p_x)
            except Exception:
                pass
        return None,None

    def _get_beta(self)->float:
        try:
            db=self._gen._db
            if hasattr(db,"_states_df") and db._states_df is not None and "beta" in db._states_df.columns:
                if self.dry_run and self._sim_bar>0:
                    idx=min(self._sim_bar-1,len(db._states_df)-1)
                    return float(db._states_df["beta"].iloc[idx])
                return float(db._states_df["beta"].iloc[-1])
            return float(db.current_beta())
        except Exception:
            return 1.0

    def _compute_zscore(self,S_t:float)->Optional[float]:
        """
        Computes rolling Z-score: Z_t = (S_t - μ_t) / σ_t.
        In dry_run: indexed by historical bar to mirror backtest state.
        In live: anchored by the calibrated end-of-training distribution.
        """
        zs=self._gen._zs
        try:
            mu_series=zs._mu
            sigma_series=zs._sigma
            if mu_series is not None and sigma_series is not None:
                if self.dry_run and self._sim_bar>0:
                    idx=min(self._sim_bar-1,len(mu_series)-1)
                    mu=float(mu_series.iloc[idx])
                    sigma=float(sigma_series.iloc[idx])
                else:
                    mu=float(mu_series.dropna().iloc[-1])
                    sigma=float(sigma_series.dropna().iloc[-1])
                if not np.isnan(mu) and not np.isnan(sigma) and sigma>1e-10:
                    return (S_t-mu)/sigma
        except Exception:
            pass

        # Rolling window fallback
        window=getattr(zs,"window",60)
        try:
            train_spread=zs._spread
            combined=list(train_spread.values[-window:])+self._s_history
        except Exception:
            combined=self._s_history
        if len(combined)<max(10,window//2):
            return None
        arr=np.array(combined[-window:])
        mu,sigma=arr.mean(),arr.std(ddof=1)
        return (S_t-mu)/sigma if sigma>1e-10 else None

    def _get_signal(self,Z_t:float)->int:
        """
        State machine signal resolver with transition-zone holding logic.
        """
        zs=self._gen._zs
        ez=zs.entry_z
        xz=zs.exit_z
        sz=zs.stop_z

        # Hard stop-loss boundary
        if Z_t>sz: return -2
        if Z_t<-sz: return 2

        # In trade: hold across transition zone until convergence target (|Z| < exit_z)
        if self._position!=0:
            if abs(Z_t)<xz: return 0
            return self._position

        # Flat: enter when spread diverges past entry_z
        if Z_t>ez: return -1
        if Z_t<-ez: return 1
        return 0

    def _place_pair_order(self,signal:int,N_Y:int,N_X:int,P_Y:float,P_X:float)->bool:
        action_y,action_x=("BUY","SELL") if signal==1 else ("SELL","BUY")
        if self.dry_run:
            print(f"  [ORDER] {action_y} {N_Y}x{self.symbol_y}@{P_Y:.2f} | {action_x} {N_X}x{self.symbol_x}@{P_X:.2f}")
            if self.enable_server_stops:
                self._place_server_stops(signal,N_Y,N_X,P_Y,P_X)
            return True
        if self._client is None: return False
        try:
            oid_y=self._client.next_orderID()
            oid_x=self._client.next_orderID()
            self._client.placeOrder(oid_y,self._contract_y,_mkt_order(action_y,N_Y))
            self._client.placeOrder(oid_x,self._contract_x,_mkt_order(action_x,N_X))
            print(f"  [ORDER] {action_y} {N_Y}x{self.symbol_y} (id={oid_y}) | {action_x} {N_X}x{self.symbol_x} (id={oid_x})")
            if self.enable_server_stops:
                self._place_server_stops(signal,N_Y,N_X,P_Y,P_X)
            return True
        except Exception as e:
            warnings.warn(f"[IBKRExecutor] Order dispatch failed: {e}")
            return False

    def _place_server_stops(self,signal:int,N_Y:int,N_X:int,P_Y:float,P_X:float):
        """
        Submits resting GTC protective stop-loss orders directly to IBKR servers.
        Ensures exchange-side liquidation even during script crash or disconnection.
        """
        if not self.enable_server_stops: return
        stop_dist=self.server_stop_pct
        if stop_dist is None:
            sigma_s=getattr(self._sl,"_spread_sigma",0.02)
            k=getattr(self._sl,"spread_stop_k",2.5)
            stop_dist=float(np.clip(k*sigma_s,0.03,0.10))

        if signal==1:
            p_y_stop=round(P_Y*(1.0-stop_dist),2)
            p_x_stop=round(P_X*(1.0+stop_dist),2)
            stop_act_y,stop_act_x="SELL","BUY"
        else:
            p_y_stop=round(P_Y*(1.0+stop_dist),2)
            p_x_stop=round(P_X*(1.0-stop_dist),2)
            stop_act_y,stop_act_x="BUY","SELL"

        if self.dry_run:
            print(f"  [SERVER STOP] Simulated GTC STP: {stop_act_y} {N_Y}x{self.symbol_y}@{p_y_stop:.2f} | {stop_act_x} {N_X}x{self.symbol_x}@{p_x_stop:.2f}")
            return
        if self._client is None: return
        try:
            self._stop_oid_y=self._client.next_orderID()
            self._stop_oid_x=self._client.next_orderID()
            self._client.placeOrder(self._stop_oid_y,self._contract_y,_stop_order(stop_act_y,N_Y,p_y_stop))
            self._client.placeOrder(self._stop_oid_x,self._contract_x,_stop_order(stop_act_x,N_X,p_x_stop))
            print(f"  [SERVER STOP] Active GTC STP: {stop_act_y} {N_Y}x{self.symbol_y}@{p_y_stop:.2f} (id={self._stop_oid_y}) | {stop_act_x} {N_X}x{self.symbol_x}@{p_x_stop:.2f} (id={self._stop_oid_x})")
        except Exception as e:
            warnings.warn(f"[IBKRExecutor] Server stop placement failed: {e}")

    def _cancel_server_stops(self):
        """
        Cancels active resting stop orders on IBKR upon deliberate position exit.
        """
        if not self.dry_run and self._client:
            if self._stop_oid_y is not None:
                try:
                    self._client.cancelOrder(self._stop_oid_y)
                    print(f"  [SERVER STOP] Cancelled resting stop order id={self._stop_oid_y} ({self.symbol_y})")
                except Exception as e:
                    warnings.warn(f"[IBKRExecutor] Cancel stop order {self._stop_oid_y} failed: {e}")
                self._stop_oid_y=None
            if self._stop_oid_x is not None:
                try:
                    self._client.cancelOrder(self._stop_oid_x)
                    print(f"  [SERVER STOP] Cancelled resting stop order id={self._stop_oid_x} ({self.symbol_x})")
                except Exception as e:
                    warnings.warn(f"[IBKRExecutor] Cancel stop order {self._stop_oid_x} failed: {e}")
                self._stop_oid_x=None

    def _check_server_stops_triggered(self)->bool:
        """
        Detects if an IBKR native stop order was executed by the broker.
        """
        if self.dry_run or self._client is None: return False
        for oid in [self._stop_oid_y,self._stop_oid_x]:
            if oid is not None and oid in self._client.order_status:
                st=self._client.order_status[oid].get("status","")
                if st=="Filled": return True
        return False

    def _close_position(self,S_exit:float,P_Y:float,P_X:float,reason:str,ts):
        """
        Closes active pair trade, computes cash-leg PnL, logs trade, and resets state.
        Cash PnL Formula:
          LONG spread  (Bought Y, Sold X) : N_Y*(P_Y_exit - P_Y_entry) + N_X*(P_X_entry - P_X_exit)
          SHORT spread (Sold Y, Bought X) : N_Y*(P_Y_entry - P_Y_exit) + N_X*(P_X_exit - P_X_entry)
        """
        self._cancel_server_stops()
        if self._P_Y_entry>0 and self._P_X_entry>0:
            if self._position==1:
                pnl=self._N_Y*(P_Y-self._P_Y_entry)+self._N_X*(self._P_X_entry-P_X)
            else:
                pnl=self._N_Y*(self._P_Y_entry-P_Y)+self._N_X*(P_X-self._P_X_entry)
        else:
            pnl=(self._N_Y*(S_exit-self._S_entry) if self._position==1 else self._N_Y*(self._S_entry-S_exit))*P_Y

        tc=(self._N_Y*P_Y+self._N_X*P_X)*self.commission_pct
        net=pnl-tc
        self._equity[-1]+=net

        close_action_y,close_action_x=("SELL","BUY") if self._position==1 else ("BUY","SELL")
        if not self.dry_run and self._client:
            try:
                oid_y=self._client.next_orderID()
                oid_x=self._client.next_orderID()
                self._client.placeOrder(oid_y,self._contract_y,_mkt_order(close_action_y,self._N_Y))
                self._client.placeOrder(oid_x,self._contract_x,_mkt_order(close_action_x,self._N_X))
            except Exception as e:
                warnings.warn(f"[IBKRExecutor] Close order execution failed: {e}")

        print(f"  [CLOSE] reason={reason} | Gross={self.currency} {pnl:+.2f} | Net={self.currency} {net:+.2f} | Cap={self.currency} {self._equity[-1]:,.2f}")

        row={"timestamp":str(ts),"direction":"LONG" if self._position==1 else "SHORT",
             "symbol_y":self.symbol_y,"symbol_x":self.symbol_x,"N_Y":self._N_Y,"N_X":self._N_X,
             "S_entry":round(self._S_entry,6),"S_exit":round(S_exit,6),"Z_entry":round(self._Z_entry,4),
             "gross_pnl":round(pnl,2),"tc":round(tc,2),"net_pnl":round(net,2),
             "bars_in_trade":self._bars_in_trade,
             "close_reason":reason,"capital_after":round(self._equity[-1],2)}
        self._closed_trades.append(row)

        # Reset state machine variables immediately prior to disk I/O
        self._position=0
        self._N_Y=0
        self._N_X=0
        self._S_entry=0.0
        self._Z_entry=0.0
        self._P_Y_entry=0.0
        self._P_X_entry=0.0
        self._bars_in_trade=0
        if hasattr(self._sl,"reset_position"):
            self._sl.reset_position()

        self._safe_write_trade_log(row)

    def _safe_write_trade_log(self,row:Dict):
        """
        Resilient disk logging. If file is locked by Excel, falls back to alternate filename.
        """
        candidates=[_LOG_FILE,_LOG_FILE.replace(".csv","_live.csv"),os.path.join(_THIS_DIR,"trade_log_fallback.csv")]
        for fpath in candidates:
            try:
                need_header=not os.path.exists(fpath) or os.path.getsize(fpath)==0
                with open(fpath,"a",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f,fieldnames=_LOG_FIELDS)
                    if need_header: w.writeheader()
                    w.writerow(row)
                if fpath!=_LOG_FILE:
                    print(f"  [NOTICE] {_LOG_FILE} is open in Excel. Appended trade to: {os.path.basename(fpath)}")
                return
            except PermissionError:
                continue
            except Exception as e:
                warnings.warn(f"[IBKRExecutor] Trade log append failed for {fpath}: {e}")
                continue
        warnings.warn("[IBKRExecutor] All trade log paths locked. Trade stored in memory.")

    def close_all(self):
        self._cancel_server_stops()
        if self._position!=0:
            P_Y,P_X=self._fetch_prices()
            if P_Y and P_X:
                S=np.log(P_Y)-self._get_beta()*np.log(P_X)
                self._close_position(S,P_Y,P_X,"manual_close",datetime.datetime.now())
        self._position=0
        print("[IBKRExecutor] All positions closed.")

    def equity_series(self)->pd.Series:
        idx=self._equity_ts if self._equity_ts else range(len(self._equity))
        return pd.Series(self._equity[:len(idx)],index=idx,name="equity")

    def closed_trades_df(self)->pd.DataFrame:
        return pd.DataFrame(self._closed_trades) if self._closed_trades else pd.DataFrame(columns=_LOG_FIELDS)

    def performance(self)->Optional[PerformanceAnalytics]:
        eq=self.equity_series()
        tl=self.closed_trades_df()
        if len(eq)<2:
            print("[IBKRExecutor] Insufficient equity data points for performance analytics.")
            return None
        rfr_map={"USD":0.045,"EUR":0.030,"GBP":0.045,"JPY":0.0025,"INR":0.065,"HKD":0.035,"SGD":0.030}
        rfr=rfr_map.get(self.currency,0.02)
        return PerformanceAnalytics(equity_curve=eq,trade_log=tl,capital=self.capital,
                                    risk_free_rate=rfr,periods_per_year=252,currency=self.currency)

    def disconnect(self):
        self.close_all()
        if not self.dry_run and self._client:
            if self._reqid_y: self._client.cancelMktData(self._reqid_y)
            if self._reqid_x: self._client.cancelMktData(self._reqid_x)
            self._client.disconnect()
            # Join the API thread so the process exits cleanly
            if self._client._api_thread and self._client._api_thread.is_alive():
                self._client._api_thread.join(timeout=5)
        print("[IBKRExecutor] Disconnected from Interactive Brokers.")

    def __repr__(self)->str:
        return (f"IBKRExecutor(pair='{self.pair_name}', pos={self._position}, "
                f"cap={self.currency} {self._equity[-1]:,.0f}, dry_run={self.dry_run})")

# =====================================================================
if __name__=="__main__":
    print("IBKRExecutor — Statistical Arbitrage Execution Engine.")
