"""
Module  : run_paper_trade.py
Project : Dark Wing — Quantitative Statistical Arbitrage Engine
Purpose : Daily Swing Pairs Trading Runner across Global Equity Markets
Style   : Institutional Quant (Compact, PEP8, Cointegration & State-Space Architecture)     """
import sys,os,warnings
import numpy as np
import pandas as pd
import yfinance as yf

# Reconfigure stdout to UTF-8 to prevent Windows cp1252 character mapping crashes
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")

_ROOT=os.path.dirname(os.path.abspath(__file__))
for _sub in ["Financial_Mathematics/Time_Series_Analysis/Spreads","Financial_Mathematics/Time_Series_Analysis/Kalman_Filter","Trade_Implement/Signals","Trade_Implement/Risk_Management","Trade_Implement/Executor","Backtesting"]:
    _p=os.path.join(_ROOT,_sub.replace("/",os.sep))
    if _p not in sys.path:
        sys.path.insert(0,_p)

from zscore import ZScore
from half_life import HalfLife
from dynamic_beta import DynamicBeta
from kalman_forecast import KalmanForecaster
from pair_signal import PairSignalGenerator
from position_sizing import PositionSizer
from stoploss import StopLossManager
from ibkr_executor import IBKRExecutor

# ==================== Global Market Presets ===========================
# 9 Multi-Market Cointegrated Pairs with Institutional Exchange Suffixes
GLOBAL_PRESETS={
    "US":{
        "NAME":"United States (NYSE / NASDAQ)",
        "SYMBOL_Y":"DAL","SYMBOL_X":"UAL",
        "EXCHANGE":"SMART","CURRENCY":"USD",
        "YF_Y":"DAL","YF_X":"UAL",
        "CAPITAL":25000.0,
    },
    "JAPAN":{
        "NAME":"Japan (Tokyo Stock Exchange)",
        "SYMBOL_Y":"7203","SYMBOL_X":"7267",
        "EXCHANGE":"SMART","CURRENCY":"JPY",
        "YF_Y":"7203.T","YF_X":"7267.T",
        "CAPITAL":3000000.0,
    },
    "GERMANY":{
        "NAME":"Germany (Deutsche Boerse XETRA)",
        "SYMBOL_Y":"BMW","SYMBOL_X":"MBG",
        "EXCHANGE":"SMART","CURRENCY":"EUR",
        "YF_Y":"BMW.DE","YF_X":"MBG.DE",
        "CAPITAL":25000.0,
    },
    "FRANCE":{
        "NAME":"France (Euronext Paris)",
        "SYMBOL_Y":"BNP","SYMBOL_X":"GLE",
        "EXCHANGE":"SMART","CURRENCY":"EUR",
        "YF_Y":"BNP.PA","YF_X":"GLE.PA",
        "CAPITAL":25000.0,
    },
    "HONG_KONG":{
        "NAME":"Hong Kong (HKEX)",
        "SYMBOL_Y":"700","SYMBOL_X":"9988",
        "EXCHANGE":"SMART","CURRENCY":"HKD",
        "YF_Y":"0700.HK","YF_X":"9988.HK",
        "CAPITAL":200000.0,
    },
    "SINGAPORE":{
        "NAME":"Singapore (Singapore Exchange SGX)",
        "SYMBOL_Y":"D05","SYMBOL_X":"U11",
        "EXCHANGE":"SMART","CURRENCY":"SGD",
        "YF_Y":"D05.SI","YF_X":"U11.SI",
        "CAPITAL":35000.0,
    },
    "ITALY":{
        "NAME":"Italy (Borsa Italiana)",
        "SYMBOL_Y":"ISP","SYMBOL_X":"UCG",
        "EXCHANGE":"SMART","CURRENCY":"EUR",
        "YF_Y":"ISP.MI","YF_X":"UCG.MI",
        "CAPITAL":25000.0,
    },
    "UK":{
        "NAME":"United Kingdom (London Stock Exchange)",
        "SYMBOL_Y":"BARC","SYMBOL_X":"LLOY",
        "EXCHANGE":"SMART","CURRENCY":"GBP",
        "YF_Y":"BARC.L","YF_X":"LLOY.L",
        "CAPITAL":20000.0,
    },
    "INDIA":{
        "NAME":"India (National Stock Exchange NSE)",
        "SYMBOL_Y":"VEDL","SYMBOL_X":"HINDALCO",
        "EXCHANGE":"NSE","CURRENCY":"INR",
        "YF_Y":"VEDL.NS","YF_X":"HINDALCO.NS",
        "CAPITAL":500000.0,
    }
}

# ==================== Active Strategy Configuration ===========================
# Target market selection from GLOBAL_PRESETS
ACTIVE_MARKET="US"
# Swing Trading parameters: daily bars with 2-year cointegration lookback window
BAR_INTERVAL="1d"
LOOKBACK_PERIOD="3y"
ENTRY_Z=1.5                     # Statistical entry boundary (2.0 sigma deviation)
EXIT_Z=0.4                      # Mean-reversion target convergence boundary
STOP_Z=3.0                      # Structural cointegration breakdown stop (3.0 sigma)
MAX_Z_VELOCITY=1.2              # Momentum shock filter: suppress entry if single-bar dZ > 1.2
MANUAL_EXIT_MODE=False          # True = alert for manual discretionary exit; False = algorithmic exit
ENABLE_SERVER_STOPS=True        # True = submit resting GTC STP orders to IBKR servers on entry
INTRADAY_MONITOR_SEC=30.0       # Polling interval (seconds) for intraday risk & regime breakdown monitoring

# Execution environment
TWS_PORT=4002           # 4002 = IB Gateway Paper | 7497 = TWS Paper
CLIENT_ID=1
DRY_RUN=True            # True = historical replay validation | False = live IBKR connection
WALK_FORWARD=True          # True = out-of-sample walk-forward validation

cfg=GLOBAL_PRESETS[ACTIVE_MARKET]
SYMBOL_Y=cfg["SYMBOL_Y"]
SYMBOL_X=cfg["SYMBOL_X"]
EXCHANGE=cfg["EXCHANGE"]
CURRENCY=cfg["CURRENCY"]
YF_Y=cfg["YF_Y"]
YF_X=cfg["YF_X"]
CAPITAL=cfg["CAPITAL"]

# ==================== Execution Pipeline ===========================
def main():
    sep="="*65
    print(sep)
    print("  Dark Wing — Quantitative Statistical Arbitrage | Pairs Trading")
    print(f"  Pair: {SYMBOL_Y}/{SYMBOL_X} [{cfg['NAME']}]")
    print(f"  Capital: {CURRENCY} {CAPITAL:,.0f} | Resolution: {BAR_INTERVAL} | Port: {TWS_PORT} | Dry-Run: {DRY_RUN}")
    print(f"  Thresholds: Entry=±{ENTRY_Z}σ | Exit=±{EXIT_Z}σ | Stop=±{STOP_Z}σ")
    print(f"  Risk Controls: Native GTC Stops={ENABLE_SERVER_STOPS} | Intraday Monitor={INTRADAY_MONITOR_SEC}s")
    print(sep)

    # Step 1: Ingest historical end-of-day price series
    print(f"\n[1/5] Ingesting historical daily data ({ACTIVE_MARKET} market | lookback={LOOKBACK_PERIOD})...")
    raw=yf.download([YF_Y,YF_X],period=LOOKBACK_PERIOD,interval=BAR_INTERVAL,auto_adjust=True,progress=False)["Close"]
    raw_y=raw[YF_Y].dropna()
    raw_x=raw[YF_X].dropna()
    common_idx=raw_y.index.intersection(raw_x.index)
    raw_y=raw_y.loc[common_idx]
    raw_x=raw_x.loc[common_idx]

    price_y=np.log(raw_y)
    price_x=np.log(raw_x)
    print(f"     {YF_Y}: {len(price_y)} daily bars | {YF_X}: {len(price_x)} daily bars")

    # Walk-forward validation branch
    if WALK_FORWARD:
        from walkforward import WalkForwardBacktest
        wf=WalkForwardBacktest(price_y=price_y,price_x=price_x,capital=CAPITAL,train_frac=0.60,entry_z=ENTRY_Z,exit_z=EXIT_Z,stop_z=STOP_Z,commission=0.0015,verbose=True)
        results=wf.run()
        print(f"\nWalk-forward complete. OOS Sharpe: {results['summary'].get('oos_sharpe','N/A')}")
        return

    # Step 2: Fit DynamicBeta State-Space Pipeline (Kalman Smoother + EM optimization)
    print("[2/5] Calibrating dynamic cointegration pipeline...")
    db=DynamicBeta(y=price_y,x=price_x,entry_z=ENTRY_Z,exit_z=EXIT_Z,stop_z=STOP_Z)
    db.fit()
    kalman_spread=db.dynamic_spread_series()

    # Estimate Ornstein-Uhlenbeck mean-reversion half-life: dS_t = κ(μ - S_t)dt + σ dW_t
    hl=HalfLife(kalman_spread)
    hl.estimate()
    hl.summary()
    half_life_days=hl.half_life_days
    zs_window=max(20,min(int(half_life_days*2),len(kalman_spread)//2))

    # Rolling Z-score normalisation: Z_t = (S_t - μ_t) / σ_t
    zs=ZScore(kalman_spread,window=zs_window,mode="rolling",entry_z=ENTRY_Z,exit_z=EXIT_Z,stop_z=STOP_Z)
    zs.compute()

    # Kalman Forecaster for direction confirmation
    kf_pairs=db.get_filter()
    forecaster=KalmanForecaster(kf=kf_pairs)

    gen=PairSignalGenerator(zscore=zs,dynamic_beta=db,kalman_forecaster=forecaster,use_forecast_filter=True,use_regime_filter=True,pair_name=f"{SYMBOL_Y}_{SYMBOL_X}",max_z_velocity=MAX_Z_VELOCITY)
    gen.generate()
    gen.summary()

    # Step 3: Instantiate Risk Management controls (Kelly / Fractional + Cooldown stops)
    print("[3/5] Initialising risk controls...")
    sizer=PositionSizer(capital=CAPITAL,method="fixed_fractional",fixed_fraction=0.4,max_trade_fraction=0.40,max_exposure=1.0)
    sl=StopLossManager(zscore=zs,half_life=half_life_days,capital=CAPITAL,max_loss_pct=0.03,spread_stop_k=2.5,time_stop_multiplier=3.0,cooldown_bars=5)

    # Step 4: Configure Execution Engine
    print("[4/5] Constructing execution engine...")
    executor=IBKRExecutor(signal_gen=gen,position_sizer=sizer,stop_loss_mgr=sl,symbol_y=SYMBOL_Y,symbol_x=SYMBOL_X,exchange=EXCHANGE,currency=CURRENCY,capital=CAPITAL,host="127.0.0.1",port=TWS_PORT,client_id=CLIENT_ID,commission_pct=0.0003,pair_name=f"{SYMBOL_Y}_{SYMBOL_X}",dry_run=DRY_RUN,sim_price_y=raw_y,sim_price_x=raw_x,manual_exit_mode=MANUAL_EXIT_MODE,enable_server_stops=ENABLE_SERVER_STOPS,intraday_check_interval_sec=INTRADAY_MONITOR_SEC,yf_symbol_y=YF_Y,yf_symbol_x=YF_X)

    # Step 5: Connect and execute
    print("[5/5] Launching execution loop...")
    if not executor.connect():
        print(f"\nERROR: Connection failed on port {TWS_PORT}. Ensure IB Gateway or TWS is running.")
        return

    executor.subscribe_prices()
    loop_interval=0.0 if DRY_RUN else 5.0
    loop_max_bars=len(raw_y) if DRY_RUN else None
    print(f"\n  Running swing execution loop (bars={loop_max_bars}). Press Ctrl+C to abort.\n{sep}")
    executor.run_loop(interval_sec=loop_interval,max_bars=loop_max_bars)

    # Performance reporting and analytics
    print("\nExecution complete. Generating quantitative audit report...")
    perf=executor.performance()
    if perf:
        perf.summary()
        chart_path=os.path.join(_ROOT,"swing_performance.png")
        perf.plot(save_path=chart_path)
        print(f"Performance chart saved to: {chart_path}")
    executor.disconnect()
    print(f"\nAudit trade log saved to: Trade_Implement/Executor/live_trade_log_ibkr.csv")

if __name__=="__main__":
    main()