import sys,os
import numpy as np
import pandas as pd
import pytest

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
for sub in ["Financial_Mathematics/Time_Series_Analysis/Spreads",
            "Financial_Mathematics/Time_Series_Analysis/Kalman_Filter",
            "Trade_Implement/Risk_Management","Backtesting"]:
    sys.path.insert(0,os.path.join(ROOT,sub.replace("/",os.sep)))

from spread_builder import SpreadBuilder
from half_life import HalfLife
from zscore import ZScore
from performance import PerformanceAnalytics
from position_sizing import PositionSizer
from stoploss import StopLossManager


def test_ols_spread_formula_sign():
    np.random.seed(0); N=200
    X=pd.Series(np.random.normal(100,5,N))
    Y=pd.Series(2.0+3.0*X.values+np.random.normal(0,0.5,N))
    sb=SpreadBuilder(y=Y,x=X,mode="ols"); sb.build()
    spread=sb.spread_series()
    assert abs(spread.mean())<1.0
    assert spread.std()<2.0

def test_halflife_ci_brackets_estimate():
    np.random.seed(42); T=500; s=np.zeros(T)
    for t in range(1,T): s[t]=s[t-1]-0.05*s[t-1]+np.random.normal(0,0.5)
    hl=HalfLife(pd.Series(s),clip_hl=(1,200)); tau=hl.estimate()
    assert hl.half_life_ci_95 is not None
    lo,hi=hl.half_life_ci_95
    assert lo<tau<hi

def test_var_cvar_coherence():
    np.random.seed(0)
    r=pd.Series(np.random.normal(0.001,0.02,300))
    eq=(1+r).cumprod()*10000
    pa=PerformanceAnalytics(eq,pd.DataFrame())
    var95=pa.var(0.95); cvar95=pa.cvar(0.95)
    cvar99=pa.cvar(0.99)
    assert var95>=0
    assert cvar95>=var95
    assert cvar99>=cvar95

def test_kelly_sizing_valid():
    np.random.seed(1)
    z=pd.Series(np.random.normal(0,1,300))
    sizer=PositionSizer(capital=100_000,method="kelly",kelly_fraction=4,kelly_window=100)
    res=sizer.compute(signal=1,beta_t=0.85,price_y=1500.0,price_x=900.0,zscore_series=z)
    assert res["N_Y"]>0
    assert res["leverage"]<=0.25

def test_stoploss_cooldown_blocks_reentry():
    spread=pd.Series(np.zeros(100))
    zs=ZScore(spread,window=10,entry_z=2.0,exit_z=0.5,stop_z=3.0); zs.compute()
    sl=StopLossManager(zscore=zs,half_life=20.0,capital=100_000,max_loss_pct=0.05,cooldown_bars=3)
    sl.enter_position(signal=1,spread_entry=0.0,n_shares=5)
    assert not sl.in_cooldown
    r=sl.check_stops(S_t=0.0,Z_t=4.0)
    assert r["stop_triggered"]
    assert sl.in_cooldown
    sl.check_stops(S_t=0.0,Z_t=0.0); assert sl.in_cooldown
    sl.check_stops(S_t=0.0,Z_t=0.0); assert sl.in_cooldown
    sl.check_stops(S_t=0.0,Z_t=0.0); assert not sl.in_cooldown
