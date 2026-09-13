"""
Strategy Comparison Backtester
================================
Project  : Dark Wing
Module   : Financial_Mathematics/Time_Series_Analysis/strategy_comparison.py

Strategies Compared
-------------------
  1. Cointegration + Kalman Filter Pairs Trading   (dynamic hedge ratio)
  2. Z-Score Static OLS Pairs Trading              (fixed OLS beta)
  3. Bollinger Band Mean Reversion                 (single stock: RELIANCE)
  4. RSI Mean Reversion                            (single stock: RELIANCE)
  5. Buy & Hold                                    (RELIANCE benchmark)

Universe  : RELIANCE.NS / TCS.NS (NSE India)
Period    : 2015-01-01 to 2026-06-30  (10 years)
Costs     : ~0.15% per trade side (brokerage + STT + SEBI + GST)
Capital   : Rs.1,00,000 base capital (normalized)
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive: saves PNG without needing a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKER_Y      = "RELIANCE.NS"
TICKER_X      = "TCS.NS"
START         = "2015-01-01"
END           = "2026-06-30"
INITIAL_CAP   = 100_000.0
COST_PER_SIDE = 0.0015        # 0.15% per side => 0.30% round-trip
TRAIN_WINDOW  = 252
ENTRY_Z       = 2.0
EXIT_Z        = 0.3
KALMAN_Q      = 1e-5
KALMAN_R      = 1e-3
BB_WINDOW     = 20
BB_STD        = 2.0
RSI_WINDOW    = 14
RSI_OB        = 70
RSI_OS        = 30
RISK_FREE     = 0.06

# ── DATA STRUCTURES ─────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_date:  object
    exit_date:   object
    entry_price: float
    exit_price:  float
    direction:   int
    pnl:         float = 0.0
    is_winner:   bool  = False

@dataclass
class BacktestResult:
    name:           str
    equity_curve:   object
    trades:         list = field(default_factory=list)
    total_return:   float = 0.0
    cagr:           float = 0.0
    sharpe:         float = 0.0
    max_drawdown:   float = 0.0
    win_rate:       float = 0.0
    n_trades:       int   = 0
    profit_factor:  float = 0.0
    total_costs:    float = 0.0
    avg_trade_days: float = 0.0
    gross_return:   float = 0.0

# ── UTILITIES ────────────────────────────────────────────────────────────────
def compute_metrics(equity, trades, initial_cap, total_costs):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    final = equity.iloc[-1]
    total_return = (final - initial_cap) / initial_cap
    cagr = (final / initial_cap) ** (1 / years) - 1 if years > 0 else 0.0
    daily_ret = equity.pct_change().dropna()
    excess = daily_ret - (RISK_FREE / 252)
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    rolling_max = equity.cummax()
    max_dd = ((equity - rolling_max) / rolling_max).min()
    closed = [t for t in trades if t.exit_date is not None]
    n_trades = len(closed)
    winners = [t for t in closed if t.pnl > 0]
    losers  = [t for t in closed if t.pnl <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0
    gp = sum(t.pnl for t in winners)
    gl = abs(sum(t.pnl for t in losers))
    profit_factor = gp / gl if gl > 0 else float("inf")
    avg_days = np.mean([(t.exit_date - t.entry_date).days for t in closed]) if closed else 0.0
    gross_return = total_return + (total_costs / initial_cap)
    return dict(total_return=total_return, cagr=cagr, sharpe=sharpe,
                max_drawdown=max_dd, win_rate=win_rate, n_trades=n_trades,
                profit_factor=profit_factor, total_costs=total_costs,
                avg_trade_days=avg_days, gross_return=gross_return)

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── KALMAN FILTER ────────────────────────────────────────────────────────────
class KF:
    def __init__(self, Q=KALMAN_Q, R=KALMAN_R, beta0=1.0, P0=1000.0):
        self.Q=Q; self.R=R; self.beta=beta0; self.P=P0
    def update(self, y, x):
        bp = self.beta; Pp = self.P + self.Q
        F = x**2 * Pp + self.R
        K = Pp * x / F
        self.beta = bp + K * (y - bp * x)
        self.P = Pp * (1 - K * x)
        if self.P <= 0: self.P = 1000.0
        return self.beta, y - self.beta * x

# ── STRATEGY 1: COINTEGRATION + KALMAN FILTER ────────────────────────────────
def run_kalman_pairs(log_y, log_x, py, px):
    kf = KF()
    equity = pd.Series(index=log_y.index, dtype=float)
    equity.iloc[0] = INITIAL_CAP
    capital = INITIAL_CAP
    trades = []; total_costs = 0.0
    spreads = []; position = 0
    entry_spread = entry_beta = entry_date = 0

    for i, date in enumerate(log_y.index):
        beta, spread = kf.update(log_y.iloc[i], log_x.iloc[i])
        spreads.append(spread)
        if i < TRAIN_WINDOW:
            equity.iloc[i] = capital; continue

        sp_arr = np.array(spreads[-60:])
        z = (spread - sp_arr.mean()) / sp_arr.std() if sp_arr.std() > 0 else 0.0
        p_y = py.iloc[i]; p_x = px.iloc[i]

        if position == 0:
            if z > ENTRY_Z:
                position = -1; entry_spread = spread; entry_date = date; entry_beta = beta
                c = COST_PER_SIDE * (p_y + beta * p_x); capital -= c; total_costs += c
            elif z < -ENTRY_Z:
                position = 1; entry_spread = spread; entry_date = date; entry_beta = beta
                c = COST_PER_SIDE * (p_y + beta * p_x); capital -= c; total_costs += c
        elif position == -1 and z < EXIT_Z:
            pnl = -(spread - entry_spread) * capital * 0.5
            c = COST_PER_SIDE * (p_y + entry_beta * p_x); capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_spread, spread, -1, pnl, pnl > 0))
            position = 0
        elif position == 1 and z > -EXIT_Z:
            pnl = (spread - entry_spread) * capital * 0.5
            c = COST_PER_SIDE * (p_y + entry_beta * p_x); capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_spread, spread, 1, pnl, pnl > 0))
            position = 0
        equity.iloc[i] = max(capital, 1.0)

    m = compute_metrics(equity.ffill(), trades, INITIAL_CAP, total_costs)
    return BacktestResult("Coint + Kalman Filter", equity.ffill(), trades, **m)

# ── STRATEGY 2: STATIC OLS Z-SCORE ──────────────────────────────────────────
def run_static_pairs(log_y, log_x, py, px):
    yt = log_y.iloc[:TRAIN_WINDOW].values; xt = log_x.iloc[:TRAIN_WINDOW].values
    Xc = np.column_stack([np.ones(len(xt)), xt])
    bols = np.linalg.lstsq(Xc, yt, rcond=None)[0]
    alpha_ols, beta_ols = float(bols[0]), float(bols[1])
    spread_full = log_y - alpha_ols - beta_ols * log_x

    equity = pd.Series(index=log_y.index, dtype=float)
    equity.iloc[0] = INITIAL_CAP
    capital = INITIAL_CAP; trades = []; total_costs = 0.0
    position = 0; entry_spread = entry_date = 0

    for i, date in enumerate(log_y.index):
        if i < TRAIN_WINDOW:
            equity.iloc[i] = capital; continue
        sw = spread_full.iloc[max(0, i-60): i]
        z = (spread_full.iloc[i] - sw.mean()) / sw.std() if sw.std() > 0 else 0.0
        sp = spread_full.iloc[i]; p_y = py.iloc[i]; p_x = px.iloc[i]

        if position == 0:
            if z > ENTRY_Z:
                position = -1; entry_spread = sp; entry_date = date
                c = COST_PER_SIDE * (p_y + beta_ols * p_x); capital -= c; total_costs += c
            elif z < -ENTRY_Z:
                position = 1; entry_spread = sp; entry_date = date
                c = COST_PER_SIDE * (p_y + beta_ols * p_x); capital -= c; total_costs += c
        elif position == -1 and z < EXIT_Z:
            pnl = -(sp - entry_spread) * capital * 0.5
            c = COST_PER_SIDE * (p_y + beta_ols * p_x); capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_spread, sp, -1, pnl, pnl > 0))
            position = 0
        elif position == 1 and z > -EXIT_Z:
            pnl = (sp - entry_spread) * capital * 0.5
            c = COST_PER_SIDE * (p_y + beta_ols * p_x); capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_spread, sp, 1, pnl, pnl > 0))
            position = 0
        equity.iloc[i] = max(capital, 1.0)

    m = compute_metrics(equity.ffill(), trades, INITIAL_CAP, total_costs)
    return BacktestResult("Static OLS Z-Score", equity.ffill(), trades, **m)

# ── STRATEGY 3: BOLLINGER BAND ───────────────────────────────────────────────
def run_bollinger(prices_y):
    ma = prices_y.rolling(BB_WINDOW).mean()
    sd = prices_y.rolling(BB_WINDOW).std()
    upper = ma + BB_STD * sd; lower = ma - BB_STD * sd

    equity = pd.Series(index=prices_y.index, dtype=float)
    equity.iloc[0] = INITIAL_CAP
    capital = INITIAL_CAP; trades = []; total_costs = 0.0
    position = 0; entry_price = entry_date = 0; shares = 0.0

    for i, date in enumerate(prices_y.index):
        if i < BB_WINDOW:
            equity.iloc[i] = capital; continue
        price = prices_y.iloc[i]; mid = ma.iloc[i]

        if position == 0:
            if price < lower.iloc[i]:
                shares = capital / price; c = capital * COST_PER_SIDE
                capital -= c; total_costs += c; position = 1; entry_price = price; entry_date = date
            elif price > upper.iloc[i]:
                shares = capital / price; c = capital * COST_PER_SIDE
                capital -= c; total_costs += c; position = -1; entry_price = price; entry_date = date
        elif position == 1 and price >= mid:
            pnl = shares * (price - entry_price); c = shares * price * COST_PER_SIDE
            capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_price, price, 1, pnl, pnl > 0))
            position = 0; shares = 0.0
        elif position == -1 and price <= mid:
            pnl = shares * (entry_price - price); c = shares * price * COST_PER_SIDE
            capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_price, price, -1, pnl, pnl > 0))
            position = 0; shares = 0.0

        mark = shares * (price - entry_price) if position != 0 else 0
        equity.iloc[i] = max(capital + mark, 1.0)

    m = compute_metrics(equity.ffill(), trades, INITIAL_CAP, total_costs)
    return BacktestResult("Bollinger Band (RELIANCE)", equity.ffill(), trades, **m)

# ── STRATEGY 4: RSI MEAN REVERSION ──────────────────────────────────────────
def run_rsi(prices_y):
    rsi_s = compute_rsi(prices_y, RSI_WINDOW)
    equity = pd.Series(index=prices_y.index, dtype=float)
    equity.iloc[0] = INITIAL_CAP
    capital = INITIAL_CAP; trades = []; total_costs = 0.0
    position = 0; entry_price = entry_date = 0; shares = 0.0

    for i, date in enumerate(prices_y.index):
        if i < RSI_WINDOW + 1 or np.isnan(rsi_s.iloc[i]):
            equity.iloc[i] = capital; continue
        price = prices_y.iloc[i]; rsi = rsi_s.iloc[i]

        if position == 0:
            if rsi < RSI_OS:
                shares = capital / price; c = capital * COST_PER_SIDE
                capital -= c; total_costs += c; position = 1; entry_price = price; entry_date = date
            elif rsi > RSI_OB:
                shares = capital / price; c = capital * COST_PER_SIDE
                capital -= c; total_costs += c; position = -1; entry_price = price; entry_date = date
        elif position == 1 and rsi > 50:
            pnl = shares * (price - entry_price); c = shares * price * COST_PER_SIDE
            capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_price, price, 1, pnl, pnl > 0))
            position = 0; shares = 0.0
        elif position == -1 and rsi < 50:
            pnl = shares * (entry_price - price); c = shares * price * COST_PER_SIDE
            capital += pnl - c; total_costs += c
            trades.append(Trade(entry_date, date, entry_price, price, -1, pnl, pnl > 0))
            position = 0; shares = 0.0

        mark = shares * (price - entry_price) if position != 0 else 0
        equity.iloc[i] = max(capital + mark, 1.0)

    m = compute_metrics(equity.ffill(), trades, INITIAL_CAP, total_costs)
    return BacktestResult("RSI Mean Reversion (RELIANCE)", equity.ffill(), trades, **m)

# ── STRATEGY 5: BUY & HOLD ──────────────────────────────────────────────────
def run_buy_hold(prices_y):
    cost = INITIAL_CAP * COST_PER_SIDE
    invested = INITIAL_CAP - cost
    shares = invested / prices_y.iloc[0]
    equity = (prices_y / prices_y.iloc[0]) * invested
    t = Trade(prices_y.index[0], prices_y.index[-1],
              prices_y.iloc[0], prices_y.iloc[-1], 1,
              shares * (prices_y.iloc[-1] - prices_y.iloc[0]) - cost,
              prices_y.iloc[-1] > prices_y.iloc[0])
    m = compute_metrics(equity, [t], INITIAL_CAP, cost)
    return BacktestResult("Buy & Hold (RELIANCE)", equity, [t], **m)

# ── REPORTING ────────────────────────────────────────────────────────────────
def print_table(results):
    sep = "=" * 115
    print(f"\n{sep}")
    print("  STRATEGY COMPARISON — Dark Wing  |  RELIANCE.NS / TCS.NS  |  2015-2026")
    print(f"  Cost model: {COST_PER_SIDE*100:.2f}% per side = {COST_PER_SIDE*2*100:.2f}% round-trip")
    print(sep)
    hdr = f"  {'Strategy':<34}{'Net Ret%':>9}{'GrossRet%':>10}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}{'WinRate%':>10}{'#Trades':>8}{'Costs Rs':>10}{'AvgDays':>9}"
    print(hdr); print("-"*115)
    for r in results:
        print(f"  {r.name:<34}"
              f"{r.total_return*100:>8.1f}%"
              f"{r.gross_return*100:>9.1f}%"
              f"{r.cagr*100:>7.1f}%"
              f"{r.sharpe:>8.2f}"
              f"{r.max_drawdown*100:>7.1f}%"
              f"{r.win_rate*100:>9.1f}%"
              f"{r.n_trades:>8d}"
              f"{r.total_costs:>9,.0f}"
              f"{r.avg_trade_days:>8.0f}d")
    print(sep)

    print("\n  -- WIN RATE DETAIL ------------------------------------------------------------")
    print(f"  {'Strategy':<34} {'Win%':>6} {'AvgWin(Rs)':>12} {'AvgLoss(Rs)':>13} {'ProfitFactor':>14}")
    print("  " + "-"*82)
    for r in results:
        closed  = [t for t in r.trades if t.exit_date is not None]
        winners = [t for t in closed if t.pnl > 0]
        losers  = [t for t in closed if t.pnl <= 0]
        aw = np.mean([t.pnl for t in winners]) if winners else 0
        al = np.mean([t.pnl for t in losers])  if losers  else 0
        pf = abs(aw / al) if al != 0 else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "Inf"
        print(f"  {r.name:<34} {r.win_rate*100:>5.1f}%  {aw:>12,.0f}   {al:>12,.0f}   {pf_s:>14}")
    print()

def plot_all(results, prices_y):
    COLORS = ["#00C9A7", "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]
    fig = plt.figure(figsize=(18, 14), facecolor="#0D0D0D")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#141414")
    for r, c in zip(results, COLORS):
        eq = r.equity_curve / INITIAL_CAP * 100
        ax1.plot(eq.index, eq.values, color=c, linewidth=1.7,
                 label=f"{r.name}  (Net {r.total_return*100:+.1f}%  |  CAGR {r.cagr*100:.1f}%)")
    ax1.axhline(100, color="#555", ls="--", lw=0.8)
    ax1.set_title("Equity Curves — Net of All Trading Costs  (Base = 100)", color="white", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio (% of Initial)", color="#CCC")
    ax1.tick_params(colors="#AAA"); ax1.grid(True, alpha=0.12, color="#444")
    ax1.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax1.legend(fontsize=8.5, facecolor="#1A1A1A", labelcolor="white", loc="upper left", framealpha=0.9)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor("#141414")
    for r, c in zip(results, COLORS):
        eq = r.equity_curve; dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.22, color=c)
        ax2.plot(dd.index, dd.values, color=c, lw=1.0, alpha=0.9,
                 label=f"{r.name}  MaxDD={r.max_drawdown*100:.1f}%")
    ax2.set_title("Drawdown (%)", color="white", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", color="#CCC")
    ax2.tick_params(colors="#AAA"); ax2.grid(True, alpha=0.12, color="#444")
    ax2.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax2.legend(fontsize=8, facecolor="#1A1A1A", labelcolor="white", loc="lower left", framealpha=0.9)

    # Panel 3: CAGR bar
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("#141414")
    short_names = [r.name.split("(")[0].strip() for r in results]
    cagrs = [r.cagr * 100 for r in results]
    bars = ax3.bar(short_names, cagrs, color=COLORS, edgecolor="#222", linewidth=0.5)
    for b, v in zip(bars, cagrs):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{v:.1f}%",
                 ha="center", va="bottom", color="white", fontsize=8)
    ax3.set_title("Net CAGR (%)", color="white", fontsize=11, fontweight="bold")
    ax3.tick_params(colors="#AAA", axis="y"); ax3.tick_params(colors="white", axis="x", labelrotation=15, labelsize=7)
    ax3.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax3.grid(True, alpha=0.12, axis="y", color="#444"); ax3.axhline(0, color="#555", lw=0.8)

    # Panel 4: Win rate + Sharpe
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor("#141414")
    x = np.arange(len(results)); w = 0.35
    wr = [r.win_rate * 100 for r in results]
    sr = [r.sharpe for r in results]
    ax4.bar(x - w/2, wr, w, label="Win Rate (%)", color=[c + "BB" for c in COLORS], edgecolor="#222")
    ax4b = ax4.twinx()
    ax4b.bar(x + w/2, sr, w, label="Sharpe", color=COLORS, alpha=0.6, edgecolor="#222")
    ax4b.set_ylabel("Sharpe", color="#AAA"); ax4b.tick_params(colors="#AAA")
    ax4b.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax4.set_xticks(x); ax4.set_xticklabels(short_names, rotation=15, fontsize=7, color="white")
    ax4.set_title("Win Rate (%) vs Sharpe Ratio", color="white", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Win Rate (%)", color="#AAA"); ax4.tick_params(colors="#AAA")
    ax4.spines[["top","right","left","bottom"]].set_edgecolor("#333")
    ax4.grid(True, alpha=0.12, axis="y", color="#444")
    h1, l1 = ax4.get_legend_handles_labels(); h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1+h2, l1+l2, fontsize=7.5, facecolor="#1A1A1A", labelcolor="white", loc="upper right")

    fig.suptitle(
        f"Dark Wing — Strategy Comparison  |  RELIANCE.NS / TCS.NS  |  2015-2026\n"
        f"Cost: {COST_PER_SIDE*100:.2f}%/side ({COST_PER_SIDE*2*100:.2f}% RT)  |  Capital: Rs.{INITIAL_CAP:,.0f}",
        color="white", fontsize=13, fontweight="bold", y=0.995)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    print(f"  Chart saved -> {out}")
    # plt.show()  # disabled for non-interactive runs; chart saved as PNG above

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Dark Wing - Strategy Comparison Backtester")
    print(f"  {TICKER_Y} / {TICKER_X}  |  {START} -> {END}")
    print("="*60 + "\n")

    raw = yf.download([TICKER_Y, TICKER_X], start=START, end=END,
                      auto_adjust=True, progress=True)["Close"].dropna()
    raw.columns = [TICKER_Y, TICKER_X]
    py = raw[TICKER_Y]; px = raw[TICKER_X]
    ly = np.log(py); lx = np.log(px)

    print(f"\n  {len(raw)} bars downloaded")
    print(f"  {TICKER_Y}: Rs.{py.iloc[0]:.1f} -> Rs.{py.iloc[-1]:.1f}")
    print(f"  {TICKER_X}: Rs.{px.iloc[0]:.1f} -> Rs.{px.iloc[-1]:.1f}\n")

    print("  [1/5] Cointegration + Kalman Filter...")
    r1 = run_kalman_pairs(ly, lx, py, px)
    print("  [2/5] Static OLS Z-Score Pairs...")
    r2 = run_static_pairs(ly, lx, py, px)
    print("  [3/5] Bollinger Band (RELIANCE)...")
    r3 = run_bollinger(py)
    print("  [4/5] RSI Mean Reversion (RELIANCE)...")
    r4 = run_rsi(py)
    print("  [5/5] Buy & Hold (RELIANCE)...")
    r5 = run_buy_hold(py)

    results = [r1, r2, r3, r4, r5]
    print_table(results)

    best_s = max(results, key=lambda r: r.sharpe)
    best_c = max(results, key=lambda r: r.cagr)
    best_w = max(results, key=lambda r: r.win_rate if r.n_trades > 1 else 0)
    print(f"  Best Sharpe  : {best_s.name}  ({best_s.sharpe:.2f})")
    print(f"  Best CAGR    : {best_c.name}  ({best_c.cagr*100:.1f}%)")
    print(f"  Best WinRate : {best_w.name}  ({best_w.win_rate*100:.1f}%)\n")

    plot_all(results, py)

if __name__ == "__main__":
    main()