flowchart TD
    subgraph FM["Financial_Mathematics/"]
        SB["SpreadBuilder\nSpreads/spread_builder.py\nBuild S_t = Y_t - β·X_t"]
        HL["HalfLife\nSpreads/half_life.py\nτ_HL = log(2)/κ"]
        ZS["ZScore\nSpreads/zscore.py\nZ_t = (S_t - μ)/σ\n[entry_z, exit_z, stop_z]"]
        KF["KalmanFilterPairs\nKalman_Filter/kalman_filter.py\nβ_t ~ N(β̂_{t|t}, P_{t|t})"]
        DB["DynamicBeta\nKalman_Filter/dynamic_beta.py\ncurrent_beta() → β_t"]
        KFC["KalmanForecaster\nKalman_Filter/kalman_forecast.py\nforecast_spread(steps=h)"]
    end

    subgraph SIGNALS["Trade_Implement/Signals/"]
        PSG["PairSignalGenerator\npair_signal.py\n[Z thresholds + CUSUM + Forecast]\n→ signal ∈ {-2,-1,0,1,2}"]
        MPSA["MultiPairSignalAggregator\nmulti_pair_signal.py\nInv-variance weights\nConsensus signal"]
    end

    subgraph RISK["Trade_Implement/Risk_Management/"]
        PSIZER["PositionSizer\nposition_sizing.py\nN_Y = f(Kelly/VolScaled/Fixed)\nN_X = round(β_t · N_Y)"]
        SLM["StopLossManager\nstoploss.py\n5 stops: Z/Dollar/Spread/Time/Trail\nenter_position() + check_stops()"]
    end

    subgraph BT["Backtesting/"]
        ENG["BacktestEngine\nengine.py\nBar-by-bar simulation loop\n→ trade_log, equity_curve"]
        PERF["PerformanceAnalytics\nperformance.py\nSharpe, Sortino, Calmar\nMDD, WinRate, ProfitFactor"]
    end

    subgraph OUT["Outputs"]
        LIVE["Live Execution\nAngel_1_Implement/\nIBKR_Implement/"]
        REPORT["Performance Report\nMetrics + Plots"]
    end

    SB -->|"S_t spread series"| ZS
    SB -->|"S_t spread series"| HL
    KF -->|"β̂_{t|t}, P_{t|t}"| DB
    DB -->|"β_t hedge ratio"| PSG
    ZS -->|"Z_t, entry/exit/stop_z"| PSG
    KFC -->|"Ŝ_{t+h}"| PSG
    HL -->|"τ_HL"| SLM
    ZS -->|"stop_z threshold"| SLM

    PSG -->|"signal DataFrame\n[z, spread, signal, position, beta]"| MPSA
    PSG -->|"signal, z, spread, beta"| ENG

    MPSA -->|"portfolio_signals()\naggregate_signal()"| ENG

    PSIZER -->|"N_Y, N_X, dollar_Y"| ENG
    SLM -->|"stop_triggered, reason, pnl"| ENG

    ENG -->|"trade_log DataFrame"| PERF
    ENG -->|"equity_curve Series"| PERF

    PERF -->|"all_metrics() dict"| REPORT
    ENG -->|"validated signals + sizing"| LIVE

    style FM fill:#1a2744,color:#cdd6f4
    style SIGNALS fill:#1a3a2a,color:#cdd6f4
    style RISK fill:#3a1a1a,color:#cdd6f4
    style BT fill:#2a1a3a,color:#cdd6f4,stroke:#7c3aed,stroke-width:2px
    style OUT fill:#1a1a2a,color:#cdd6f4