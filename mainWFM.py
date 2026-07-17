from src.srt.diagnostics.wfmAnalysis import wfmAnalysis
import quantstats as qs

fileName = "Strategy 13.2.64.csv"

analysis1 = wfmAnalysis(fileName, returnsFreq="Weekly", startDate="2008-01-01 00:00:00", endDate="2025-12-31 23:59:59")

analysis1.getReturnsSeries()

for b, backtest in analysis1.namesListInd.items():
    ret = analysis1.backtestDfs[backtest]["Returns"]
    ret = ret.set_axis(ret.index.to_timestamp())
    
    #print(analysis1.backtestDfs[backtest])
    # Calculate risk metrics
    volatility = qs.stats.volatility(ret, periods=52)
    sharpe = qs.stats.sharpe(ret)
    var = qs.stats.var(ret)
    cvar = qs.stats.cvar(ret)
    skew = qs.stats.skew(ret)
    kurtosis = qs.stats.kurtosis(ret)
    print(f"Run - {backtest}:")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Annualized Volatility: {volatility:.4f}")
    print(f"Value at Risk (95%): {var:.4f}")
    print(f"Conditional VaR (95%): {cvar:.4f}")
    print(f"Return Distribution Skewness: {skew:.4f}")
    print(f"Return Distribution Kurtosis: {kurtosis:.4f}")
    print("\n")
    #a = qs.stats.rolling_sharpe(ret, rolling_period=26, periods_per_year=52)
    #qs.reports.html(ret, "SPY")
    
#analysis1.plotEquityCurves(mainName="Main: XAUUSD_DukasM1_Infinox/H1", capital=100000.0, figsize=(12, 6))

analysis1.plotDegradation(mainName="Main: XAUUSD_DukasM1_Infinox/H1", metric="annualReturn", method="ratio_baseline", window=52, baselineWindows=4*52)
