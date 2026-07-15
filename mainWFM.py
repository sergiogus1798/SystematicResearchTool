from src.srt.diagnostics.wfmAnalysis import wfmAnalysis
import quantstats as qs

fileName = "Strategy 13.2.64.csv"

analysis1 = wfmAnalysis(fileName, returnsFreq="Weekly", startDate="2008-01-01 00:00:00", endDate="2025-12-31 23:59:59")

analysis1.getReturnsSeries()

for b, backtest in analysis1.namesListInd.items():
    ret = analysis1.backtestDfs[backtest]["PnL"]
    ret = ret.set_axis(ret.index.to_timestamp())
    
    #print(analysis1.backtestDfs[backtest])
    # Calculate risk metrics
    volatility = qs.stats.volatility(ret, periods=52)
    var = qs.stats.var(ret)
    cvar = qs.stats.cvar(ret)
    skew = qs.stats.skew(ret)
    kurtosis = qs.stats.kurtosis(ret)
    print(f"Run - {backtest}:")
    print(f"Annualized Volatility: {volatility:.4f}")
    print(f"Value at Risk (95%): {var:.4f}")
    print(f"Conditional VaR (95%): {cvar:.4f}")
    print(f"Return Distribution Skewness: {skew:.4f}")
    print(f"Return Distribution Kurtosis: {kurtosis:.4f}")
    print("\n")