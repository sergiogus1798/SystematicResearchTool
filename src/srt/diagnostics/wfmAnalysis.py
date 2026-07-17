import numpy as np
import pandas as pd
import os
import quantstats as qs
import matplotlib.pyplot as plt
from src.srt.metrics.metrics import metrics

class wfmAnalysis:
    def __init__(self, fileName, returnsFreq="Weekly", startDate="2003-01-01 00:00:00", endDate="2025-12-31 23:59:59"):
        self.fileName = os.path.join("DataWFM", fileName)
        
        # Checking file exists
        if not os.path.exists(self.fileName):
            raise Exception(f"File with name {fileName} does not exist in DataWFM directory.")
            
        self.rawFileData = pd.read_csv(self.fileName, sep=";", parse_dates=["Close time"], date_format="%Y.%m.%d %H:%M:%S")
        
        self.initialBalance = self.getInitialBalance()
        self.getNamesBacktests()
        self.nRuns = len(self.namesListInd)
        self.returnsFreq = returnsFreq
        self.startDate = startDate
        self.endDate = endDate
        self.metrics = metrics(returnsFreq=self.returnsFreq)
    
    def getInitialBalance(self):
        firstTrade = self.rawFileData["Profit/Loss"][0]    
        firstBalance = self.rawFileData["Balance"][0]
        if firstTrade <= 0.0:
            return firstBalance + abs(firstTrade)
        else:
            return firstBalance - abs(firstTrade)
        
        
    def getNamesBacktests(self):
        namesList = list(pd.unique(self.rawFileData["Result name"]))
        self.namesListInd = {index: name for index, name in enumerate(namesList)}


    def getReturnsSeries(self):
        
        if self.returnsFreq == "Weekly":
            conversion = "W"
        elif self.returnsFreq == "Daily":
            conversion = "D"
        elif self.returnsFreq == "Monthly":
            conversion = "ME"
        else:
            raise ValueError(f"returns must be 'Daily', 'Weekly', or 'Monthly', got {self.returnsFreq!r}")

        periodPartitionIndex = pd.period_range(start=self.startDate, end=self.endDate, freq=conversion)
        self.backtestDfs = {}
        for b, backtest in self.namesListInd.items():
            resultData = self.rawFileData[self.rawFileData["Result name"] == backtest]
            resultData = resultData[["Close time", "Profit/Loss", "Sample type"]]
            resultData['Close time'] = resultData['Close time'].dt.to_period(conversion)
            
            pnl = resultData.groupby('Close time')["Profit/Loss"].sum()
            returns = pnl / self.initialBalance
           
            label = resultData.groupby("Close time")["Sample type"].first()
            
            df = pd.DataFrame({"PnL": pnl, "Returns": returns, "Sample type": label})
            df = df.reindex(periodPartitionIndex)
            df["PnL"] = df["PnL"].fillna(0.0)
            df["Returns"] = df["Returns"].fillna(0.0)
            
            rowsIS = df.index[df["Sample type"].str.contains("IS", na=False)]
            rowsOOS = df.index[df["Sample type"].str.contains("OOS", na=False)]
            
            if len(rowsIS) != 0:
                df.loc[:rowsIS.max(), "sample_type"] = "IS"
            if len(rowsOOS) != 0:
                df.loc[rowsOOS.min():, "sample_type"] = "OOS"
        
            df["Sample type"] = (df["Sample type"] == "OOS").astype(int)
            self.backtestDfs[backtest] = df[["PnL", "Returns", "Sample type"]]


    def filterReturnsData(self, rawData, dataType="All", countZeroWeeks=True):
        
        if dataType == "All":  
            returns = rawData[:, 0]
        elif dataType == "IS":
            returns = returns[rawData[:, 1] == 0]
        elif dataType == "OOS":
            returns = returns[rawData[:, 1] == 1]
        
        nonzeroReturns = np.nonzero(returns)[0]
        first, last = nonzeroReturns.min(), nonzeroReturns.max()
        activeReturns = returns[first:last+1]
        if not countZeroWeeks:
            activeReturns = activeReturns[activeReturns != 0]
            
        return activeReturns
                
    def getSignificanceMetrics(self):
            
        self.sharpeSeries = np.zeros((self.nRuns, 3)) # 0->IS, 1->OOS, 2->ALL
        self.PSRSeries = np.zeros((self.nRuns, 3)) # 0->IS, 1->OOS, 2->ALL
        
        self.lengthRuns_OOS = np.zeros(self.nRuns)
        self.minTRLSeries_OOS = np.zeros(self.nRuns)
        for i in range(self.nRuns):
            returns_All = self.filterReturnsData(rawData=self.backtestData[i, :, :], dataType="All", countZeroWeeks=True)
            returns_IS = self.filterReturnsData(rawData=self.backtestData[i, :, :], dataType="IS", countZeroWeeks=True)
            returns_OOS = self.filterReturnsData(rawData=self.backtestData[i, :, :], dataType="OOS", countZeroWeeks=True)
            
            self.sharpeSeries[i, 0] = self.metrics.sharpe(tradesSeries=returns_All)
            self.sharpeSeries[i, 1] = self.metrics.sharpe(tradesSeries=returns_IS)
            self.sharpeSeries[i, 2] = self.metrics.sharpe(tradesSeries=returns_OOS)
            
            self.PSRSeries[i, 0] = self.metrics.PSR(tradesSeries=returns_All, annualizedBenchmarkSharpe=0.0)
            self.PSRSeries[i, 1] = self.metrics.PSR(tradesSeries=returns_IS, annualizedBenchmarkSharpe=0.0)
            self.PSRSeries[i, 2] = self.metrics.PSR(tradesSeries=returns_OOS, annualizedBenchmarkSharpe=0.0)
            
            self.lengthRuns_OOS[i] = len(returns_OOS)
            self.minTRLSeries_OOS[i] = self.metrics.minTRL(candidateReturns=returns_OOS, confidence=95, annualizedBenchmarkSharpe=0.0)
            
        valid = ~np.isnan(self.sharpeSeries_OOS)
        deflatedBenchmarkSR = self.metrics.deflatedBenchmark(sharpeSeries=self.sharpeSeries_OOS[valid])
        bestSharpeIndex_OOS = np.nanargmax(self.sharpeSeries_OOS)
        returnsBest_OOS = self.filterReturnsData(rawData=self.backtestData[bestSharpeIndex_OOS, :, :], dataType="OOS", countZeroWeeks=True)
        self.deflatedSR = self.metrics.DSR(candidateData=returnsBest_OOS, deflatedSR=deflatedBenchmarkSR)
        
    #def getDegradation(self):
        
    def rollingCalculation(self, name, metric="sharpe", window=52, periodsYear=52, capital=None):

        if capital == None:
            capital = self.initialBalance
          
        s = self.backtestDfs[name]["Returns"]
        s.index = s.index.to_timestamp()

        if metric == 'sharpe':
            r = qs.stats.rolling_sharpe(s, rolling_period=window, periods_per_year=periodsYear)
        elif metric == 'sortino':
            r = qs.stats.rolling_sortino(s, rolling_period=window, periods_per_year=periodsYear)
        elif metric == 'volatility':
            r = qs.stats.rolling_volatility(s, rolling_period=window, periods_per_year=periodsYear)
        elif metric == 'profitFactor':
            pos = s.clip(lower=0).rolling(window).sum()
            neg = -s.clip(upper=0).rolling(window).sum()
            r = pos / neg.replace(0, np.nan)
        elif metric == 'annualReturn':
            # fixed-notional -> arithmetic: mean weekly PnL scaled to a year
            r = s.rolling(window).mean() * periodsYear
        else:
            raise ValueError(metric)

        return r.replace([np.inf, -np.inf], np.nan).dropna()


    def normalizeDegradation(self, r, method='diff_baseline', baselineWindows=52):
        """Normalize one rolling series against a reference. Pure: Series -> Series."""
        if len(r) < baselineWindows + 10:
            return None
        base = r.iloc[:baselineWindows].mean()
        if not np.isfinite(base):
            return None

        if method == 'diff_baseline':
            return r - base
        if method == 'ratio_baseline':
            return r / base if base > 1e-9 else None
        if method == 'zscore_baseline':
            sd = r.iloc[:baselineWindows].std(ddof=1)
            return (r - base) / sd if sd > 1e-9 else None
        if method == 'expanding_ref':
            return r - r.expanding().mean()
        if method == 'rank':
            return r.rank(pct=True)
        if method == 'none':
            return r
        raise ValueError(method)


    def plotEquityCurves(self, mainName=None, capital=100000.0, figsize=(12, 6)):
        """
        Plot cumulative equity for every run. Main is highlighted; WF runs are thin/grey.
        Cumulative SUM (not product) — fixed-notional risk means the account is arithmetic.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if mainName is None:
            mainName = next(n for n in self.backtestDfs if n.startswith('Main'))

        # WF runs first so Main draws on top
        for name, df in self.backtestDfs.items():
            if name == mainName:
                continue
            eq = df["Returns"].cumsum() * 100
            ax.plot(eq.index.to_timestamp(), eq.values,
                    color='#9aa0a6', linewidth=0.8, alpha=0.45, zorder=1)

        eq = self.backtestDfs[mainName]["Returns"].cumsum() * 100
        ax.plot(eq.index.to_timestamp(), eq.values,
                color='#c0392b', linewidth=2.0, zorder=3, label='Main')

        ax.plot([], [], color='#9aa0a6', linewidth=0.8, alpha=0.6,
                label=f'WF runs (n={len(self.backtestDfs)-1})')
        ax.axhline(0, color='#555', linewidth=0.7, linestyle='--', zorder=0)
        ax.set_xlabel('date')
        ax.set_ylabel('cumulative return (% of initial capital)')
        ax.legend(frameon=False, loc='upper left')
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()
        

    def plotDegradation(self, metric='sharpe', window=52, method='diff_baseline',
                    baselineWindows=52, mainName=None, figsize=(12, 6)):
        """Orchestration: loop runs, normalize, draw."""
        if mainName is None:
            mainName = next(n for n in self.backtestDfs if n.startswith('Main'))

        curves = {}
        for name in self.backtestDfs:
            r = self.rollingCalculation(name, metric, window)
            z = self.normalizeDegradation(r, method, baselineWindows)
            if z is not None:
                curves[name] = z

        fig, ax = plt.subplots(figsize=figsize)
        for name, z in curves.items():
            if name != mainName:
                ax.plot(z.index, z.values, color='#9aa0a6', lw=0.8, alpha=0.35, zorder=1)

        if curves:
            stacked = pd.concat(curves.values(), axis=1)
            ax.plot(stacked.index, stacked.median(axis=1).values,
                    color='#1f4e9c', lw=2.2, zorder=4, label='median of runs')
            ax.fill_between(stacked.index, stacked.quantile(.25, axis=1),
                            stacked.quantile(.75, axis=1),
                            color='#1f4e9c', alpha=0.12, zorder=0, label='IQR')

        if mainName in curves:
            z = curves[mainName]
            ax.plot(z.index, z.values, color='#c0392b', lw=2.0, zorder=3, label='Main')

        ref = {'ratio_baseline': 1.0, 'rank': 0.5}.get(method, 0.0)
        ax.axhline(ref, color='#333', lw=1.0, ls='--', zorder=2)
        ax.set_ylabel(f'{metric} ({window}w) — {method}')
        ax.legend(frameon=False, loc='upper right')
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        plt.show()