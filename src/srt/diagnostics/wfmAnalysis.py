import numpy as np
import pandas as pd
import os
import quantstats as qs
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


    def getReturnsSeries(self, normalizeReturns=True):
        
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
            if normalizeReturns:
                pnl = pnl / self.initialBalance
            label = resultData.groupby("Close time")["Sample type"].first()
            
            df = pd.DataFrame({"PnL": pnl, "Sample type": label})
            df = df.reindex(periodPartitionIndex)
            df["PnL"] = df["PnL"].fillna(0.0)
            
            rowsIS = df.index[df["Sample type"].str.contains("IS", na=False)]
            rowsOOS = df.index[df["Sample type"].str.contains("OOS", na=False)]
            
            if len(rowsIS) != 0:
                df.loc[:rowsIS.max(), "sample_type"] = "IS"
            if len(rowsOOS) != 0:
                df.loc[rowsOOS.min():, "sample_type"] = "OOS"
        
            df["Sample type"] = (df["Sample type"] == "OOS").astype(int)
            self.backtestDfs[backtest] = df[["PnL", "Sample type"]]


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
        