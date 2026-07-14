import numpy as np
import pandas as pd
import os
import quantstats as qs
from src.srt.metrics.metrics import metrics

class wfmAnalysis:
    def __init__(self, fileName, returnsFreq="Weekly"):
        self.fileName = os.path.join("DataWFM", fileName)
        
        # Checking file exists
        if not os.path.exists(self.fileName):
            raise Exception(f"File with name {fileName} does not exist in DataWFM directory.")
            
        self.rawFileData = pd.read_csv(self.fileName, sep=";", parse_dates=["Close time"], date_format="%Y.%m.%d %H:%M:%S")
        self.getNamesBacktests()
        self.nRuns = len(self.namesListInd)
        
        self.metrics = metrics(returnsFreq=returnsFreq)
        
    def getNamesBacktests(self):
        namesList = list(pd.unique(self.rawFileData["Result name"]))
        self.namesListInd = {index: name for index, name in enumerate(namesList)}

    def getReturnsSeries(self, startDate="2003-01-01 00:00:00", endDate="2025-12-31 23:59:59", returns="Weekly"):
        
        if returns == "Weekly":
            conversion = "W"
        elif returns == "Daily":
            conversion = "D"
        elif returns == "Monthly":
            conversion = "ME"
        else:
            raise ValueError(f"returns must be 'Daily', 'Weekly', or 'Monthly', got {returns!r}")

        periodPartitionIndex = pd.period_range(start=startDate, end=endDate, freq=conversion)
        self.nBacktests = len(self.namesListInd)
        self.backtestData = np.zeros((self.nBacktests, len(periodPartitionIndex), 2))
        for b, backtest in self.namesListInd.items():
            print(f"{b}: {backtest}")
            resultData = self.rawFileData[self.rawFileData["Result name"] == backtest]
            resultData = resultData[["Close time", "Profit/Loss", "Sample type"]]
            resultData['Close time'] = resultData['Close time'].dt.to_period(conversion)
            resultData = resultData.groupby('Close time').sum()
            resultData = resultData.reindex(periodPartitionIndex, fill_value=0.00)    

            rowsIS = resultData.index[resultData["Sample type"].str.contains("IS", na=False)]
            rowsOOS = resultData.index[resultData["Sample type"].str.contains("OOS", na=False)]
            
            lastIS = rowsIS.max()
            firstOOS = rowsOOS.min()
            
            resultData.loc[:lastIS, "Sample type"] = "IS"
            
            if len(rowsOOS) != 0:
                resultData.loc[firstOOS:, "Sample type"] = "OOS"

            resultData["Sample type"] = (resultData["Sample type"] == "OOS").astype(int)
            
            self.backtestData[b] = resultData.to_numpy()
            
        print(self.backtestData.shape)
        
    
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