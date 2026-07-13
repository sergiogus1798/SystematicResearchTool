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

    def getReturnsSeries(self, resultsName, startDate="2003-01-01 00:00:00", endDate="2025-12-31 23:59:59", returns="Weekly"):
        
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
        
        
    def getSignificanceMetrics(self, countZeroWeeks=True):
            
        self.sharpeSeries_All = np.zeros(self.nRuns)
        self.sharpeSeries_IS = np.zeros(self.nRuns)
        self.sharpeSeries_OOS = np.zeros(self.nRuns)
        self.PSRSeries_IS = np.zeros(self.nRuns)
        self.PSRSeries_OOS = np.zeros(self.nRuns)
        
        self.lengthRuns_OOS = np.zeros(self.nRuns)
        self.minTRLSeries_OOS = np.zeros(self.nRuns)
        for i in range(self.nRuns):
            returns = self.backtestData[i, :, 0]
            returns_IS = returns[self.backtestData[i, :, 1] == 0]
            returns_OOS = returns[self.backtestData[i, :, 1] == 1]
            
            nonzero_pos = np.nonzero(returns)[0]
            nonzero_pos_IS = np.nonzero(returns_IS)[0]     # positions of active (nonzero) weeks
            nonzero_pos_OOS = np.nonzero(returns_OOS)[0]     # positions of active (nonzero) weeks
            first, last = nonzero_pos.min(), nonzero_pos.max()
            firstIS, lastOOS = nonzero_pos_IS.min(), nonzero_pos_OOS.max()
            activeReturns = returns[first:last+1]
            activeReturns_IS = returns_IS[firstIS:]
            activeReturns_OOS = returns_OOS[:lastOOS+1]
            if not countZeroWeeks:
                activeReturns = activeReturns[activeReturns != 0]
                activeReturns_IS = activeReturns_IS[activeReturns_IS != 0]
                activeReturns_OOS = activeReturns_OOS[activeReturns_OOS != 0]
                
            self.sharpeSeries_All[i] = self.metrics.sharpe(tradesSeries=activeReturns)
            self.sharpeSeries_IS[i] = self.metrics.sharpe(tradesSeries=activeReturns_IS)
            self.sharpeSeries_OOS[i] = self.metrics.sharpe(tradesSeries=activeReturns_OOS)
            self.PSRSeries_IS[i] = self.metrics.PSR(tradesSeries=activeReturns_IS, annualizedBenchmarkSharpe=0.0)
            self.PSRSeries_OOS[i] = self.metrics.PSR(tradesSeries=activeReturns_OOS, annualizedBenchmarkSharpe=0.0)
            
            self.lengthRuns_OOS[i] = len(activeReturns_OOS)
            self.minTRLSeries_OOS[i] = self.metrics.minTRL(candidateReturns=activeReturns_OOS, confidence=95, annualizedBenchmarkSharpe=0.0)
            
        valid = ~np.isnan(self.sharpeSeries_OOS)
        deflatedBenchmarkSR = self.metrics.deflatedBenchmark(sharpeSeries=self.sharpeSeries_OOS[valid])
        bestSharpeIndex_OOS = np.nanargmax(self.sharpeSeries_OOS)
        returnsBest_OOS = self.backtestData[bestSharpeIndex_OOS, :, 0][self.backtestData[bestSharpeIndex_OOS, :, 1] == 1]
        nonzero_pos_OOS = np.nonzero(returnsBest_OOS)[0]     # positions of active (nonzero) weeks
        lastOOS = nonzero_pos_OOS.max()
        activereturnsBest_OOS = returnsBest_OOS[:lastOOS+1]
        if not countZeroWeeks:
            activereturnsBest_OOS = activereturnsBest_OOS[activereturnsBest_OOS != 0]

        self.deflatedSR = self.metrics.DSR(candidateData=activereturnsBest_OOS, deflatedSR=deflatedBenchmarkSR)