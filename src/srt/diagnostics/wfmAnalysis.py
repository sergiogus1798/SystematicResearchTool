import numpy as np
import pandas as pd
import os
import quantstats as qs

class wfmAnalysis:
    def __init__(self, fileName):
        self.fileName = os.path.join("DataWFM", fileName)
        
        # Checking file exists
        if not os.path.exists(self.fileName):
            raise Exception(f"File with name {fileName} does not exist in DataWFM directory.")
            
        self.rawFileData = pd.read_csv(self.fileName, sep=";", parse_dates=["Close time"], date_format="%Y.%m.%d %H:%M:%S")
        self.getNamesBacktests()

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