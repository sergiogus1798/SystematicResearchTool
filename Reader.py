import pandas as pd
import os

class Reader:
    def __init__(self, stratsDir):
        self.dataDir = os.path.join("Data", stratsDir)
        
        # Checking path exists
        if not os.path.exists(self.dataDir):
            raise Exception(f"Strategies directory {stratsDir} does not exist.")
            
        self.stratNames = os.listdir(self.dataDir)
        self.nStrats = len(self.stratNames)
        
    def readStrats(self, startDate="2003-01-01 00:00:00", endDate="2025-12-31 23:59:59", returns="Weekly"):
        
        if returns == "Weekly":
            conversion = "W"
        elif returns == "Daily":
            conversion = "D"
        elif returns == "Monthly":
            conversion = "M"
            
        periodPartitionIndex = pd.period_range(start=startDate, end=endDate, freq=conversion)
        stratCleanData = pd.DataFrame()
        for i, strat in enumerate(self.stratNames):
            stratPath = os.path.join(self.dataDir, strat)
            dataStrat = pd.read_csv(stratPath, sep=";", parse_dates=["Close time"], date_format="%Y.%m.%d %H:%M:%S")
            dataStrat = dataStrat[['Profit/Loss', 'Close time']]
            dataStrat['Close time'] = dataStrat['Close time'].dt.to_period(conversion)
            dataStrat = dataStrat.groupby('Close time').sum()
            dataStrat = dataStrat.reindex(periodPartitionIndex, fill_value=0.00)
            stratName = strat.removesuffix(".csv")
            dataStrat = dataStrat.rename(columns={'Profit/Loss': stratName})
            stratCleanData = pd.concat([stratCleanData, dataStrat], axis=1)
 
        # Checks just in case
        periodsDF, stratsDF = stratCleanData.shape()
        
        if stratsDF != self.nStrats:
            raise Exception(f"Number of strategies ({self.nStrats}) differ from number of dataframe columns ({stratsDF}).")
            