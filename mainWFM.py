from src.srt.diagnostics.wfmAnalysis import wfmAnalysis

fileName = "Strategy 13.2.64.csv"

analysis1 = wfmAnalysis(fileName)

analysis1.getReturnsSeries(resultsName=analysis1.namesListInd[1], returns="Weekly")
