import numpy as np
import itertools as it
import math

class cscv:
    def __init__(self, originalData, S = 16):
        self.originalData = originalData
        self.periods = self.originalData.index
        self.strat_names = self.originalData.columns
        self.origDataNumpy = self.originalData.to_numpy()
        print(self.periods)
        self.S = S
        self.nCombinations = math.comb(int(self.S), int(self.S/2))
        print("Combinatorially symmetric cross-validation (CSCV):")
        print(f"Total number of combinations for CSCV: {self.nCombinations}")
        
        self.nRows, self.nColumns = self.originalData.shape
        subrowsIndices = self.subRows()
        self.submatrices = self.createSubmatrices(subrowsIndices)

        self.runCSCV(self.submatrices)
        
    def subRows(self):
        self.nSubRows = round(self.nRows/self.S, 0)
        nLeftRows = self.nRows - self.S * self.nSubRows
        if nLeftRows == 0:
            print(f"Number of total rows ({self.nRows}) is divisible by number of submatrices ({self.S})")
        else:
            print(f"Number of total rows ({self.nRows}) is NOT divisible by number of submatrices ({self.S}), dropping {int(nLeftRows)} rows from the beginning")
            
        rowsIndices = np.arange(nLeftRows, self.nRows + self.nSubRows, self.nSubRows, dtype=np.int_) # Added "+ self.nSubRows" cause otherwise array lacks last value
        return rowsIndices
    
    def createSubmatrices(self, rowIndices):
        submatrices = np.zeros([int(self.S), int(self.nSubRows), int(self.nColumns)])
        for s in range(len(rowIndices)-1):
            initialIndex = rowIndices[s]
            lastIndex = rowIndices[s+1]
            submatrices[s,:,:] = self.origDataNumpy[initialIndex:lastIndex, :]
        
        return submatrices    
        
    def runCSCV(self, submatrices):
        
        combinationsIndices = list(it.combinations(range(int(self.S)), int(self.S/2)))
        origN = np.arange(0, 16, 1, dtype=np.int_)
        iter = 1
        for c in combinationsIndices:
            j1Index = np.array(c)
            j2Index = np.setdiff1d(origN, j1Index)

            j1Matrix = submatrices[j1Index].reshape(-1, self.nColumns)
            j2Matrix = submatrices[j2Index].reshape(-1, self.nColumns)
            
            print(f"Tested {iter}/{self.nCombinations} combinations, ({100 * (iter/self.nCombinations)} %)")
            iter = iter + 1