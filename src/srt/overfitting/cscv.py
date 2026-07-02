import numpy as np
import itertools as it
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as statmodels
from patsy import dmatrices

class cscv:
    def __init__(self, originalData, S = 16, returnsFreq = "Weekly", performanceMetric = "Sharpe"):
        self.originalData = originalData
        self.periods = self.originalData.index
        self.strat_names = self.originalData.columns
        self.origDataNumpy = self.originalData.to_numpy()

        self.S = S
        self.nCombinations = math.comb(int(self.S), int(self.S/2))
        print("Combinatorially symmetric cross-validation (CSCV):")
        print(f"Total number of combinations for CSCV: {self.nCombinations}")
        print(f"Performance in {returnsFreq} returns")
        
        self.sharpeFactor = np.sqrt(52)
        self.performanceMetric = performanceMetric
        if self.performanceMetric not in ["Sharpe", "Sortino", "PorfitFactor", "PnL"]:
            raise Exception(f"Selected wrong performance metric for the CSCV ({self.performanceMetric}), available: (Sharpe, Sortino, ProfitFactor and PnL)")
        
        if returnsFreq == "Monthly":
            self.sharpeFactor = np.sqrt(12)
        elif returnsFreq == "Daily":
            self.sharpeFactor = np.sqrt(252)
            
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
    
    def performanceCalculation(self, backtestsMatrix):
        if self.performanceMetric == "Sharpe":
            mean = backtestsMatrix.mean(axis=0)
            std = backtestsMatrix.std(axis=0)
            return (mean / std) * self.sharpeFactor
        elif self.performanceMetric == "Sortino":
            negMatrix = backtestsMatrix[backtestsMatrix < 0.00]
            mean = backtestsMatrix.mean(axis=0)
            std = negMatrix.std(axis=0)
            return (mean / std) * self.sharpeFactor
        elif self.performanceMetric == "ProfitFactor":
            positiveTotal = backtestsMatrix[backtestsMatrix > 0.00].sum(axis=0)
            negativeTotal = backtestsMatrix[backtestsMatrix < 0.00].sum(axis=0)
            return positiveTotal / negativeTotal
        elif self.performanceMetric == "PnL":
            return backtestsMatrix.sum(axis=0)
        
            
    def runCSCV(self, submatrices):
        
        combinationsIndices = list(it.combinations(range(int(self.S)), int(self.S/2)))
        origN = np.arange(0, 16, 1, dtype=np.int_)
        
        iter = 0
        self.performanceIS = np.zeros((self.nCombinations))
        self.performanceOOS = np.zeros(self.nCombinations)
        self.lambda_c = np.zeros((self.nCombinations))
        for c in combinationsIndices:
            j1Index = np.array(c)
            j2Index = np.setdiff1d(origN, j1Index)

            j1Matrix = submatrices[j1Index].reshape(-1, self.nColumns)
            j2Matrix = submatrices[j2Index].reshape(-1, self.nColumns)
            
            R1 = self.performanceCalculation(j1Matrix)
            R2 = self.performanceCalculation(j2Matrix)
            
            indexR1max = np.argmax(R1)
            self.performanceIS[iter] = R1[indexR1max]
            self.performanceOOS[iter] = R2[indexR1max]
            
            rankedR2 = stats.rankdata(R2, method="average")
            omega_c = rankedR2[indexR1max] / (self.nColumns + 1)
            self.lambda_c[iter] = math.log(omega_c / (1 - omega_c))
            print(f"Tested {iter+1}/{self.nCombinations} combinations, ({100 * ((iter+1)/self.nCombinations):.3f} %)")
            iter = iter + 1
        
        positiveLogits = self.lambda_c[self.lambda_c > 0.00]
        self.PBO = len(positiveLogits) / self.nCombinations
            
    def plotHistogramLogits(self, nBins=50):
        
        plt.hist(x=self.lambda_c, bins=nBins, density=True)       
        plt.title("Probability density function of the logits (lambda_c)")
        plt.xlabel("Logits")
        plt.ylabel("Frequency")
        #plt.grid(show=True)
        ax = plt.gca()
        ax.text(0.95, 0.95, f"PBO = {self.PBO:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.show()   
            
    def plotPerformanceMetricComparison(self):
        
        plt.scatter(x=self.performanceIS, y=self.performanceOOS, marker=".", s=4)
        plt.xlabel("In-sample Performance", fontsize=12)
        plt.ylabel("Out-of-sample Performance", fontsize=12)
        #plt.grid(show=True)
        plt.title(f"Candidate strategies: IS vs OOS Performance ({self.performanceMetric})",
             fontsize=14, fontweight="bold", pad=14)
        
        # Fitting a OLS model
        X = statmodels.add_constant(self.performanceIS)
        model = statmodels.OLS(self.performanceOOS, X)
        res = model.fit()
        print(res.summary())
        
        alpha, beta = res.params            # [const, slope]

        x_line = np.linspace(self.performanceIS.min(), self.performanceIS.max(), 100)
        y_line = alpha + beta * x_line
        plt.plot(x_line, y_line, color="black", linewidth=2,
            label=f"OLS: y = {alpha:.2f} + {beta:.2f}x")
        plt.legend()
        plt.show()
