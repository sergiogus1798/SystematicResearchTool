import numpy as np
import scipy
import math

class metrics:
    def __init__(self, returnsFreq="Weekly"):
        self.sharpeFactor = np.sqrt(52)
        
        if returnsFreq == "Monthly":
            self.sharpeFactor = np.sqrt(12)
        elif returnsFreq == "Daily":
            self.sharpeFactor = np.sqrt(252)
    
    def profitFactor(self, tradesSeries):
        posTotal = tradesSeries[tradesSeries > 0.00].sum()
        negTotal = tradesSeries[tradesSeries < 0.00].sum()
        
        return posTotal / abs(negTotal)
    
    def netProfit(self, tradesSeries):
        return tradesSeries.sum()
    
    def sharpe(self, tradesSeries):
        return (np.mean(tradesSeries) / np.std(tradesSeries)) * self.sharpeFactor
    
    def sortino(self, tradesSeries):
        negTrades = tradesSeries[tradesSeries < 0.00]
        return (np.mean(tradesSeries) / np.std(negTrades)) * self.sharpeFactor
    
    def PSR(self, tradesSeries, annualizedBenchmarkSharpe=0.0):
        """
        Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): the probability that the
        strategy's true Sharpe exceeds a benchmark, given a finite, non-normal return sample.
        Corrects the naive Sharpe for track-record length (n) and for skew/kurtosis, which
        inflate estimation uncertainty. Returns a probability in [0, 1]; higher = more
        confident the edge is real. Inputs are per-period (weekly); benchmark is de-annualized
        internally. Use as a confidence gate on a single return series, not a cross-trial test.
        """
    
        sharpeWeekly = np.mean(tradesSeries) / np.std(tradesSeries, ddof=1)
        benchSharpeWeekly = annualizedBenchmarkSharpe / self.sharpeFactor
        
        gamma3 = scipy.stats.skew(a=tradesSeries, bias=True)
        gamma4 = scipy.stats.kurtosis(a=tradesSeries, bias=True)
        
        n = len(tradesSeries)
        numer = (sharpeWeekly - benchSharpeWeekly) * np.sqrt(n - 1)
        denom = np.sqrt(1 - gamma3 * sharpeWeekly + ((gamma4 - 1) / 4) * pow(sharpeWeekly, 2.0))
        z = numer / denom
        return scipy.stats.norm.cdf(z)
    
    def deflatedBenchmark(self, sharpeSeries):
        V = np.var(sharpeSeries, ddof=1)
        N = len(sharpeSeries)
        EULER_MASC = 0.5772156649
        emax = ((1 - EULER_MASC) * scipy.stats.norm.ppf(1 - 1/N) + EULER_MASC * scipy.stats.norm.ppf(1 - 1/(N * math.e)))
        return np.sqrt(V) * emax
        
    def DSR(self, candidateData, deflatedSR):
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado): PSR with the benchmark raised to the
        Sharpe the luckiest of N trials would post by chance. Corrects for selection across the
        N configurations searched. trialSharpes = per-period Sharpes of all N runs (incl. the
        candidate). Returns P(true Sharpe > luck bar) in [0, 1]; low = the winner may be noise-mining.
        NOTE: assumes independent trials; overlapping WF data means effective N < len(trialSharpes),
        so this is a conservative (upper-bound-N) deflation and excludes the SQX generation search.
        """
        return self.PSR(candidateData, annualizedBenchmarkSharpe=deflatedSR*self.sharpeFactor)
        
    def minTRL(self, candidateReturns, confidence=95, annualizedBenchmarkSharpe=0.0):
        """
        Minimum Track Record Length (Bailey & Lopez de Prado): the number of return periods
        (weeks) needed for the Sharpe to exceed the benchmark at confidence 1-alpha, given the
        sample's skew/kurtosis. Compare against actual OOS length: if MinTRL > weeks available,
        the track record is too short to claim the edge at this confidence. Output is in weeks.
        """
        activeReturns = candidateReturns[candidateReturns != 0]
        sharpeWeekly = np.mean(activeReturns) / np.std(activeReturns, ddof=1)
        benchSharpeWeekly = annualizedBenchmarkSharpe / self.sharpeFactor
        
        if sharpeWeekly <= benchSharpeWeekly:
            return np.inf

        gamma3 = scipy.stats.skew(a=activeReturns, bias=True)
        gamma4 = scipy.stats.kurtosis(a=activeReturns, bias=True)
        
        z_alpha = scipy.stats.norm.ppf(confidence/100)
        firstTerm = (1 - gamma3 * sharpeWeekly + pow(sharpeWeekly, 2.0) * ((gamma4 - 1)/4))
        secondTerm = pow(z_alpha/(sharpeWeekly - benchSharpeWeekly), 2.0)
        return 1 + firstTerm * secondTerm