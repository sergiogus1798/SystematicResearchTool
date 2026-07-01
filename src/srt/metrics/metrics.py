import numpy as np

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