import quantstats as qs
import pandas as pd

import quantstats as qs
import pandas as pd
import yfinance as yf

# Download historical data for a stock
stock_data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

# Extract the returns series
returns = stock_data['Adj Close'].pct_change().dropna()

# Calculate basic performance metrics
cagr = qs.stats.cagr(returns)
sharpe = qs.stats.sharpe(returns)
sortino = qs.stats.sortino(returns)
calmar = qs.stats.calmar(returns)

print(f"CAGR: {cagr:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Sortino Ratio: {sortino:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")
