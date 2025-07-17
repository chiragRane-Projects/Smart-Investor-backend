import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError("No data found for given ticker or time range")
    
    data.reset_index(inplace=True)
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return data