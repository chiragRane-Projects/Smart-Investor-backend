import pandas as pd

def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Price Change'] = df['Close'] - df['Open']
    df['High-Low Diff'] = df['High'] - df['Low']
    df['5 Day MA'] = df['Close'].rolling(window=5).mean()
    df['10 Day MA'] = df['Close'].rolling(window=10).mean()
    
    df.dropna(inplace=True)
    
    return df

def create_features_targets(df: pd.DataFrame):
    df = df.copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    X = df[['Open', 'High', 'Low', 'Volume', 'Price Change', 'High-Low Diff', '5 Day MA', '10 Day MA']]
    y = df['Target']
    
    return X, y