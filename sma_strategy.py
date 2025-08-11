import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import sys

def get_stock_data(ticker, start_date, end_date, auto_adjust=True):
    """Fetch historical stock data from Yahoo Finance."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
        if df.empty:
            raise ValueError("No data returned. Check ticker symbol or date range.")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        print(f"Data fetched: {df.shape[0]} rows.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

def add_indicators(df):
    """Add SMA50, SMA200, and RSI indicators."""
    print("Calculating indicators...")
    try:
        close = df[['Close']].astype(float).squeeze().values  # Force 1D NumPy array
        df['SMA50'] = talib.SMA(close, timeperiod=50)
        df['SMA200'] = talib.SMA(close, timeperiod=200)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        sys.exit(1)

def generate_signals(df):
    """Generate buy/sell signals based on SMA crossovers."""
    print("Generating signals...")
    df['Signal'] = 0
    df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
    df.loc[df['SMA50'] < df['SMA200'], 'Signal'] = -1
    df['Signal'] = df['Signal'].fillna(0)

    # Detect crossovers for trade signals (when the signal changes)
    df['Trade Signal'] = df['Signal'].diff()

    return df

def backtest(df):
    """Backtest the strategy and compute cumulative returns."""
    print("Running backtest...")
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Returns'] * df['Signal'].shift(1)
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
    return df

def plot_results(df, ticker):
    """Plot cumulative returns and buy/sell signals."""
    print("Plotting results...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot cumulative returns
    ax1.plot(df.index, df['Cumulative Returns'], label='Market', color='blue')
    ax1.plot(df.index, df['Cumulative Strategy'], label='Strategy', color='green')
    ax1.set_title(f'Cumulative Returns vs. Strategy Returns for {ticker}')
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend()
    ax1.grid(True)

    # Plot price with SMAs
    ax2.plot(df.index, df['Close'], label='Close Price', color='black')
    ax2.plot(df.index, df['SMA50'], label='SMA50', color='orange', alpha=0.8)
    ax2.plot(df.index, df['SMA200'], label='SMA200', color='purple', alpha=0.8)

    # Buy and sell arrows
    buy_signals = df[df['Trade Signal'] == 2]
    sell_signals = df[df['Trade Signal'] == -2]

    ax2.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='green', label='Buy Signal')
    ax2.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='red', label='Sell Signal')

    ax2.set_title(f'{ticker} Price with SMA Crossovers and Trade Signals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=True)  # Keeps plot open

def main():
    print("Starting SMA strategy backtest...")

    # === Strategy Parameters ===
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2024-01-01"
    auto_adjust = True

    # === Strategy Pipeline ===
    df = get_stock_data(ticker, start_date, end_date, auto_adjust=auto_adjust)
    df = add_indicators(df)
    df = generate_signals(df)
    df = backtest(df)
    plot_results(df, ticker)

    print("Backtest complete.")

if __name__ == "__main__":
    main()
