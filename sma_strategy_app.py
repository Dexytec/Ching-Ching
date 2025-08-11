import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date, auto_adjust=True):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def add_indicators(df):
    close = df[['Close']].astype(float).squeeze().values
    df['SMA50'] = talib.SMA(close, timeperiod=50)
    df['SMA200'] = talib.SMA(close, timeperiod=200)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    return df

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
    df.loc[df['SMA50'] < df['SMA200'], 'Signal'] = -1
    df['Signal'] = df['Signal'].fillna(0)
    df['Trade Signal'] = df['Signal'].diff()
    return df

def backtest(df):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Returns'] * df['Signal'].shift(1)
    df['Cumulative Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
    return df

def plot_results(df, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(df.index, df['Cumulative Returns'], label='Market', color='blue')
    ax1.plot(df.index, df['Cumulative Strategy'], label='Strategy', color='green')
    ax1.set_title(f'Cumulative Returns vs. Strategy Returns for {ticker}')
    ax1.set_ylabel('Cumulative Returns')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, df['Close'], label='Close Price', color='black')
    ax2.plot(df.index, df['SMA50'], label='SMA50', color='orange', alpha=0.8)
    ax2.plot(df.index, df['SMA200'], label='SMA200', color='purple', alpha=0.8)

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
    return fig

# Streamlit UI
st.title("SMA Strategy Backtest App 📈")

ticker = st.text_input("Enter ticker symbol:", "AAPL")
start_date = st.date_input("Start date:", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End date:", pd.to_datetime("2024-01-01"))

if st.button("Run Backtest"):
    with st.spinner("Fetching data..."):
        try:
            df = get_stock_data(ticker, start_date, end_date)
            df = add_indicators(df)
            df = generate_signals(df)
            df = backtest(df)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("Backtest complete!")
    st.write(df.tail(10))  # Show last 10 rows

    fig = plot_results(df, ticker)
    st.pyplot(fig)

