import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import os

def Logarithmic_regression(df):
    df['price_y'] = np.log(df['Close'])
    df['x'] = np.arange(len(df))
    b, a = np.polyfit(df['x'], df['price_y'], 1)
    df['priceTL'] = b * df['x'] + a
    df['y-TL'] = df['price_y'] - df['priceTL']
    df['SD'] = np.std(df['y-TL'])
    df['TL-2SD'] = df['priceTL'] - 2 * df['SD']
    df['TL-SD'] = df['priceTL'] - df['SD']
    df['TL+SD'] = df['priceTL'] + df['SD']
    df['TL+2SD'] = df['priceTL'] + 2 * df['SD']
    return df

def plot_chart(df, symbol):
    fig, (ax1, ax2) = plt.subplots(dpi=600, nrows=2, sharex=True)
    ax1.grid(True, color='silver', linewidth=0.5)
    ax2.grid(True, color='silver', linewidth=0.5)
    ax1.set_ylabel('Price')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Indicator')
    plt.suptitle(f'{symbol} Stock Price Trend with Logarithmic Regression', fontsize=10)
    ax2.xaxis.set_major_formatter(DateFormatter("%m/%y"))

    ax1.plot(df['Date'], df['Close'], color='blue', linewidth=0.5)
    ax2.plot(df['Date'], df['price_y'], color='black', linewidth=0.5)

    colors = ['hotpink', 'orange', 'gold', 'yellowgreen', 'lightgreen']
    ax2.plot(df['Date'], df['TL+2SD'], color=colors[0], linewidth=0.5)
    ax2.plot(df['Date'], df['TL+SD'], color=colors[1], linewidth=0.5)
    ax2.plot(df['Date'], df['priceTL'], color=colors[2], linewidth=0.5)
    ax2.plot(df['Date'], df['TL-SD'], color=colors[3], linewidth=0.5)
    ax2.plot(df['Date'], df['TL-2SD'], color=colors[4], linewidth=0.5)

    ax2.fill_between(df['Date'], df['TL+2SD'], df['TL+SD'], facecolor=colors[1], alpha=0.6)
    ax2.fill_between(df['Date'], df['TL+SD'], df['priceTL'], facecolor=colors[2], alpha=0.6)
    ax2.fill_between(df['Date'], df['priceTL'], df['TL-SD'], facecolor=colors[3], alpha=0.6)
    ax2.fill_between(df['Date'], df['TL-SD'], df['TL-2SD'], facecolor=colors[4], alpha=0.6)

    return fig

def run_log_regression(symbol):
    print(f"Fetching all historical data for {symbol}...")
    df = yf.Ticker(symbol).history(period="max", interval='1d').reset_index()

    if df.empty or 'Close' not in df.columns:
        print(f"No valid data returned for '{symbol}'.")
        return

    df = Logarithmic_regression(df)
    fig = plot_chart(df, symbol)

    filename = f'{symbol}_log_regression.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    print("Saved to:", os.path.abspath(filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot and save a logarithmic regression chart for a stock ticker.')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol (e.g., AAPL, MSFT, UNH)')

    args = parser.parse_args()
    run_log_regression(args.symbol.upper())