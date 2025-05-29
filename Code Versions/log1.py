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
    if 'Date' not in df.columns:
        df = df.reset_index()
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

def Rolling_Log_Regression(df, window):
    df['price_y'] = np.log(df['Close'])
    df['x'] = np.arange(len(df))

    def fit_regression(x):
        y = df['price_y'].iloc[x.index]
        return np.polyfit(x, y, 1)[0] * x.iloc[-1] + np.polyfit(x, y, 1)[1]  # prediction at t

    df['priceTL'] = df['x'].rolling(window).apply(lambda x: fit_regression(x), raw=False)
    df['y-TL'] = df['price_y'] - df['priceTL']
    df['SD'] = df['y-TL'].rolling(window).std()
    df['TL-2SD'] = df['priceTL'] - 2 * df['SD']
    df['TL-SD'] = df['priceTL'] - df['SD']
    df['TL+SD'] = df['priceTL'] + df['SD']
    df['TL+2SD'] = df['priceTL'] + 2 * df['SD']
    return df

def run_log_regression(symbol, start, end, interval, rolling=None, save_csv=False):
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, start=start, end=end, interval=interval).reset_index()

    if df.empty or 'Close' not in df.columns:
        print(f"No valid data returned for '{symbol}'.")
        return

    df = Rolling_Log_Regression(df, rolling) if rolling else Logarithmic_regression(df)
    fig = plot_chart(df, symbol)

    filename = f'{symbol}_log_regression.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    print("Saved chart to:", os.path.abspath(filename))

    if save_csv:
        csv_name = f'{symbol}_log_regression.csv'
        df.to_csv(csv_name, index=False)
        print("Saved data to:", os.path.abspath(csv_name))


# Command line arguments below


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Financial analysis tool using log regression and technical indicators.')

    # Accept one or more symbols
    parser.add_argument('symbols', nargs='+', help='One or more stock ticker symbols (e.g., AAPL MSFT TSLA)')

    # Optional date range and interval
    parser.add_argument('--start', type=str, default='2000-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    valid_intervals = [
        '1m', '2m', '5m', '15m', '30m', '60m', '90m',
        '1d', '5d', '1wk', '1mo', '3mo'
    ]
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        choices=valid_intervals,
        help=(
            'Data interval from yfinance. Intraday (1m–90m) data only supports recent periods. '
            'Examples: 1m, 5m, 1d, 1wk, 1mo, 3mo.'
        )
    )
    # Optional rolling regression
    parser.add_argument('--rolling', type=int, default=None, help='Rolling window size for trendline (e.g., 252)')

    # Optional CSV export
    parser.add_argument('--csv', action='store_true', help='Save enriched data to CSV file')

    args = parser.parse_args()

    for symbol in args.symbols:
        symbol = symbol.upper()
        run_log_regression(
            symbol=symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            rolling=args.rolling,
            save_csv=args.csv
        )

