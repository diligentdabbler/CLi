import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
import os

def logarithmic_regression(df):
    df = df.copy()
    df['price_y'] = np.log(df['Close'])
    df['x'] = np.arange(len(df))
    b, a = np.polyfit(df['x'], df['price_y'], 1)
    df['priceTL'] = b * df['x'] + a
    df['y_TL'] = df['price_y'] - df['priceTL']
    df['SD'] = np.std(df['y_TL'])
    df['TL_2SD'] = df['priceTL'] - 2 * df['SD']
    df['TL_SD'] = df['priceTL'] - df['SD']
    df['TLpSD'] = df['priceTL'] + df['SD']
    df['TLp2SD'] = df['priceTL'] + 2 * df['SD']
    return df

def rolling_log_regression(df, window):
    df = df.copy()
    df['price_y'] = np.log(df['Close'])
    df['x'] = np.arange(len(df))

    def fit_regression(x):
        y = df['price_y'].iloc[x.index]
        b, a = np.polyfit(x, y, 1)
        return b * x.iloc[-1] + a

    df['priceTL'] = df['x'].rolling(window).apply(lambda x: fit_regression(x), raw=False)
    df['y_TL'] = df['price_y'] - df['priceTL']
    df['SD'] = df['y_TL'].rolling(window).std()
    df['TL_2SD'] = df['priceTL'] - 2 * df['SD']
    df['TL_SD'] = df['priceTL'] - df['SD']
    df['TLpSD'] = df['priceTL'] + df['SD']
    df['TLp2SD'] = df['priceTL'] + 2 * df['SD']
    return df

def plot_chart(df, symbol, percent_gain=None, date_range=None):
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)

    fig, (ax1, ax2) = plt.subplots(dpi=600, nrows=2, sharex=True)
    ax1.grid(True, color='silver', linewidth=0.5)
    ax2.grid(True, color='silver', linewidth=0.5)
    ax1.set_ylabel('Price')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Log Price and Trend')

    title = f'{symbol} Log Regression ({percent_gain:+.2f}%)' if percent_gain is not None else f'{symbol} Log Regression'
    plt.suptitle(title, fontsize=10)
    if percent_gain is not None and date_range:
        fig.text(0.99, 0.94, f'*{date_range}*', fontsize=6, style='italic', ha='right')

    ax2.xaxis.set_major_formatter(DateFormatter("%m/%y"))

    ax1.plot(df['Date'], df['Close'], color='blue', linewidth=0.5)
    ax2.plot(df['Date'], df['price_y'], color='black', linewidth=0.5)
    ax2.plot(df['Date'], df['TLp2SD'], color='hotpink', linewidth=0.5)
    ax2.plot(df['Date'], df['TLpSD'], color='orange', linewidth=0.5)
    ax2.plot(df['Date'], df['priceTL'], color='gold', linewidth=0.5)
    ax2.plot(df['Date'], df['TL_SD'], color='yellowgreen', linewidth=0.5)
    ax2.plot(df['Date'], df['TL_2SD'], color='lightgreen', linewidth=0.5)

    ax2.fill_between(df['Date'], df['TLp2SD'], df['TLpSD'], facecolor='orange', alpha=0.6)
    ax2.fill_between(df['Date'], df['TLpSD'], df['priceTL'], facecolor='gold', alpha=0.6)
    ax2.fill_between(df['Date'], df['priceTL'], df['TL_SD'], facecolor='yellowgreen', alpha=0.6)
    ax2.fill_between(df['Date'], df['TL_SD'], df['TL_2SD'], facecolor='lightgreen', alpha=0.6)

    # Add SD level labels to right side
    last_date = df['Date'].iloc[-1]
    ax2.text(last_date, df['TLp2SD'].iloc[-1], '+2', va='center', ha='left', fontsize=6)
    ax2.text(last_date, df['TLpSD'].iloc[-1], '+1', va='center', ha='left', fontsize=6)
    ax2.text(last_date, df['priceTL'].iloc[-1], '0', va='center', ha='left', fontsize=6)
    ax2.text(last_date, df['TL_SD'].iloc[-1], '-1', va='center', ha='left', fontsize=6)
    ax2.text(last_date, df['TL_2SD'].iloc[-1], '-2', va='center', ha='left', fontsize=6)

    return fig

def run_log_regression(symbol, start, end, interval, rolling=None, save_csv=False, args=None):
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, start=start, end=end, interval=interval).reset_index()

    if df.empty or 'Close' not in df.columns:
        print(f"No valid data returned for '{symbol}'.")
        return

    percent_gain = None
    date_range = None
    if args and (args.pergl or getattr(args, 'PERGL', False)):
        try:
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            percent_gain = ((end_price - start_price) / start_price) * 100
            date_range = f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}"
            print(f"{symbol} percent gain from {start} to {end}: {percent_gain:.2f}%")
        except Exception as e:
            print(f"Error calculating percent gain for {symbol}: {e}")

    df = rolling_log_regression(df, rolling) if rolling and rolling < len(df) else logarithmic_regression(df)
    fig = plot_chart(df, symbol, percent_gain=percent_gain, date_range=date_range)

    filename = f'{symbol}_log_regression.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("Saved chart to:", os.path.abspath(filename))

    if save_csv or getattr(args, 'CSV', False):
        csv_name = f'{symbol}_log_regression.csv'
        df.to_csv(csv_name, index=False)
        print("Saved data to:", os.path.abspath(csv_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logarithmic regression analysis for stock tickers.')
    parser.add_argument('tickers', nargs='+', help='Stock ticker symbols (e.g., AAPL MSFT TSLA)')
    parser.add_argument('--start', type=str, default='2000-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')

    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
    parser.add_argument('--intrv', '--INTRV', dest='inter', type=str, default='1d', choices=valid_intervals, help='Data interval (e.g., 1d, 1wk, 1mo)')
    parser.add_argument('--smooth', type=int, default=None, help='Rolling window for regression (e.g., 252)')
    parser.add_argument('--csv', '--CSV', dest='csv', action='store_true', help='Save the enriched data to CSV')
    parser.add_argument('--pergl', '--PERGL', dest='pergl', action='store_true', help='Display percent gain over the date range')

    args = parser.parse_args()
    for symbol in args.tickers:
        run_log_regression(
            symbol=symbol.upper(),
            start=args.start,
            end=args.end,
            interval=args.inter,
            rolling=args.smooth,
            save_csv=args.csv,
            args=args
        )

