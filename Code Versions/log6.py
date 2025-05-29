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

    # --------- Run Log Regression ---------


def run_log_regression(symbol, start, end, interval, rolling=None, save_csv=False, args=None):
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, start=start, end=end, interval=interval).reset_index()

    if df.empty or 'Close' not in df.columns:
        print(f"No valid data returned for '{symbol}'.")
        return

# Set default dates based on actual data if not provided
    if start is None:
        start = df['Date'].iloc[0].strftime('%Y-%m-%d')
    if end is None:
        end = df['Date'].iloc[-1].strftime('%Y-%m-%d')

    percent_gain = None
    date_range = None
    if args and (args.perc or getattr(args, 'PERC', False)):
        try:
            start_price = float(df['Close'].iloc[0].item())
            end_price = float(df['Close'].iloc[-1].item())
            percent_gain = ((end_price - start_price) / start_price) * 100
            date_range = f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}"
            print(f"{symbol} percent gain from {start} to {end}: {percent_gain:.2f}%")
        except Exception as e:
            print(f"Error calculating percent gain for {symbol}: {e}")

    df = rolling_log_regression(df, rolling) if rolling and rolling < len(df) else logarithmic_regression(df)
    fig = plot_chart(df, symbol, percent_gain=percent_gain, date_range=date_range)

    # Save PNG + CSV for the regression chart
    filename = f'{symbol}_log_regression.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("Saved chart to:", os.path.abspath(filename))

    if save_csv or getattr(args, 'CSV', False):
        csv_name = f'{symbol}_log_regression.csv'
        df.to_csv(csv_name, index=False)
        print("Saved data to:", os.path.abspath(csv_name))

    # --------- New Logic Starts Here ---------
    std_dev = round(df['y_TL'].std(), 4) if 'y_TL' in df else None
    avg_pe = None
    avg_div = None

    ticker_obj = yf.Ticker(symbol)
    hist = ticker_obj.history(start=start, end=end)

    # ----- PE RATIO -----
    if getattr(args, 'pe', False):
        try:
            pe_ratio = ticker_obj.info.get('trailingPE', None)
            if pe_ratio and pe_ratio > 0:
                avg_pe = round(pe_ratio, 2)
                if args.csv:
                    pe_df = pd.DataFrame({'Date': [end], 'Trailing PE': [avg_pe]})
                    pe_df.to_csv(f'{symbol}_pe.csv', index=False)
                    print(f"Saved P/E data to: {symbol}_pe.csv")

                # Plot static value over time for visual consistency
                plt.figure(dpi=600)
                plt.plot(df['Date'], [avg_pe] * len(df), label='Trailing PE')
                plt.title(f'{symbol} P/E Ratio (Static Trailing PE)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{symbol}_pe_chart.png')
                plt.close()
        except Exception as e:
            print(f"Could not calculate PE for {symbol}: {e}")

    # ----- DIVIDENDS -----
    if getattr(args, 'div', False):
        try:
            dividends = ticker_obj.dividends

            if not dividends.empty:
                if dividends.index.tz is None:
                    # Localize dividend index to UTC if not already timezone-aware
                    dividends.index = dividends.index.tz_localize('UTC')

                start_ts = pd.to_datetime(start).tz_localize(dividends.index.tz)
                dividends = dividends[dividends.index >= start_ts]

            avg_div = round(dividends.mean(), 4) if not dividends.empty else 0.0
            div_df = pd.DataFrame({'Date': dividends.index, 'Dividend': dividends.values})

            if args.csv:
                div_df.to_csv(f'{symbol}_dividends.csv', index=False)
                print(f"Saved dividend data to: {symbol}_dividends.csv")

            plt.figure(dpi=600)
            plt.plot(div_df['Date'], div_df['Dividend'], label='Dividend')
            plt.title(f'{symbol} Dividend Yield Over Time')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{symbol}_div_chart.png')
            plt.close()

        except Exception as e:
            print(f"Could not retrieve dividend data for {symbol}: {e}")

    # --------- Return summary metrics if needed ---------
    return {
        'Ticker': symbol,
        'Percent Gain': round(percent_gain, 2) if percent_gain is not None else None,
        'Std Dev': std_dev,
        'Average Dividend Yield': avg_div,
        'Average PE Ratio': avg_pe
    }


#   ----------   #
# Argparse Setup #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logarithmic regression analysis for stock tickers.', add_help=False
                                     # disables auto -h/--help
                                     )

    # Argument Definitions #
    parser.add_argument('-h', '--help', action='store_true', help='Show help and usage instructions')
    parser.add_argument('tickers', nargs='*', help='Stock ticker symbols (e.g., AAPL MSFT TSLA)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')

    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
    parser.add_argument('--intrv', '--INTRV', dest='inter', type=str, default='1d', choices=valid_intervals,
                        help='Data interval (e.g., 1d, 1wk, 1mo)')
    parser.add_argument('--smooth', type=int, default=None, help='Rolling window for regression (e.g., 252)')
    parser.add_argument('--perc', '--PERC', dest='perc', action='store_true',
                        help='Display percent gain over the date range')

    parser.add_argument('--compare', action='store_true', help='Compare all tickers on key metrics in one PNG and CSV')
    parser.add_argument('--pe', action='store_true', help='Chart P/E ratio and output raw values')
    parser.add_argument('--div', action='store_true', help='Chart dividends and output raw values')
    parser.add_argument('--sd', action='store_true', help='Output standard deviations from mean')
    parser.add_argument('--csv', '--CSV', dest='csv', action='store_true', help='Save the enriched data to CSV')

# Parse Arguments #
import sys

pre_args, _ = parser.parse_known_args()

if pre_args.help:
    print("""
    ================
    CLI TOOLs: GUIDE
    ================

    File Architecture:
      cd ~/cl                 Changes directory to Command Line
      python3 script.py       Defines path to .py Script 

    Usage:
      python3 script.py       TICKER1 TICKER2 ... [OPTIONS]

    Positional Arguments:
      Tickers                 Ticker symbols (SPY GOLD USD-BTC)

    Optional Arguments:
      --help -h               Open this Guide
      --start YYYY-MM-DD      Set the start date for data (default: 2000-01-01)
      --end YYYY-MM-DD        Set the end date for data (default: today)
      --intrv                 Data interval (e.g., 1d, 1wk, 1mo, etc.)
      --smooth                Rolling regression window (e.g., 252 for 1-year daily)
      --compare(x)-self-ticker-group         Compare(x's) one or all tickers to either -self or a specific ticker or group of tickers on any defined optional arguments metrics such as sd, perc, div, pe and saves this analysis to a PNG chart and CSV
      --sd                    Output standard deviations from mean and summary
      --perc                 Display percent gain/loss over the selected date range
      --div                   Chart dividends and output raw values
      --pe                    Chart P/E ratio and output raw values
      --csv                   Save to a CSV file

    [PASTE]:


    Example:
      python3 script.py SPY GOLD USD-BTC --intrv 1d --smooth 252 --perc --div --pe --sd --compare --top10 --start 1900-01-01 --end 2100-01-01

    Note:
      Use multiple arguments simultaneously for combined outputs.

    """)
    sys.exit()

args = parser.parse_args()

if not args.tickers:
    print("❌ Error: You must specify at least one ticker symbol.\nUse --help to view usage instructions.")
    sys.exit(1)

# Prep for comparison
comparison_data = []

# Ticker Loop
for symbol in args.tickers:
    summary = None
    try:
        summary = run_log_regression(
            symbol=symbol.upper(),
            start=args.start,
            end=args.end,
            interval=args.inter,
            rolling=args.smooth,
            save_csv=args.csv,
            args=args
        )
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

    if args.compare and summary:
        comparison_data.append(summary)

# ---- Finalize and export comparison if requested ---- #
if args.compare and comparison_data:
    df_compare = pd.DataFrame(comparison_data)

    # Optional: reorder/select known columns
    columns = ['Ticker', 'Percent Gain', 'Std Dev', 'Average Dividend Yield', 'Average PE Ratio']
    df_compare = df_compare[[col for col in columns if col in df_compare.columns]]

    # Sort by best performers (optional cosmetic improvement)
    df_compare = df_compare.sort_values(by='Percent Gain', ascending=False)

    # Save summary CSV
    df_compare.to_csv('comparison_table.csv', index=False)
    print("Saved comparison summary to:", os.path.abspath('comparison_table.csv'))

    # Plot sorted % gain bar chart
    plt.figure(dpi=600)
    plt.bar(df_compare['Ticker'], df_compare['Percent Gain'], color='skyblue')

    # Add total ticker count to title
    plt.title(f'Percent Gain Comparison ({len(df_compare)} Tickers)')
    plt.ylabel('% Gain')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    plt.close()
    print("Saved comparison chart to:", os.path.abspath('comparison_chart.png'))


