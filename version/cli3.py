import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import TextArea, AnnotationBbox
# from datetime import datetime # why is this here?
# import matplotlib.font_manager as fm #
import os
import sys


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

    # --------- Plot Chart --------- #


def plot_chart(df, symbol, percent_gain=None, date_range=None, avg_div_yield=None, show_log=False):
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)

    if show_log:
        fig, (ax1, ax2) = plt.subplots(dpi=600, nrows=2, sharex=True)
    else:
        fig, ax1 = plt.subplots(dpi=600)
        ax2 = None

    ax1.grid(True, color='silver', linewidth=0.5)
    ax1.set_ylabel('Price')
    ax1.plot(df['Date'], df['Close'], color='black', linewidth=0.25)
    ax1.set_title('Linear Chart', loc='center', fontsize=9)

    title = f'{symbol} ({percent_gain:+.2f}%)' if percent_gain is not None else f'{symbol}'
    plt.suptitle(title, fontsize=11, weight='bold')

    if percent_gain is not None and date_range:
        fig.text(0.99, 0.94, f'*{date_range}*', fontsize=6, style='italic', ha='right')

    if avg_div_yield is not None:
        fig.text(0.99, 0.91, f'Avg Annual Yield: {avg_div_yield:.2f}%', fontsize=6, style='italic', ha='right')

    if show_log and ax2:
        ax2.grid(True, color='silver', linewidth=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Log Price')
        ax2.set_title('Log Chart', loc='center', fontsize=9)
        ax2.xaxis.set_major_formatter(DateFormatter("%m/%y"))

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

        last_date = df['Date'].iloc[-1]
        ax2.text(last_date, df['TLp2SD'].iloc[-1], '+2', va='center', ha='left', fontsize=6)
        ax2.text(last_date, df['TLpSD'].iloc[-1], '+1', va='center', ha='left', fontsize=6)
        ax2.text(last_date, df['priceTL'].iloc[-1], '0', va='center', ha='left', fontsize=6)
        ax2.text(last_date, df['TL_SD'].iloc[-1], '-1', va='center', ha='left', fontsize=6)
        ax2.text(last_date, df['TL_2SD'].iloc[-1], '-2', va='center', ha='left', fontsize=6)


    # --Regression SUMMARY BOX-- #
    try:
        # Calculate summary statistics
        today_log_price = df['price_y'].iloc[-1]
        expected_log_price = df['priceTL'].iloc[-1]
        percent_error = ((today_log_price - expected_log_price) / expected_log_price) * 100
        std_devs_from_line = (today_log_price - expected_log_price) / df['SD'].iloc[-1]

        # Compose the summary content
        summary_title = "REGRESSION SUMMARY"
        summary_lines = [
            f"Todays log: {today_log_price:.2f}",
            f"Estimate: {expected_log_price:.2f}",
            f"% Error: {percent_error:.2f}%",
            f"STDev: {std_devs_from_line:.2f}σ"
        ]
        full_text = [summary_title] + summary_lines

        ax = fig.axes[-1]  # log chart axis
        box_x, box_y = 1.06, 0.5   # move further right
        width, height = 0.11, 0.4  # make skinnier
        fontsize = 6
        spacing = height / (len(full_text) + 1)

        # Draw white rounded background box
        bbox = FancyBboxPatch(
            (box_x, box_y - height / 2),
            width, height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.05",
            linewidth=0.5,
            edgecolor='black',
            facecolor='black',
            alpha=0.85,
            zorder=10,
            clip_on=False
        )
        ax.add_patch(bbox)

        # Center text inside the box
        for i, line in enumerate(full_text):
            ax.text(
                box_x -.035,  # Smaller padding for tighter left alignment
                box_y + height / 2 - (i + 1) * spacing,
                line,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontfamily='monospace',
                color='white',
                ha='left',
                va='center',
                zorder=11
            )


    except Exception as e:
        print(f"Could not generate log regression summary box: {e}")

    plt.subplots_adjust(right=0.91)

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
    avg_div_yield = None  # <--- ✅ Define here to avoid UnboundLocalError

    if args and (args.perc or getattr(args, 'PERC', False)):
        try:
            start_price = float(df['Close'].iloc[0].item())
            end_price = float(df['Close'].iloc[-1].item())
            percent_gain = ((end_price - start_price) / start_price) * 100
            date_range = f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}"
            print(f"{symbol} percent gain from {start} to {end}: {percent_gain:.2f}%")
        except Exception as e:
            print(f"❌Error calculating percent gain for {symbol}: {e}")

    df = rolling_log_regression(df, rolling) if rolling and rolling < len(df) else logarithmic_regression(df)
    fig = plot_chart(df, symbol, percent_gain=percent_gain, date_range=date_range,
                     avg_div_yield=avg_div_yield if args.div else None,
                     show_log=getattr(args, 'log', False))

    # Save PNG + CSV for the regression chart
    filename = f'{symbol}_log_regression.png'
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print("✔️Saved chart to:", os.path.abspath(filename))

    if save_csv:
        csv_name = f'{symbol}_log_regression.csv'
        df.to_csv(csv_name, index=False)
        print("✔️Saved data to:", os.path.abspath(csv_name))

    # --------- New Logic Starts Here ---------
    std_dev = round(df['y_TL'].std(), 4) if 'y_TL' in df else None
    avg_pe = None
    avg_div = None

    ticker_obj = yf.Ticker(symbol)
    hist = ticker_obj.history(start=start, end=end)

    # ----- PE RATIO BLOCK ----- #
    avg_pe = None

    if getattr(args, 'pe', False):
        try:
            hist = ticker_obj.history(start=start, end=end, interval=interval)
            earnings_per_share = ticker_obj.info.get('trailingEps')

            if 'Close' in hist.columns and earnings_per_share and earnings_per_share > 0:
                hist['PE'] = hist['Close'] / earnings_per_share
                pe_series = hist['PE'].dropna()

                if not pe_series.empty:
                    avg_pe = round(pe_series.mean(), 2)
                    start_pe = pe_series.iloc[0]
                    end_pe = pe_series.iloc[-1]
                    pe_change = round(((end_pe - start_pe) / start_pe) * 100, 2)

                    if save_csv or getattr(args, 'csv', False):
                        hist_pe_df = pd.DataFrame({'Date': pe_series.index, 'PE': pe_series.values})
                        hist_pe_df.to_csv(f'{symbol}_pe.csv', index=False)
                        print(f"✔️Saved P/E data to: {symbol}_pe.csv")

                    plt.figure(dpi=600)
                    plt.plot(pe_series.index, pe_series, label='P/E Ratio', linewidth=0.5)
                    plt.axhline(avg_pe, linestyle='--', color='red', label=f'Avg PE: {avg_pe}')
                    plt.title(f'{symbol} P/E Ratio Over Time', fontsize=9)
                    plt.xlabel('Date')
                    plt.ylabel('P/E')
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(fontsize=6)

                    summary_text = (
                        f"Start PE: {start_pe:.2f}\n"
                        f"End PE: {end_pe:.2f}\n"
                        f"Change: {pe_change:+.2f}%\n"
                        f"Avg PE: {avg_pe:.2f}"
                    )
                    plt.gcf().text(0.99, 0.90, summary_text,
                                   ha='right', va='top', fontsize=6,
                                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(f'{symbol}_pe_chart.png')
                    plt.close()
            else:
                print(f"Not enough info to calculate PE for {symbol}.")
        except Exception as e:
            print(f"❌ Could not calculate PE for {symbol}: {e}")

    # ----- DIVIDENDS BLOCK ----- #
    avg_div_yield = None

    if getattr(args, 'div', False):
        try:
            dividends = ticker_obj.dividends
            hist_prices = ticker_obj.history(start=start, end=end)['Close']

            if not dividends.empty and not hist_prices.empty:
                if dividends.index.tz is None:
                    dividends.index = dividends.index.tz_localize('UTC')
                start_ts = pd.to_datetime(start).tz_localize(dividends.index.tz)
                dividends = dividends[dividends.index >= start_ts]

                price_on_div_dates = hist_prices[hist_prices.index.isin(dividends.index)]
                matched_prices = price_on_div_dates.reindex(dividends.index, method='nearest')

                dividend_yield_series = (dividends / matched_prices) * 100
                avg_div_yield = round(dividend_yield_series.mean(), 2)

                start_yield = dividend_yield_series.iloc[0]
                end_yield = dividend_yield_series.iloc[-1]
                div_yield_change = round(((end_yield - start_yield) / start_yield) * 100, 2)

                div_df = pd.DataFrame({
                    'Date': dividends.index,
                    'Dividend': dividends.values,
                    'Yield (%)': dividend_yield_series.values
                })

                if save_csv or getattr(args, 'csv', False):
                    div_df.to_csv(f'{symbol}_dividends.csv', index=False)
                    print(f"✔️Saved dividend data to: {symbol}_dividends.csv")

                # Chart plotting
                plt.figure(dpi=600)
                plt.plot(div_df['Date'], div_df['Yield (%)'], label='Dividend Yield (%)', linewidth=0.5)
                plt.axhline(avg_div_yield, color='red', linestyle='--', linewidth=0.5,
                            label=f'Avg: {avg_div_yield:.2f}%')
                plt.title(f'{symbol} Dividend Yield Over Time', fontsize=9)
                plt.xlabel('Date')
                plt.ylabel('Yield (%)')
                plt.legend(fontsize=6)
                plt.grid(True, linestyle='--', alpha=0.5)

                # Add summary box
                summary_text = (
                    f"Start Yield: {start_yield:.2f}%\n"
                    f"End Yield: {end_yield:.2f}%\n"
                    f"Change: {div_yield_change:+.2f}%"
                )
                plt.gcf().text(0.99, 0.90, summary_text,
                               ha='right', va='top', fontsize=6,
                               bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

                plt.tight_layout()
                plt.savefig(f'{symbol}_div_chart.png')
                plt.close()
            else:
                avg_div_yield = 0.0
        except Exception as e:
            print(f"❌ Could not retrieve dividend data for {symbol}: {e}")

    # --------- NEW: Generate Histogram Plots ---------
    if getattr(args, 'normdist', False):
        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            create_histograms(symbol, df, start_dt, end_dt, args)
        except Exception as e:
            print(f"❌ Could not run histogram generation for {symbol}: {e}")

    # --------- Return summary metrics if needed ---------
    return {
        'Ticker': symbol,
        'Percent Gain': round(percent_gain, 2) if percent_gain is not None else None,
        'Std Dev': std_dev,
        'Average Dividend Yield': avg_div_yield,
        'Average PE Ratio': avg_pe
    }

    # --------- Normal dist Histogram Function --------- #


def create_histograms(symbol, df, start_dt, end_dt, args):
    try:
        print(f"📊 Creating histogram(s) for {symbol}...")

        metrics = []
        titles = []
        y_labels = []
        today_values = []

        # Daily Percent Change
        if getattr(args, 'perc', False):
            df['Pct Change'] = df['Close'].pct_change() * 100
            pct_series = df['Pct Change'].dropna()
            if not pct_series.empty:
                metrics.append(pct_series)
                titles.append(f'{symbol} Daily % Change')
                y_labels.append('% Change')
                today_values.append(pct_series.iloc[-1])

        # Dividends
        if getattr(args, 'div', False):
            dividends = yf.Ticker(symbol).dividends
            if not dividends.empty:
                if dividends.index.tz is None:
                    dividends.index = dividends.index.tz_localize('UTC')
                start_dt = start_dt.tz_localize('UTC') if start_dt.tzinfo is None else start_dt
                end_dt = end_dt.tz_localize('UTC') if end_dt.tzinfo is None else end_dt
                dividends = dividends[(dividends.index >= start_dt) & (dividends.index <= end_dt)]
                if not dividends.empty:
                    metrics.append(dividends)
                    titles.append(f'{symbol} Dividend Amounts')
                    y_labels.append('Dividend $')
                    today_values.append(dividends.iloc[-1])

        # P/E Ratio
        if getattr(args, 'pe', False):
            ticker_obj = yf.Ticker(symbol)
            earnings = ticker_obj.info.get('trailingEps')
            if earnings and earnings > 0:
                hist = ticker_obj.history(start=start_dt, end=end_dt, interval=args.inter)
                hist['PE'] = hist['Close'] / earnings
                pe_series = hist['PE'].dropna()
                if not pe_series.empty:
                    metrics.append(pe_series)
                    titles.append(f'{symbol} P/E Ratio')
                    y_labels.append('P/E')
                    today_values.append(pe_series.iloc[-1])

        # Plot each histogram (✅ now inside the try block)
        for i in range(len(metrics)):
            series = metrics[i]
            title = titles[i]
            ylabel = y_labels[i]
            today_val = today_values[i]

            plt.figure(dpi=600)
            plt.hist(series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(today_val, color='red', linestyle='--', linewidth=1)
            plt.text(today_val, plt.ylim()[1] * 0.9, f'Today: {today_val:.2f}', color='red', ha='left', fontsize=6)

            plt.title(f'{title} Distribution')
            plt.xlabel(ylabel)
            plt.ylabel('Frequency')
            plt.tight_layout()

            # Safe filename generation
            import re
            safe_title = re.sub(r'[^A-Za-z0-9_]+', '_', title.lower())
            filename = f"{symbol}_{safe_title}_normdist.png"

            plt.savefig(filename)
            plt.close()
            print(f"✔️Saved histogram to: {os.path.abspath(filename)}")

    except Exception as e:
        print(f"❌ Error creating histograms for {symbol}: {e}")


# --------- CLI Parser --------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logarithmic regression analysis for stock tickers.', add_help=False)

    # Argparse Setup #
    parser.add_argument('-h', '--help', action='store_true', help='Show help and usage instructions')

    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
    parser.add_argument('--intrv', '--INTRV', dest='inter', type=str, default='1d', choices=valid_intervals,
                        help='Data interval (e.g., 1d, 1wk, 1mo)')
    parser.add_argument('--smooth', type=int, default=None, help='Rolling window for regression (e.g., 252)')

    parser.add_argument('--log', action='store_true', help='Include log regression chart view')

    parser.add_argument('--perc', '--PERC', dest='perc', action='store_true',
                        help='Display percent gain over the date range')
    parser.add_argument('--div', action='store_true', help='Chart dividends and output raw values')
    parser.add_argument('--pe', action='store_true', help='Chart P/E ratio and output raw values')
    parser.add_argument('--normdist', action='store_true', help='Create histogram of metric distributions')
    parser.add_argument('--compare', action='store_true', help='Compare all tickers on key metrics in one PNG and CSV')
    parser.add_argument('--csv', '--CSV', dest='csv', action='store_true', help='Save the enriched data to CSV')

    parser.add_argument('--comparex', nargs='+', help='Tickers to compare against --vs group')
    parser.add_argument('--vs', nargs='+', help='Group of tickers to compare to base tickers using selected metrics')
    parser.add_argument('tickers', nargs='*', help='Stock ticker symbols (e.g., AAPL MSFT TSLA)')

    # Pre-Parse Arguments for --help
    pre_args, _ = parser.parse_known_args()
    if pre_args.help:
        print("""
        =========
        CLI TOOLs
        =========

  --> Change directory:
cd ~/path/to/your/folder

  --> Run python script:
python3 cli3.py SPY GOLD USD-BTC 
--log --normdist --intrv 1d 
--smooth 252 --pe --div --compare 
--start 1900-01-01 --end 2100-01-01 --csv 
    
        ========
        COMMANDS
        ========

Positional Arguments:
Ticker (default)        Ticker must come first (SPY GOLD USD-BTC)
--comparex              Can be used as a beginning argument or an optional argument               

Optional Arguments:
"blank"                 Returns the default linear chart for X ticker
--log                   Returns the logarithmic chart in addition to the default linear chart
--normdist              Plots a .png distribution of daily % inc/dec for (perc, div, pe)
--intrv                 Data interval (e.g., 1d, 1wk, 1mo, etc.)
--smooth                Rolling regression window (e.g., 252 for 1-year daily)         
--perc                  Display percent gain/loss over the selected date range (defaults to included)
--pe                    Chart P/E ratio and output raw values
--div                   Chart dividends and output raw values
--compare               v1. Produces a comparison .png output for the tickers and metrics inputted over defined date range
--comparex              v2. Compares x ticker (--vs ticker) or group thereof
--vs                    Conditional argument for --comparex above ex. (--comparex aapl msft nvda --vs spy)
--start YYYY-MM-DD      Set the start date for data (default: 2000-01-01)
--end YYYY-MM-DD        Set the end date for data (default: today)
--csv                   Save to a .CSV file
--help -h               Opens this Guide        

        =============
        [COPY/PASTE]:
        =============

COMMANDS:
---------------]
python3 cli3.py
--log
--normdist
--intrv 1d
--smooth 1008
--perc
--pe
--div
--compare
--comparex
--vs
--start 1900-01-01
--end 2100-01-01
--csv
------------------]
SMOOTHING: 
(over N trading days for 1d intervals)
-------------------------]
--smooth 252     ≈ 1 year
--smooth 504     ≈ 2 years
--smooth 756     ≈ 3 years
--smooth 1008    ≈ 4 years
--smooth 1260    ≈ 5 years
--smooth 2520    ≈ 10 years
--smooth 5040    ≈ 20 years
--smooth 7560    ≈ 30 years
--smooth 10080   ≈ 40 years
--smooth 12600   ≈ 50 years
--smooth 15120   ≈ 60 years
--smooth 17640   ≈ 70 years
--smooth 20160   ≈ 80 years
--smooth 22680   ≈ 90 years
--smooth 25200   ≈ 100 years
----------------------------]
CURRENT SCRIPTS:
python3 cli3.py --log --normdist --intrv 1d --smooth 1008 --perc --pe --div --compare --vs --start 1900-01-01 --end 2100-01-01 --csv

        """)
        sys.exit()

    args = parser.parse_args()

    # Ensure --perc gain/loss is calculated by default unless explicitly turned off
    if not getattr(args, 'perc', False):
        args.perc = True

    # Handle --comparex logic
    if args.comparex and args.vs:
        args.tickers = args.comparex + args.vs
        args.compare = True
        base_tickers = args.comparex
        vs_tickers = args.vs
    elif args.comparex and not args.vs:
        print("❌ Error: --comparex must be used with --vs and at least one metric (e.g., --perc, --div, --pe)")
        sys.exit(1)
    else:
        args.tickers = getattr(args, 'tickers', []) or []
        base_tickers = []
        vs_tickers = []

    if not args.tickers:
        print("❌Error: You must specify at least one ticker symbol.\nUse --help to view usage instructions.")
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
            print(f"❌Error processing {symbol}: {e}")

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

        # Save summary CSV only if --csv is included
        if args.csv:
            df_compare.to_csv('comparison_table.csv', index=False)
            print("✔️Saved comparison summary to:", os.path.abspath('comparison_table.csv'))

        # Generate individual metric charts with enhanced visualization
        for metric in columns[1:]:  # Skip 'Ticker'
            if metric in df_compare.columns:
                plt.figure(figsize=(10, 6), dpi=600)
                bars = plt.bar(df_compare['Ticker'], df_compare[metric], color='skyblue', edgecolor='black')

                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.2f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 5),
                                 textcoords='offset points',
                                 ha='center', va='bottom', fontsize=6)

                if args.comparex:
                    plt.title(f'{metric} Comparison: {" vs. ".join(args.comparex)} vs. Others', fontsize=10)
                else:
                    plt.title(f'{metric} Comparison ({len(df_compare)} Tickers)', fontsize=10)

                plt.xlabel('Ticker')
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right', fontsize=6)
                plt.yticks(fontsize=6)
                plt.grid(True, axis='y', linestyle='--', alpha=0.5)

                # Add summary stats as a text box
                stats_text = (
                    f"Mean: {df_compare[metric].mean():.2f}\n"
                    f"Max: {df_compare[metric].max():.2f}\n"
                    f"Min: {df_compare[metric].min():.2f}"
                )
                plt.gcf().text(0.98, 0.8, stats_text, fontsize=7, va='top', ha='right',
                               bbox=dict(boxstyle="round", facecolor='white', edgecolor='gray'))

                plt.tight_layout()
                file_safe_metric = metric.replace(" ", "_").lower()
                filename = f'comparison_{file_safe_metric}.png'
                plt.savefig(filename)
                plt.close()
                print(f"✔️Saved comparison chart to:", os.path.abspath(filename))


