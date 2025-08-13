# --- Import Libraries --- #
# ------------------------ #
import argparse
import pandas as pd
import yfinance as yf

# ==== yfinance rate-limit and caching wrappers (inserted by assistant) ====
# These wrappers aim to reduce burstiness and add gentle retries so terminal runs
# behave like PyCharm runs without changing your call sites or logic.

import time as _yl_time
import random as _yl_rand
import threading as _yl_threading
import requests as _yl_requests

try:
    import yfinance as _yl_yf
except Exception:
    _yl_yf = None

# Shared HTTP session
_YL_SESSION = _yl_requests.Session()

# Allow disabling wrappers via env var if needed
import os as _yl_os
if _yl_os.getenv("YF_WRAPPERS_DISABLED", "").lower() not in {"1", "true", "yes"} and _yl_yf is not None:
    _yl_download_orig = _yl_yf.download

    # Cache dicts to avoid duplicate refetches within a run
    _yl_download_cache = {}
    _yl_history_cache = {}

    # Simple token bucket to keep ~1–1.5 RPS; adjust via env if desired
    _yl_rate = float(_yl_os.getenv("YF_RPS", "0.5"))
    _yl_burst = int(float(_yl_os.getenv("YF_BURST", "1")))
    _yl_tokens = _yl_burst
    _yl_last = _yl_time.monotonic()
    _yl_lock = _yl_threading.Lock()
    _yl_cooldown_until = 0.0
    _yl_min_cooldown = float(_yl_os.getenv("YF_COOLDOWN", "60"))

    def _yl_acquire():
        # Basic token bucket with global cooldown on 429
        global _yl_tokens, _yl_last
        while True:
            with _yl_lock:
                now = _yl_time.monotonic()
                # respect cooldown
                if now < _yl_cooldown_until:
                    sleep_for = _yl_cooldown_until - now
                    _yl_time.sleep(sleep_for)
                    continue
                # refill
                _yl_tokens = min(_yl_burst, _yl_tokens + (now - _yl_last) * _yl_rate)
                _yl_last = now
                if _yl_tokens >= 1.0:
                    _yl_tokens -= 1.0
                    return
            _yl_time.sleep(0.02)

    def _is_rate_limit_exception(e):
        s = str(e).lower()
        return ("429" in s or "rate" in s or "too many" in s or "temporarily unavailable" in s)

    def _normalize_kwargs(kwargs):
        # Convert dict to a tuple of sorted items; pandas objects excluded
        key_items = []
        for k, v in sorted(kwargs.items()):
            # Avoid caching on objects that are clearly not hashable or huge
            if hasattr(v, "to_pydatetime") or hasattr(v, "to_timestamp"):
                v = str(v)
            key_items.append((k, str(v)))
        return tuple(key_items)

    def _cache_key(prefix, *args, **kwargs):
        return (prefix, tuple(args), _normalize_kwargs(kwargs))

    def _retry_call(fn, *args, **kwargs):
        delay = _yl_min_cooldown
        max_retries = int(_yl_os.getenv("YF_MAX_RETRIES", "5"))
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_exception(e):
                    _yl_time.sleep(delay + _yl_rand.random())
                    delay = min(delay * 2.0, 32.0)
                    continue
                raise
        # final try
        return fn(*args, **kwargs)

    def _wrapped_download(*args, **kwargs):
        # Enforce gentler settings
        kwargs.setdefault("threads", False)
        kwargs.setdefault("progress", False)
        # Build cache key (ignore session object)
        key = _cache_key("download", *args, **{k:v for k,v in kwargs.items() if k != "session"})
        if key in _yl_download_cache:
            return _yl_download_cache[key]
        _yl_acquire()
        kwargs.setdefault("session", _YL_SESSION)
        result = _retry_call(_yl_download_orig, *args, **kwargs)
        _yl_download_cache[key] = result
        return result

    # Monkeypatch download
    _yl_yf.download = _wrapped_download

    # Monkeypatch Ticker.history to add retries + caching + pacing
    try:
        import yfinance.ticker as _yl_ticker_mod
        _yl_history_orig = _yl_ticker_mod.Ticker.history

        def _wrapped_history(self, *args, **kwargs):
            # Cache by ticker symbol + args/kwargs that define the window
            sym = getattr(self, "ticker", None)
            # Exclude 'auto_adjust' and similar toggles from over-broad caching
            cache_kwargs = {k: v for k, v in kwargs.items() if k not in ("auto_adjust", "actions")}
            key = _cache_key(("history", sym), *args, **cache_kwargs)
            if key in _yl_history_cache:
                return _yl_history_cache[key]
            _yl_acquire()
            result = _retry_call(_yl_history_orig, self, *args, **kwargs)
            _yl_history_cache[key] = result
            return result

        _yl_ticker_mod.Ticker.history = _wrapped_history
    except Exception:
        # If monkeypatch fails, just continue; core download wrapper still helps.
        pass
# ==== end wrappers ====

# ==== centralized fetch & reuse helpers ====
# Resolve original yfinance download function (pre-wrapped if available)
try:
    _YL_ORIG_DOWNLOAD = _yl_download_orig  # from the wrapper above, if present
except NameError:
    _YL_ORIG_DOWNLOAD = yf.download

from datetime import datetime
import pandas as _pf_pd

_PRICE_CACHE = {}        # key: (symbol, interval) -> {"range": (start,end), "df": DataFrame}
_DIVIDEND_CACHE = {}     # key: symbol -> Series

def _to_ts(x):
    if x is None:
        return None
    try:
        return _pf_pd.Timestamp(x)
    except Exception:
        return _pf_pd.Timestamp(str(x))

def _range_union(a, b):
    s1, e1 = a
    s2, e2 = b
    s = min([t for t in (s1, s2) if t is not None]) if (s1 is not None or s2 is not None) else None
    e = max([t for t in (e1, e2) if t is not None]) if (e1 is not None or e2 is not None) else None
    return (s, e)

def _slice_df(df, start=None, end=None):
    if start is None and end is None:
        return df
    st = _to_ts(start) if start is not None else None
    en = _to_ts(end) if end is not None else None
    out = df
    if st is not None:
        out = out[out.index >= st]
    if en is not None:
        out = out[out.index <= en]
    return out

def get_price_history(symbol, start=None, end=None, interval="1d", **kwargs):
    """Fetch price history once per symbol/interval; reuse and slice for subsequent calls."""
    sym = str(symbol).replace(".", "-")
    key = (sym, interval)
    req = (_to_ts(start), _to_ts(end))
    entry = _PRICE_CACHE.get(key)

    need_fetch = True
    if entry is not None:
        have_start, have_end = entry["range"]
        if (req[0] is None or (have_start is not None and have_start <= req[0])) and \
           (req[1] is None or (have_end is not None and have_end >= req[1])):
            need_fetch = False

    if need_fetch:
        fetch_range = req if entry is None else _range_union(entry["range"], req)
        df_new = _YL_ORIG_DOWNLOAD(sym, start=fetch_range[0], end=fetch_range[1],
                                   interval=interval, progress=False, threads=False, actions=True, session=_YL_SESSION, **kwargs)
        if not isinstance(df_new, _pf_pd.DataFrame):
            return df_new
        if entry is None:
            entry = {"range": fetch_range, "df": df_new}
        else:
            df_old = entry["df"]
            df_merged = _pf_pd.concat([df_old, df_new]).sort_index()
            df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
            entry = {"range": fetch_range, "df": df_merged}
        _PRICE_CACHE[key] = entry

    return _slice_df(entry["df"], start, end)

def get_price_history_from_ticker(ticker_obj, start=None, end=None, interval="1d", **kwargs):
    sym = getattr(ticker_obj, "ticker", None) or getattr(getattr(ticker_obj, "info", {}), "get", lambda *_: None)("symbol", None)
    return get_price_history(sym, start=start, end=end, interval=interval, **kwargs)

def get_dividends(symbol):
    sym = str(symbol).replace(".", "-")
    if sym in _DIVIDEND_CACHE:
        return _DIVIDEND_CACHE[sym]
    # Pull actions via price history to avoid .dividends property
    try:
        df_actions = get_price_history(sym, interval="1d", period="max", actions=True)
        div = None
        if isinstance(df_actions, _pf_pd.DataFrame) and not df_actions.empty and "Dividends" in df_actions.columns:
            ser = df_actions["Dividends"]
            div = ser[ser != 0.0]
    except Exception:
        div = None
    _DIVIDEND_CACHE[sym] = div
    return div

def get_dividends_from_ticker(ticker_obj):
    sym = getattr(ticker_obj, "ticker", None) or getattr(getattr(ticker_obj, "info", {}), "get", lambda *_: None)("symbol", None)
    return get_dividends(sym)
# ==== end helpers ====
# ==== end helpers ====


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.patches import FancyBboxPatch
import os
import sys

# -------- --log Function -------------- #
# -------------------------------------- #
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


# --------- --Smooth Function ---------- #
# -------------------------------------- #
def smooth_log_regression(df, window):
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


# ------------- Plot Chart ------------- #
# -------------------------------------- #
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


# ----------- Regression Summary Stats ----------- #
# ------------------------------------------------ #
    try:
        # Calculate summary statistics
        today_log_price = df['price_y'].iloc[-1]
        expected_log_price = df['priceTL'].iloc[-1]

        # Todays % Error
        percent_error = ((today_log_price - expected_log_price) / expected_log_price) * 100

        # Mean absolute percent error over all available data
        percent_errors = ((df['price_y'] - df['priceTL']) / df['priceTL']) * 100
        mape = percent_errors.abs().mean()

        # Distance from regression line in std dev.
        std_devs_from_line = (today_log_price - expected_log_price) / df['SD'].iloc[-1]

    # Transform today_log_price to Linear (for summary content)
        today_linear_price = np.exp(today_log_price)

        # Compose summary content
        summary_title = "REGRESSION SUMMARY:"
        summary_lines = [
            f"Today linear: {today_linear_price:.2f}",
            f"Today log: {today_log_price:.4f}",
            f"Estimate:  {expected_log_price:.4f}",
            f"Error:    {percent_error:.2f}%",
            f"MAPE:      {mape:.2f}%",
            f"STDev:    {std_devs_from_line:.2f}σ"
        ]
        full_summary_text = [summary_title] + summary_lines

        ax = fig.axes[-1]  # log chart axis
        box_x, box_y = 1.07, 0.765
        width, height = 0.11, 0.4
        fontsize = 6
        spacing = height / (len(full_summary_text) + 1)

    # --- Draw background BOX --- #
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

    # --------- Conditional Formatting for Summary Stats --------- #
    # ------------------------------------------------------------ #
        for i, line in enumerate(full_summary_text):
            color = 'white'

            if i == 4:  # Today % Error
                if -2.5 <= percent_error <= 2.5:
                    color = 'green'
                elif -5 <= percent_error <= 5:
                    color = 'lightgreen'
                elif -10 <= percent_error <= 10:
                    color = 'pink'
                elif percent_error < -10 or percent_error > 10:
                    color = 'red'

            elif i == 5:  # MAPE (Mean Absolute % Error)
                if -2.5 <= mape <= 2.5:
                    color = 'green'
                elif -5 <= mape <= 5:
                    color = 'lightgreen'
                elif -10 <= mape <= 10:
                    color = 'pink'
                elif mape < -10 or mape > 10:
                    color = 'red'

            elif i == 6:  # STDev
                if 0 >= std_devs_from_line >= -1:
                    color = 'lightgreen'
                elif std_devs_from_line <= -1:
                    color = 'green'
                elif 0 <= std_devs_from_line <= 1:
                    color = 'pink'
                elif std_devs_from_line >= 1:
                    color = 'red'

            ax.text(
                box_x - 0.035,
                box_y + height / 1.75 - (i + 1) * spacing,
                line,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontfamily='monospace',
                ha='left',
                va='center',
                color=color,
                zorder=11
            )
    # ------------- END ------------- #


# ------------- Conditional Format LEGEND ------------- #
# ----------------------------------------------------- #
        legend_title = "LEGEND:"
        legend_lines = [
            ("Error ±2.5%", "green"),
            ("Error ±5%", "lightgreen"),
            ("Error ±10%", "pink"),
            ("Error > ±10%", "red"),
            ("------------", "white"),
            ("STDev < -1σ", "green"),
            ("STDev -1σ to 0", "lightgreen"),
            ("STDev 0 to 1σ", "pink"),
            ("STDev > 1σ", "red"),
        ]

        ax = fig.axes[-1]  # log chart axis
        box_x, box_y = 1.07, 0.235
        width, height = 0.11, 0.4
        fontsize = 5.2
        spacing = height / (len(legend_lines) + .40)  # +2 for top and bottom padding

        # --- Draw background BOX --- #
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

        # --- Draw the TITLE --- #
        ax.text(box_x - 0.035,
                box_y + 0.21,
                legend_title,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontfamily='monospace',
                color='white',
                ha='left',
                va='top',
                zorder=11)

        # --- Draw th TEXT legend line with its corresponding color --- #
        for i, (label, color) in enumerate(legend_lines):
            ax.text(box_x - 0.035,
                    box_y + height / 1.99 - (i + 1) * spacing,
                    label,
                    transform=ax.transAxes,
                    fontsize=fontsize,
                    fontfamily='monospace',
                    color=color,
                    ha='left',
                    va='top',
                    zorder=11)

    except Exception as e:
        print(f"Could not generate regression summary legend box: {e}")

    plt.subplots_adjust(right=0.91)

    return fig


# ------------ Run Log Regression ------------ #
# -------------------------------------------- #
def run_log_regression(symbol, start, end, interval, rolling=None, save_csv=False, args=None):
    print(f"Fetching data for {symbol}...")
    df = get_price_history(symbol, start=start, end=end, interval=interval).reset_index()

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

    df = smooth_log_regression(df, rolling) if rolling and rolling < len(df) else logarithmic_regression(df)
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
# ---------------------------------------------------------------- #


    # --------- New Logic Starts Here ---------
    std_dev = round(df['y_TL'].std(), 4) if 'y_TL' in df else None
    avg_pe = None
    avg_div = None

    ticker_obj = yf.Ticker(symbol)
    hist = get_price_history_from_ticker(ticker_obj, start=start, end=end)

    # ----- PE RATIO BLOCK ----- #
    avg_pe = None

    if getattr(args, 'pe', False):
        try:
            hist = get_price_history_from_ticker(ticker_obj, start=start, end=end, interval=interval)
            earnings_per_share = ticker_obj.info.get('trailingEps')

            if 'Close' in hist.columns and earnings_per_share and earnings_per_share > 0:
                hist['PE'] = hist['Close'] / earnings_per_share
                pe_series = hist['PE'].dropna()

                if not pe_series.empty:
                    avg_pe = round(pe_series.mean(), 2)
                    start_pe = pe_series.iloc[0]
                    end_pe = pe_series.iloc[-1]
                    pe_change = round(((end_pe - start_pe) / start_pe) * 100, 2)

# pe edit to add price and eps below
# FIX why eps is same for everyday?
                    if save_csv or getattr(args, 'csv', False):
                        hist_pe_df = pd.DataFrame({
                            'Date': pe_series.index,
                            'Price': hist.loc[pe_series.index, 'Close'].values,
                            'EPS': [earnings_per_share] * len(pe_series),
                            'PE': pe_series.values
                        })
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
            dividends = get_dividends_from_ticker(ticker_obj)
            hist_prices = get_price_history_from_ticker(ticker_obj, start=start, end=end)['Close']

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

        # DIVIDEND CSV output:

                div_df = pd.DataFrame({
                    'Date': dividends.index,
                    'Share Price': matched_prices.values,
                    'Dividend': dividends.values,
                    'Yield (%)': dividend_yield_series.values,
                })

                # Step 1: Add temporary year column
                div_df['Year'] = div_df['Date'].dt.year

                # Step 2: Create empty Annual Yield column
                div_df['Annual Yield'] = np.nan
                projected_yields = {}

                # Step 3: Project or compute annual yield for each year, assign to last row
                for year, group in div_df.groupby('Year'):
                    idx_last = group.index[-1]
                    n_divs = group['Yield (%)'].count()
                    y_sum = group['Yield (%)'].sum()
                    annual_yield = (y_sum / n_divs) * 4 if n_divs < 4 else y_sum
                    projected_yields[year] = annual_yield
                    div_df.loc[idx_last, 'Annual Yield'] = annual_yield

                # Step 4: Average annual yield (from valid rows only)
                avg_annual_yield = div_df['Annual Yield'].mean()
                div_df['Average Annual Yield'] = np.nan
                div_df.loc[div_df.index[0], 'Average Annual Yield'] = avg_annual_yield

                # Step 5: Difference from Average (Projects Annual yield for incomplete most recent year)
                div_df['Diff from Average'] = np.nan
                for year, group in div_df.groupby('Year'):
                    idx_last = group.index[-1]
                    diff = projected_yields[year] - avg_annual_yield
                    div_df.loc[idx_last, 'Diff from Average'] = diff

                # Step 6: Add % Difference from Average
                div_df['% Diff from Average'] = np.nan

                for year, group in div_df.groupby(div_df['Date'].dt.year):
                    idx_last = group.index[-1]
                    projected_yield = div_df.loc[idx_last, 'Annual Yield']

                    # Avoid division by zero (edge case)
                    if avg_annual_yield != 0 and not pd.isna(projected_yield):
                        percent_diff = (projected_yield - avg_annual_yield) / avg_annual_yield
                        div_df.loc[idx_last, '% Diff from Average'] = percent_diff

                # Step 6.1: Format % Diff from Average as percentage in Excel
                div_df['% Diff from Average'] = div_df['% Diff from Average'].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ''
                )

                # Final cleanup
                div_df.drop(columns='Year', inplace=True)

        # Merge dividend csv analysis with log_regression #

                # Step 7: Add % above/below log reg trend to dividend csv output
                # Ensure TL_Price exists
                df['TL_Price'] = np.exp(df['priceTL'])

                # Step 7.1: Ensure 'Date' is a **flat column**, not in index or multi-level column
                if 'Date' not in div_df.columns:
                    div_df = div_df.reset_index()
                if 'Date' not in df.columns:
                    df = df.reset_index()

                # Step 7.2: Remove MultiIndex on columns (some operations create it)
                div_df.columns = div_df.columns.get_level_values(0)
                df.columns = df.columns.get_level_values(0)

                # Step 7.3: Normalize 'Date' to same type (datetime.date)
                div_df['Date'] = pd.to_datetime(div_df['Date']).dt.date
                df['Date'] = pd.to_datetime(df['Date']).dt.date

                # Step 7.4: Merge TL_Price from df (log regression) into div_df (dividend data)
                div_df = pd.merge(div_df, df[['Date', 'TL_Price']], on='Date', how='left')

                # Step 7.5: Calculate and format % Above/Below Trendline
                div_df['% Above/Below Trendline'] = (
                        (div_df['Share Price'] - div_df['TL_Price']) / div_df['TL_Price']
                ).map(lambda x: f"{x:.2%}" if pd.notna(x) else '')

            # Debugging for Step 7:
            #    print("div_df.dtypes:\n", div_df.dtypes)
            #    print("df.dtypes:\n", df.dtypes)
            #    print("div_df.columns:\n", div_df.columns)
            #    print("df.columns:\n", df.columns)

        # ------------------------------------------------------------------------------- #

                if save_csv or getattr(args, 'csv', False):
                    div_df.to_csv(f'{symbol}_dividends.csv', index=False)
                    print(f"✔️Saved dividend data to: {symbol}_dividends.csv")

# -------------------------------------------------------------------------------------------------- #

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


# --------- --Normdist Histogram Function ------------ #
# ---------------------------------------------------- #
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
            dividends = get_dividends(symbol)
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
                hist = get_price_history_from_ticker(ticker_obj, start=start_dt, end=end_dt, interval=args.inter)
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


# ------COMPAREX NEW LOGIC------- #
# ------------------------------- #

# ------ --comparex with --pe ------- #
# ------------------------------------ #
def comparex_pe_summary(base_ticker, vs_tickers, args):
    tickers = [base_ticker] + vs_tickers
    pe_data = {}
    eps_data = {}

    for t in tickers:
        try:
            obj = yf.Ticker(t)
            eps = obj.info.get('trailingEps', None)
            price = get_price_history_from_ticker(obj, period='1d')['Close'].iloc[-1]
            if eps and eps > 0:
                pe = price / eps
                pe_data[t] = pe
                eps_data[t] = eps
        except Exception:
            continue

    if base_ticker not in pe_data or len(pe_data) <= 1:
        print(f"❌ Not enough valid P/E data to compare {base_ticker} vs group.")
        return

    base_pe = pe_data[base_ticker]
    group_pes = [pe for t, pe in pe_data.items() if t != base_ticker]
    group_mean = np.mean(group_pes)
    group_std = np.std(group_pes)
    z_score = (base_pe - group_mean) / group_std if group_std > 0 else 0
    percentile = sum(1 for x in group_pes if x < base_pe) / len(group_pes) * 100
    percent_diff = ((base_pe - group_mean) / group_mean) * 100

    # Bar Chart
    plt.figure(figsize=(10, 6), dpi=600)
    plt.bar(pe_data.keys(), pe_data.values(), color='skyblue', edgecolor='black')
    plt.axhline(group_mean, linestyle='--', color='gray', label=f'Group Mean: {group_mean:.2f}')
    plt.axhline(base_pe, linestyle='--', color='red', label=f'{base_ticker} P/E: {base_pe:.2f}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('P/E Ratio')
    plt.title(f"{base_ticker} vs Group P/E Ratio", fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=6)

    # Summary Box
    summary_title = "P/E COMPARISON SUMMARY"
    lines = [
        f"{base_ticker} P/E: {base_pe:.2f}",
        f"Group Mean: {group_mean:.2f}",
        f"Group StdDev: {group_std:.2f}",
        f"Z-Score: {z_score:.2f}",
        f"Percentile: {percentile:.1f}%",
        f"% Diff from Mean: {percent_diff:+.2f}%"
    ]
    full_text = [summary_title] + lines

    box_x, box_y = 1.03, 0.5
    width, height = 0.14, 0.35
    spacing = height / (len(full_text) + 1)
    fontsize = 6

    ax = plt.gca()
    bbox = FancyBboxPatch(
        (box_x, box_y - height / 2), width, height,
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

    for i, line in enumerate(full_text):
        ax.text(
            box_x - 0.035,
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

    plt.subplots_adjust(right=0.89)
    filename = f'comparex_{base_ticker.lower()}_pe_chart.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✔️Saved comparex P/E chart to: {os.path.abspath(filename)}")

    # Optional CSV
    if getattr(args, 'csv', False):
        pd.DataFrame({
            'Ticker': list(pe_data.keys()),
            'PE Ratio': list(pe_data.values())
        }).to_csv(f'comparex_{base_ticker.lower()}_pe.csv', index=False)
        print(f"✔️Saved comparex P/E CSV to: {os.path.abspath(f'comparex_{base_ticker.lower()}_pe.csv')}")


# ------ --comparex with --div ------- #
# ------------------------------------ #

def comparex_div_summary(base_ticker, vs_tickers, args):
    tickers = [base_ticker] + vs_tickers
    div_yields = {}

    print(f"🟡 Running comparex_div_summary for {base_ticker} vs {vs_tickers}")

    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            dividends = get_dividends_from_ticker(ticker_obj)
            hist_prices = get_price_history_from_ticker(ticker_obj, start=args.start, end=args.end)['Close']

            if not dividends.empty and not hist_prices.empty:
                if dividends.index.tz is None:
                    dividends.index = dividends.index.tz_localize('UTC')
                start_ts = pd.to_datetime(args.start).tz_localize(dividends.index.tz)
                dividends = dividends[dividends.index >= start_ts]

                price_on_div_dates = hist_prices[hist_prices.index.isin(dividends.index)]
                matched_prices = price_on_div_dates.reindex(dividends.index, method='nearest')

                div_yield_series = (dividends / matched_prices) * 100
                avg_yield = round(div_yield_series.mean(), 2)
                div_yields[t] = avg_yield
        except Exception:
            continue

    if base_ticker not in div_yields or len(div_yields) <= 1:
        print(f"❌ Not enough dividend yield data to compare {base_ticker} vs group.")
        return

    base_yield = div_yields[base_ticker]
    group_yields = [y for t, y in div_yields.items() if t != base_ticker]
    group_mean = np.mean(group_yields)
    group_std = np.std(group_yields)
    z_score = (base_yield - group_mean) / group_std if group_std > 0 else 0
    percentile = sum(1 for x in group_yields if x < base_yield) / len(group_yields) * 100
    percent_diff = ((base_yield - group_mean) / group_mean) * 100

    # Plot
    plt.figure(figsize=(10, 6), dpi=600)
    plt.bar(div_yields.keys(), div_yields.values(), color='skyblue', edgecolor='black')
    plt.axhline(group_mean, linestyle='--', color='gray', label=f'Group Mean: {group_mean:.2f}%')
    plt.axhline(base_yield, linestyle='--', color='red', label=f'{base_ticker} Yield: {base_yield:.2f}%')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Avg Dividend Yield (%)')
    plt.title(f"{base_ticker} vs Group Dividend Yields", fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=6)

    # Summary Box
    summary_title = "DIVIDEND COMPARISON SUMMARY"
    lines = [
        f"{base_ticker} Yield: {base_yield:.2f}%",
        f"Group Mean: {group_mean:.2f}%",
        f"Group StdDev: {group_std:.2f}",
        f"Z-Score: {z_score:.2f}",
        f"Percentile: {percentile:.1f}%",
        f"% Diff from Mean: {percent_diff:+.2f}%"
    ]
    full_text = [summary_title] + lines

    box_x, box_y = 1.03, 0.5
    width, height = 0.14, 0.35
    spacing = height / (len(full_text) + 1)
    fontsize = 6

    ax = plt.gca()
    bbox = FancyBboxPatch(
        (box_x, box_y - height / 2), width, height,
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

    for i, line in enumerate(full_text):
        ax.text(
            box_x - 0.035,
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

    plt.subplots_adjust(right=0.89)
    filename = f'comparex_{base_ticker.lower()}_div_chart.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"✔️Saved comparex dividend chart to: {os.path.abspath(filename)}")

    # Optional CSV
    if getattr(args, 'csv', False):
        pd.DataFrame({
            'Ticker': list(div_yields.keys()),
            'Avg Dividend Yield (%)': list(div_yields.values())
        }).to_csv(f'comparex_{base_ticker.lower()}_div.csv', index=False)
        print(f"✔️Saved comparex dividend CSV to: {os.path.abspath(f'comparex_{base_ticker.lower()}_div.csv')}")


# ------------ Argparser ------------ #
# ----------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logarithmic regression analysis for stock tickers.', add_help=False)

    # Add Arguments #
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
Description:
This is a command-line tool that facilitates the analysis of financial market data available via the yfinance python library.
It requires yfinance, pandas, numpy, matplotlib for data manipulation/visualization and argparse to create a "syntax" for the commandline arguments.
The intended use of this script is to act as a quantitative pre/post-requisite for making investment decisions.
In other words, the qualitative fundamentals, financials and future prospects of said asset must be sound in order for this to be used as an "indicator of timing".

Indicators/Assumptions:
When you run --log on a given ticker(s) the linear and log charts will be outputted in one png file for each respective ticker, accompanied by a summary statistics box and legend.
The legend defines our assumptions, the summary statistics box presents the values derived and their assumed positive/negative indicators with conditional color formatting.
For example, with the current version (cli15.py) if a given tickers actual price today is <= -2 std devs below the estimate of the regression line,
and the mean absolute percentage error (MAPE) of that regression line is < 2.5%,
then both metrics are identified as positive (dark green) and subsequently the ticker is identified as undervalued, --> *(add text that presents on .png as "overvalued" or "undervalued")*
due to both the current position in # of std devs and the accuracy of the regression being run.

~None of these assumptions are set in stone, they are to be tested and refined.

        =========
        CLI Setup
        =========

REQUIRED libraries:
pip install yfinance pandas numpy matplotlib argparse

[----------------------------------]
BASH Instructions:
From terminal --> Change directory
cd ~/path/to/your/folder

From directory --> Run python script
python3 cli15.py SPY GOLD USD-BTC
[----------------------------------]
VENV Instructions:
From venv --> Run python script
python3 cli15.py SPY GOLD USD-BTC
[----------------------------------]
GIT Push/Pull Instructions:
cd /path/to/your/project (only from terminal)
git status               (venv start here)
git add -A
git status               (to check that correct changes are staged)
git commit -m "commit message"
git push origin main
[----------------------------------]

        ========
        COMMANDS
        ========

POSITIONAL Arguments:
Ticker (default)        Ticker must come first (SPY GOLD USD-BTC)
--comparex              Can be used as a positional argument or an optional argument

OPTIONAL Arguments:
"blank"                 Returns the default linear chart for x ticker
--log                   Returns the logarithmic chart in addition to the default linear chart
--norm / --normdist     Plots a distribution of daily % gain/loss over a given date range
--intrv                 Data interval (1d, 1wk, 1mo, etc.)
--smooth                Rolling regression window ex. (252 for 1-year daily)
--pe                    Chart P/E ratios over defined date range
--div                   Chart dividends over defined date range
--compare               v1. Produces a comparison .png output for the tickers and metrics inputted (--pe, --div, ect.) over defined date range
--comparex              v2. Compares x ticker --vs a group of tickers (using the --comparex ticker as the basis to compare the --vs group against) *only functional on --pe metric currently*
--vs                    Conditional argument for --comparex ex. (--comparex nvda --vs amd msft adi --pe)
--start YYYY-MM-DD      Set the start date for data (default: inception)
--end YYYY-MM-DD        Set the end date for data (default: today)
--csv                   Save raw data to .csv file
--help -h               Opens this guide

        ==========
        COPY/PASTE
        ==========

COMMANDS:
-----------------]
python3 cli15.py
--log
--normdist
--intrv 1d
--smooth 1008
--pe
--div
--compare
--comparex
--vs
--start 1900-01-01 (Alias for inception)
--end 2100-01-01 (Alias for latest available data)
--csv
------------------]

--INTRV:
| Interval | Max Period Fetchable | Notes                                              |
| -------- | -------------------- | -------------------------------------------------- |
| `1m`     | 7 days               | Only available for recent data, no pre/post-market |
| `2m`     | 60 days              | Intraday                                           |
| `5m`     | 60 days              | Intraday                                           |
| `15m`    | 60 days              | Intraday                                           |
| `30m`    | 60 days              | Intraday                                           |
| `60m`    | 730 days (2 years)   | Only returns data for market hours                 |
| `90m`    | 60 days              | Limited use                                        |
| `1h`     | Same as `60m`        | Alias for `60m`                                    |
| `1d`     | Full history         | Highest level of granularity for full history      |
| `1w`     | Full history         | Medium level of granularity for full history       |
| `1mo`    | Full history         | Lowest level of granularity for full history       |
---------------------------------------------------------------------------------------|

--SMOOTH:
(over N trading days for 1d intervals)
---------------------------]
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

v6. CURRENT CAPABILITY Examples:
python3 cli15.py --log --normdist --intrv 1d --smooth 252 --start 1900-01-01 --end 2100-01-01 --pe --div --compare --csv
python3 cli15.py --comparex nvda --vs amd msft adi --pe
-----------------------------------------------------------------------------------------------------------------------]
        """)
        sys.exit()

    args = parser.parse_args()

    # Ensure --perc gain/loss is calculated by default unless explicitly turned off
    if not getattr(args, 'perc', False):
        args.perc = True

    # --- Insert control logic --- #
    if args.comparex and args.vs and args.pe:
        base_ticker = args.comparex[0].upper()
        vs_tickers = [t.upper() for t in args.vs]
        comparex_pe_summary(base_ticker, vs_tickers, args)
        # Exit after running this comparison to avoid duplicate processing
        sys.exit(0)

    elif args.comparex and not args.vs:
        print("❌ Error: --comparex must be used with --vs and at least one metric (e.g., --perc, --div, --pe)")
        sys.exit(1)
    else:
        args.tickers = getattr(args, 'tickers', []) or []
        base_tickers = []
        vs_tickers = []

    # Skip this check if --comparex is used
    if not args.tickers and not args.comparex:
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


# ---------- Compare Section END ---------- #
# ----------------------------------------- #
