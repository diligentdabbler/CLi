import yfinance as yf
import sys

if len(sys.argv) < 2:
    print("Usage: python yfinance_debugger.py TICKER1 TICKER2 ...")
    sys.exit()

tickers = sys.argv[1:]

for symbol in tickers:
    print(f"\n🔍 Checking {symbol}...")

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="max")

        if history.empty:
            print(f"⚠️  No data returned for {symbol}. Possibly delisted or temporarily unavailable on Yahoo Finance.")
        else:
            print(f"✅  {symbol} returned {len(history)} rows.")
            print(history.tail(2))  # Show last 2 rows for reference

    except Exception as e:
        print(f"❌  Error retrieving data for {symbol}: {e}")
