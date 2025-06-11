Description:

This is a command-line tool that facilitates the analysis of financial market data available via the yfinance python library.
It requires yfinance, pandas, numpy, matplotlib for data manipulation/visualization and argparse to create a "syntax" for your commandline arguments.
The broad purpose of this project is to identify if a purely analytical investment strategy holds any merit.
The current version (cli5.py) is capable of performing logarithmic regressions, charting distributions of % gain/loss in an interval over a given date range,
and the charting and comparison of dividends and pe ratios (other available metrics and functionality will be integrated in future versions).

Indicators/Assumptions:

When you run --log on a given ticker there will be a summary statistics box and legend.
The legend defines our assumptions, and the summary statistics present the assumed positive/negative indicators.
For example, if a given tickers actual price today is 2 std deus below the estimate of the log regression line,
and the mean absolute percentage error (MAPE) of that regression line is less than 2.5%,
than the given asset is undervalued based purely on historical data,
with the assumption that the trend holds true into the future.

~ None of these assumptions are set in stone, it is my hope that people experiment with and build ontop of what is currently in place.

        =========
        CLI TOOLs
        =========

REQUIRED libraries:
pip install yfinance pandas numpy matplotlib argparse

BASH Instructions:
From terminal --> Change directory
cd ~/path/to/your/folder

From directory --> Run script
python3 cli5.py SPY GOLD USD-BTC

[----------------------------------]

VENV Instructions:
From venv --> Run the python script
python3 cli5.py SPY GOLD USD-BTC

        ========
        COMMANDS
        ========

POSITIONAL Arguments:
Ticker (default)        Ticker must come first (SPY GOLD USD-BTC)
--comparex              Can be used as a positional argument or an optional argument

OPTIONAL Arguments:
"blank"                 Returns the default linear chart for X ticker
--log                   Returns the logarithmic chart in addition to the default linear chart
--norm / --normdist     Plots a .png distribution of daily % gain/loss over a given date range
--intrv                 Data interval (1d, 1wk, 1mo, etc.)
--smooth                Rolling regression window (e.g., 252 for 1-year daily)
--perc                  Display percent gain/loss over the selected date range (defaults to included)
--pe                    Chart P/E ratio and output raw values
--div                   Chart dividends and output raw values
--compare               v1. Produces a comparison .png output for the tickers and metrics inputted over defined date range
--comparex              v2. Compares x ticker (--vs ticker) or group thereof (using x ticker as the average to compare the --vs group against)
--vs                    Conditional argument for --comparex above ex. (--comparex aapl msft nvda --vs spy)
--start YYYY-MM-DD      Set the start date for data (default: 2000-01-01)
--end YYYY-MM-DD        Set the end date for data (default: today)
--csv                   Save to a .CSV file
--help -h               Opens this guide

        =============
        [COPY/PASTE]:
        =============

COMMANDS:
---------------]
python3 cli3.py
--log
--norm / --normdist
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

INTRV:
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
---------------------------------------------------------------------------------------]

SMOOTH:
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

v5. CURRENT CAPABILITY:
python3 cli5.py --log --normdist --intrv 1d --smooth 1008 --pe --div --compare --vs --start 1900-01-01 --end 2100-01-01 --csv

--COMPAREX CAPABILITY:
python3 cli5.py --comparex nvda --vs amd msft adi --pe --log --normdist --intrv 1d --smooth 1008 --start 1900-01-01 --end 2100-01-01 --csv
python3 cli5.py --comparex nvda --vs amd msft adi --pe
python3 cli5.py --comparex eix --vs pcg sre duk nee so --pe

