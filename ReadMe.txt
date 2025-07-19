
Description:
This is a command-line tool that facilitates the analysis of financial market data available via the yfinance python library.
It requires yfinance, pandas, numpy, matplotlib for data manipulation/visualization and argparse to create a "syntax" for the commandline arguments.
The intended use of this script is to act as a quantitative pre/post-requisite for making investment decisions.
In other words, the qualitative fundamentals, financials and future prospects of said asset must be sound in order for this to be used as an "indicator of timing".

Indicators/Assumptions:
When you run --log on a given ticker(s) the linear and log charts will be outputted in one png file for each respective ticker, accompanied by a summary statistics box and legend.
The legend defines our assumptions, the summary statistics box presents the values derived and their assumed positive/negative indicators with conditional color formatting.
For example, with the current version (cli6.py) if a given tickers actual price today is <= -2 std devs below the estimate of the regression line,
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
python3 cli6.py SPY GOLD USD-BTC
[----------------------------------]
VENV Instructions:
From venv --> Run python script
python3 cli6.py SPY GOLD USD-BTC
[----------------------------------]
GIT Push/Pull Instructions:
cd /path/to/your/project (only from terminal)
git status               (venv start here)
git add -A
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
python3 cli6.py
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
python3 cli6.py --log --normdist --intrv 1d --smooth 252 --start 1900-01-01 --end 2100-01-01 --pe --div --compare --csv
python3 cli6.py --comparex nvda --vs amd msft adi --pe
-----------------------------------------------------------------------------------------------------------------------]
