Description:

This Python script is a command-line tool for performing a logarithmic regression analysis on the historical stock price data of a given ticker symbol. It requires the yfinance library, pandas, numpy and matplotlib for data manipulation and visualization of the results. Saved as a .png file with the path ~/PycharmProjects/commandline 



Bash Instructions:

# "change directory" identifies the folder #
cd ~/PycharmProjects/commandline
------------------------------------------------------
# Runs the log_regression script tool for any TICKER #
python3 log_regression.py AAPL


Run:

  v0.

cd ~/PycharmProjects/commandline

python3 log_regression.py 

  v1. 

cd ~/PycharmProjects/commandline

python3 log_regression_1.py 

  v2. 

cd ~/PycharmProjects/commandline

python3 log2.py 

  v3. 

cd ~/PycharmProjects/commandline

python3 log3.py 



Commands: 

Multiple Tickers:	#Ticker is positional#

aapl msft btc-usd 

% gain/loss:

--percent-gain

Date range:

--start 1900-01-01 --end 2100-01-01

Interval:

--interval 1wk

Rolling regression:

--rolling 252

Save CSV:

--csv

Multiple Commands:	# (Ticker is positional and must come first) #

python3 log2.py aapl msft btc-usd --start 1900-01-01 --end 2100-01-01 --interval 1d --rolling 252 --csv 


Combining commands:

Rolling Window	Best Interval	Description
--rolling 252	--interval 1d	~1-year rolling window (daily)
--rolling 1008	--interval 1d	~4-year political rolling window (daily) *
--rolling 1260	--interval 1d	~5-year rolling window (daily)
--rolling 2520	--interval 1d	~10-year rolling window (daily)
--rolling 7560	--interval 1d	~30-year rolling window (daily)

--rolling 52	--interval 1wk	~1-year rolling window (weekly)
--rolling 260	--interval 1wk	~5-year rolling window (weekly)
--rolling 520	--interval 1wk	~10-year rolling window (weekly)
--rolling 1560	--interval 1wk	~30-year rolling window (weekly)

--rolling 12	--interval 1mo	~1-year rolling window (monthly)
--rolling 60	--interval 1mo	~5-year rolling window (monthly)
--rolling 120	--interval 1mo	~10-year rolling window (monthly)
--rolling 360	--interval 1mo	~30-year rolling window (monthly)


v3. #current capability 

python3 log3.py eix swk btc-usd --percent-gain --interval 1d --rolling 252 --start 1900-01-01 --end 2100-01-01








	







