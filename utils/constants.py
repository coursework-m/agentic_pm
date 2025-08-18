"""Constants for Agentic PM"""
import os
from datetime import datetime
PG_USERNAME = "ekwiti"
PG_PASSWORD = "scnknfvnslicewoevn"
PG_DATABASE = "langgraph_db"
PG_HOST = "localhost"
PG_PORT = 5432
DB_URI = f"postgresql://{PG_USERNAME}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
# Set API Key
API_KEY = os.environ['API_KEY']
# List of tickers to fetch data for
TICKERS = ["AAPL", "GOOG", "MSFT", "META", "TSLA", "NVDA", "AMZN", "NFLX"]
# Define date range (YYYYMMDD format)
START_DATE = "20200101"  # Jan 1, 2020
END_DATE = datetime.now().strftime('%Y%m%d')
TODAY = datetime.now().strftime('%Y%m%d') # Today
# API Base URL
BASE_URL = "https://www.alphavantage.co/query"
# Define the path for the backtest or forwardtest
# This can be set to "backtest" or "forwardtest" based on the context
BACKTEST = True # Set to True for backtesting, False for forward testing
PATH = "backtest" if BACKTEST else "forwardtest"
# Directory for saving output
OUTPUT_DIR = f"./data/{PATH}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR10 = f"{OUTPUT_DIR}/messages"
os.makedirs(OUTPUT_DIR10, exist_ok=True)
