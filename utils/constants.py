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
# Define the path for the backtest or forwardtest
# This can be set to "backtest" or "forwardtest" based on the context
BACKTEST = True # Set to True for backtesting, False for forward testing
PATH = "backtest" if BACKTEST else "forwardtest"
# Directory for saving output
OUTPUT_DIR = f"./data/{PATH}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR2 = f"./data/{PATH}/securities_data"
os.makedirs(OUTPUT_DIR2, exist_ok=True)
OUTPUT_DIR3 = f"./data/{PATH}/analysts_summary"
os.makedirs(OUTPUT_DIR3, exist_ok=True)
OUTPUT_DIR4 = f"./data/{PATH}/research_summary"
os.makedirs(OUTPUT_DIR4, exist_ok=True)
OUTPUT_DIR5 = f"./data/{PATH}/transactions"
os.makedirs(OUTPUT_DIR5, exist_ok=True)
OUTPUT_DIR6 = f"./data/{PATH}/portfolio_summary"
os.makedirs(OUTPUT_DIR6, exist_ok=True)
OUTPUT_DIR7 = f"./data/{PATH}/analyst_response/"
os.makedirs(OUTPUT_DIR7, exist_ok=True)
OUTPUT_DIR8 = f"./data/{PATH}/research_response/"
os.makedirs(OUTPUT_DIR8, exist_ok=True)
# OUTPUT_DIR9 = f"./data/{path}/messages/"
# os.makedirs(OUTPUT_DIR9, exist_ok=True)
# API Base URL
BASE_URL = "https://www.alphavantage.co/query"
