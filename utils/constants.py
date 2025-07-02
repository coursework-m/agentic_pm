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
END_DATE = datetime.now().strftime('%Y%m%d') # Today
TODAY = datetime.now() # Today

# Directory for saving output
OUTPUT_DIR = "./data/alpha_vantage_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR2 = "./data/securities_data"
os.makedirs(OUTPUT_DIR2, exist_ok=True)
OUTPUT_DIR3 = "./data/analysts_summary"
os.makedirs(OUTPUT_DIR3, exist_ok=True)
OUTPUT_DIR4 = "./data/research_summary"
os.makedirs(OUTPUT_DIR4, exist_ok=True)
# API Base URL
BASE_URL = "https://www.alphavantage.co/query"
