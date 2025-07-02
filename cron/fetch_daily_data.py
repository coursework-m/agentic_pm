"""fetch_daily_data"""
from tools.tools import fetch_securities_data
from utils.constants import TICKERS
def daily_run():
    """Run daily cron job"""
    print("[CRON] Fetching market data...")
    market_data = fetch_securities_data(TICKERS)
    with open("logs/latest_market_data.json", "w", encoding='utf') as f:
        f.write(market_data)

if __name__ == "__main__":
    daily_run()

# CRONTAB EXAMPLE
# Add this line to crontab with `crontab -e`
# 0 6 * * * /usr/bin/python3 ~/workspace/agentic_pm/fetch_daily_data.py >> /cron.log 2>&1
