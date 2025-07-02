"""Backtesting script for Agentic PM"""
from datetime import datetime, timedelta
from main import daily_run

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

def backtest():
    """simulate backtest"""
    date = START_DATE
    while date <= END_DATE:
        print(f"\nRunning backtest for {date.strftime('%Y-%m-%d')}")
        daily_run(today=date.strftime('%Y-%m-%d'), checkpoint=False, backtest=True)
        date += timedelta(days=1)

if __name__ == '__main__':
    backtest()
