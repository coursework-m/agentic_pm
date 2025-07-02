"""Backtesting script for Agentic PM"""
from datetime import datetime, timedelta
from main import daily_run

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

def backtest(start_date=START_DATE, end_date=END_DATE):
    """simulate backtest"""
    date = start_date
    while date <= end_date:
        print(f"\nRunning backtest for {date.strftime('%Y-%m-%d')}")
        daily_run(today=date.strftime('%Y-%m-%d'), checkpoint=False, backtest=True)
        date += timedelta(days=1)

if __name__ == '__main__':
    backtest(START_DATE, END_DATE)
