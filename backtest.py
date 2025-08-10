"""Backtesting script for Agentic PM"""
from datetime import datetime, timedelta
from models.llm_setup import get_llm
from main import daily_run

START_DATE = datetime(2025, 5, 7)
END_DATE = datetime(2025, 8, 8)

def backtest(start_date=START_DATE, end_date=END_DATE, llm=None, react=False):
    """simulate backtest"""
    date = start_date
    thread_id=f"backtest_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    while date <= end_date:
        try:
            print(f"""\nRunning backtest for {date.strftime('%Y-%m-%d')}""")
            daily_run(
                today=date.strftime('%Y-%m-%d'),
                checkpoint=False,
                backtest=True,
                thread_id=thread_id,
                llm=llm,
                react=react
            )
            date += timedelta(days=1)
        except FileNotFoundError as e:
            print(f"Error on {date.strftime('%Y-%m-%d')}: {e}")
            date += timedelta(days=1)

if __name__ == '__main__':
    # To run a backtest, set BACKTEST to True in utils/constants.py
    # ollama_llm = get_llm('ollama')  # Use 'ollama' backend for (ReAct enabled) LLM agents
    hf_llm = get_llm('hf')  # Use 'hf' backend for chat LLMs
    TIMEIT = True  # Set to True to enable timing
    if TIMEIT:
        import time
        start_time = time.time()
        backtest(START_DATE, END_DATE, llm=hf_llm, react=False)
        print(f"Backtest execution time: {time.time() - start_time} seconds")
    else:
        # Run the backtest without timing
        backtest(START_DATE, END_DATE, llm=hf_llm, react=False)
