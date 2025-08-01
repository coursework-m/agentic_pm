"""Backtesting script for Agentic PM"""
from datetime import datetime, timedelta
from models.llm_setup import get_llm
from main import daily_run

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2020, 6, 30)

def backtest(start_date=START_DATE, end_date=END_DATE, llm=None):
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
                llm=llm
            )
            date += timedelta(days=1)
        except FileNotFoundError as e:
            print(f"Error on {date.strftime('%Y-%m-%d')}: {e}")
            date += timedelta(days=1)

if __name__ == '__main__':
    # ollama_llm = get_llm('ollama')  # Use 'ollama' backend for (ReAct enabled) LLM agents
    hf_llm = get_llm('hf')  # Use 'hf' backend for chat LLMs
    TIMEIT = False  # Set to True to enable timing
    if TIMEIT:
        import time
        start_time = time.time()
        backtest(START_DATE, END_DATE, llm=hf_llm)
        print(f"Backtest execution time: {time.time() - start_time} seconds")
    else:
        # Run the backtest without timing
        backtest(START_DATE, END_DATE, llm=hf_llm)
