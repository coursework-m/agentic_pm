"""Backtesting script for Agentic PM"""
import random
import string
import time
from pathlib import Path
from datetime import datetime, timedelta
from models.llm_setup import get_llm
from main import daily_run
from utils.constants import OUTPUT_DIR

def find_resume_date(thread_id, start_date, end_date, model_name):
    """Find the last completed date"""
    output_dir6 = f"{OUTPUT_DIR}/{model_name}/portfolio_history"
    folder = Path(output_dir6) / thread_id
    print(f"Folder is {folder}")
    if not folder.exists():
        return start_date
    # Gather all JSON files that match YYYY-MM-DD.json
    existing_dates = []
    for file in folder.glob("*.json"):
        print(f"File is {file}")
        try:
            existing_dates.append(datetime.strptime(file.stem, "%Y-%m-%d"))
        except ValueError:
            continue  # skip any that don't match format

    if not existing_dates:
        return start_date
    last_date = max(d for d in existing_dates if start_date <= d <= end_date)
    return last_date + timedelta(days=1)

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 8, 8)

def backtest(start_date=START_DATE,
             end_date=END_DATE,
             llm=None,
             react=False,
             last_code=None,
             model_config=None):
    """Simulate backtest"""
    if last_code:
        code = last_code
    else:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    thread_id=f"{code}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    date = find_resume_date(thread_id, start_date, end_date, model_config['model'])
    print(f"Start date is {date}")
    while date <= end_date:
        print(f"While date is {date}")
        retries = 3
        try:
            print(f"""\nRunning backtest for {date.strftime('%Y-%m-%d')}""")
            daily_run(
                today=date.strftime('%Y-%m-%d'),
                checkpoint=False,
                backtest=True,
                thread_id=thread_id,
                llm=llm,
                react=react,
                end_date=end_date.strftime('%Y-%m-%d'),
                model_config=model_config
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error on {date.strftime('%Y-%m-%d')}: {e}")
            if isinstance(e, FileNotFoundError):
                continue
            # Otherwise
            print(f"Retrying {date.strftime('%Y-%m-%d')} Retries remaining {retries}")
            if retries > 0:
                retries -= 1
                daily_run(
                    today=date.strftime('%Y-%m-%d'),
                    checkpoint=False,
                    backtest=True,
                    thread_id=thread_id,
                    llm=llm,
                    react=react,
                    end_date=end_date.strftime('%Y-%m-%d'),
                    model_config=model_config
                )
        finally:
            date += timedelta(days=1)

if __name__ == '__main__':
    # ollama ids
    # mistral:v0.3
    # llama3:latest
    # qwen3:latest
    # gemma3:latest
    # deepseek-r1:latest
    # gpt-oss:20b
    # ///////////////////// #
    # HF ids
    # "openai/gpt-oss-20b"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "Qwen/Qwen3-4B"
    # "google/gemma-3-270m"
    # "google/gemma-3-4b-it"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    llm_config = {
        "model": "google/gemma-3-4b-it",
        "max_new_tokens": 2028,
        "temperature": 0.15,
        "backend": "hf" # Use 'ollama' backend for REACT LLM
    }
    # To run a backtest, set BACKTEST to True in utils/constants.py
    hf_llm = get_llm(llm_config['backend'], llm_config['model'], llm_config)
    PREVIOUS = "UZB81" # reuse GWEN3 backtest from previous run
    PREVIOUS = "F8DNU" # reuse Gemma3 from previous run
    TIMEIT = True  # Set to True to enable timing
    if TIMEIT:
        start_time = time.time()
        backtest(START_DATE,
                 END_DATE,
                 llm=hf_llm,
                 react=False,
                 last_code=PREVIOUS,
                 model_config=llm_config)
        print(f"Backtest execution time: {time.time() - start_time} seconds")
    else:
        # Run the backtest without timing
        backtest(START_DATE,
                 END_DATE,
                 llm=hf_llm,
                 react=False,
                 last_code=PREVIOUS,
                 model_config=llm_config)
