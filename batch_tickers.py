"""Batch tickers"""
import os
import json
from transformers import AutoTokenizer
from utils.constants import OUTPUT_DIR2

def est_token_count(token_data=None):
    """Estimate token count."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokens = tokenizer.encode(token_data)
    print(f"Input token count: {len(tokens)}")

def batch_tickers(prompt_data):
    """Batch data by ticker."""
    batched_data = []
    for data in prompt_data:
        print(f"Processing ticker: {data['ticker']}")
        # print(f"Data for {ticker}: {data[ticker]}")
        # est_token_count(json.dumps(data))
        batched_data.append(data)
    return batched_data

if __name__ == "__main__":
    TODAY = "20250102"
    filename = os.path.join(OUTPUT_DIR2, f"{TODAY}.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Backtest data file not found: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        ticker_data = json.load(f)
    batched = batch_tickers(ticker_data)
    print(f"Total batched tickers: {batched}")
