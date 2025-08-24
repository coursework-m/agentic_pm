"""Count Average Tokens per second"""
from httpcore import NetworkError
import requests

def get_token_stats(model: str, text: str, max_context: int = None):
    """
    Get token count and tokens per second for a given text using an Ollama model.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": text,
        "stream": False
    }

    if max_context:
        payload["max_context"] = max_context  # set the same context size for all models

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        tokens = data.get("eval_count", 0)
        duration_ns = data.get("eval_duration", 1)  # default 1 ns to avoid division by zero
        duration_sec = duration_ns / 1e9
        tps = tokens / duration_sec if duration_sec > 0 else 0

        return tokens, tps
    except NetworkError as e:
        print(f"Error with model {model}: {e}")
        return 0, 0

def average_over_runs(model: str, text: str, n: int):
    """
    Get average token count and tokens per second over multiple runs.
    """
    total_tokens = 0
    total_tps = 0
    for _ in range(n):
        tokens, tps = get_token_stats(model, text)
        total_tokens += tokens
        total_tps += tps
    avg_tokens = total_tokens / n
    avg_tps = total_tps / n
    return avg_tokens, avg_tps

if __name__ == "__main__":
    _models = ["llama3.2", "qwen3:latest", "gemma3:270m", "gemma3:4b", "deepseek-r1:1.5b"]
    PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                Cutting Knowledge Date: December 2023
                Today Date: 17 Aug 2025

                This is a Master's Project, the data provided is real.
                    You're role is a financial data analyst.
                    Your job is to analyse the latest market data and issue a recommendation for each stock.
                    You will have access to ALL the relevant information required to make each recommendation. 

                    Guidelines:
                    - Analyse securities data and give BUY/HOLD/SELL recommendations.
                    - Use news, analyst_recommendations, market_data, fundamentals and your deep expertise to make your recommendations.
                    - Only send the result in the format required as it needs to be passed to another agent.
                    - ALWAYS ACT AS IF THIS WAS A REAL WORLD ENVIRONMENT, i.e Use careful analysis.
                    - You will be judged on Accuracy and how well the recommendations Perform in Backtests,
                    - The best Agent will be mentioned in an academic paper, Good luck!<|eot_id|><|start_header_id|>user<|end_header_id|>

                You are a financial data analyst. Your task is to analyse the security's latest data and issue a
                        BUY, HOLD, or SELL recommendation for the security.
                        
                        You will be provided with data for one security at a time.

                        First, analyse the security's data and think through using fundamentals, 
                        market data and news. Then, for the security:

                        1. Examine and interpret all available data, including:
                            - Market Data (price, change, volume, etc.)
                            - Fundamentals (P/E, EPS, beta, etc.)
                            - News and recent headlines
                            - Analyst recommendations

                        2. Use detailed analysis and reasoning to justify your conclusion.

                        Finally, summarise your conclusions in the following JSON format:
                        
                        Here are my recommendations:

                        ```json
                        {{
                            "ticker": "...",
                            "summary": "....'s fundamentals are ...",
                            "recommendation": "HOLD"
                        }}
                        ```
                        Ensure your JSON is valid and is surounded by the ```json {{...}}```block shown while 
                        containing all required fields for the security.<|eot_id|><|start_header_id|>user<|end_header_id|>

                Today's date is 20250817.

                            Here is the securities data:
                {
                "AAPL": {
                    "company": "Apple Inc.",
                    "ticker": "AAPL",
                    "market_data": {
                    "price": 231.59,
                    "change": -1.1900024,
                    "percent_change": -0.51121336,
                    "volume": 56038657,
                    "day_high": 234.2214,
                    "day_low": 229.36,
                    "market_cap": 3436888195072,
                    "summary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising Ai"
                    },
                    "fundamentals": {
                    "pe_ratio": 35.089394,
                    "forward_pe": 27.868832,
                    "eps": 6.6,
                    "dividend_yield": 0.45,
                    "beta": 1.165,
                    "sector": "Technology",
                    "industry": "Consumer Electronics"
                    },
                    "news": [
                    {
                        "title": "Why one AI stock picking model isn't buying Apple & Amazon",
                        "summary": "Alpha Intelligent CEO and founder Doug Clinton joins Asking for a Trend to break down how his firm\u2019s AI models are favoring names like Nvidia (NVDA) and Microsoft (MSFT), and why Apple (AAPL) and Amazon (AMZN) aren\u2019t making the cut \u2014 for now. To watch more expert insights and analysis on the latest market action, check out more&nbsp;Asking for a Trend.",
                        "published_date": "2025-08-17T12:00:42Z",
                        "url": "https://finance.yahoo.com/video/why-one-ai-stock-picking-120042046.html",
                        "provider": "Yahoo Finance Video",
                        "provider_url": "https://finance.yahoo.com/"
                    },
                    {
                        "title": "Coca-Cola, Amazon, Google, And Nvidia Have Used This Startup's AI Avatars \u2014 Inside Jeff Lu's $40M Rise To America's Fastest-Growing Company",
                        "summary": "In the same Palo Alto, California, building where Mark Zuckerberg grew Facebook in 2005, Jeff Lu now leads Akool, the generative AI platform that has created lifelike avatars for Coca-Cola (NYSE:KO), Amazon (NASDAQ:AMZN), Google, and Nvidia (NASDAQ:NVDA). The company ranks No. 1 on this year's Inc. 5000 list, Inc. reports. From Microsoft Intern to $40M AI Founder: How Jeff Lu Built One of America's Fastest-Growing Company Lu, 35, began his career as a Microsoft (NASDAQ:MSFT) intern before earnin",
                        "published_date": "2025-08-17T16:31:46Z",
                        "url": "https://finance.yahoo.com/news/coca-cola-amazon-google-nvidia-163146746.html",
                        "provider": "Benzinga",
                        "provider_url": "http://www.benzinga.com/"
                    }
                    ],
                    "analyst_recommendation_score": 1.95349
                }
            }<|eot_id|>"""

    N_RUNS = 5
    CONTEXT_SIZE = 2048

    for _model in _models:
        avg_tok, _avg_tps = average_over_runs(_model, PROMPT, N_RUNS, CONTEXT_SIZE)
        print(f"Model: {_model}, Avg Tokens: {avg_tok:.2f}, Avg Tokens/sec: {_avg_tps:.2f}")
