"""Financial data tools for Agentic PM. NB. Docs and parameters were added by AI"""
import os
import json
import asyncio
from pprint import pprint
from datetime import datetime
from typing import List, Dict
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from utils.constants import API_KEY, BASE_URL, START_DATE, OUTPUT_DIR2

async def get_ticker_data(ticker: str, delay: float = 2.5) -> Dict:
    """Fetches financial and news data for a single ticker using Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.
        delay (float, optional): Delay in seconds before fetching data. Defaults to 2.5.

    Returns:
        Dict: Dictionary containing company info, market data, fundamentals, and news.
    """
    await asyncio.sleep(delay)

    def fetch_blocking():
        t = yf.Ticker(ticker)
        info = t.info
        company_name = info.get("longName", ticker)

        # Safely try to get news; fallback to empty list if unavailable
        try:
            news = []
            for story in t.news[:2]:
                article = {
                    "title": story['content']['title'],
                    "summary": story['content']['summary'],
                    "published_date": story['content']['pubDate'],
                    "url": story['content']['canonicalUrl']['url'],
                    "provider": story['content']['provider']['displayName'],
                    "provider_url": story['content']['provider']['url']
                }
                news.append(article)
        except ConnectionError:
            print(t.news)

        return {
            "company": company_name,
            "ticker": ticker,
            "market_data": {
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "percent_change": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "market_cap": info.get("marketCap"),
                "summary": info.get("longBusinessSummary", "")[:300]
            },
            "fundamentals": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry")
            },
            "news": news,
            "analyst_recommendation_score": info.get("recommendationMean"),
        }

    try:
        return await asyncio.to_thread(fetch_blocking)
    except ConnectionError as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return {
            "ticker": ticker,
            "market_data": {},
            "fundamentals": {},
            "news": [],
            "analyst_recommendations": []
        }

async def get_full_securities_data_async(
        tickers: List[str],
        throttle: float = 2.5) -> List[Dict]:
    """Fetches financial data for a list of tickers asynchronously.

    Args:
        tickers (List[str]): List of stock ticker symbols.
        throttle (float, optional): Delay multiplier between requests. Defaults to 2.5.

    Returns:
        List[Dict]: List of dictionaries containing ticker data.
    """
    tasks = [get_ticker_data(ticker, delay=throttle * i) for i, ticker in enumerate(tickers)]
    results = await asyncio.gather(*tasks)
    # pprint(results)
    results = [{result["ticker"]: result} for result in results]
    # pprint(results)
    filename = os.path.join(OUTPUT_DIR2, f"{datetime.now().strftime('%Y%m%d')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

@tool
def fetch_securities_data(tickers: List[str]) -> Dict[str, Dict]:
    """Fetches financial data for a list of tickers using Yahoo Finance.

    Args:
        tickers (List[str]): List of stock ticker symbols.

    Returns:
        List[Dict]: List of dictionaries containing ticker data.
    """
    return asyncio.run(get_full_securities_data_async(tickers))

@tool
def fetch_news_sentiment(ticker: str) -> List[Dict]:
    """Fetches news sentiment data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        List[Dict]: List of dictionaries containing news articles with sentiment analysis.
    """
    print(f"Fetching news sentiment for {ticker}...")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": API_KEY,
        "limit": 1000,
        "time_from": START_DATE,
        "sort": "LATEST"
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    data = response.json()

    if "feed" not in data:
        print(f"No news data found for {ticker}")
        return None

    articles = []
    for article in data["feed"]:
        published_date = article["time_published"][:8]
        sentiment_score = float(article["overall_sentiment_score"])

        if sentiment_score <= -0.35:
            sentiment_label = "bearish"
        elif -0.35 < sentiment_score <= -0.15:
            sentiment_label = "somewhat-bearish"
        elif -0.15 < sentiment_score < 0.15:
            sentiment_label = "neutral"
        elif 0.15 <= sentiment_score < 0.35:
            sentiment_label = "somewhat-bullish"
        else:
            sentiment_label = "bullish"

        article_data = {
            "Ticker": ticker,
            "Title": article.get("title", ""),
            "Published": published_date,
            "Sentiment": sentiment_label.capitalize(),
            "Sentiment Score": article.get("overall_sentiment_score", ""),
            "Summary": article.get("summary", ""),
            "Source": article.get("source", ""),
            "Source_domain": article["source_domain"],
            "URL": article.get("url", ""),
            "Full_Article": None
        }
        try:
            article_response = requests.get(article["url"], timeout=5)
            soup = BeautifulSoup(article_response.text, "html.parser")
            paragraphs = soup.find_all("p")
            full_text = "\n".join([p.get_text() for p in paragraphs])
            article_data["Full_Article"] = full_text
            articles.append(article_data)
        except ConnectionError as e:
            article_data["Full_Article"] = f"Error fetching article: {e}"
    return articles

@tool
def fetch_economic_data() -> list:
    """Fetches key economic indicators.

    Returns:
        list: List of dictionaries containing economic indicator data.
    """
    print("Fetching economic data...")

    economic_indicators = ["REAL_GDP", "INFLATION", "UNEMPLOYMENT", "TREASURY_YIELD"]
    all_data = []

    for indicator in economic_indicators:
        params = {
            "function": indicator,
            "apikey": API_KEY,
            "interval": "quarterly"
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        if "data" in data:
            for entry in data["data"]:
                print(entry)
                all_data.append({
                    "Indicator": indicator,
                    "Date": entry.get("date", ""),
                    "Value": entry.get("value", "")
                })
    return all_data

@tool
def scan_market(tickers, min_volume=1e6, min_change=0.02) -> list:
    """Scans the market for tickers with significant volume and price change.

    Args:
        tickers (list): List of stock ticker symbols.
        min_volume (float, optional): Minimum volume threshold. Defaults to 1e6.
        min_change (float, optional): Minimum percent change threshold. Defaults to 0.02.

    Returns:
        list: Sorted list of tickers meeting the criteria.
    """
    results = []
    for t in tickers:
        df = yf.download(t, period="2d", interval="1d")
        if len(df) < 2:
            continue
        pct_change = (df.iloc[-1]['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']
        if df.iloc[-1]['Volume'] > min_volume and abs(pct_change) > min_change:
            results.append({
                "ticker": t,
                "change": round(pct_change, 4),
                "volume": int(df.iloc[-1]['Volume'])
            })
    return sorted(results, key=lambda x: abs(x['change']), reverse=True)

@tool
def search_news(ticker: str, company_name: str) -> list:
    """Searches for recent news headlines for a given stock ticker or company name.

    Args:
        ticker (str): The stock ticker symbol.
        company_name (str): The company name.

    Returns:
        list: List of news articles with title, publisher, and link.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"{ticker, company_name} stock news"
    try:
        response = tavily_client.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=5,
            days=7,
            include_answer=False
        )
        results = response.get("results", [])
        return [
            {
                "title": item.get("title"),
                "publisher": item.get("source"),
                "link": item.get("url")
            }
            for item in results
            if item.get("title") and item.get("url")
        ]
    except ConnectionError as e:
        print(f"Error fetching news from Tavily: {e}")
        return []

@tool
def search(query: str):
    """Performs a web search using DuckDuckGo.

    Args:
        query (str): The search query.

    Returns:
        Any: Search results from DuckDuckGo.
    """
    return DuckDuckGoSearchResults(tool_input=query)
