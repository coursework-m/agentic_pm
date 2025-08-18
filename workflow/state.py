"""Agentic PM Custom State"""
from typing import Dict, List, Any
from langgraph.graph import MessagesState

class CustomState(MessagesState, total=False):
    """Custom State"""
    remaining_steps: int
    tickers: List[str]
    transactions: List[Dict[str, Any]]
    securities_data: Dict[str, Any]
    analysis_summary: List[Dict[str, Any]]
    research_summary: List[Dict[str, Any]]
    portfolio: Dict[str, Any]
    portfolio_value: float
    portfolio_history: List[Dict[str, Any]]
    total_asset_value: float
    chat_history: List[Any]
    structured_response: List[Any]
    holdings: Dict[str, Dict[str, float]] # e.g., {"AAPL": {"quantity": 10, "buy_price": 175.0}}
    holdings_value: float
    cash: float
    daily_realised_pnl: float
    cum_realised_pnl: float
    unrealised_pnl: float
    reasoning: List[str]
    total_market_value: float
    total_pnl: float
    analysis_response: str
    research_response: str
    prices: Dict[str, Any]
    portfolio_summary: Dict[str, Any]
    metrics: Dict[str, Any]
    daily_pnl: float
