"""
Agentic PM Nodes"""
import os
import json
import traceback
from datetime import datetime
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage #, ToolMessage
from workflow.state import CustomState
from tools.tools import search, search_news, scan_market, fetch_securities_data
from utils.utils import parse_summary
from utils.constants import TICKERS, OUTPUT_DIR3, OUTPUT_DIR4
from prompts.prompts import analyst_system_prompt, researcher_system_prompt

def router_node(state: CustomState):
    """Should Continue Router"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return "trader_node"

def memory_node(state: dict, config: dict) -> dict:
    """Load Store Data"""
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    store = config.get("configurable", {}).get("store", None)
    print(thread_id)
    print(store)
    def safe_get(key, default):
        try:
            result = store.get(("portfolio", thread_id), key)
            value = result.value if result is not None else default
            print(f"[DEBUG] Value for '{key}':", value)
            print("Type:", type(value))
            return value
        except KeyError as e:
            print(f"[WARN] Failed to load key '{key}': {e}")
            traceback.print_exc()
            return default

    holdings = safe_get("holdings", {})
    transactions = safe_get("transactions", [])
    transactions = transactions["transactions"] \
        if not isinstance(transactions, list) else transactions
    cash = safe_get("cash", 100_000.0)
    cash = cash["cash"] if not isinstance(cash, float) else cash
    realised_pnl = safe_get("realised_pnl", 0.0)
    realised_pnl = realised_pnl["realised_pnl"] \
        if not isinstance(realised_pnl, float) else realised_pnl

    print(f"[{thread_id}] Portfolio loaded from store.")

    return {
        **state,
        "holdings": holdings,
        "transactions": transactions,
        "cash": cash,
        "realised_pnl": realised_pnl,
        "messages": state.get("messages", [])[-1:]  # Retain just last message
    }

def data_node(state: CustomState, config: dict) -> dict:
    """Data Node"""
    backtest = config.get("configurable", {}).get("backtest", False)
    today = config.get("configurable", {}).get("today", datetime.now())

    if backtest:
        # For backtesting, we can use a static data file or mock data
        # Here we assume a static file exists with pre-fetched data
        filename = os.path.join(OUTPUT_DIR3, f"{today}.json")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Backtest data file not found: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        print(f"Loaded backtest data from {filename}")
    else:
        # For live data, we fetch the latest securities data
        print("Fetching live securities data...")
        # Ensure the fetch_securities_data tool is available
        if not hasattr(fetch_securities_data, 'invoke'):
            raise RuntimeError("fetch_securities_data tool is not properly configured.")
        data = fetch_securities_data.invoke({"tickers": TICKERS})
        # Serialize the original data structure as JSON
        json_data = json.dumps(data, indent=2)

    # Create an AIMessage with the raw data
    data_message = AIMessage(
        content=f"Here is the raw securities data:\n```json\n{json_data}\n```",
        name="data node"
    )

    # Return updated state including the message and stored raw data
    return {
        **state,
        "tickers": TICKERS,
        "securities_data": data,
        "messages": state["messages"] + [data_message]
    }

# def analyst_node(state: CustomState, config: dict) -> dict:
#     """Call Analyst Node"""
#     analyst_config = config
#     securities_data = state["securities_data"]
#     analyst_prompt = HumanMessage(content=f"""
#         You are a financial data analyst. Your task is to analyse the latest securities data and issue a
#         BUY, HOLD, or SELL recommendation for each.

#         First, analyse the securities data and think through each stock using fundamentals, 
#         market data and news. Then, for each security:

#         1. Examine and interpret all available data, including:
#             - Market Data (price, change, volume, etc.)
#             - Fundamentals (P/E, EPS, beta, etc.)
#             - News and recent headlines
#             - Analyst recommendations

#         2. Use detailed analysis and reasoning to justify your conclusion.

#         Finally, summarize your conclusions in the following JSON format:
                                  
#         Here are my recommendations:
        
#         [
#             {{
#                 "ticker": "...",
#                 "summary": "...",
#                 "recommendation": "BUY/HOLD/SELL"
#             }},
#             ...
#         ]

#         Here is the securities data:         
#         {securities_data}
#         """,
#         name="analyst"
#     )

#     analyst_messages = [analyst_system_prompt, analyst_prompt]
#     llm = analyst_config["configurable"]["llm"]
#     analyst_response = llm.invoke({"messages": analyst_messages}, analyst_config)
#     # analyst_response = llm.invoke(analyst_messages, analyst_config)
#     # print(analyst_response.tool_calls)
#     content = analyst_response.content
#     analyst_summary = parse_summary(content)
#     filename = os.path.join(OUTPUT_DIR3, f"{datetime.now().strftime('%Y%m%d')}.json")
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(analyst_summary, f, indent=2)
#     return {
#         **state,
#         "analysis_summary": analyst_summary,
#         "messages": state["messages"] + [analyst_response]
#     }

def analyst_node(state: CustomState, config: dict) -> dict:
    """Call Analyst Node"""

    analyst_config = config
    securities_data = state["securities_data"]
    analyst_prompt = HumanMessage(content=f"""
        You are a financial data analyst. Your task is to analyse the latest securities data and issue a
        BUY, HOLD, or SELL recommendation for each.

        First, analyse the securities data and think through each stock using fundamentals, 
        market data and news. Then, for each security:

        1. Examine and interpret all available data, including:
            - Market Data (price, change, volume, etc.)
            - Fundamentals (P/E, EPS, beta, etc.)
            - News and recent headlines
            - Analyst recommendations

        2. Use detailed analysis and reasoning to justify your conclusion.

        Finally, summarise your conclusions in the following JSON format:
        
        Here are my recommendations:

        ```json
        [
            {{
                "ticker": "AAPL",
                "summary": "Apple Inc. is showing ...",
                "recommendation": "BUY"
            }},
            {{
                "ticker": "GOOG",
                "summary": "Google's fundamentals are ...",
                "recommendation": "HOLD"
            }}
            // ... one object per ticker
        ]
        ```
        Here is the securities data:         
        {securities_data}
        """,
        name="analyst"
    )

    analyst_messages = [analyst_system_prompt, analyst_prompt]
    llm = analyst_config["configurable"]["llm"]

    # For AgentExecutor, pass a string prompt; for chat models, pass messages
    try:
        # Try as agent (AgentExecutor)
        analyst_response = llm.invoke({"messages": [analyst_prompt.content]}, analyst_config)
        # AgentExecutor returns a dict with 'output' or 'return_values'
        if isinstance(analyst_response, dict):
            content = analyst_response.get("response") or analyst_response.get("return_values", {}).get("output")
            if not content:
                # fallback: try to get first value
                content = next(iter(analyst_response.values()))
    except Exception:
        # Fallback: try as chat model
        analyst_response = llm.invoke(analyst_messages, analyst_config)
        content = analyst_response.content

    try:
        analyst_summary = parse_summary(content)
    except Exception:
        print("Failed to parse summary. LLM output was:\n", content)
        raise

    filename = os.path.join(OUTPUT_DIR3, f"{datetime.now().strftime('%Y%m%d')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analyst_summary, f, indent=2)
    return {
        **state,
        "analysis_summary": analyst_summary,
        "messages": state["messages"] + [analyst_response]
    }

def researcher_node(state: CustomState, config: dict) -> dict:
    """Call Research Node"""
    researcher_config = config
    securities_data = state["securities_data"]
    analysis_summary = state["analysis_summary"]
    researcher_prompt = HumanMessage(content=f"""You are a financial research analyst.
        Your task is to analyze the latest securities data and analysis summary then issue an APPROVED/DENIED recommendation for each.
        
        First, analyse the securities data and think through each stock using fundamentals, market data and news. 
        
        For each security:

        1. Examine and interpret all available data, including:
            - Market Data (price, change, volume, etc.)
            - Fundamentals (P/E, EPS, beta, etc.)
            - News and recent headlines
            - Analyst recommendations
            - The Analysis summary

        2. Use detailed analysis and reasoning to justify your conclusion.
        
        For each security, return:
            - "ticker"
            - "approved": true or false
            - "target_allocation_percent": % of total portfolio (float)
            - "reasoning": your reasoning for approval/denial

        Finally, summarize your conclusions in the following JSON format:
        
        Here are my recommendations:
        
        ```json
        [
            {{
                "ticker": "AAPL",
                "approved": true,
                "target_allocation_percent": 4.0,
                "reasoning": "AAPL is stable and aligns with a conservative portfolio."
            }},
            ...
        ]
                                     
        Here is the securities data:         
        {securities_data}

        Here is the analysis data:
        {analysis_summary}
        """,
        name="researcher"
    )

    researcher_messages = [researcher_system_prompt, researcher_prompt]
    llm = researcher_config["configurable"]["llm"]
    researcher_response = llm.invoke(researcher_messages, researcher_config)
    print(researcher_response.tool_calls)
    content = researcher_response.content
    research_summary = parse_summary(content)
    filename = os.path.join(OUTPUT_DIR4, f"{datetime.now().strftime('%Y%m%d')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(research_summary, f, indent=2)
    return {
        **state,
        "research_summary": research_summary,
        "messages": state["messages"] + [researcher_response]
    }

def trader_node(state: CustomState):
    """Call Trader node with full rebalance logic"""
    try:
        research_summary = state["research_summary"]
    except KeyError as e:
        return {
            "messages": state["messages"] + [AIMessage(
                content=f"Error parsing research summary: {e}", name="trader")]
        }

    securities_data = state["securities_data"]
    analysis_summary = {rec["ticker"]: rec for rec in state.get("analysis_summary", [])}
    holdings = dict(state.get("holdings", {}))
    cash = state.get("cash", 100_000.0)
    realised_pnl = state.get("realised_pnl", 0.0)
    transactions = state.get("transactions", [])

    # Compute total portfolio value
    portfolio_value = cash + sum(
        holdings[t]["quantity"] * securities_data[t]["market_data"]["price"]
        for t in holdings if t in securities_data
    )

    for record in research_summary:
        ticker = record["ticker"]
        approved = record["approved"]
        allocation_pct = record.get("target_allocation_percent", 0.0) / 100.0
        price = securities_data.get(ticker, {}).get("market_data", {}).get("price")

        if price is None or price <= 0.0:
            continue

        target_value = allocation_pct * portfolio_value
        holding = holdings.get(ticker, {"quantity": 0.0, "buy_price": price})
        current_qty = holding["quantity"]
        current_value = current_qty * price

        analyst_rec = analysis_summary.get(ticker, {}).get("recommendation", "HOLD")

        # --- SELL (Full or Rebalance) ---
        if approved and analyst_rec == "SELL":
            if current_qty > 0:
                proceeds = current_qty * price
                original_cost = current_qty * holding["buy_price"]
                realized_pnl += proceeds - original_cost
                cash += proceeds
                transactions.append({
                    "ticker": ticker,
                    "type": "SELL",
                    "price": price,
                    "quantity": current_qty,
                    "total": proceeds
                })
                del holdings[ticker]
                print(f"Sold {current_qty:.2f} of {ticker} (SELL recommendation)")
            continue

        if not approved:
            continue  # Skip unapproved tickers

        # --- Rebalance Logic ---
        diff = current_value - target_value

        if diff > 0.01:  # Over-allocated → SELL
            amount_to_sell = diff
            qty_to_sell = amount_to_sell / price
            qty_to_sell = min(qty_to_sell, current_qty)

            proceeds = qty_to_sell * price
            original_cost = qty_to_sell * holding["buy_price"]
            realised_pnl += proceeds - original_cost
            cash += proceeds
            remaining_qty = current_qty - qty_to_sell

            if remaining_qty <= 0:
                del holdings[ticker]
            else:
                holdings[ticker]["quantity"] = remaining_qty

            transactions.append({
                "ticker": ticker,
                "type": "SELL",
                "price": price,
                "quantity": qty_to_sell,
                "total": proceeds
            })
            print(f"Rebalanced: Sold {qty_to_sell:.2f} of {ticker} (over-allocated)")

        elif diff < -0.01:  # Under-allocated → BUY
            amount_to_buy = min(cash, -diff)
            qty_to_buy = amount_to_buy / price
            new_total_qty = current_qty + qty_to_buy
            new_total_cost = (current_qty * holding["buy_price"]) + (qty_to_buy * price)
            new_avg_price = new_total_cost / new_total_qty if new_total_qty > 0 else price

            holdings[ticker] = {
                "quantity": new_total_qty,
                "buy_price": new_avg_price
            }
            cash -= amount_to_buy

            transactions.append({
                "ticker": ticker,
                "type": "BUY",
                "price": price,
                "quantity": qty_to_buy,
                "total": amount_to_buy
            })
            print(f"Rebalanced: Bought {qty_to_buy:.2f} of {ticker} (under-allocated)")

        else:
            print(f"No rebalancing needed for {ticker} (within target)")

    trader_message = HumanMessage(
        content=f"""Rebalanced portfolio.\nHoldings:\n```json\n{
            json.dumps(holdings, indent=2)
        }\n```\n
        Transactions:\n```json\n{json.dumps(transactions, indent=2)}\n```""",
        name="trader"
    )

    return {
        **state,
        "holdings": holdings,
        "cash": cash,
        "realised_pnl": realised_pnl,
        "transactions": transactions,
        "messages": state["messages"] + [trader_message],
    }

def portfolio_node(state: dict) -> dict:
    """Save Portfolio Data and display a P&L summary."""

    holdings = state.get("holdings", {})
    securities_data = state.get("securities_data", {})
    cash = state.get("cash", 0.0)
    realised_pnl = state.get("realised_pnl", 0.0)

    # Compute values
    total_market_value = 0.0
    unrealised_pnl = 0.0
    summary_lines = []

    for ticker, pos in holdings.items():
        quantity = pos.get("quantity", 0)
        buy_price = pos.get("buy_price", 0)
        market_price = securities_data.get(ticker, {}).get("market_data", {}).get("price", 0)

        if quantity <= 0 or market_price <= 0:
            continue

        market_value = quantity * market_price
        cost_basis = quantity * buy_price
        unrealised = market_value - cost_basis

        total_market_value += market_value
        unrealised_pnl += unrealised

        summary_lines.append(
            f"{ticker}: Qty={quantity:.2f}, Buy=${buy_price:.2f}, Market=${market_price:.2f}, "
            f"Value=${market_value:,.2f}, Unrealised PnL=${unrealised:,.2f}"
        )

    total_portfolio_value = total_market_value + cash
    total_pnl = realised_pnl + unrealised_pnl

    pnl_summary = (
        f"--- Portfolio Summary ---\n"
        f"Cash: ${cash:,.2f}\n"
        f"Market Value: ${total_market_value:,.2f}\n"
        f"Realised PnL: ${realised_pnl:,.2f}\n"
        f"Unrealised PnL: ${unrealised_pnl:,.2f}\n"
        f"Total Portfolio Value: ${total_portfolio_value:,.2f}\n"
        f"Total PnL: ${total_pnl:,.2f}\n"
        f"-------------------------\n\n" +
        "\n".join(summary_lines)
    )

    # Append message
    save_message = HumanMessage(
        content=f"Portfolio PnL.\n```text\n{pnl_summary}\n```",
        name="show_portfolio"
    )

    return {
        **state,
        "messages": state["messages"] + [save_message]
    }

def store_node(state: dict, config: dict) -> dict:
    """Save Portfolio Data"""
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    store = config.get("configurable", {}).get("store", None)
    print(thread_id)
    # Uncomment to delete portfolio tables from store (Useful for corrupted data)
    # store.delete(("portfolio", thread_id), "cash")
    # store.delete(("portfolio", thread_id), "realized_pnl")
    # store.delete(("portfolio", thread_id), "transactions")
    # store.delete(("portfolio", thread_id), "holdings")
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")

    store.put(
        ("portfolio", thread_id), "holdings", state.get("holdings", {})
    )
    store.put(
        ("portfolio", thread_id), "transactions", {"transactions": state.get("transactions", [])}
    )
    store.put(
        ("portfolio", thread_id), "cash", {"cash": state.get("cash", 100_000.0)}
    )
    store.put(
        ("portfolio", thread_id), "realized_pnl", {"realized_pnl": state.get("realized_pnl", 0.0)}
    )

    print(f"[{thread_id}] Portfolio saved: holdings, transactions, cash, and P&L.")
    return state

tools = [search, search_news, scan_market, fetch_securities_data]
tool_node = ToolNode(tools)
