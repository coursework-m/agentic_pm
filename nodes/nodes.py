"""
Agentic PM Nodes"""
import os
import json
import traceback
from datetime import datetime
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage #, ToolMessage
from workflow.state import CustomState
from tools.tools import search, search_news, scan_market, fetch_securities_data
from prompts.prompts import (
    analyst_system_prompt,
    analyst_prompt,
    researcher_system_prompt,
    researcher_prompt
)
from prompts.react_prompts import react_researcher_prompt
from utils.utils import parse_summary
from utils.constants import (
    TICKERS,
    OUTPUT_DIR2,
    OUTPUT_DIR3,
    OUTPUT_DIR4,
    OUTPUT_DIR5,
    OUTPUT_DIR6,
    OUTPUT_DIR7,
    OUTPUT_DIR8
)
import pandas as pd
import numpy as np

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
            print(f"[DEBUG] Loading key '{key}' for thread '{thread_id}'")
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
    unrealised_pnl = safe_get("unrealised_pnl", 0.0)
    unrealised_pnl = unrealised_pnl["unrealised_pnl"] \
        if not isinstance(unrealised_pnl, float) else unrealised_pnl
    total_market_value = safe_get("total_market_value", 0.0)
    total_market_value = total_market_value["total_market_value"] \
        if not isinstance(total_market_value, float) else total_market_value
    total_pnl = safe_get("total_pnl", 0.0)
    total_pnl = total_pnl["total_pnl"] \
        if not isinstance(total_pnl, float) else total_pnl
    print(f"[{thread_id}] Portfolio loaded from store.")

    return {
        **state,
        "holdings": holdings,
        "total_market_value": total_market_value,
        "total_pnl": total_pnl,
        "transactions": transactions,
        "cash": cash,
        "realised_pnl": realised_pnl,
        "unrealised_pnl": unrealised_pnl,
        "portfolio_summary": state.get("portfolio_summary", {}),
        "messages": state.get("messages", [])[-1:]  # Retain just last message
    }

def data_node(state: CustomState, config: dict) -> dict:
    """Data Node"""
    backtest = config.get("configurable", {}).get("backtest", False)
    today = config.get("configurable", {}).get("today", datetime.now())

    if backtest:
        try:
            # For backtest, we load the pre-saved securities data
            filename = os.path.join(OUTPUT_DIR2, f"{today}.json")
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Backtest data file not found: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded backtest data from {filename}")
            print(f"Data type is: {type(data)}")
        except FileNotFoundError as e:
            print(f"Error loading backtest data: {e}")
            raise e
    else:
        # For live data, we fetch the latest securities data
        print("Fetching live securities data...")
        data = fetch_securities_data.invoke({"tickers": TICKERS})
        print("Loaded live data")
        print(f"Data type is: {type(data)}")
    # Serialize the original data structure as JSON string
    json_str = json.dumps(data, indent=2)

    # Create an AIMessage with the raw data
    data_message = AIMessage(
        content=f"Here is the securities data:\n{json_str}",
        name="data node"
    )

    # Return updated state including the message and stored raw data
    return {
        **state,
        "tickers": TICKERS,
        "securities_data": data,
        "messages": state["messages"] + [data_message]
    }

def analyst_node(state: CustomState, config: dict) -> dict:
    """Call Analyst Node"""
    analyst_config = config
    today = config.get("configurable", {}).get("today", datetime.now())
    securities_data = state["securities_data"]
    llm = analyst_config["configurable"]["llm"]

    analyst_summary = []
    analyst_responses = []
    contents = []
    for sec in securities_data:
        sec_data_str = json.dumps(sec, indent=2)
        # Add data to the analyst prompt
        ticker_message = HumanMessage(
            content=f"""Today's date is {today}.\n
            Here is the securities data:\n{sec_data_str}""",
            name="analyst"
        )
        analyst_messages = [analyst_system_prompt, analyst_prompt, ticker_message]
        analyst_response = llm.invoke(analyst_messages, analyst_config)
        content = analyst_response.content
        parsed = parse_summary(content)
        # Append to results as a dict
        analyst_summary.append(parsed)
        analyst_responses.append(analyst_response)
        contents.append(content)

    filename = os.path.join(OUTPUT_DIR3, f"{today}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analyst_summary, f, indent=2)
    with open(f"{OUTPUT_DIR7}/{today}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(contents))
    return {
        **state,
        "analysis_summary": analyst_summary,
        "analyst_response": contents,
        "messages": state["messages"] + analyst_responses
    }

def researcher_node(state: CustomState, config: dict) -> dict:
    """Call Research Node"""
    researcher_config = config
    today = config.get("configurable", {}).get("today", datetime.now())
    securities_data = {k: v for d in state.get("securities_data", []) for k, v in d.items()}
    analysis_summary = state["analysis_summary"]
    llm = researcher_config["configurable"]["llm"]
    react = researcher_config["configurable"].get("react", False)
    content = str
    research_summary = []
    research_responses = []
    contents = []
    for analysis in analysis_summary:
        if analysis is None:
            continue
        ticker = analysis["ticker"]
        analyst_data_str = json.dumps(analysis, indent=2)
        sec_data_str = json.dumps(securities_data.get(ticker, {}), indent=2)

        if react:
            react_research_message = PromptTemplate(
                input_variables=react_researcher_prompt.input_variables,
                template=react_researcher_prompt.template + f"""Today's date is {today}.\n
                Here is the Security data:\n{sec_data_str}\n
                Here is the Analyst's summary:\n{analyst_data_str}"""
            )
            research_response = llm.invoke({"input": react_research_message}, researcher_config)
            if isinstance(research_response, dict):
                content = research_response.get("response") \
                    or research_response.get("return_values", {}).get("output") \
                    or next(iter(research_response.values()), None)
        else:
            ticker_message = HumanMessage(
                content=f"""Today's date is {today}.\n
                Here is the Security data:\n{sec_data_str}\n
                Here is the Analyst's summary:\n{analyst_data_str}""",
                name="researcher"
            )
            research_messages = [researcher_system_prompt, researcher_prompt, ticker_message]
            research_response = llm.invoke(research_messages, researcher_config)
            content = research_response.content

        parsed = parse_summary(content)
        research_summary.append(parsed)
        research_responses.append(research_response)
        contents.append(content)

    filename = os.path.join(OUTPUT_DIR4, f"{today}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(research_summary, f, indent=2)
    with open(f"{OUTPUT_DIR8}/{today}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(contents))
    return {
        **state,
        "research_summary": research_summary,
        "researcher_response": contents,
        "messages": state["messages"] + research_responses
    }

def trader_node(state: CustomState, config: dict) -> dict:
    """Call Trader node with full rebalance logic"""
    try:
        research_summary = state["research_summary"]
    except KeyError as e:
        return {
            "messages": state["messages"] + [AIMessage(
                content=f"Error parsing research summary: {e}", name="trader")]
        }
    today = config.get("configurable", {}).get("today", datetime.now())
    securities_data = {k: v for d in state.get("securities_data", []) for k, v in d.items()}
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
        if record['ticker']:
            ticker = record["ticker"]
        else:
            continue
        approved = record["approved"]
        allocation_pct = record.get("target_allocation_percent", 0.0) / 100.0
        price = securities_data.get(ticker, {}).get("market_data", {}).get("price", 0)

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
                realised_pnl += proceeds - original_cost
                cash += proceeds
                transactions.append({
                    "ticker": ticker,
                    "type": "SELL",
                    "price": price,
                    "quantity": current_qty,
                    "total": proceeds,
                    "date": today
                })
                del holdings[ticker]
                print(f"Sold {current_qty:.2f} of {ticker} (SELL recommendation)")
            continue

        if not approved:
            continue  # Skip unapproved tickers

        # Rebalance Logic
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
                "total": proceeds,
                "date": today
            })
            print(f"Rebalanced: Sold {qty_to_sell:.2f} of {ticker} (over-allocated)")

        elif diff < -0.01:  # Under allocated → BUY
            amount_to_buy = min(cash, -diff)
            qty_to_buy = amount_to_buy / price
            new_total_qty = current_qty + qty_to_buy
            new_total_cost = (current_qty * holding["buy_price"]) + (qty_to_buy * price)
            new_avg_price = new_total_cost / new_total_qty if new_total_qty > 0 else price

            holdings[ticker] = {
                "quantity": new_total_qty,
                "buy_price": new_avg_price,
                "date": today
            }
            cash -= amount_to_buy

            transactions.append({
                "ticker": ticker,
                "type": "BUY",
                "price": price,
                "quantity": qty_to_buy,
                "total": amount_to_buy,
                "date": today
            })
            print(f"Rebalanced: Bought {qty_to_buy:.2f} of {ticker} (under-allocated)")

        else:
            print(f"No rebalancing needed for {ticker} (within target)")

    filename = os.path.join(OUTPUT_DIR5, f"{today}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2)
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

def portfolio_node(state: dict, config: dict) -> dict:
    """Save Portfolio Data and display a P&L summary."""

    holdings = state.get("holdings", {})
    transactions = state.get("transactions", [])
    securities_data = {k: v for d in state.get("securities_data", []) for k, v in d.items()}
    cash = state.get("cash", 0.0)
    realised_pnl = state.get("realised_pnl", 0.0)
    today = config.get("configurable", {}).get("today", datetime.now().strftime('%Y%m%d'))
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
        f"TotalMarket Value: ${total_market_value:,.2f}\n"
        f"Realised PnL: ${realised_pnl:,.2f}\n"
        f"Unrealised PnL: ${unrealised_pnl:,.2f}\n"
        f"Total Portfolio Value: ${total_portfolio_value:,.2f}\n"
        f"Total PnL: ${total_pnl:,.2f}\n"
        f"-------------------------\n\n" +
        "\n".join(summary_lines)
    )
    pnl_summary_dict = {}
    pnl_summary_dict = {
        "cash": cash,
        "total_market_value": total_market_value,
        "realised_pnl": realised_pnl,
        "unrealised_pnl": unrealised_pnl,
        "total_portfolio_value": total_portfolio_value,
        "total_pnl": total_pnl,
        "holdings": holdings,
        "transactions": transactions,
        "date": today,
        "portfolio_summary": pnl_summary,
    }

    # Save PnL summary to file
    filename = os.path.join(OUTPUT_DIR6, f"{today}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(pnl_summary_dict, f, indent=2)
    print(f"Portfolio summary saved to {filename}")

    # Append message
    save_message = HumanMessage(
        content=f"Portfolio PnL.\n```text\n{pnl_summary}\n```",
        name="show_portfolio"
    )

    return {
        **state,
        "portfolio_history": pnl_summary_dict,
        **{key: value for key, value in pnl_summary_dict.items()},
        "messages": state["messages"] + [save_message]
    }

def store_node(state: dict, config: dict) -> dict:
    """Save Portfolio Data"""
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    store = config.get("configurable", {}).get("store", None)
    # Uncomment to delete portfolio tables from store (Useful for corrupted data)
    # store.delete(("portfolio", thread_id), "cash")
    # store.delete(("portfolio", thread_id), "realised_pnl")
    # store.delete(("portfolio", thread_id), "transactions")
    # store.delete(("portfolio", thread_id), "holdings")
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    store.put(
        ("portfolio_history", thread_id), "portfolio_history", state.get("portfolio_history", {})
    )
    store.put(
        ("portfolio", thread_id), "holdings", state.get("holdings", {})
    )
    store.put(
        (
            "portfolio", thread_id),
            "transactions", {"transactions": state.get("transactions", [])
        }
    )
    store.put(
        (
            "portfolio", thread_id),
            "unrealised_pnl",
            {"unrealised_pnl": state.get("unrealised_pnl", 0.0)
        }
    )
    store.put(
        ("portfolio", thread_id),
        "cash", {
            "cash": state.get("cash", 100_000.0)
        }
    )
    store.put(
        ("portfolio", thread_id),
        "realised_pnl", {
            "realised_pnl": state.get("realised_pnl", 0.0)
        }
    )
    store.put(
        ("portfolio", thread_id),
        "total_market_value", {
            "total_market_value": state.get("total_market_value", 0.0)
        }
    )
    store.put(
        ("portfolio", thread_id),
        "total_pnl", {
            "total_pnl": state.get("total_pnl", 0.0)
        }
    )
    store.put(
        ("portfolio", thread_id),
        "portfolio_summary", {
            "portfolio_summary": state.get("portfolio_summary", {})
        }
    )
    store.put(
        ("portfolio", thread_id),
        "reasoning", state.get("reasoning", [])
    )
    store.put(
        ("portfolio", thread_id),
        "analysis_summary", state.get("analysis_summary", [])
    )
    store.put(
        ("portfolio", thread_id),
        "research_summary", state.get("research_summary", [])
    )
    store.put(
        ("portfolio", thread_id),
        "analysis_response", state.get("analysis_response", "")
    )
    store.put(
        ("portfolio", thread_id),
        "research_response", state.get("research_response", "")
    )

    print(f"[{thread_id}] Portfolio saved to store.")
    return state

def performance_node(config: dict, store) -> dict:
    """A node to calculate performance"""
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")

    # Load historical portfolio values
    history = store.get(("portfolio_history", thread_id))  # daily snapshots
    df = pd.DataFrame(history)  # Expected cols: date, total_portfolio_value
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # Calculate daily returns
    df['daily_return'] = df['total_portfolio_value'].pct_change()

    # Metrics
    avg_daily_return = df['daily_return'].mean()
    daily_volatility = df['daily_return'].std()

    sharpe_ratio = (
        avg_daily_return / daily_volatility
    ) * np.sqrt(252) if daily_volatility != 0 else np.nan

    # Drawdown
    df['cum_max'] = df['total_portfolio_value'].cummax()
    df['drawdown'] = df['total_portfolio_value'] / df['cum_max'] - 1
    max_drawdown = df['drawdown'].min()

    # CAGR
    total_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365
    cagr = (
        df['total_portfolio_value'].iloc[-1] / df['total_portfolio_value'].iloc[0]
    )**(1/total_years) - 1

    # Calmar ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "calmar_ratio": calmar_ratio,
        "volatility": daily_volatility
    }

    # Store metrics
    store.put(("portfolio_metrics", thread_id), "metrics", metrics)

    print(f"[{thread_id}] Performance metrics updated: {metrics}")
    return metrics

tools = [search, search_news, scan_market, fetch_securities_data]
tool_node = ToolNode(tools)
