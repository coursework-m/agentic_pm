"""
Agentic PM Nodes"""
import os
import json
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from utils.constants import TICKERS
# Define the path for the backtest or forwardtest
# This can be set to "backtest" or "forwardtest" based on the context
BACKTEST = True # Set to True for backtesting, False for forward testing
PATH = "backtest" if BACKTEST else "forwardtest"
# Directory for saving output
OUTPUT_DIR = f"./data/{PATH}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR10 = f"{OUTPUT_DIR}/messages"
os.makedirs(OUTPUT_DIR10, exist_ok=True)

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
    daily_pnl = safe_get("daily_pnl", 0.0)
    daily_pnl = daily_pnl["daily_pnl"] if not isinstance(daily_pnl, float) else daily_pnl
    unrealised_pnl = safe_get("unrealised_pnl", 0.0)
    unrealised_pnl = unrealised_pnl["unrealised_pnl"] \
        if not isinstance(unrealised_pnl, float) else unrealised_pnl
    total_market_value = safe_get("total_market_value", 0.0)
    total_market_value = total_market_value["total_market_value"] \
        if not isinstance(total_market_value, float) else total_market_value
    total_pnl = safe_get("total_pnl", 0.0)
    total_pnl = total_pnl["total_pnl"] \
        if not isinstance(total_pnl, float) else total_pnl
    portfolio_history = safe_get("portfolio_history", [])
    portfolio_history = portfolio_history["portfolio_history"] \
        if not isinstance(portfolio_history, list) else portfolio_history
    memory_message = AIMessage(
        content=f"[{thread_id}] Portfolio loaded from store.",
        name="memory node"
    )

    return {
        **state,
        "holdings": holdings,
        "total_market_value": total_market_value,
        "total_pnl": total_pnl,
        "transactions": transactions,
        "cash": cash,
        "unrealised_pnl": unrealised_pnl,
        "portfolio_history": portfolio_history,
        "messages": state["messages"] + [memory_message]
    }

def data_node(state: CustomState, config: dict) -> dict:
    """Data Node"""
    backtest = config.get("configurable", {}).get("backtest", False)
    today = config.get("configurable", {}).get("today", datetime.now())
    # thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    if backtest:
        try:
            # For backtest, we load the pre-saved securities data
            OUTPUT_DIR2 = f"{OUTPUT_DIR}/securities_data"
            os.makedirs(OUTPUT_DIR2, exist_ok=True)
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
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    securities_data = state["securities_data"]
    llm = analyst_config["configurable"]["llm"]
    model_config = config.get("configurable", {}).get("model_config", {})
    model_name = model_config["model"]

    analyst_summary = []
    analyst_responses = []
    contents = []
    for sec in securities_data:
        sec_data_str = json.dumps(sec, indent=2)
        # Add data to the analyst prompt
        ticker_message = HumanMessage(
            content=f"""Today's date is {today}.\n
            Here is the securities data:\n{sec_data_str}""",
            name=f"{model_name} analyst node"
        )
        analyst_messages = [analyst_system_prompt, analyst_prompt, ticker_message]
        analyst_response = llm.invoke(analyst_messages, analyst_config)
        content = analyst_response.content
        parsed = parse_summary(content)
        # Append to results as a dict
        analyst_summary.append(parsed)
        analyst_responses.append(analyst_response)
        contents.append(content)

    OUTPUT_DIR3 = f"{OUTPUT_DIR}/{model_name}/analysts_summary"
    os.makedirs(OUTPUT_DIR3, exist_ok=True)
    OUTPUT_DIR7 = f"{OUTPUT_DIR}/{model_name}/analyst_response"
    os.makedirs(OUTPUT_DIR7, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR3 + f"/{thread_id}", f"{today}.json")
    os.makedirs(OUTPUT_DIR3 + f"/{thread_id}", exist_ok=True)
    os.makedirs(OUTPUT_DIR7 + f"/{thread_id}", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analyst_summary, f, indent=2)
    with open(f"{OUTPUT_DIR7}/{thread_id}/{today}.txt", "w", encoding="utf-8") as f:
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
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    securities_data = {k: v for d in state.get("securities_data", []) for k, v in d.items()}
    analysis_summary = state["analysis_summary"]
    llm = researcher_config["configurable"]["llm"]
    react = researcher_config["configurable"].get("react", False)
    model_config = config.get("configurable", {}).get("model_config", {})
    model_name = model_config["model"]
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
                name=f"{model_config['model']} researcher node"
            )
            research_messages = [researcher_system_prompt, researcher_prompt, ticker_message]
            research_response = llm.invoke(research_messages, researcher_config)
            content = research_response.content

        parsed = parse_summary(content)
        research_summary.append(parsed)
        research_responses.append(research_response)
        contents.append(content)

    OUTPUT_DIR4 = f"{OUTPUT_DIR}/{model_name}/research_summary"
    os.makedirs(OUTPUT_DIR4, exist_ok=True)
    OUTPUT_DIR8 = f"{OUTPUT_DIR}/{model_name}/research_response"
    os.makedirs(OUTPUT_DIR8, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR4 + f"/{thread_id}", f"{today}.json")
    os.makedirs(OUTPUT_DIR4 + f"/{thread_id}", exist_ok=True)
    os.makedirs(OUTPUT_DIR8 + f"/{thread_id}", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(research_summary, f, indent=2)
    with open(f"{OUTPUT_DIR8}/{thread_id}/{today}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(contents))
    return {
        **state,
        "research_summary": research_summary,
        "researcher_response": contents,
        "messages": state["messages"] + research_responses
    }


def trader_node(state: CustomState, config: dict) -> dict:
    """Execute rebalancing and trades"""
    today = config.get("configurable", {}).get("today", datetime.now())
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    research_summary = state.get("research_summary", [])
    securities_data = {k: v for d in state.get("securities_data", []) for k, v in d.items()}
    holdings = state.get("holdings", {})
    cash = state.get("cash", 0.0)
    transactions = state.get("transactions", [])

    prices = {
        ticker: data.get("market_data", {}).get("price") for ticker, data in securities_data.items()
    }
    holdings_value = sum(pos["quantity"] * prices.get(t, 0.0) for t, pos in holdings.items())
    portfolio_value = holdings_value + cash
    daily_pnl = 0.0
    for rec in research_summary:
        if rec is None or "ticker" not in rec:
            continue
        daily_proceeds = 0.0
        ticker = rec["ticker"]
        target_pct = rec.get("target_allocation_percent", 0.0)
        approved = rec.get("approved", False)
        price = prices.get(ticker)

        if not approved or price is None or price <= 0:
            continue

        current_qty = holdings.get(ticker, {}).get("quantity", 0.0)
        buy_price = holdings.get(ticker, {}).get("buy_price", 0.0)
        current_value = current_qty * price
        # 0% allocation → Sell all
        if target_pct == 0.0:
            if current_qty > 0:
                revenue = current_qty * price
                cash += revenue
                proceeds = (price - buy_price) * current_qty
                daily_proceeds += proceeds
                daily_pnl += daily_proceeds
                transactions.append({
                    "ticker": ticker,
                    "action": "sell_all",
                    "quantity": current_qty,
                    "price": price,
                    "pnl": proceeds,
                    "date": today
                })
                del holdings[ticker]
            continue

        target_value = (target_pct / 100) * portfolio_value
        diff_value = target_value - current_value

        if abs(diff_value) < 1e-6:
            continue

        qty_change = diff_value / price

        if qty_change > 0:  # Buy
            cost = qty_change * price
            if cost > cash:
                qty_change = cash / price
                cost = qty_change * price

            total_cost_existing = current_qty * buy_price
            total_cost_new = qty_change * price
            new_qty_total = current_qty + qty_change
            new_avg_cost = (
                total_cost_existing + total_cost_new
            ) / new_qty_total if new_qty_total > 0 else 0.0

            holdings[ticker] = {
                "quantity": new_qty_total,
                "buy_price": new_avg_cost,
                "last_updated": today
            }
            cash -= cost
            transactions.append({
                "ticker": ticker,
                "action": "buy",
                "quantity": qty_change,
                "price": price,
                "date": today
            })

        elif qty_change < 0:  # Sell
            sell_qty = min(abs(qty_change), current_qty)
            revenue = sell_qty * price
            cash += revenue
            proceeds = (price - buy_price) * sell_qty
            daily_proceeds += proceeds
            daily_pnl += daily_proceeds
            holdings[ticker]["quantity"] -= sell_qty
            if holdings[ticker]["quantity"] <= 0:
                del holdings[ticker]

            transactions.append({
                "ticker": ticker,
                "action": "sell",
                "quantity": sell_qty,
                "price": price,
                "pnl": proceeds,
                "date": today
            })
    state.update({
        "holdings": holdings,
        "cash": cash,
        "prices": prices,
        "transactions": transactions,
        "daily_pnl": daily_pnl,
    })

    OUTPUT_DIR5 = f"{OUTPUT_DIR}/transactions"
    os.makedirs(OUTPUT_DIR5, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR5 + f"/{thread_id}", f"{today}.json")
    os.makedirs(OUTPUT_DIR5 + f"/{thread_id}", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2)

    trader_message = HumanMessage(
        content=f"""Rebalanced portfolio.\nHoldings:\n```json\n{
            json.dumps(holdings, indent=2)
        }\n```\n
        Transactions:\n```json\n{json.dumps(transactions, indent=2)}\n```""",
        name="trader node"
    )

    return {
        **state,
        "holdings": holdings,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "transactions": transactions,
        "messages": state["messages"] + [trader_message],
    }

def portfolio_node(state: CustomState, config: dict) -> dict:
    """Calculate portfolio metrics: realised PnL, unrealised PnL, and total value."""
    today = config.get("configurable", {}).get("today", datetime.now().strftime('%Y%m%d'))
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    holdings = state.get("holdings", {})
    transactions = state.get("transactions", [])
    history = state.get("portfolio_history", [])
    cash = state.get("cash", 0.0)
    daily_pnl = state.get("daily_pnl", 0.0)
    prices = state.get("prices", {})
    holdings_value = 0.0
    unrealised_pnl = 0.0

    for ticker, pos in holdings.items():
        qty = pos["quantity"]
        buy_price = pos["buy_price"]
        market_price = prices.get(ticker)

        if market_price is None or qty <= 0:
            continue

        position_value = qty * market_price
        holdings_value += position_value

        unrealised_pnl += (market_price - buy_price) * qty

    cum_realised_pnl = sum(
        t["pnl"]
        for t in transactions
        if t["action"] == "sell"
    )

    portfolio_value = holdings_value + cash
    total_pnl = cum_realised_pnl + unrealised_pnl


    state.update({
        "holdings_value": holdings_value,
        "portfolio_value": portfolio_value,
        "unrealised_pnl": unrealised_pnl,
        "total_pnl": total_pnl,
        "cash": cash
    })

    snapshot = {
        "timestamp": today,
        "portfolio_value": portfolio_value,
        "holdings_value": holdings_value,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "cum_realised_pnl": cum_realised_pnl,
        "unrealised_pnl": unrealised_pnl,
        "total_pnl": total_pnl,
        "holdings": holdings,
        "transactions": transactions
    }

    history.append(snapshot)

    portfolio_message = HumanMessage(
        content=(
            f"  Portfolio metrics for {today}:\n"
            f"  Portfolio value: {portfolio_value:,.2f}\n"
            f"  Cash: {cash:,.2f}\n"
            f"  Holdings value: {holdings_value:,.2f}\n"
            f"  Realised PnL: {cum_realised_pnl:,.2f}\n"
            f"  Daily PnL: {daily_pnl:,.2f}\n"
            f"  Unrealised PnL: {unrealised_pnl:,.2f}\n"
            f"  Total PnL: {total_pnl:,.2f}\n"
            f"  Transactions: {transactions}\n"
        ),
        name="portfolio node"
    )

    # Save PnL history to file
    OUTPUT_DIR6 = f"{OUTPUT_DIR}/portfolio_history"
    os.makedirs(OUTPUT_DIR6, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR6 + f"/{thread_id}", f"{today}.json")
    os.makedirs(OUTPUT_DIR6 + f"/{thread_id}", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Portfolio history saved to {filename}")

    return {
        **state,
        "holdings_value": holdings_value,
        "portfolio_value": portfolio_value,
        "unrealised_pnl": unrealised_pnl,
        "cum_realised_pnl": cum_realised_pnl,
        "total_pnl": total_pnl,
        "cash": cash,
        "porfolio_history": history,
        "messages": state["messages"] + [portfolio_message]

    }

def store_node(state: CustomState, config: dict) -> dict:
    """Save Portfolio Data"""
    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    store = config.get("configurable", {}).get("store", None)

    store.put(
        (
            "portfolio", thread_id), 
            "holdings", state.get("holdings", {})
        )
    store.put(
        (
            "portfolio", thread_id), 
            "transactions", {"transactions": state.get("transactions", [])}
        )
    store.put(
        (
            "portfolio", thread_id),
            "unrealised_pnl", {"unrealised_pnl": state.get("unrealised_pnl", 0.0)}
        )
    store.put(
        (
            "portfolio", thread_id),
            "cash", {"cash": state.get("cash", 100_000.0)}
        )
    store.put(
        (
            "portfolio", thread_id),
            "daily_pnl", {"daily_pnl": state.get("daily_pnl", 0.0)}
        )
    store.put(
        (
            "portfolio", thread_id),
            "total_market_value", {"total_market_value": state.get("total_market_value", 0.0)}
        )
    store.put(
        (
            "portfolio", thread_id),
            "total_pnl", {"total_pnl": state.get("total_pnl", 0.0)}
        )
    store.put(
        (
            "portfolio", thread_id),
            "reasoning", state.get("reasoning", [])
        )
    store.put(
        (
            "portfolio", thread_id),
            "analysis_summary", state.get("analysis_summary", [])
        )
    store.put(
        (
            "portfolio", thread_id),
            "research_summary", state.get("research_summary", [])
        )
    store.put(
        (
            "portfolio", thread_id),
            "analysis_response", state.get("analysis_response", "")
        )
    store.put(
        (
            "portfolio", thread_id),
            "research_response", state.get("research_response", "")
        )
    store.put(
        (
            "portfolio", thread_id),
            "portfolio_history", {"portfolio_history": state.get("portfolio_history", [])}
        )
    store_message = HumanMessage(
        content=(
            f"[{thread_id}] Portfolio saved to store."
        ),
        name="store node"
    )
    return {
        **state,
        "messages": state["messages"] + [store_message]
    }

def performance_node(state: CustomState, config: dict, store) -> dict:
    """A node to calculate performance"""
    def plot_drawdown(metrics_df, drawdown_info):
        """Plot the drawdown metric"""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot portfolio value
        ax1.plot(
            metrics_df.index, metrics_df['portfolio_value'], color='blue', label='Portfolio Value')
        ax1.set_ylabel('Portfolio Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Highlight peak, trough, and recovery
        ax1.scatter(drawdown_info['peak_date'],
                    metrics_df.loc[drawdown_info['peak_date'], 'portfolio_value'],
                    color='green', marker='^', s=100, label='Peak')

        ax1.scatter(drawdown_info['trough_date'],
                    metrics_df.loc[drawdown_info['trough_date'], 'portfolio_value'],
                    color='red', marker='v', s=100, label='Trough')

        if drawdown_info['recovery_date'] is not None:
            ax1.scatter(drawdown_info['recovery_date'],
                        metrics_df.loc[drawdown_info['recovery_date'], 'portfolio_value'],
                        color='orange', marker='o', s=100, label='Recovery')

        # Drawdown subplot
        ax2 = ax1.twinx()
        ax2.fill_between(metrics_df.index, metrics_df['drawdown_pct'] * 100, 0,
                        color='grey', alpha=0.3, label='Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')

        # Title and legends
        plt.title(f"Max Drawdown: {drawdown_info['max_drawdown_pct']*100:.2f}% "
                f"(${drawdown_info['max_drawdown_dollars']:.2f})")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        fig.savefig("drawdown_chart.png", dpi=300, bbox_inches='tight')


    thread_id = config.get("configurable", {}).get("thread_id", "default-thread")
    today = config.get("configurable", {}).get("today", datetime.now().strftime('%Y%m%d'))
    backtest = config.get("configurable", {}).get("backtest", datetime.now().strftime('%Y%m%d'))
    end_date = config.get("configurable", {}).get("end_date", datetime.now().strftime('%Y%m%d'))
    metrics = {}
    performance_message = HumanMessage(
        content=(
            f"  [{thread_id}] Performance metrics only calculated for backtests"
        ),
        name="performance node"
    )
    if backtest and end_date == today:
        # Load historical portfolio values
        history = state.get("portfolio_history", []) # daily snapshots
        metrics_df = pd.DataFrame(history)
        print(metrics_df.head())
        print(metrics_df.tail())
        metrics_df['date'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df.sort_values('date', inplace=True)
        # Sharpe ratio (https://en.wikipedia.org/wiki/Sharpe_ratio)
        # In finance, the Sharpe ratio (also known as the Sharpe index, the Sharpe measure,
        # and the reward-to-variability ratio) measures the performance of an investment
        # such as a security or portfolio compared to a risk-free asset, after adjusting
        # for its risk.
        metrics_df['daily_return'] = metrics_df['portfolio_value'].pct_change()
        risk_free_annual = 0.05  # example 5% annual risk-free rate
        risk_free_daily = risk_free_annual / 252
        # Average daily return
        # Mean percent chage of the portfolio value
        avg_daily_excess_return = metrics_df['daily_return'].mean() - risk_free_daily
        # Daily Volatility (https://www.stockopedia.com/ratios/daily-volatility-12000)
        # The Daily Volatility of a security is the standard deviation
        # of its daily return.
        daily_volatility = metrics_df['daily_return'].std()
        # Sharpe rartio;
        if daily_volatility != 0:
            sharpe_ratio = (avg_daily_excess_return / daily_volatility) * np.sqrt(252)
        else:
            sharpe_ratio = np.nan

        metrics_df['sharpe_ratio'] = sharpe_ratio

        # Drawdown (https://en.wikipedia.org/wiki/Drawdown_(economics))
        # Drawdown is the measure of the decline from a historical peak in some
        # variable (typically the cumulative profit or total open equity of a financial
        # trading strategy. The maximum drawdown (MDD) up to time T is the maximum of the
        # drawdown over the history of the variable.
        # CumMax finds running Peak
        metrics_df['cum_max'] = metrics_df['portfolio_value'].cummax()

        # Percentage drawdown (will be negative from peak)
        metrics_df['drawdown_pct'] = metrics_df['portfolio_value'] / metrics_df['cum_max'] - 1

        # Cash drawdown value
        metrics_df['drawdown_cash'] = metrics_df['portfolio_value'] - metrics_df['cum_max']

        # Max drawdown percentage and cash value
        max_dd_pct = -metrics_df['drawdown_pct'].min()
        max_dd_cash = -metrics_df['drawdown_cash'].min()

        # Find trough date (max drawdown point)
        trough_date = metrics_df['drawdown_pct'].idxmin()

        # Find peak date
        peak_date = metrics_df.loc[:trough_date, 'portfolio_value'].idxmax()

        # Find recovery date (if any)
        recovery_mask = metrics_df.loc[trough_date:, 'portfolio_value'] \
        >= metrics_df.loc[peak_date, 'portfolio_value']
        recovery_date = recovery_mask[recovery_mask].index.min() if recovery_mask.any() else None

        # Drawdown metrics for chart
        drawdown_info = {
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_dollars': max_dd_cash,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date
        }

        # Only possible where data is grouped by year - TODO: add a yearly option
        # CAGR (https://en.wikipedia.org/wiki/Compound_annual_growth_rate)
        # (https://www.investopedia.com/terms/c/cagr.asp%23)
        # To calculate the CAGR of an investment: Divide the value of an investment at the end
        # of the period by its value at the beginning of that period. Raise the result to an
        # exponent of one divided by the number of years. Subtract one from the subsequent result.
        # total_years = (metrics_df['date'].iloc[-1] - metrics_df['date'].iloc[0]).days / 365
        # cagr = (
        #     metrics_df['portfolio_value'].iloc[-1] / metrics_df['portfolio_value'].iloc[0]
        # )**(1/total_years) - 1

        # Calmar ratio (https://en.wikipedia.org/wiki/Calmar_ratio)
        # calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

        metrics["sharpe_ratio"] = sharpe_ratio
        metrics["max_drawdown"] = max_dd_pct
        metrics["max_drawdown_cash"] = max_dd_cash
        metrics["daily_volatility"] = daily_volatility
        # metrics["cagr"] = cagr
        # metrics["calmar_ratio"] = calmar_ratio
        # Save Performance Metrics history to file as JSON
        filename = os.path.join(OUTPUT_DIR9 + f"/{thread_id}", f"{today}.json")
        os.makedirs(OUTPUT_DIR9 + f"/{thread_id}", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        # Save metrics_df as CSV
        metrics_df.to_csv(OUTPUT_DIR9 + f"/{thread_id}/{today}.csv")
        print(f"Performance metrics saved to {filename}")

        # Store metrics
        store.put(("portfolio", thread_id), "metrics", metrics)

        # Plot the drawdown metrics for run
        plot_drawdown(metrics_df, drawdown_info)
        performance_message = HumanMessage(
            content=(
                f"  [{thread_id}] Performance metrics updated: {metrics}"
                f"  Performance metrics for {today}:\n"
                f"  Sharpe ratio: {sharpe_ratio:,.2f}\n"
                f"  Max drawdown %: {max_dd_pct:,.2f}\n"
                f"  Max drawdown cash: {max_dd_cash:,.2f}\n"
                f"  Daily volatility: {daily_volatility:,.2f}"
                # f"  Cagr: {cagr:,.2f}\n"
                # f"  Calmar ratio: {calmar_ratio:,.2f}\n"
            ),
            name="performance node"
        )
    return {
        **state,
        "metrics": metrics,
        "messages": state["messages"] + [performance_message]
    }

tools = [search, search_news, scan_market, fetch_securities_data]
tool_node = ToolNode(tools)
