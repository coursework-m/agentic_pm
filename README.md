# Agentic Portfolio Management (agentic_pm)

Agentic Portfolio Management (Agentic PM) is a multi-agent system for automated portfolio management, leveraging large language models (LLMs) and real-world financial data. The system simulates a team of agents (analyst, researcher, trader, etc.) collaborating to analyze market data, generate recommendations, and manage a portfolio, with a focus on accuracy and backtest performance.

## Features

- **Multi-Agent Workflow:** Modular nodes for memory, data ingestion, analysis, research, trading, and portfolio management.
- **LLM Integration:** Uses state-of-the-art LLMs (e.g., Llama 3) for financial analysis and decision-making.
- **Backtesting:** Simulate historical performance over custom date ranges.
- **Extensible Tools:** Integrates with external data sources and custom tools for market data, news, and more.
- **Checkpointing:** Supports workflow checkpointing and state persistence using PostgreSQL.

## Project Structure
. ├── backtest.py # Script for running historical backtests ├── main.py # Main entry point for daily agent workflow ├── requirements.txt # Python dependencies ├── prompts/ # System and agent prompt templates ├── models/ # LLM setup and loading ├── nodes/ # Agent node logic ├── tools/ # Data fetching and utility tools ├── utils/ # Constants and utility functions ├── workflow/ # Workflow orchestration and state management ├── data/ # Market and research data └── cron/ # Scheduled data fetching scripts


## Installation

1. **Clone the repository:**
```sh
   git clone https://github.com/yourusername/agentic_pm.git
   cd agentic_pm
```
2. **Install dependencies:**
```sh
pip install -r requirements.txt
```
3. **Set up environment variables:**
DB_URI: PostgreSQL connection string for checkpointing and state storage.
HF_TOKEN: (Optional) HuggingFace token for LLM access.
You can use a .env file or export variables in your shell.

## Usage
Daily Run
Run the daily agent workflow for the current day:
```sh
python main.py
```
To specify a custom date or enable checkpointing/backtesting, modify the call to daily_run in main.py.

## Cron Job
Set up a cron job to run the daily workflow automatically. Edit your crontab with:
```sh
crontab -e
```
Add the following line to run the script every day at 9 AM:
```sh0 9 * * * /usr/bin/python3 /path/to/agentic_pm/main.py
```
Make sure to adjust the path to your Python interpreter and the script location.
## Data Fetching
Run the data fetching script to update market and research data:
```sh
python cron/fetch_data.py
```
This script can be scheduled to run periodically (e.g., daily) to keep your data up-to-date. You can also run it manually to fetch the latest data.

## Checkpointing
Agentic PM supports workflow checkpointing using PostgreSQL. This allows you to save the state of the workflow and resume later. To enable checkpointing, ensure your DB_URI environment variable is set correctly. The system will automatically save the state after each major step in the workflow.

## Backtest Setup
To run backtests, ensure you have historical market data available in the `data/` directory. The backtest script will use this data to simulate the agent's decisions over a specified date range. You can download historical data from various financial data providers or use your own datasets.

## Backtesting
Run a backtest for a specific date range:
```sh
python backtest.py --start 2025-01-01 --end 2025-12-31
```
You can adjust the start and end dates as needed. The backtest will simulate the agent's decisions over the specified period and output performance metrics.

## Customization
Prompts: Edit agent instructions in prompts/prompts.py and prompts/react_prompts.py.
LLM Setup: Configure models in models/llm_setup.py and models/hf_setup.py.
Tools: Add or modify data tools in tools/tools.py.
Nodes: Implement custom agent logic in nodes/agent_nodes.py.
Workflow: Adjust workflow orchestration in workflow/workflow.py.

## Contributing
For academic and research use. Contributions welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Inspired by the principles of multi-agent systems and automated trading.
- Utilises advanced LLMs for financial analysis and decision-making.
- Thanks to the open-source community for tools and libraries that made this possible.