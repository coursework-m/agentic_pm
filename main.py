"""Multi agent Portfolio Management"""
import random
import string
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from utils.constants import TODAY, DB_URI
from prompts.prompts import system_prompt, start_prompt
from workflow.workflow import build_workflow
from models.llm_setup import get_llm

# Uncomment the following lines to enable debug mode for the yf module
# import yf
# yf.enable_debug_mode()

def daily_run(today=None,
              checkpoint=False,
              backtest=False,
              thread_id=None,
              llm=None,
              react=False,
              end_date=None,
              model_config=None):
    """Run the daily cycle of the Agentic PM workflow."""

    # Ensure the database URI is set correctly
    if not DB_URI:
        raise ValueError("DB_URI is not set. Please check configuration.")

    with (
        PostgresStore.from_conn_string(DB_URI) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # Uncomment the following lines on first run to set up the database.
        # store.setup()
        # checkpointer.setup()

        app = build_workflow(store, checkpointer, checkpoint=checkpoint)
        # Set the thread ID for the run
        config = {"configurable": {"thread_id": thread_id}}
        if thread_id and backtest:
            config["configurable"]["thread_id"] = thread_id
        # Set the LLM configuration
        config["configurable"]["model_config"] = model_config
        config["configurable"]["today"] = today
        config["configurable"]["end_date"] = end_date
        config["configurable"]["store"] = store
        config["configurable"]["backtest"] = backtest
        config["configurable"]["llm"] = llm
        config["configurable"]["react"] = react
        messages = [
            system_prompt,
            start_prompt,
        ]
        seen = set()

        for event in app.stream({"messages": messages}, config, stream_mode="values"):
            for msg in event["messages"]:
                if msg.content not in seen:
                    seen.add(msg.content)
                    msg.pretty_print()

if __name__ == "__main__":
    # ollama ids
    # mistral:v0.3
    # llama3:latest
    # qwen3:latest
    # gemma3:latest
    # deepseek-r1:latest
    # gpt-oss:20b
    # ///////////////////// #
    # HF ids
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "openai/gpt-oss-20b"
    # "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-4B-Thinking-2507-FP8"
    # "google/gemma-3-4b-it"
    llm_config = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.15,
        "backend": "hf" # Use 'ollama' backend for REACT LLM
    }
    model = get_llm(llm_config['backend'], llm_config['model'], llm_config)
    TIMEIT = True  # Set to True to enable timing
    CODE = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    THREAD_ID = f"agent_run_daily_{CODE}"
    if TIMEIT:
        import time
        start_time = time.time()
        daily_run(
            TODAY,
            False,
            backtest=False,
            thread_id=THREAD_ID,
            llm=model,
            react=False,
            end_date=None,
            model_config=llm_config
        )
        print(f"Execution time: {time.time() - start_time} seconds")
    else:
        # Run the daily workflow without timing
        print(f"Running daily workflow for {TODAY}")
        daily_run(
            TODAY,
            False,
            backtest=False,
            thread_id=THREAD_ID,
            llm=model,
            react=False,
            end_date=None,
            model_config=llm_config
        )
