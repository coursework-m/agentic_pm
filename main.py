"""Multi agent Portfolio Management"""
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from utils.constants import TODAY, DB_URI
from prompts.prompts import system_prompt, start_prompt
from workflow.workflow import build_workflow
from models.llm_setup import get_llm

# Uncomment the following lines to enable debug mode for the yf module
# import yf
# yf.enable_debug_mode()

def daily_run(today=TODAY,
              checkpoint=False,
              backtest=False,
              thread_id=None,
              llm=None):
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
        else:
            # Default thread ID if not provided
            config["configurable"]["thread_id"] = "agent_run_daily"
        # Set the LLM configuration
        config["configurable"]["today"] = today
        config["configurable"]["store"] = store
        config["configurable"]["backtest"] = backtest
        config["configurable"]["llm"] = llm
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
    # ollama_llm = get_llm('ollama')  # Use 'ollama' backend for LLM
    hf_llm = get_llm('hf')  # Use 'hf' backend for LLM
    TIMEIT = False  # Set to True to enable timing
    if TIMEIT:
        import time
        start_time = time.time()
        daily_run(TODAY, False, llm=hf_llm, thread_id="agent_run_daily_003")
        print(f"Execution time: {time.time() - start_time} seconds")
    else:
        # Run the daily workflow without timing
        print(f"Running daily workflow for {TODAY}")
        daily_run(TODAY, False, llm=hf_llm, thread_id="agent_run_daily_003")
