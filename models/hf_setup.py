"""Setup LLM model for Agentic PM"""
import os
from models.load_model import load_llm
from tools.tools import search, search_news, fetch_securities_data

hf_access_token=os.getenv("HF_TOKEN")
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
llm = load_llm(model_id=MODEL_NAME)

tools = [search, search_news, fetch_securities_data]
llm = llm.bind_tools(tools)
