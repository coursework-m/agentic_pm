"""Setup LLM model for Agentic PM"""
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
# from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentOutputParser
from pydantic import BaseModel
from models.load_model import load_llm
from tools.tools import fetch_economic_data
from tools.tools import search_news, search
from prompts.react_prompts import system_prompt
# from utils.utils import parse_summary

class AnalystSchema(BaseModel):
    """Schema for the Analyst's recommendation output."""
    response: list


def get_llm(backend: str = "hf", model_name: str = "google/gemma-3-4b-it", llm_config=None):
    """Initialize and return the LLM model based on the configuration."""
    tools = [fetch_economic_data, search_news, search]
    if backend == "ollama":
        llm = ChatOllama(
            # format="json",
            model=model_name,
            temperature=llm_config.get("temperature", 0.15),
            num_ctx=llm_config.get("max_new_tokens", 8192),
            num_gpu=1
        )
        # llm = llm.bind_tools(tools)
        llm_agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=system_prompt,
        )
        llm_executor = AgentExecutor.from_agent_and_tools(
            agent=llm_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        return llm_executor
    if backend == "hf":
        llm = load_llm(
            model_id=model_name,
            temperature=llm_config['temperature'],
            max_new_tokens=llm_config['max_new_tokens']
        )
        llm = llm.bind_tools(tools)
        return llm
