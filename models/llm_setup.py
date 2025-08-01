"""Setup LLM model for Agentic PM"""
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
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


def get_llm(backend: str = "hf", model_name: str = "gemma3:latest"):
    """Initialize and return the LLM model based on the configuration."""
    # mistral:v0.3
    # llama3:latest
    # qwen3:latest
    # gemma3:latest
    # deepseek-r1:latest
    tools = [fetch_economic_data, search_news, search]
    if backend == "ollama":
        llm = ChatOllama(
            # format="json",
            model=model_name,
            temperature=0.15,
            num_ctx=4096,
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
    elif backend == "hf":
        model_name="meta-llama/Llama-3.2-3B-Instruct"
        llm = load_llm(model_id=model_name)
        llm = llm.bind_tools(tools)
        return llm
