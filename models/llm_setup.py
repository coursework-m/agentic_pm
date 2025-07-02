"""Setup LLM model for Agentic PM"""
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from models.load_model import load_llm
from tools.tools import scan_market

class AnalystSchema(BaseModel):
    """Schema for the Analyst's recommendation output."""
    ticker: str
    summary: str
    recommendation: str

# mistral:v0.3
# llama3.2
def get_llm(backend: str = "hf", model_name: str = "llama3.2"):
    """Initialize and return the LLM model based on the configuration."""
    tools = [scan_market]
    if backend == "ollama":
        llm = ChatOllama(
            model=model_name,
            temperature=0.15,
            num_ctx=2048,
            num_gpu=1
        )
        # llm = llm.bind_tools(tools)
        llm_agent = create_react_agent(
            model=llm,
            tools=tools,
            response_format=AnalystSchema,
            debug=True
        )
        llm_executor = AgentExecutor(agent=llm_agent, tools=tools, verbose=True)
        return llm_executor
    elif backend == "hf":
        model_name="meta-llama/Llama-3.2-3B-Instruct"
        llm = load_llm(model_id=model_name)
        llm = llm.bind_tools(tools)
        return llm
