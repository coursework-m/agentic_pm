"""Setup LLM model for Agentic PM"""
from langchain_community.llms import Ollama

model_name = "llama3"  # Specify the model name you want to use

llm = Ollama(
    model=model_name,
    temperature=0.15,
    max_tokens=1024,
    model_kwargs={"num_gpus": 1}
)  
