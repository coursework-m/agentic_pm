"""Load a Hugging Face LLM model for text generation."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
# from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF

def load_llm(model_id: str, temperature=0.15, max_new_tokens=2028):
    """Load the LLM"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    token = os.getenv("HF_TOKEN")  # export HF_TOKEN=token before running

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map={"": device},
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )

    wrapped_llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=wrapped_llm, model_id=model_id)