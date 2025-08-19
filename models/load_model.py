"""Load a Hugging Face LLM model for text generation."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline

def load_llm(model_id: str, temperature=0.15, max_new_tokens=2028):
    """Load the LLM"""
    # Select device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16  # safer than float16 on MPS
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device} with dtype {dtype}")
    token = os.getenv("HF_TOKEN")  # export HF_TOKEN=token before running

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
        attn_implementation="sdpa",
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # task="image-text-to-text" if "gemma" in model_id.lower() else "text-generation"
    task = "text2text-generation" if "gemma" in model_id.lower() else "text-generation"
    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        device_map="auto" if device == "cuda" else None,
    )

    wrapped_llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=wrapped_llm, model_id=model_id)
