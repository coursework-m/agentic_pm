"""Utility functions for agentic_pm"""
import json
import re
from typing import Union
from json import JSONDecodeError
from termcolor import colored
from langchain.schema import HumanMessage, AIMessage

def print_json(json_data: Union[dict, list], indent: int = 2):
    """Print JSON data in a pretty format."""
    if isinstance(json_data, (dict, list)):
        print(json.dumps(json_data, indent=indent))
    else:
        raise ValueError("Input must be a dictionary or a list.")

def print_conversation(history, last_n=1):
    """Print"""
    filtered = [m for m in history if isinstance(m, (HumanMessage, AIMessage))]
    for msg in filtered[-last_n:]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        color = "blue" if role == "User" else "green"
        print(colored(f"{role}:", color))
        print(msg.name, msg.content.strip())
        print()

def extract_json_block(text: str):
    """Extract JSON Block"""
    start = text.find("```json")
    if start == -1:
        return []

    end = text.find("```", start + 7)
    if end == -1:
        return []

    json_str = text[start + 7:end].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return []

def regex_json_block(text):
    """Extract and parse the last valid JSON block from text."""
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not blocks:
        raise ValueError("No JSON block found.")

    for block in reversed(blocks):  # Try most recent first
        try:
            return json.loads(block)
        except JSONDecodeError as e:
            print(f"[WARN] Skipping invalid block: {e}\nBlock:\n{block}")
            continue

    raise ValueError("No valid JSON found in code blocks.")

def reverse_regex_json_block(text):
    """Extract the *last* valid JSON block (by reversing the string)"""
    reversed_text = text[::-1]

    # Regex for reversed JSON code block: matches ```json ... ```
    reversed_blocks = re.findall(r"```.*?nosj```", reversed_text, re.DOTALL)

    for block in reversed_blocks:
        # Flip the string back to normal
        normal_block = block[::-1]

        # Strip off the ```json and ``` markers
        match = re.search(r"```json\s*(.*?)\s*```", normal_block, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON block found.")

def parse_summary(content: str):
    """Try Parsing content"""
    try:
        # Split on a stable prefix that comes before the code block
        if "following recommendations:" in content:
            raw_json = content.split("made the following recommendations:")[1]
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            return json.loads(raw_json)
        else:
            raise ValueError("Expected phrase not found in content.")
    except (IndexError, JSONDecodeError, ValueError) as e:
        print(f"[INFO] Phrase-based method failed: {e}. Falling back to regex parser...")
        return reverse_regex_json_block(content)
