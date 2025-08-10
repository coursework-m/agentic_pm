"""Utility functions for agentic_pm"""
import json
import re
from json import JSONDecodeError

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
    """Parse content"""
    try:
        # Attempt to parse using regex
        return reverse_regex_json_block(content)
    except (IndexError, JSONDecodeError, ValueError) as e:
        print(f"[INFO] regex parser failed: {e} for content: {content}")
