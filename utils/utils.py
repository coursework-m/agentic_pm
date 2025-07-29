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
        print(f"[INFO] Phrase-based method failed: {e}. \
              Falling back to regex parser...")
        return reverse_regex_json_block(content)
