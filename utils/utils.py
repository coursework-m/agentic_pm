"""Utility functions for agentic_pm"""
import json
import logging
import re
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reverse_regex_json_block(text):
    """Extract the last valid JSON block (by reversing the string)"""
    reversed_text = text[::-1]

    # Regex for reversed JSON code block: matches ```json ... ```
    reversed_json = re.findall(r"```.*?nosj```", reversed_text, re.DOTALL)

    for block in reversed_json:
        # Flip the string back to normal
        normal_block = block[::-1]
        # Strip off the ```json and ``` markers
        match = re.search(r"```json\s*(.*?)\s*```", normal_block, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.info(f"[INFO] JSON decode error: {e}")
                logger.info(f"[INFO] Invalid JSON block: {match.group(1)}")
                continue

    raise ValueError("No valid JSON block found.")

def parse_summary(content: str):
    """Parse content"""
    try:
        # Attempt to parse using regex
        return reverse_regex_json_block(content)
    except (IndexError, JSONDecodeError, ValueError) as e:
        logger.info(f"[INFO] regex parser failed: {e} for content: {content}")
