"""Utility functions for resolution recommendation analysis."""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def clean_json_string(json_string: str) -> str:
    """
    Remove markdown JSON fences if present.

    Handles JSON strings wrapped in markdown code blocks (```json ... ```).

    Args:
        json_string: Potentially markdown-wrapped JSON string

    Returns:
        Cleaned JSON string without markdown wrappers

    Examples:
        >>> clean_json_string('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    pattern = r"^```json\s*(.*?)\s*```$"
    cleaned_string = re.sub(pattern, r"\1", json_string.strip(), flags=re.DOTALL)
    return cleaned_string.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response with automatic markdown cleaning.

    Attempts to parse JSON, automatically handling common LLM response
    formats including markdown code blocks.

    Args:
        text: JSON string, potentially with markdown wrappers

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If text cannot be parsed as valid JSON

    Examples:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}

        >>> safe_json_loads('```json\\n[{"label": 0}]\\n```')
        [{'label': 0}]
    """
    cleaned = clean_json_string(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.error(f"Original text (first 500 chars): {text[:500]}")
        raise
