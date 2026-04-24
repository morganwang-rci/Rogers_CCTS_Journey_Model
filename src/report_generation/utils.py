"""Utility functions for Level Reports Generation."""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JSONProcessingError(Exception):
    """Raised when JSON processing fails."""

    pass


def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """
    Safely parse JSON string, handling common formatting issues.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON dictionary

    Raises:
        JSONProcessingError: If JSON cannot be parsed after cleanup
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to clean the string
        cleaned = json_str.strip()

        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.startswith("\n"):
            cleaned = cleaned[1:]

        if cleaned.endswith("\n```"):
            cleaned = cleaned[:-4]
        elif cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as retry_error:
            logger.error(f"Failed to parse JSON after cleanup: {retry_error}")
            raise JSONProcessingError(
                f"Failed to parse JSON: {str(e)}\n\nCleaned content: {cleaned}"
            ) from retry_error


def clean_openai_response(raw_response: str) -> str:
    """
    Clean OpenAI response by removing markdown formatting.

    Args:
        raw_response: Raw response from OpenAI API

    Returns:
        Cleaned response string
    """
    # Remove opening markers
    if raw_response.startswith("```json\n"):
        raw_response = raw_response[8:]
    elif raw_response.startswith("```json"):
        raw_response = raw_response[7:]
    elif raw_response.startswith("```\n"):
        raw_response = raw_response[4:]
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:]

    # Remove closing markers
    if raw_response.endswith("\n```"):
        raw_response = raw_response[:-4]
    elif raw_response.endswith("```"):
        raw_response = raw_response[:-3]

    return raw_response.strip()


def create_interaction_identifier(case_number: str, interaction_sequence: int) -> str:
    """
    Create a standardized interaction identifier.

    Args:
        case_number: The case number
        interaction_sequence: Sequence number (1-based)

    Returns:
        Formatted interaction identifier
    """
    return f"Case {case_number}_Interaction {interaction_sequence}"


def validate_dataframe_columns(df: "pd.DataFrame", required_columns: list) -> None:
    """
    Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in dataframe: {', '.join(missing)}"
        )
