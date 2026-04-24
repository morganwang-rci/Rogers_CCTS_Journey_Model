"""Main interaction analyzer module for processing CCTS complaint interactions."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import AzureOpenAI

from .config import AzureOpenAIConfig, ProcessingConfig
from .prompts import (
    get_agent_evaluation_prompt,
    get_interaction_analysis_prompt,
    get_journey_analysis_prompt,
)
from .utils import (
    JSONProcessingError,
    clean_openai_response,
    create_interaction_identifier,
    safe_json_loads,
    validate_dataframe_columns,
)

logger = logging.getLogger(__name__)


class InteractionAnalyzer:
    """Analyzer for customer interactions and complaint data."""

    # Required columns for analysis
    REQUIRED_COLUMNS = [
        "attr_account_number",
        "calendar_date",
        "Case Number",
        "File Number",
        "Brand",
        "Product Line",
        "Product",
        "media_type",
        "Customer Issue",
        "Root Cause",
        "Notes",
        "full_transcript",
        "emp_id",
    ]

    def __init__(
        self,
        client: AzureOpenAI,
        processing_config: ProcessingConfig,
    ):
        """
        Initialize the InteractionAnalyzer.

        Args:
            client: Azure OpenAI client
            processing_config: Processing configuration

        Raises:
            ValueError: If client or config is invalid
        """
        if not client:
            raise ValueError("Azure OpenAI client is required")
        if not processing_config:
            raise ValueError("Processing config is required")

        self.client = client
        self.config = processing_config
        logger.info("InteractionAnalyzer initialized")

    def create_context_prompt(self, row: pd.Series) -> str:
        """
        Create context string from interaction data.

        Args:
            row: DataFrame row containing interaction data

        Returns:
            Formatted context string
        """
        context = f"""
        This is information from CCTS complaints
        - Account Number: {row['attr_account_number']}
        - Conversation Date: {row['calendar_date']}
        - Case Number: {row['Case Number']}
        - Brand: {row['Brand']}
        - Product Line: {row['Product Line']} 
        - Product: {row['Product']}
        - Media Type: {row['media_type']}
        - Customer Issue: {row['Customer Issue']}
        - Potential Root Cause: {row['Root Cause']}
        - Notes: {row['Notes']}
        """
        return context

    def _call_openai_with_retry(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.1
    ) -> str:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: Message list for the API call
            max_tokens: Maximum tokens in response
            temperature: Temperature parameter

        Returns:
            Response content

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.config.api_retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                logger.debug(f"OpenAI API call successful on attempt {attempt + 1}")
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.config.api_retry_attempts - 1:
                    wait_time = self.config.api_retry_delay * (attempt + 1)
                    logger.warning(
                        f"API call attempt {attempt + 1} failed. "
                        f"Retrying in {wait_time} seconds. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API call failed after {self.config.api_retry_attempts} attempts: {str(e)}"
                    )
                    raise

    def analyze_interaction(self, row: pd.Series) -> Tuple[str, str]:
        """
        Analyze a single interaction.

        Args:
            row: DataFrame row containing interaction data

        Returns:
            Tuple of (interaction_analysis, agent_evaluation) as JSON strings

        Raises:
            JSONProcessingError: If JSON parsing fails
            Exception: If API calls fail
        """
        try:
            context = self.create_context_prompt(row)
            interaction_prompt = get_interaction_analysis_prompt(row.to_dict())
            agent_prompt = get_agent_evaluation_prompt(row.to_dict(), context)

            logger.debug(
                f"Analyzing interaction for case {row['Case Number']}, "
                f"date {row['calendar_date']}"
            )

            raw_response = self._call_openai_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert customer service analyst specializing in "
                            "complaint resolution and customer journey analysis. "
                            "Provide detailed, objective analysis based on the interaction data."
                        ),
                    },
                    {"role": "user", "content": interaction_prompt},
                ],
                max_tokens=self.config.max_tokens_interaction,
            )

            time.sleep(self.config.request_delay)

            agent_eval = self._call_openai_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert customer service analyst specializing in "
                            "complaint resolution and customer journey analysis. "
                            "Provide detailed, objective analysis based on the interaction data."
                        ),
                    },
                    {"role": "user", "content": agent_prompt},
                ],
                max_tokens=self.config.max_tokens_interaction,
            )

            logger.debug(f"Analysis completed for case {row['Case Number']}")
            return raw_response, agent_eval

        except Exception as e:
            logger.error(
                f"Error analyzing interaction for case {row['Case Number']}: {str(e)}"
            )
            raise

    def analyze_all_interactions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze all interactions in a dataframe.

        Args:
            df: DataFrame containing interaction data

        Returns:
            List of analysis results

        Raises:
            ValueError: If required columns are missing
            JSONProcessingError: If JSON parsing fails
        """
        validate_dataframe_columns(df, self.REQUIRED_COLUMNS)

        results = []
        df = df.sort_values("conversation_start", ascending=True).reset_index(drop=True)
        df["calendar_date"] = df["calendar_date"].astype(str)

        logger.info(f"Starting analysis of {len(df)} interactions")

        for idx, row in df.iterrows():
            try:
                raw_response, agent_eval = self.analyze_interaction(row)

                raw_response = clean_openai_response(raw_response).strip()
                agent_eval = clean_openai_response(agent_eval).strip()

                try:
                    parsed_response = safe_json_loads(raw_response)
                except JSONProcessingError as e:
                    logger.error(f"Failed to parse interaction response for index {idx}: {str(e)}")
                    parsed_response = {"error": f"Failed to parse JSON: {str(e)}"}

                try:
                    agent_eval_parsed = safe_json_loads(agent_eval)
                except JSONProcessingError as e:
                    logger.error(f"Failed to parse agent evaluation for index {idx}: {str(e)}")
                    agent_eval_parsed = {"error": f"Failed to parse JSON: {str(e)}"}

                interaction_sequence = idx + 1
                interaction_identifier = create_interaction_identifier(
                    row["Case Number"], interaction_sequence
                )

                interaction_result = {
                    "interaction_sequence": interaction_sequence,
                    "interaction_identifier": interaction_identifier,
                    "calendar_date": row["calendar_date"],
                    "case_number": row["Case Number"],
                    "Conversational analysis": parsed_response,
                    "Agent Evaluation": agent_eval_parsed,
                }

                results.append(interaction_result)

                logger.info(
                    f"Completed analysis {interaction_sequence}/{len(df)} "
                    f"for case {row['Case Number']}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing interaction {idx} for case {row['Case Number']}: {str(e)}"
                )
                continue

        logger.info(f"Analysis complete. Processed {len(results)} interactions")
        return results

    def generate_summary_report(self, results: List[Dict[str, Any]], file_number: str) -> Dict[str, Any]:
        """
        Generate overall summary across all interactions.

        Args:
            results: List of analysis results
            file_number: File number for the case

        Returns:
            Summary analysis as dictionary

        Raises:
            Exception: If API call fails
        """
        if not results:
            logger.warning("No results provided for summary report generation")
            return {"error": "No results provided"}

        logger.info(f"Generating journey summary for case number {file_number}")

        journey_prompt = get_journey_analysis_prompt(results, file_number)

        try:
            response = self._call_openai_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior customer experience analyst. "
                            "Provide strategic insights and actionable recommendations."
                        ),
                    },
                    {"role": "user", "content": journey_prompt},
                ],
                max_tokens=self.config.max_tokens_summary,
            )

            cleaned_response = clean_openai_response(response).strip()

            try:
                summary_json = safe_json_loads(cleaned_response)
                logger.info(f"Journey summary successfully generated for {file_number}")
                return summary_json
            except JSONProcessingError as e:
                logger.error(f"Failed to parse journey summary JSON for {file_number}: {str(e)}")
                return {"error": f"Failed to parse JSON: {str(e)}"}

        except Exception as e:
            logger.error(f"Error generating journey summary for {file_number}: {str(e)}")
            raise


def save_analysis_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    file_number: str,
    conversation_dir: Path,
    journey_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save analysis results to files.

    Args:
        results: Analysis results to save
        summary: Summary report to save
        file_number: File number for naming
        conversation_dir: Directory for interaction analysis
        journey_dir: Directory for journey summary

    Returns:
        Tuple of (interaction_path, summary_path)

    Raises:
        IOError: If file write fails
    """
    try:
        conversation_dir.mkdir(parents=True, exist_ok=True)
        journey_dir.mkdir(parents=True, exist_ok=True)

        interaction_path = conversation_dir / f"interaction_analysis_{file_number}.json"
        summary_path = journey_dir / f"journey_summary_{file_number}.json"

        with open(interaction_path, "w") as f:
            json.dump(results, f, indent=2)
            logger.info(f"Saved interaction analysis to {interaction_path}")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            logger.info(f"Saved journey summary to {summary_path}")

        return interaction_path, summary_path

    except IOError as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        raise
