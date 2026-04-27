import logging
import re
import time
from typing import Any

from .config import Config

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AzureOpenAI = None

logger = logging.getLogger(__name__)


class RelevancyChecker:
    """Handles relevancy checking between complaints and interactions using Azure OpenAI."""

    def __init__(self, config: Config):
        """Initialize RelevancyChecker."""
        self.config = config
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for relevancy checking")

        self.client = AzureOpenAI(
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_endpoint=config.azure_endpoint,
        )

    def check_relevancy(self, complaint: str, interaction: str) -> str:
        """Check if interaction is relevant to complaint."""
        prompt = f"""
        You are a senior customer experience analyst. Determine if the interaction is directly relevant to the customer's complaint.

        Complaint:
        {complaint}

        Interaction:
        {interaction}

        Please answer with only "yes" or "no".
        """

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior customer experience analyst. "
                        "Decide whether the complaint and interaction are related."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o",
            max_tokens=10,
            temperature=0,
        )

        text = response.choices[0].message.content.strip().lower()
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"
        logger.warning("Relevancy model returned unexpected text, defaulting to 'no': %s", text)
        return "no"

    def process_interactions_relevancy(self, df_interactions: Any) -> Any:
        """Process all interactions for relevancy checking."""
        logger.info("Starting relevancy checking for %d interactions", len(df_interactions))

        if "Customer Verbatim" not in df_interactions.columns:
            raise ValueError("Missing Customer Verbatim column required for relevancy analysis")

        df = df_interactions.copy()
        df["relevancy"] = "no"

        for index, row in df.iterrows():
            complaint = str(row.get("Customer Verbatim", "")).strip()
            interaction = str(row.get("full_transcript", "")).strip()

            try:
                relevancy = self.check_relevancy(complaint, interaction)
                time.sleep(self.config.relevancy_sleep_time)
                logger.debug("Processed row %d: %s", index, relevancy)
                df.at[index, "relevancy"] = relevancy
            except Exception as exc:
                logger.error("Error processing row %d: %s", index, exc)
                df.at[index, "relevancy"] = "no"

        df = df[df["relevancy"].isin(["yes", "no"])].reset_index(drop=True)
        logger.info(
            "Completed relevancy checking. %d interactions processed successfully",
            len(df),
        )
        return df
