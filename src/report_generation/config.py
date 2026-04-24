"""Configuration management for Level Reports Generation."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""

    api_key: str
    api_version: str
    azure_endpoint: str
    model: str = "gpt-4o"


@dataclass
class DataConfig:
    """Data paths and settings configuration."""

    pickle_file_path: str
    conversation_output_dir: str
    journey_output_dir: str


@dataclass
class ProcessingConfig:
    """Processing-related configuration."""

    max_tokens_interaction: int = 10240
    max_tokens_summary: int = 8192
    temperature: float = 0.1
    api_retry_attempts: int = 3
    api_retry_delay: int = 2
    request_delay: float = 2.0


class ConfigManager:
    """Manages application configuration from environment variables."""

    @staticmethod
    def load_azure_config() -> AzureOpenAIConfig:
        """Load Azure OpenAI configuration from environment."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")

        model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

        return AzureOpenAIConfig(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            model=model,
        )

    @staticmethod
    def load_data_config() -> DataConfig:
        """Load data configuration from environment."""
        pickle_path = os.getenv("DATA_PICKLE_PATH")
        if not pickle_path:
            raise ValueError("DATA_PICKLE_PATH environment variable not set")

        conversation_dir = os.getenv("CONVERSATION_OUTPUT_DIR", "./output/conversations")
        journey_dir = os.getenv("JOURNEY_OUTPUT_DIR", "./output/journeys")

        return DataConfig(
            pickle_file_path=pickle_path,
            conversation_output_dir=conversation_dir,
            journey_output_dir=journey_dir,
        )

    @staticmethod
    def load_processing_config() -> ProcessingConfig:
        """Load processing configuration from environment."""
        return ProcessingConfig(
            max_tokens_interaction=int(
                os.getenv("MAX_TOKENS_INTERACTION", "10240")
            ),
            max_tokens_summary=int(os.getenv("MAX_TOKENS_SUMMARY", "8192")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            api_retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "3")),
            api_retry_delay=int(os.getenv("API_RETRY_DELAY", "2")),
            request_delay=float(os.getenv("REQUEST_DELAY", "2.0")),
        )
