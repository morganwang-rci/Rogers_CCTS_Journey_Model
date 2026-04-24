"""Report Generation Module - Step 1 of CCTS Complaint Analysis Pipeline.

This module handles the generation of detailed complaint reports from interaction data,
including:
- Interaction analysis with complaint correlation
- Agent performance evaluation
- Customer journey mapping
- Root cause identification
"""

from .analyzer import InteractionAnalyzer, save_analysis_results
from .config import AzureOpenAIConfig, ConfigManager, DataConfig, ProcessingConfig
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

__version__ = "2.0"
__all__ = [
    "InteractionAnalyzer",
    "save_analysis_results",
    "ConfigManager",
    "AzureOpenAIConfig",
    "DataConfig",
    "ProcessingConfig",
    "clean_openai_response",
    "safe_json_loads",
    "create_interaction_identifier",
    "validate_dataframe_columns",
    "JSONProcessingError",
    "get_interaction_analysis_prompt",
    "get_agent_evaluation_prompt",
    "get_journey_analysis_prompt",
]
