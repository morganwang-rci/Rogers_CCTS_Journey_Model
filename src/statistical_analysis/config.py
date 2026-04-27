import os


class Config:
    """Configuration class for the analysis pipeline."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.analysis_period = os.getenv("ANALYSIS_PERIOD", "Nov 2025")
        self.ccts_start = os.getenv("CCTS_START", "2025/11/01")
        self.ccts_end = os.getenv("CCTS_END", "2025/11/30")

        # File paths
        self.ccts_csv_path = os.getenv(
            "CCTS_CSV_PATH",
            "/Workspace/Users/morgan.wang@rci.rogers.ca/share_workspeace/consolidated_tracker.csv",
        )
        self.chat_pickle_path = os.getenv("CHAT_PICKLE_PATH", "Nov2025_chat_interactions.pkl")
        self.voice_pickle_path = os.getenv("VOICE_PICKLE_PATH", "Nov2025_voice_interaction.pkl")
        self.all_interactions_pickle_path = os.getenv(
            "ALL_INTERACTIONS_PICKLE_PATH", "Nov2025_all_interactions.pkl"
        )
        self.queue_pickle_path = os.getenv("QUEUE_PICKLE_PATH", "Nov2025_interaction_queues.pkl")

        # Databricks Unity Catalog configuration
        self.databricks_catalog = os.getenv("DATABRICKS_CATALOG", "cex_prod")
        self.databricks_schema = os.getenv("DATABRICKS_SCHEMA", "vw_genesys")
        self.databricks_reporting_schema = os.getenv("DATABRICKS_REPORTING_SCHEMA", "vw_reporting")

        # Azure OpenAI
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        # Processing
        self.relevancy_sleep_time = float(os.getenv("RELEVANCY_SLEEP_TIME", "0.5"))
