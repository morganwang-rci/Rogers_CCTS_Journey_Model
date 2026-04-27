import logging
from typing import List

import pandas as pd

from .config import Config

try:
    from pyspark.sql import SparkSession
    _active_spark = SparkSession.getActiveSession()
    DATABRICKS_SPARK_AVAILABLE = _active_spark is not None
except Exception:
    DATABRICKS_SPARK_AVAILABLE = False
    _active_spark = None

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from Databricks Unity Catalog."""

    def __init__(self, config: Config):
        """Initialize DataLoader with configuration."""
        self.config = config
        self.spark = _active_spark
        self._detect_environment()

    def _detect_environment(self) -> None:
        """Verify Databricks Spark environment."""
        if not DATABRICKS_SPARK_AVAILABLE:
            raise RuntimeError(
                "Databricks Spark session not available. "
                "This pipeline must run within a Databricks environment."
            )
        logger.info("Running in Databricks environment with Spark")

    def load_ccts_data(self) -> pd.DataFrame:
        """Load and preprocess CCTS complaint data."""
        logger.info("Loading CCTS data from %s", self.config.ccts_csv_path)

        df = pd.read_csv(self.config.ccts_csv_path, encoding="windows-1252")
        df["Customer Verbatim"] = (
            df["Customer Issue"].fillna("") + " " + df["Customers Request"].fillna("")
        )
        df.rename(
            columns={
                "CCTS Case: Case Number": "Case Number",
                "Primary Account (Customer)": "attr_account_number",
            },
            inplace=True,
        )
        df["Email Date"] = pd.to_datetime(df["Email Date"], format="%m/%d/%Y", errors="coerce")
        df["attr_account_number"] = df["attr_account_number"].astype(str).str.replace("-", "", regex=False)

        df = df.loc[
            (df["Email Date"] >= pd.to_datetime(self.config.ccts_start))
            & (df["Email Date"] <= pd.to_datetime(self.config.ccts_end))
        ].reset_index(drop=True)

        logger.info("Loaded %d CCTS complaints for %s", len(df), self.config.analysis_period)
        return df

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query using Spark and return a pandas DataFrame."""
        try:
            result_df = self.spark.sql(query)
            df = result_df.toPandas()
            logger.debug("Executed query using Spark SQL")
            return df
        except Exception as exc:
            logger.error("Spark SQL query failed: %s", exc)
            raise

    def load_chat_interactions(self, account_numbers: List[str]) -> pd.DataFrame:
        """Load chat interactions for the given account numbers."""
        if not account_numbers:
            logger.warning("No account numbers provided for chat interaction lookup.")
            return pd.DataFrame()

        logger.info("Loading chat interactions for %d accounts", len(account_numbers))
        account_filter = ",".join(f"'{acc}'" for acc in account_numbers)

        query = f"""
        SELECT  B.queue_name, B.attr_account_number, A.calendar_date, B.talk_time,
                A.full_transcript, A.emp_id, A.participant_ordinal, A.media_type,
                B.session_direction, B.disconnect_type, B.attr_chat_curation_selection,
                B.attr_chat_intent, B.attr_chat_origination_page, B.conversation_start
        FROM `{self.config.databricks_catalog}`.`{self.config.databricks_schema}`.`transcripts_digital` A
        JOIN (
            SELECT c.*
            FROM `{self.config.databricks_catalog}`.`{self.config.databricks_schema}`.`call_detail` c
            WHERE c.attr_account_number IN ({account_filter})
        ) B ON A.conversation_id = B.conversation_id AND A.emp_id = B.emp_id
        """

        df = self._execute_query(query)
        df["media_type"] = df.get("media_type", "chat")
        return self._normalize_interaction_frame(df)

    def load_voice_interactions(self, account_numbers: List[str]) -> pd.DataFrame:
        """Load voice interactions for the given account numbers."""
        if not account_numbers:
            logger.warning("No account numbers provided for voice interaction lookup.")
            return pd.DataFrame()

        logger.info("Loading voice interactions for %d accounts", len(account_numbers))
        account_filter = ",".join(f"'{acc}'" for acc in account_numbers)

        query = f"""
        SELECT  B.queue_name, B.attr_account_number, A.calendar_date, B.talk_time,
                A.full_transcript, A.emp_id, A.participant_ordinal, B.session_direction,
                B.disconnect_type, B.attr_chat_curation_selection, B.attr_chat_intent,
                B.attr_chat_origination_page, B.conversation_start
        FROM `{self.config.databricks_catalog}`.`{self.config.databricks_schema}`.`transcripts_no_hold` A
        JOIN (
            SELECT c.*
            FROM `{self.config.databricks_catalog}`.`{self.config.databricks_schema}`.`call_detail` c
            WHERE c.attr_account_number IN ({account_filter})
        ) B ON A.conversation_id = B.conversation_id AND A.emp_id = B.emp_id
        """

        df = self._execute_query(query)
        df["media_type"] = "voice"
        return self._normalize_interaction_frame(df)

    def load_queue_hierarchy(self) -> pd.DataFrame:
        """Load queue hierarchy data from Databricks Unity Catalog."""
        logger.info("Loading queue hierarchy from Databricks")
        query = f"SELECT * FROM `{self.config.databricks_catalog}`.`{self.config.databricks_reporting_schema}`.`hierarchy_queue_hierarchy`"
        df = self._execute_query(query)
        logger.info("Loaded queue hierarchy with %d records", len(df))
        return df

    def _normalize_interaction_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "calendar_date" in df.columns:
            df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
        if "attr_account_number" in df.columns:
            df["attr_account_number"] = (
                df["attr_account_number"].astype(str).str.replace("-", "", regex=False)
            )
        return df.dropna(subset=["attr_account_number", "calendar_date"]).reset_index(drop=True)
