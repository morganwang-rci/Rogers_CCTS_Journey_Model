"""Data processing utilities for complaint journey analysis and text extraction."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import json
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class DataProcessor:
    """
    Handles data extraction and processing for complaint journey analysis.
    
    Provides robust methods for loading, parsing, and extracting structured
    complaint data from JSON files and dictionaries.
    """

    @staticmethod
    def safe_get(dct: Dict, *keys: str, default=None) -> Any:
        """
        Safely navigate nested dictionaries without raising KeyError.

        Args:
            dct: Dictionary to navigate
            *keys: Keys to traverse in order
            default: Default value if key path not found

        Returns:
            Value at the nested key path or default value

        Raises:
            TypeError: If dct is not a dictionary at any level
        """
        if not isinstance(dct, dict):
            logger.debug(f"Expected dict, got {type(dct).__name__}")
            return default

        current = dct
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    @staticmethod
    def _load_json_file(file_path: Path) -> Dict:
        """
        Load and parse JSON file with comprehensive error handling.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON as dictionary

        Raises:
            FileNotFoundError: If file does not exist
            DataProcessingError: If file cannot be read or JSON is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {file_path.name}: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
        except IOError as e:
            error_msg = f"Cannot read file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e

    @staticmethod
    def extract_case_journey_analysis(data_or_file: Union[Dict, str, Path]) -> Dict[str, Any]:
        """
        Extract all relevant fields from a case-level complaint journey analysis JSON.

        Parameters
        ----------
        data_or_file : dict or str or Path
            Either:
            - a loaded Python dictionary containing the JSON data
            - or a path to a JSON file

        Returns
        -------
        dict
            Flattened case-level record
        """
        try:
            # Load JSON if a file path is provided
            if isinstance(data_or_file, (str, Path)):
                file_path = Path(data_or_file)
                try:
                    data = DataProcessor._load_json_file(file_path)
                except DataProcessingError as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
                    return {}
                file_name = file_path.name
            elif isinstance(data_or_file, dict):
                data = data_or_file
                file_name = None
            else:
                logger.error(f"Invalid input type: {type(data_or_file).__name__}. Expected dict, str, or Path.")
                return {}

            ccts_journey = DataProcessor.safe_get(data, "ccts_complaint_journey_analysis", default={}) or {}
            customer_complaint_genesis = DataProcessor.safe_get(ccts_journey, "customer_complaint_genesis", default={}) or {}
            response_assessment = DataProcessor.safe_get(ccts_journey, "response_assessment", default={}) or {}
            journey_failure_points = DataProcessor.safe_get(ccts_journey, "journey_failure_points", default={}) or {}

            value_gap_analysis = DataProcessor.safe_get(data, "value_gap_analysis", default={}) or {}
            rationality_assessment = DataProcessor.safe_get(value_gap_analysis, "rationality_assessment", default={}) or {}

            prevention_opportunity_analysis = DataProcessor.safe_get(data, "prevention_opportunity_analysis", default={}) or {}

            resolution_recommendations = DataProcessor.safe_get(data, "resolution_recommendations", default={}) or {}
            root_cause_identification = DataProcessor.safe_get(resolution_recommendations, "root_cause_identification", default={}) or {}

            record = {
                # File metadata
                "file_name": file_name,

                # ccts_complaint_journey_analysis
                'case_number': ccts_journey.get("case_number"),

                # customer_complaint_genesis
                "primary_complaint_issue": customer_complaint_genesis.get("primary_complaint_issue"),
                "issue_evolution": customer_complaint_genesis.get("issue_evolution"),
                "unresolved_issues": customer_complaint_genesis.get("unresolved_issues", []),

                # response_assessment
                "solutions_offered": response_assessment.get("solutions_offered", []),
                "implementation_gaps": response_assessment.get("implementation_gaps", []),
                "consistency_of_handling": response_assessment.get("consistency_of_handling"),

                # journey_failure_points
                "critical_breakdown_moments": journey_failure_points.get("critical_breakdown_moments", []),
                "repeat_contact_pattern": journey_failure_points.get("repeat_contact_pattern"),
                "final_straw_incident": journey_failure_points.get("final_straw_incident"),

                # value_gap_analysis
                "offer_vs_expectation_matrix": value_gap_analysis.get("offer_vs_expectation_matrix"),

                # rationality_assessment
                "customer_demand_rationality": rationality_assessment.get("customer_demand_rationality"),
                "customer_demand_rationality_justification": rationality_assessment.get("customer_demand_rationality_justification"),
                "company_offer_adequacy": rationality_assessment.get("company_offer_adequacy"),
                "company_offer_adequacy_justification": rationality_assessment.get("company_offer_adequacy_justification"),

                # prevention_opportunity_analysis
                "proactive_outreach": prevention_opportunity_analysis.get("proactive_outreach", []),
                "compensation_timing": prevention_opportunity_analysis.get("compensation_timing", []),
                "escalation_management": prevention_opportunity_analysis.get("escalation_management", []),

                # resolution_recommendations / root_cause_identification
                "primary_root_cause": root_cause_identification.get("primary_root_cause"),
                "contributing_factors": root_cause_identification.get("contributing_factors", []),
                "systemic_vs_individual": root_cause_identification.get("systemic_vs_individual"),
                "cause_explanation": root_cause_identification.get("cause_explanation"),
                "evidence_base": root_cause_identification.get("evidence_base", []),

                # strategic recommendations
                "strategic_recommendations": resolution_recommendations.get("strategic_recommendations", []),

                # Raw blocks for traceability
                "ccts_complaint_journey_analysis_raw": ccts_journey,
                "value_gap_analysis_raw": value_gap_analysis,
                "prevention_opportunity_analysis_raw": prevention_opportunity_analysis,
                "resolution_recommendations_raw": resolution_recommendations,
            }

            return record

        except Exception as e:
            logger.error(f"Unexpected error extracting case journey analysis: {e}", exc_info=True)
            return {}

    @staticmethod
    def process_case_journey_folder(folder_path: Union[str, Path], pattern: str = "*.json") -> pd.DataFrame:
        """
        Process all case journey analysis JSON files in a folder and return a DataFrame.

        Args:
            folder_path: Path to folder containing JSON files
            pattern: Glob pattern for files to process

        Returns:
            DataFrame with extracted records
        """
        records = []

        for file_path in Path(folder_path).glob(pattern):
            record = DataProcessor.extract_case_journey_analysis(file_path)
            if record:
                records.append(record)

        return pd.DataFrame(records)
    
    @staticmethod
    def save_dataframe_to_journey_table(
        df: pd.DataFrame,
        spark: Any,
        table_name: str,
        created_by: str,
        updated_by: str,
        processing_date: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        mode: str = "overwrite",
    ) -> str:
        """
        Save a DataFrame as a Databricks table using a Spark session.

        Args:
            df: DataFrame to save.
            spark: Active Spark session for Databricks.
            table_name: Target table name.
            created_by: Creator identifier.
            updated_by: Updater identifier.
            processing_data: Optional processing metadata.
            created_at: Optional creation timestamp; defaults to current UTC.
            updated_at: Optional update timestamp; defaults to current UTC.
            mode: Spark save mode, such as "overwrite" or "append".

        Returns:
            Table name that was saved.

        Raises:
            DataProcessingError: If the DataFrame is invalid or save fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise DataProcessingError("Input must be a pandas DataFrame.")
        if spark is None:
            raise DataProcessingError("A Spark session is required to save to a Databricks table.")
        if not table_name:
            raise DataProcessingError("A table_name is required to save to a Databricks table.")

        created_at = created_at or datetime.utcnow().isoformat()
        updated_at = updated_at or datetime.utcnow().isoformat()

        df = df.copy()
        df["processing_date"] =processing_date or datetime.utcnow().isoformat()
        # if "created_at" not in df.columns or df["created_at"].isna().all():
        #     df["created_at"] = created_at
        # if "created_by" not in df.columns or df["created_by"].isna().all():
        #     df["created_by"] = created_by
        df["updated_at"] = updated_at
        df["updated_by"] = updated_by

        try:
            spark_df = spark.createDataFrame(df)
            if mode.lower() == "append" and spark.catalog.tableExists(table_name):
                spark_df.write.mode("append").saveAsTable(table_name)
            else:
                spark_df.write.mode(mode).saveAsTable(table_name)
            return table_name
        except Exception as e:
            logger.error(f"Failed to save DataFrame as Databricks table {table_name}: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to save Databricks table: {e}") from e