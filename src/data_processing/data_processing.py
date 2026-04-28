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
    

    @staticmethod
    def extract_case_interaction_analysis(data_or_file: Union[Dict, str, Path]) -> List[Dict[str, Any]]:
        """
        Extract interaction-level and agent-level data from case interaction JSON.
        
        Processes the nested structure containing conversational analysis and 
        agent evaluations, returning one record per agent evaluation (or one 
        interaction-level record if no agent evaluations exist).
        
        Parameters
        ----------
        data_or_file : dict or str or Path
            Either:
            - a loaded Python dictionary containing the JSON data
            - or a path to a JSON file
            
        Returns
        -------
        list[dict]
            List of flattened records. One record per agent evaluation, or one
            interaction-level record if no agent evaluations exist.
        """
        try:
            # Load JSON if a file path is provided
            if isinstance(data_or_file, (str, Path)):
                file_path = Path(data_or_file)
                try:
                    data = DataProcessor._load_json_file(file_path)
                except DataProcessingError as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
                    return []
                file_name = file_path.name
            elif isinstance(data_or_file, dict):
                data = data_or_file
                file_name = None
            else:
                logger.error(f"Invalid input type: {type(data_or_file).__name__}. Expected dict, str, or Path.")
                return []
            
            records = []
            
            # Normalize data to a list of interactions
            interactions = data if isinstance(data, list) else [data]
            
            for interaction in interactions:
                # -----------------------------
                # Top-level blocks
                # -----------------------------
                conversational_analysis = DataProcessor.safe_get(interaction, "Conversational analysis", default={}) or {}
                agent_evaluation_block = DataProcessor.safe_get(interaction, "Agent Evaluation", default={}) or {}
                
                interaction_metadata = DataProcessor.safe_get(conversational_analysis, "interaction_metadata", default={}) or {}
                interaction_summary = DataProcessor.safe_get(conversational_analysis, "interaction_summary", default={}) or {}
                escalation_factors = DataProcessor.safe_get(conversational_analysis, "escalation_factors", default={}) or {}
                journey_insights = DataProcessor.safe_get(conversational_analysis, "journey_insights", default={}) or {}
                
                # -----------------------------
                # Base interaction-level record
                # -----------------------------
                base_record = {
                    # File / identifiers
                    "file_name": file_name,
                    "interaction_sequence": interaction.get("interaction_sequence"),
                    "interaction_identifier": interaction.get("interaction_identifier"),
                    "calendar_date": interaction.get("calendar_date"),
                    "case_number": interaction.get("case_number"),
                    
                    # Conversational analysis - interaction metadata
                    "interaction_date": interaction_metadata.get("interaction_date"),
                    "metadata_case_number": interaction_metadata.get("case_number"),
                    "file_number": interaction_metadata.get("file_number"),
                    "ccts_customer_issue": interaction_metadata.get("ccts_customer_issue"),
                    "customer_issue": interaction_metadata.get("ccts_customer_issue"),  # alias for convenience
                    "key_topics_discussed": interaction_metadata.get("key_topics_discussed", []),
                    "notable_moments": interaction_metadata.get("notable_moments", []),
                    
                    # Conversational analysis - interaction summary
                    "structured_summary": interaction_summary.get("structured_summary"),
                    "identified_main_issue": interaction_summary.get("identified_main_issue"),
                    "customer_intent": interaction_summary.get("customer_intent"),
                    "agent_response": interaction_summary.get("agent_response"),
                    
                    # Keep raw summary block too
                    "interaction_summary_raw": interaction_summary,
                    
                    # Conversational analysis - escalation factors
                    "escalation_risk_score": escalation_factors.get("escalation_risk_score"),
                    "contributing_factors": escalation_factors.get("contributing_factors", []),
                    "customer_frustration_points": escalation_factors.get("customer_frustration_points", []),
                    "internal_transfer": escalation_factors.get("interal_transfer") or escalation_factors.get("internal_transfer"),
                    "dropped_calls": escalation_factors.get("dropped_calls"),
                    "customer_callbacks": escalation_factors.get("customer_callbacks"),
                    
                    # Keep raw escalation block too
                    "escalation_factors_raw": escalation_factors,
                    
                    # Conversational analysis - journey insights
                    "journey_stage": journey_insights.get("journey_stage"),
                    "patterns_identified": journey_insights.get("patterns_identified", []),
                    "interaction_value_for_journey_analysis": journey_insights.get("interaction_value_for_journey_analysis"),
                    "unresolved_items": journey_insights.get("unresolved_items", []),
                    "recommended_journey_tags": journey_insights.get("recommended_journey_tags", []),
                    
                    # Keep raw journey block too
                    "journey_insights_raw": journey_insights,
                    
                    # Optional raw conversational analysis
                    "conversational_analysis_raw": conversational_analysis,
                }
                
                # -----------------------------
                # Agent evaluations
                # -----------------------------
                agent_evaluations = agent_evaluation_block.get("agent_evaluations", [])
                
                # If there are no agent evaluations, still return the interaction-level record
                if not agent_evaluations:
                    no_agent_record = base_record.copy()
                    no_agent_record.update({
                        "agent_identifier": None,
                        
                        # Performance evaluation
                        "overall_performance": None,
                        "key_strengths": [],
                        "improvement_areas": [],
                        "evaluation_of_next_steps": None,
                        
                        # Communication skills
                        "communication_skills_rating": None,
                        "communication_skills_justification": None,
                        
                        # Empathy
                        "empathy_level_rating": None,
                        "empathy_level_justification": None,
                        
                        # Professionalism
                        "professionalism_level_rating": None,
                        "professionalism_level_justification": None,
                        
                        # Resolution efficiency
                        "resolution_efficiency_rating": None,
                        "resolution_efficiency_justification": None,
                        
                        # Optional raw performance block
                        "performance_evaluation_raw": {},
                        
                        # Infraction assessment
                        "agent_infraction": None,
                        "infraction_rationale": None,
                        "educational_gap_detected": [],
                        "educational_gap_details": None,
                        "unactioned_threat_detection": None,
                        "threat_handling_assessment": None,
                        
                        # Optional raw infraction block
                        "infraction_assessment_raw": {},
                        
                        # Internal escalation
                        "escalation_appropriate": None,
                        "escalation_timing": None,
                        "escalation_effectiveness": None,
                        
                        # Optional raw escalation block
                        "internal_escalation_raw": {},
                    })
                    records.append(no_agent_record)
                    continue
                
                # One row per agent evaluation
                for agent_eval in agent_evaluations:
                    performance_evaluation = agent_eval.get("performance_evaluation", {}) or {}
                    infraction_assessment = agent_eval.get("infraction_assessment", {}) or {}
                    internal_escalation = agent_eval.get("internal_escalation", {}) or {}
                    
                    # Nested rating objects
                    communication_skills = performance_evaluation.get("communication_skills", {}) or {}
                    empathy_level = performance_evaluation.get("empathy_level", {}) or {}
                    professionalism_level = performance_evaluation.get("professionalism_level", {}) or {}
                    resolution_efficiency = performance_evaluation.get("resolution_efficiency", {}) or {}
                    
                    agent_record = base_record.copy()
                    agent_record.update({
                        # Agent ID
                        "agent_identifier": agent_eval.get("agent_identifier"),
                        
                        # Performance evaluation
                        "overall_performance": performance_evaluation.get("overall_performance"),
                        "key_strengths": performance_evaluation.get("key_strengths", []),
                        "improvement_areas": performance_evaluation.get("improvement_areas", []),
                        "evaluation_of_next_steps": performance_evaluation.get("evaluation_of_next_steps"),
                        
                        # Communication skills
                        "communication_skills_rating": communication_skills.get("rating"),
                        "communication_skills_justification": communication_skills.get("justification"),
                        
                        # Empathy
                        "empathy_level_rating": empathy_level.get("rating"),
                        "empathy_level_justification": empathy_level.get("justification"),
                        
                        # Professionalism
                        "professionalism_level_rating": professionalism_level.get("rating"),
                        "professionalism_level_justification": professionalism_level.get("justification"),
                        
                        # Resolution efficiency
                        "resolution_efficiency_rating": resolution_efficiency.get("rating"),
                        "resolution_efficiency_justification": resolution_efficiency.get("justification"),
                        
                        # Optional raw performance block
                        "performance_evaluation_raw": performance_evaluation,
                        
                        # Infraction assessment
                        "agent_infraction": infraction_assessment.get("agent_infraction"),
                        "infraction_rationale": infraction_assessment.get("infraction_rationale"),
                        "educational_gap_detected": infraction_assessment.get("educational_gap_detected", []),
                        "educational_gap_details": infraction_assessment.get("educational_gap_details"),
                        "unactioned_threat_detection": infraction_assessment.get("unactioned_threat_detection"),
                        "threat_handling_assessment": infraction_assessment.get("threat_handling_assessment"),
                        
                        # Optional raw infraction block
                        "infraction_assessment_raw": infraction_assessment,
                        
                        # Internal escalation
                        "escalation_appropriate": internal_escalation.get("escalation_appropriate"),
                        "escalation_timing": internal_escalation.get("escalation_timing"),
                        "escalation_effectiveness": internal_escalation.get("escalation_effectiveness"),
                        
                        # Optional raw escalation block
                        "internal_escalation_raw": internal_escalation,
                    })
                    
                    records.append(agent_record)
            
            return records
            
        except Exception as e:
            logger.error(f"Unexpected error extracting case interaction analysis: {e}", exc_info=True)
            return []

    @staticmethod
    def process_case_interaction_folder(folder_path: Union[str, Path], pattern: str = "*.json") -> pd.DataFrame:
        """
        Process all case interaction JSON files in a folder and return a DataFrame.
        
        Each file can contain one or multiple interactions, and each interaction
        can have zero or more agent evaluations. Returns one row per agent evaluation
        (or one row per interaction if no agent evaluations exist).
        
        Args:
            folder_path: Path to folder containing JSON files
            pattern: Glob pattern for files to process
            
        Returns:
            DataFrame with extracted records
        """
        records = []
        
        for file_path in Path(folder_path).glob(pattern):
            file_records = DataProcessor.extract_case_interaction_analysis(file_path)
            records.extend(file_records)
        
        return pd.DataFrame(records)
    
    @staticmethod
    def load_agent_improvement_data(data_folder: Union[str, Path]) -> pd.DataFrame:
        """
        Load and process agent improvement data from JSON files.
        
        Uses extract_case_interaction_analysis to get comprehensive interaction and 
        agent evaluation data, then explodes improvement_areas to create one row 
        per improvement item for clustering analysis.
        
        Args:
            data_folder: Path to folder containing JSON files
            
        Returns:
            DataFrame with exploded improvement_areas (one row per improvement)
            
        Raises:
            FileNotFoundError: If no JSON files found in folder
            DataProcessingError: If no valid records extracted
        """
        data_folder = Path(data_folder)
        logger.info(f"Loading agent improvement data from: {data_folder}")
        
        # Use the comprehensive case interaction extraction
        df = DataProcessor.process_case_interaction_folder(data_folder)
        
        if df.empty:
            raise DataProcessingError("No valid records extracted from JSON files")
        
        logger.info(f"Loaded {len(df)} total agent evaluation records")
        
        # Extract and explode improvement_areas
        if 'improvement_areas' not in df.columns:
            raise DataProcessingError("No 'improvement_areas' column found in data")
        
        # Drop rows without improvement areas
        df_with_improvements = df.dropna(subset=['improvement_areas'])
        logger.info(f"Records with improvements: {len(df_with_improvements)}")
        
        # Explode the list column
        df_exploded = df_with_improvements.explode('improvement_areas')
        df_exploded = df_exploded.dropna(subset=['improvement_areas'])
        df_exploded = df_exploded[df_exploded['improvement_areas'].str.strip().astype(bool)]
        
        logger.info(f"After explosion: {len(df_exploded)} improvement items")
        
        return df_exploded.reset_index(drop=True)





