import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from openai import AzureOpenAI
from ccts_theme_driver_analysis import ThemeAnalyzer
from report_generation import ConfigManager
from data_processing.data_processing import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Journey_dir = "/Workspace/Users/morgan.wang@rci.rogers.ca/share_workspeace/Levels_report_generation/Journey_2512_full" 
def load_theme_config() -> Dict[str, Any]:
    """Load theme analysis configuration from environment variables."""
    data_folder = os.getenv("JOURNEY_DIR", "./output")
    if not data_folder:
        raise ValueError("THEME_DATA_FOLDER environment variable is required")

    text_column = os.getenv("THEME_TEXT_COLUMN", "primary_complaint_issue")
    clustering_method = os.getenv("THEME_CLUSTERING_METHOD", "leiden")
    dim_reduction_method = os.getenv("THEME_DIM_REDUCTION_METHOD", "umap")
    # reduce_dimensions = os.getenv("THEME_REDUCE_DIMENSIONS", "False").strip().lower() in (
    #     "1",
    #     "true",
    #     "yes",
    # )
    norm = os.getenv("THEME_EMBEDDING_NORMALIZATION", "True")
    top_n_representatives = int(os.getenv("TOP_N_REPRESENTATIVES", "15"))
    
    # Output configuration
    output_path = os.getenv("THEME_OUTPUT_PATH", "./output/theme_analysis/")

    
    # Clustering parameters
    clustering_params = {}
    if clustering_method in ("kmeans", "auto"):
        n_clusters_env = os.getenv("THEME_KMEANS_N_CLUSTERS", "None")
        if n_clusters_env and n_clusters_env.strip().lower() not in ("none", "", "null"):
            clustering_params["n_clusters"] = int(n_clusters_env)
        else:
            # Let the clustering analyzer auto-determine k
            clustering_params["auto_k"] = True 


    elif clustering_method in ("dbscan", "auto"):
        clustering_params["min_cluster_size"] = int(os.getenv("THEME_DBSCAN_MIN_CLUSTER_SIZE", "30"))
        clustering_params["min_samples"] = int(os.getenv("THEME_DBSCAN_MIN_SAMPLES", "10"))
        clustering_params["dbscan_metric"] = os.getenv("THEME_DBSCAN_METRIC", "euclidean")
        
    elif clustering_method in ("leiden", "auto"):
        clustering_params["k"] = int(os.getenv("THEME_LEIDEN_K", "30"))
        clustering_params["resolution_parameter"] = float(os.getenv("THEME_LEIDEN_RESOLUTION", "0.7"))
        clustering_params["use_snn"] = os.getenv("THEME_LEIDEN_USE_SNN", "True").strip().lower() in ("1", "true", "yes")
        clustering_params["leiden_metric"] = os.getenv("THEME_LEIDEN_METRIC", "cosine")
        clustering_params["random_state"] = int(os.getenv("THEME_LEIDEN_RANDOM_STATE", "42"))
        clustering_params["return_graph"] =  os.getenv("THEME_LEIDEN_RETURN_GRAPH", "False").strip().lower() in ("1", "true", "yes")
    # For "auto" method, no specific parameters needed

    return {
        "text_column": text_column,
        "data_folder": data_folder,
        "clustering_method": clustering_method,
        "dim_reduction_method": dim_reduction_method,
        "norm": norm,
        "top_n_representatives": top_n_representatives,
        "output_path": output_path,
        "clustering_params": clustering_params,
    }






def save_theme_results(results: Dict[str, Any], output_path: str) -> None:
    """Save a lightweight theme analysis summary to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        # "clustering_method": results.get(""),
        "clustering_method": results.get("clustering_method", results.get("clustering", {}).get("method", "unknown")),
        "n_clusters": results.get("n_clusters", []),
        "raw_response": results.get("topics", []),
        "themes": results.get("themes", []),
        "cluster_payloads": results.get("cluster_payloads", []),
        "raw_results": results
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Theme analysis summary saved to %s", output_file)



def build_theme_databricks_rows(
    results: Dict[str, Any],
    process_date: str,
    created_by: str,
    updated_by: str,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Build a table-ready DataFrame for theme results with metadata.

    Args:
        results: Theme analysis results from run_theme_analysis.
        process_date: Processing date to store in the table.
        created_by: Creator identifier.
        updated_by: Updater identifier.

    Returns:
        DataFrame containing rows ready for Databricks save.
    """
    total_texts = len(results.get("texts", []))
    topic_data = results.get("topics", {})
    
    # Extract topics/recommendations from the topic analysis result
    # The topics can be in different formats depending on parsing
    topics = []
    
    # Check if topics are directly in the topic_data
    if isinstance(topic_data, dict):
        # Try to get topics from various possible keys
        topics = (
            # topic_data.get("recommendations") or 
            topic_data.get("topics") or 
            []
        )
    elif isinstance(topic_data, list):
        topics = topic_data
    
    # If topics is still empty, try to parse from raw_response
    if not topics and isinstance(topic_data, dict):
        raw_response = topic_data.get("raw_response")
        if raw_response:
            try:
                import json
                from data_processing.utils import safe_json_loads
                parsed = safe_json_loads(raw_response)
                if isinstance(parsed, list):
                    topics = parsed
                elif isinstance(parsed, dict):
                    topics = parsed.get("recommendations") or parsed.get("topics") or []
            except Exception as e:
                logger.warning(f"Failed to parse raw_response: {e}")

    if not topics:
        logger.warning("No topics found in results")
        return pd.DataFrame()

    cluster_payloads = {
        item.get("label"): item
        for item in (topic_data.get("cluster_payloads", []) if isinstance(topic_data, dict) else [])
        if isinstance(item, dict) and item.get("label") is not None
    }

    rows = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue

        label = topic.get("label")
        theme = topic.get("topic") or topic.get("theme") or f"cluster_{label}"
        description = topic.get("description")
        reason = topic.get("reason")
        short_example = topic.get("short_example")

        cluster_info = cluster_payloads.get(label, {})
        cluster_size = cluster_info.get("cluster_size")
        path_materiality_percentage = (
            float(cluster_size) / total_texts * 100
            if total_texts and cluster_size is not None
            else None
        )

        rows.append({
            "process_date": process_date,
            "label": label,
            "theme": theme,
            "description": description,
            "reason": reason,
            "short_example": short_example,
            "cluster_size": cluster_size,
            "path_materiality_percentage": path_materiality_percentage,
            "created_at": created_at or datetime.utcnow().isoformat(),
            "created_by": created_by,
            "updated_at": updated_at or datetime.utcnow().isoformat(),
            "updated_by": updated_by,
        })

    return pd.DataFrame(rows)




# def theme_driver_analysis() -> Dict[str, Any]:
#     """Run the theme/driver analysis pipeline.
    
#     Supports custom Leiden clustering parameters via environment variables:
#     - THEME_LEIDEN_K: Number of neighbors (default: 30)
#     - THEME_LEIDEN_RESOLUTION: Resolution parameter (default: 0.7)
#     - THEME_LEIDEN_USE_SNN: Use Shared Nearest Neighbor (default: True)
#     - THEME_LEIDEN_METRIC: Distance metric (default: cosine)
#     - THEME_LEIDEN_RANDOM_STATE: Random seed (default: 42)
#     """
#     logger.info("Loading theme analysis configuration...")
#     azure_config = ConfigManager.load_azure_config()
#     theme_config = load_theme_config()

#     logger.info("Initializing Azure OpenAI client...")
#     client = AzureOpenAI(
#         api_key=azure_config.api_key,
#         api_version=azure_config.api_version,
#         azure_endpoint=azure_config.azure_endpoint,
#     )

#     logger.info("Initializing ThemeAnalyzer...")
#     analyzer = ThemeAnalyzer(client)

#     logger.info("Running complete theme analysis...")
#     results = analyzer.run_theme_analysis(
#         data_folder=theme_config["data_folder"],
#         text_column=theme_config["text_column"],
#         clustering_method=theme_config["clustering_method"],
#         reduce_dimensions=theme_config["dim_reduction_method"],
#         norm = theme_config["norm"],
#         **theme_config["clustering_params"],
#     )

#     logger.info(
#         "Theme analysis completed. Found %s clusters.",
#         results.get("clustering", {}).get("n_clusters"),
#     )

#     output_path = theme_config["output_path"]
#     if output_path:
#         save_theme_results(results, output_path)

#     return results

def save_theme_results_to_databricks(
    spark: Any,
    results: Dict[str, Any],
    table_name: str,
    process_date: str,
    created_by: str,
    updated_by: str,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
    mode: str = "append",
) -> str:
    """
    Save theme analysis rows into a Databricks table.

    Args:
        spark: Spark session.
        results: Theme analysis results from run_theme_analysis.
        table_name: Databricks table name.
        process_date: Processing date value for each row.
        created_by: Creator identifier.
        updated_by: Updater identifier.
        mode: Save mode, typically "append" or "overwrite".

    Returns:
        The table name that was saved.
    """
    table_df = build_theme_databricks_rows(results, process_date, created_by, updated_by)
    if table_df.empty:
        raise ValueError("No theme rows available to save to Databricks table.")

    logger.info(f"Saving {len(table_df)} theme records to Databricks table: {table_name}")
    
    return DataProcessor.save_dataframe_to_databricks_table(
        table_df,
        spark,
        table_name,
        created_by=created_by,
        updated_by=updated_by,
        processing_date= datetime.utcnow().date().isoformat(),
        created_at=datetime.utcnow(),
        updated_at= datetime.utcnow(),
        mode=mode,
    )


def theme_driver_analysis(spark: Any = None, save_to_table: bool = False) -> Dict[str, Any]:
    """Run the theme/driver analysis pipeline.
    
    Supports custom Leiden clustering parameters via environment variables:
    - THEME_LEIDEN_K: Number of neighbors (default: 30)
    - THEME_LEIDEN_RESOLUTION: Resolution parameter (default: 0.7)
    - THEME_LEIDEN_USE_SNN: Use Shared Nearest Neighbor (default: True)
    - THEME_LEIDEN_METRIC: Distance metric (default: cosine)
    - THEME_LEIDEN_RANDOM_STATE: Random seed (default: 42)
    - THEME_OUTPUT_TABLE: Optional Databricks table name to save results.
    """
    logger.info("Loading theme analysis configuration...")
    azure_config = ConfigManager.load_azure_config()
    theme_config = load_theme_config()

    logger.info("Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.azure_endpoint,
    )

    logger.info("Initializing ThemeAnalyzer...")
    analyzer = ThemeAnalyzer(client)

    logger.info("Running complete theme analysis...")
    text_column = theme_config.get("text_column") or "primary_complaint_issue"
    if "text_column" not in theme_config:
        logger.warning("THEME_TEXT_COLUMN was not set; using default '%s'", text_column)

    results = analyzer.run_theme_analysis(
        data_folder=theme_config["data_folder"],
        text_column=text_column,
        clustering_method=theme_config["clustering_method"],
        reduce_dimensions=theme_config["dim_reduction_method"],
        norm=theme_config["norm"],
        **theme_config["clustering_params"],
    )

    logger.info(
        "Theme analysis completed. Found %s clusters.",
        results.get("clustering", {}).get("n_clusters"),
    )

    output_path = theme_config["output_path"]
    if output_path:
        save_theme_results(results, output_path)

    output_table = os.getenv("THEME_OUTPUT_TABLE")
    if save_to_table and spark is not None and output_table:
        process_date = datetime.utcnow().date().isoformat()
        created_by = os.getenv("THEME_CREATED_BY", "unknown")
        created_at = os.getenv("THEME_CREATED_AT", process_date)
        updated_by = os.getenv("THEME_UPDATED_BY", created_by)
        updated_at = os.getenv("THEME_UPDATED_AT", datetime.utcnow().isoformat())
        save_theme_results_to_databricks(
            spark=spark,
            results=results,
            table_name=output_table,
            process_date=process_date,
            created_by=created_by,
            updated_by=updated_by,
            updated_at=updated_at,
            mode=os.getenv("THEME_SAVE_MODE", "append"),
        )
        logger.info("Theme results saved to Databricks table %s", output_table)

    return results

if __name__ == "__main__":
    try:
        theme_driver_analysis()
    except Exception as exc:
        logger.exception("Theme driver analysis failed: %s", exc)
        sys.exit(1)
