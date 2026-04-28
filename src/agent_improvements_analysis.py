"""
Agent Improvements Analysis Driver Script.

This script provides a command-line interface for running agent improvements
analysis with configuration via environment variables.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from openai import AzureOpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_improvements import AgentImprovementAnalyzer
from report_generation import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def load_agent_improvemen_config() -> Dict[str, Any]:
    """Load theme analysis configuration from environment variables."""
    data_folder = os.getenv("JOURNEY_DIR", "./output")
    if not data_folder:
        raise ValueError("AGENT_DATA_FOLDER environment variable is required")

    text_column = os.getenv("AGENT_TEXT_COLUMN", "improvement_areas")
    clustering_method = os.getenv("AGENT_CLUSTERING_METHOD", "auto")
    dim_reduction_method = os.getenv("AGENT_DIM_REDUCTION_METHOD", "umap")
    # reduce_dimensions = os.getenv("THEME_REDUCE_DIMENSIONS", "False").strip().lower() in (
    #     "1",
    #     "true",
    #     "yes",
    # )
    norm = os.getenv("AGENT_EMBEDDING_NORMALIZATION", "False")
    top_n_representatives = int(os.getenv("TOP_N_REPRESENTATIVES", "15"))
    
    # Output configuration
    output_path = os.getenv("AGENT_OUTPUT_PATH", "./output/agent_improvement/")

    
    # Clustering parameters
    clustering_params = {}
    if clustering_method in ("kmeans", "auto"):
        n_clusters_env = os.getenv("AGENT_KMEANS_N_CLUSTERS", "None")
        if n_clusters_env and n_clusters_env.strip().lower() not in ("none", "", "null"):
            clustering_params["n_clusters"] = int(n_clusters_env)
        else:
            # Let the clustering analyzer auto-determine k
            clustering_params["auto_k"] = True 


    elif clustering_method in ("dbscan", "auto"):
        clustering_params["min_cluster_size"] = int(os.getenv("AGENT_DBSCAN_MIN_CLUSTER_SIZE", "30"))
        clustering_params["min_samples"] = int(os.getenv("AGENT_DBSCAN_MIN_SAMPLES", "10"))
        clustering_params["dbscan_metric"] = os.getenv("AGENT_DBSCAN_METRIC", "euclidean")
        
    elif clustering_method in ("leiden", "auto"):
        clustering_params["k"] = int(os.getenv("AGENT_LEIDEN_K", "30"))
        clustering_params["resolution_parameter"] = float(os.getenv("AGENT_LEIDEN_RESOLUTION", "0.7"))
        clustering_params["use_snn"] = os.getenv("AGENT_LEIDEN_USE_SNN", "True").strip().lower() in ("1", "true", "yes")
        clustering_params["leiden_metric"] = os.getenv("AGENT_LEIDEN_METRIC", "cosine")
        clustering_params["random_state"] = int(os.getenv("AGENT_LEIDEN_RANDOM_STATE", "42"))
        clustering_params["return_graph"] =  os.getenv("AGNT_LEIDEN_RETURN_GRAPH", "False").strip().lower() in ("1", "true", "yes")
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




# def load_agent_improvement_config() -> Dict[str, Any]:
#     """
#     Load agent improvement analysis configuration from environment variables.
    
#     Returns:
#         Configuration dictionary with analysis parameters
#     """
#     config = {
#         "data_folder": os.getenv("AI_DATA_FOLDER"),
#         "output_path": os.getenv("AI_OUTPUT_PATH"),
#         "clustering_method": os.getenv("AI_CLUSTERING_METHOD", "auto"),
#         "min_cluster_size": int(os.getenv("AI_MIN_CLUSTER_SIZE", "10")),
#         "min_samples": int(os.getenv("AI_MIN_SAMPLES", "5")),
#         "norm": os.getenv("AI_NORM", "l2"),
#         "dim_reduction_method": os.getenv("AI_DIM_REDUCTION", "umap"),
#         "top_n_representative": int(os.getenv("AI_TOP_N", "15")),
#         "visualize": os.getenv("AI_VISUALIZE", "true").lower() == "true",
#     }
    
#     if not config["data_folder"]:
#         raise ValueError("AI_DATA_FOLDER environment variable is required")
    
#     return config


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


def agent_improvements_analysis(
    spark: Any = None,
    save_to_table: bool = False
) -> Dict[str, Any]:
    """
    Run the agent improvements analysis pipeline.
    
    Supports configuration via environment variables:
    - AI_DATA_FOLDER: Path to JSON data folder (required)
    - AI_OUTPUT_PATH: Path to save results (optional)
    - AI_CLUSTERING_METHOD: Clustering method (default: auto)
    - AI_MIN_CLUSTER_SIZE: Min cluster size (default: 10)
    - AI_MIN_SAMPLES: Min samples for DBSCAN (default: 5)
    - AI_NORM: Normalization type (default: l2)
    - AI_DIM_REDUCTION: Dimension reduction method (default: umap)
    - AI_TOP_N: Representative points per cluster (default: 15)
    - AI_VISUALIZE: Generate visualizations (default: true)
    - AI_OUTPUT_TABLE: Optional Databricks table name (if save_to_table=True)
    
    Args:
        spark: Optional Spark session for Databricks integration
        save_to_table: Whether to save results to Databricks table
        
    Returns:
        Analysis results dictionary
    """
    logger.info("="*80)
    logger.info("Starting Agent Improvements Analysis")
    logger.info("="*80)
    
    # Load configuration
    logger.info("Loading configuration...")
    azure_config = ConfigManager.load_azure_config()
    ai_config = load_agent_improvemen_config()
    
    logger.info(f"Data folder: {ai_config['data_folder']}")
    logger.info(f"Clustering method: {ai_config['clustering_method']}")
    logger.info(f"Dimension reduction: {ai_config['dim_reduction_method']}")
    
    # Initialize Azure OpenAI client
    logger.info("Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.azure_endpoint,
    )
    
    # Create analyzer
    logger.info("Initializing Agent Improvement Analyzer...")
    analyzer = AgentImprovementAnalyzer(
        azure_client=client,
        llm_model="gpt-4o",
        embedding_model="text-embedding-ada-002"
    )
    
    # Run analysis
    logger.info("Running complete agent improvements analysis...")
    results = analyzer.run_agentperformance_analysis(
        data_folder=ai_config["data_folder"],
        clustering_method=ai_config["clustering_method"],
        norm=ai_config["norm"],
        dim_reduction_method=ai_config["dim_reduction_method"],
        **theme_config["clustering_params"],
        # top_n_representative=ai_config["top_n_representative"],
        # visualize=ai_config["visualize"],
        # output_path=ai_config.get("output_path"),

    )
    
    logger.info(
        "Analysis completed. Found %s clusters and extracted %s topics.",
        results["metadata"]["n_clusters"],
        results["metadata"]["n_topics"],
    )
    
    # Save to file if output path specified
    output_path = ai_config.get("output_path")
    if output_path:
        logger.info(f"Results saved to: {output_path}")
    
    # TODO: Implement Databricks table save if needed
    # if save_to_table and spark is not None:
    #     output_table = os.getenv("AI_OUTPUT_TABLE")
    #     if output_table:
    #         # Implement save logic similar to theme_driver_analysis
    #         pass
    
    logger.info("="*80)
    logger.info("Agent Improvements Analysis Completed Successfully")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    try:
        results = agent_improvements_analysis()
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total improvement areas: {results['metadata']['n_texts']}")
        print(f"Clusters found: {results['metadata']['n_clusters']}")
        print(f"Topics extracted: {results['metadata']['n_topics']}")
        print("="*80)
        
    except Exception as exc:
        logger.exception("Agent improvements analysis failed: %s", exc)
        sys.exit(1)
