import logging
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict

from openai import AzureOpenAI
from ccts_theme_driver_analysis import ThemeAnalyzer
from report_generation import ConfigManager

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
        "recommendations": results.get("recommendations", []),
        "cluster_payloads": results.get("cluster_payloads", []),
        "raw_results": results
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Theme analysis summary saved to %s", output_file)


def theme_driver_analysis() -> Dict[str, Any]:
    """Run the theme/driver analysis pipeline.
    
    Supports custom Leiden clustering parameters via environment variables:
    - THEME_LEIDEN_K: Number of neighbors (default: 30)
    - THEME_LEIDEN_RESOLUTION: Resolution parameter (default: 0.7)
    - THEME_LEIDEN_USE_SNN: Use Shared Nearest Neighbor (default: True)
    - THEME_LEIDEN_METRIC: Distance metric (default: cosine)
    - THEME_LEIDEN_RANDOM_STATE: Random seed (default: 42)
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
    results = analyzer.run_theme_analysis(
        data_folder=theme_config["data_folder"],
        text_column=theme_config["text_column"],
        clustering_method=theme_config["clustering_method"],
        reduce_dimensions=theme_config["dim_reduction_method"],
        norm = theme_config["norm"],
        **theme_config["clustering_params"],
    )

    logger.info(
        "Theme analysis completed. Found %s clusters.",
        results.get("clustering", {}).get("n_clusters"),
    )

    output_path = theme_config["output_path"]
    if output_path:
        save_theme_results(results, output_path)

    return results


if __name__ == "__main__":
    try:
        theme_driver_analysis()
    except Exception as exc:
        logger.exception("Theme driver analysis failed: %s", exc)
        sys.exit(1)
