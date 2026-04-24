import logging
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict

from openai import AzureOpenAI
from resolution_recommendation import ResolutionRecommendationAnalyzer
from report_generation import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default data folder for resolution recommendations
Journey_dir = "/Workspace/Users/morgan.wang@rci.rogers.ca/share_workspeace/Levels_report_generation/Journey_2512_full"


def load_rr_config() -> Dict[str, Any]:
    """Load resolution recommendation analysis configuration from environment variables."""
    data_folder = os.getenv("JOURNEY_FOLDER", Journey_dir)
    if not data_folder:
        raise ValueError("JOURNEY_FOLDER environment variable is required")

    # Clustering configuration
    clustering_method = os.getenv("RR_CLUSTERING_METHOD", "auto")
    dim_reduction_method = os.getenv("RR_DIM_REDUCTION_METHOD", "umap")
    normalize_embeddings = os.getenv("RR_NORMALIZE_EMBEDDINGS", "False").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    
    # Topic extraction parameters
    top_n_representatives = int(os.getenv("TOP_N_REPRESENTATIVES", "15"))
    
    # Output configuration
    output_path = os.getenv("RR_OUTPUT_PATH", "./output/resolution_recommendations/")
    
    # Clustering method-specific parameters
    clustering_params = {}
    if clustering_method in ("kmeans", "auto"):
        n_clusters_env = os.getenv("RR_KMEANS_N_CLUSTERS", "None")
        if n_clusters_env and n_clusters_env.strip().lower() not in ("none", "", "null"):
            clustering_params["n_clusters"] = int(n_clusters_env)
        else:
            # Let the clustering analyzer auto-determine k
            clustering_params["auto_k"] = True 


    elif clustering_method in ("dbscan", "auto"):
        clustering_params["min_cluster_size"] = int(os.getenv("RR_DBSCAN_MIN_CLUSTER_SIZE", "30"))
        clustering_params["min_samples"] = int(os.getenv("RR_DBSCAN_MIN_SAMPLES", "10"))
        clustering_params["dbscan_metric"] = os.getenv("RR_DBSCAN_METRIC", "euclidean")
        
    elif clustering_method in ("leiden", "auto"):
        clustering_params["k"] = int(os.getenv("RR_LEIDEN_K", "30"))
        clustering_params["resolution_parameter"] = float(os.getenv("RR_LEIDEN_RESOLUTION", "0.7"))
        clustering_params["use_snn"] = os.getenv("RR_LEIDEN_USE_SNN", "True").strip().lower() in ("1", "true", "yes")
        clustering_params["leiden_metric"] = os.getenv("RR_LEIDEN_METRIC", "cosine")
        clustering_params["random_state"] = int(os.getenv("RR_LEIDEN_RANDOM_STATE", "42"))
        clustering_params["return_graph"] =  os.getenv("RR_LEIDEN_RETURN_GRAPH", "False").strip().lower() in ("1", "true", "yes")
    # For "auto" method, no specific parameters needed

    return {
        "data_folder": data_folder,
        "clustering_method": clustering_method,
        "dim_reduction_method": dim_reduction_method,
        "normalize_embeddings": normalize_embeddings,
        "top_n_representatives": top_n_representatives,
        "output_path": output_path,
        "clustering_params": clustering_params,
    }


def save_rr_results(results: Dict[str, Any], output_path: str) -> None:
    """Save resolution recommendation analysis results to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a summary with key information
    summary = {
        # "clustering_method": results.get(""),
        "clustering_method": results.get("clustering_method", results.get("clustering", {}).get("method", "unknown")),
        "n_clusters": results.get("n_clusters", []),
        "raw_response": results.get("topics"),
        "recommendations": results.get("recommendations", []),
        "cluster_payloads": results.get("cluster_payloads", []),
        "raw_results": results
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Resolution recommendation analysis saved to %s", output_file)


def resolution_recommendation_analysis() -> Dict[str, Any]:
    """Run the resolution recommendation analysis pipeline.
    
    Supports custom clustering parameters via environment variables:
    - RR_CLUSTERING_METHOD: Clustering algorithm ("auto", "kmeans", "dbscan", "leiden")
    - RR_DIM_REDUCTION_METHOD: Dimension reduction method ("umap", "pca", "auto")
    - RR_NORMALIZE_EMBEDDINGS: Whether to normalize embeddings (default: True)
    - RR_TOP_N_REPRESENTATIVES: Number of representative points per cluster (default: 15)
    
    KMeans parameters:
    - RR_KMEANS_N_CLUSTERS: Number of clusters (default: 7)
    
    DBSCAN parameters:
    - RR_DBSCAN_MIN_CLUSTER_SIZE: Minimum cluster size (default: 30)
    - RR_DBSCAN_MIN_SAMPLES: Minimum samples (default: 10)
    - RR_DBSCAN_METRIC: Distance metric (default: euclidean)
    
    Leiden parameters:
    - RR_LEIDEN_K: Number of neighbors (default: 30)
    - RR_LEIDEN_RESOLUTION: Resolution parameter (default: 0.7)
    - RR_LEIDEN_USE_SNN: Use Shared Nearest Neighbor (default: True)
    - RR_LEIDEN_METRIC: Distance metric (default: cosine)
    - RR_LEIDEN_RANDOM_STATE: Random seed (default: 42)
    """
    logger.info("Loading resolution recommendation analysis configuration...")
    azure_config = ConfigManager.load_azure_config()
    rr_config = load_rr_config()

    logger.info("Initializing Azure OpenAI client...")
    client = AzureOpenAI(
        api_key=azure_config.api_key,
        api_version=azure_config.api_version,
        azure_endpoint=azure_config.azure_endpoint,
    )

    logger.info("Initializing ResolutionRecommendationAnalyzer...")
    analyzer = ResolutionRecommendationAnalyzer(
        azure_client=client,
        llm_model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    )

    logger.info("Running resolution recommendation analysis pipeline...")
    results = analyzer.run_recommendation_pipeline(
        data_folder=rr_config["data_folder"],
        clustering_method=rr_config["clustering_method"],
        dim_reduction_method=rr_config["dim_reduction_method"],
        normalize_embeddings=rr_config["normalize_embeddings"],
        **rr_config["clustering_params"],
    )

    logger.info(
        "Resolution recommendation analysis completed. Found %s topics.",
        len(results.get("topics", []))
    )

    # Save results
    output_path = rr_config["output_path"]
    if output_path:
        save_rr_results(results, output_path)

    return results


if __name__ == "__main__":
    try:
        resolution_recommendation_analysis()
    except Exception as exc:
        logger.exception("Resolution recommendation analysis failed: %s", exc)
        sys.exit(1)
