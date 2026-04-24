"""Main clustering analyzer that combines all clustering methods."""

import logging
from typing import Dict, Any, Optional, Literal, List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .kmeans import KMeansClustering
from .dbscan import DBSCANClustering
from .leiden import LeidenClustering

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Main analyzer for performing clustering analysis using multiple methods."""

    def __init__(self):
        """Initialize the ClusteringAnalyzer."""
        self.kmeans_clusterer = KMeansClustering()
        self.dbscan_clusterer = DBSCANClustering()
        self.leiden_clusterer = LeidenClustering()
        logger.info("ClusteringAnalyzer initialized with all methods")

    def apply_kmeans_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None,
        auto_k: bool = True
    ) -> Dict[str, Any]:
        """
        Apply KMeans clustering.

        Args:
            embeddings: Input embedding vectors
            n_clusters: Number of clusters (if None and auto_k=True, will be determined automatically)
            auto_k: Whether to determine k automatically

        Returns:
            Dictionary with model, labels, and metadata
        """
        model, labels, n_clusters_used, kmeans_metrics = self.kmeans_clusterer.cluster(embeddings, n_clusters, auto_k)
        return {
            'method': 'kmeans',
            'model': model,
            'n_clusters': n_clusters,
            'labels': labels,
            'embeddings_shape': embeddings.shape,
            **kmeans_metrics
        }

    def apply_dbscan_clustering(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        metric: str = 'euclidean'
    ) -> Dict[str, Any]:
        """
        Apply DBSCAN clustering.

        Args:
            embeddings: Input embedding vectors
            min_cluster_size: Minimum samples in a cluster
            min_samples: Minimum samples threshold
            metric: Distance metric

        Returns:
            Dictionary with model, labels, and metadata
        """
        model, labels, dbscan_metrics = self.dbscan_clusterer.cluster(embeddings, min_cluster_size, min_samples, metric)

        # Calculate noise percentage
        noise_count = np.sum(labels == -1)
        noise_percentage = (noise_count / len(labels)) * 100 if len(labels) > 0 else 0

        return {
            'method': 'dbscan',
            'model': model,
            'labels': labels,
            'noise_percentage': noise_percentage,
            'n_clusters': len(np.unique(labels[labels != -1])),
            'embeddings_shape': embeddings.shape,
            **dbscan_metrics
        }

    def apply_leiden_clustering(
        self,
        embeddings: np.ndarray,
        k: int = 30,
        use_snn: bool = True,
        resolution_parameter: float = 0.7,
        metric: Literal["cosine", "euclidean"] = "cosine",
        return_graph: bool = False,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Apply Leiden clustering.

        Args:
            embeddings: Input embedding vectors
            k: Number of nearest neighbors
            resolution_parameter: Resolution parameter for Leiden
            metric: Distance metric
            random_state: Random seed

        Returns:
            Dictionary with labels and comprehensive results
        """
        labels, results = self.leiden_clusterer.cluster(
            embeddings, k,use_snn,  resolution_parameter, metric, return_graph, random_state
        )

        return {
            'method': 'leiden',
            'labels': labels,
            **results
        }

    def select_best_clustering_method(
        self,
        kmeans_result: Dict[str, Any],
        dbscan_result: Dict[str, Any],
        leiden_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        methods: List[Dict[str, Any]] = [
        {
            "method": "kmeans",
            "n_clusters": kmeans_result.get("n_clusters", 0),
            "noise": 0,
            "composite_score": kmeans_result.get("composite_score", 0),
            "result": kmeans_result,
        },
        {
            "method": "leiden",
            "n_clusters": leiden_result.get("n_clusters", 0),
            "noise": 0,
            "composite_score": leiden_result.get("metrics", {}).get("composite_score", 0)
            if "metrics" in leiden_result
            else leiden_result.get("composite_score", 0),
            "result": leiden_result,
        },
        {
            "method": "dbscan",
            "n_clusters": dbscan_result.get("n_clusters", 0),
            "noise": dbscan_result.get("noise_percentage", 0),
            "composite_score": dbscan_result.get("composite_score", 0),
            "result": dbscan_result,
        },
        ]

    # -----------------------------
    # Apply eligibility rules
    # -----------------------------
        eligible_methods = [
            m for m in methods
            if 3<= m["n_clusters"] <= 20 and m["noise"] < 30
        ]

        # -----------------------------
        # Select best method
        # -----------------------------
        if eligible_methods:
            best = max(eligible_methods, key=lambda m: m["composite_score"])
            decision_rule = "ELIGIBLE_MAX_COMPOSITE_SCORE"
        else:
            best = max(methods, key=lambda m: m["composite_score"])
            decision_rule = "FALLBACK_NO_METHOD_ELIGIBLE"

        return {
            "best_method": best["method"],
            "decision_rule": decision_rule,
            "explainability_score": round(best["composite_score"], 2),
            "n_clusters": best["n_clusters"],
            "noise_percentage": best["noise"],
            "metrics": best["result"].get("metrics", {}),
            "raw_result": best["result"]
        }