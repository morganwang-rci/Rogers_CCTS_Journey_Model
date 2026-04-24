"""Evaluation utilities for clustering results."""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """Handles evaluation of clustering results."""

    @staticmethod
    def evaluate_clustering_result(
        embeddings: np.ndarray,
        labels: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a clustering result.

        For methods like DBSCAN with noise label -1:
        - metrics are computed on non-noise points only
        - noise ratio is reported separately

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            model_name: Name of the model for reporting

        Returns:
            Dictionary with evaluation metrics
        """
        unique_labels = set(labels)
        has_noise = -1 in unique_labels

        n_noise = int(np.sum(labels == -1)) if has_noise else 0
        noise_ratio = n_noise / len(labels)

        # Number of clusters excluding noise
        n_clusters = len(unique_labels - {-1})

        result = {
            "model": model_name,
            "n_samples": len(labels),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
        }

        # Remove noise points for evaluation
        if has_noise:
            mask = labels != -1
            X_eval = embeddings[mask]
            labels_eval = labels[mask]
        else:
            X_eval = embeddings
            labels_eval = labels

        unique_eval_clusters = np.unique(labels_eval)

        # Need at least 2 clusters
        if len(unique_eval_clusters) < 2:
            result["silhouette"] = np.nan
            result["calinski_harabasz"] = np.nan
            result["davies_bouldin"] = np.nan
            result["valid_for_metric"] = False
            return result

        result["silhouette"] = silhouette_score(X_eval, labels_eval)
        result["calinski_harabasz"] = calinski_harabasz_score(X_eval, labels_eval)
        result["davies_bouldin"] = davies_bouldin_score(X_eval, labels_eval)
        result["valid_for_metric"] = True

        return result