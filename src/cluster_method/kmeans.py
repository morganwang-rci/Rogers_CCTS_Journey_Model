"""K-Means clustering implementation."""

import logging
from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator

logger = logging.getLogger(__name__)


class KMeansClustering:
    """K-Means clustering implementation with automatic k determination."""

    def __init__(self):
        """Initialize KMeansClustering."""
        pass

    def determine_optimal_k(self, embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 10) -> int:
        """
        Determine the optimal number of clusters using both the elbow method and silhouette scores.

        Args:
            embeddings: Input embedding vectors
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider

        Returns:
            Optimal number of clusters
        """
        # Ensure we don't try more clusters than we have samples
        max_possible_clusters = min(max_clusters, len(embeddings) - 1)
        if max_possible_clusters <= min_clusters:
            return min_clusters

        # Calculate inertia (for elbow method) and silhouette scores
        inertia_values = []
        silhouette_values = []
        k_values = range(min_clusters, max_possible_clusters + 1)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertia_values.append(kmeans.inertia_)

            # Only calculate silhouette if we have enough samples and more than one cluster
            if len(embeddings) > k and k > 1:
                silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
                silhouette_values.append(silhouette_avg)
            else:
                silhouette_values.append(0)

        # Try to find the elbow point using the KneeLocator
        try:
            kneedle = KneeLocator(
                list(k_values),
                inertia_values,
                S=1.0,
                curve="convex",
                direction="decreasing",
            )
            elbow_k = kneedle.elbow
        except Exception:
            elbow_k = None

        # Find the k with the highest silhouette score
        best_silhouette_k = (
            k_values[np.argmax(silhouette_values)] if silhouette_values else None
        )

        # Decision logic for optimal k
        if elbow_k and best_silhouette_k:
            # If both methods agree, use that value
            if elbow_k == best_silhouette_k:
                optimal_k = elbow_k
            # Otherwise, prefer silhouette if it's reasonable
            elif best_silhouette_k >= min_clusters and best_silhouette_k <= max_clusters:
                optimal_k = best_silhouette_k
            else:
                optimal_k = elbow_k
        elif elbow_k:
            optimal_k = elbow_k
        elif best_silhouette_k:
            optimal_k = best_silhouette_k
        else:
            # Default to middle of range if methods fail
            optimal_k = (min_clusters + max_possible_clusters) // 2

        # Ensure the optimal k is within our desired range
        optimal_k = max(min_clusters, min(optimal_k, max_possible_clusters))

        return optimal_k

    def cluster(self, embeddings: np.ndarray, n_clusters: Optional[int] = None, auto_k: bool = True) -> Tuple[KMeans, np.ndarray, int]:
        """
        Apply KMeans clustering.

        Args:
            embeddings: Input embedding vectors
            n_clusters: Number of clusters (if None and auto_k=True, will be determined automatically)
            auto_k: Whether to determine k automatically

        Returns:
            Tuple of (model, labels, n_clusters)
        """

        if n_clusters is None and auto_k:
            print("  Determining optimal k...")
            n_clusters = self.determine_optimal_k(embeddings, min_clusters=2, max_clusters=10)
        elif n_clusters is None:
            n_clusters = 4  # Default
            print(f"  Using default k={n_clusters}")
        else:
            print(f"  Using user-specified k={n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        print(f"  ✓ KMeans complete: {len(np.unique(kmeans_labels))} clusters created")
        kmeans_metrics = {}
        kmeans_metrics['n_clusters'] = n_clusters
        kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
        kmeans_silhouette_normalized = ((kmeans_silhouette + 1) / 2) * 100
        kmeans_metrics['silhouette_score'] = kmeans_silhouette_normalized

        kmeans_db_index = davies_bouldin_score(embeddings, kmeans_labels)
        kmeans_metrics['davies_bouldin_normalized'] = 100 / (1 + kmeans_db_index)

        kmeans_ch_index = calinski_harabasz_score(embeddings, kmeans_labels)
        kmeans_metrics['calinski_harabasz_normalized'] = min(kmeans_ch_index / 10, 100)

        kmeans_counts = np.bincount(kmeans_labels)
        kmeans_balance = 100 - (np.std(kmeans_counts) / np.mean(kmeans_counts) * 100)
        kmeans_balance = max(0, min(100, kmeans_balance))
        kmeans_metrics['cluster_balance_score'] = kmeans_balance

        kmeans_cluster_count_score = 100 - abs((n_clusters - 5) * 10)
        kmeans_cluster_count_score = max(0, min(100, kmeans_cluster_count_score))
        kmeans_metrics['cluster_count_score'] = kmeans_cluster_count_score
        kmeans_metrics['n_clusters'] = n_clusters

        kmeans_composite = (
            kmeans_silhouette_normalized * 40 +
            kmeans_metrics['davies_bouldin_normalized'] * 30 +
            kmeans_metrics['calinski_harabasz_normalized'] * 30
        ) / 100
        kmeans_metrics['composite_score'] = kmeans_composite

        return kmeans, kmeans_labels, n_clusters, kmeans_metrics