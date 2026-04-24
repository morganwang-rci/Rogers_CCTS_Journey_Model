"""DBSCAN/HDBSCAN clustering implementation."""

import logging
from typing import Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional dependency
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

logger = logging.getLogger(__name__)


class DBSCANClustering:
    """DBSCAN/HDBSCAN clustering implementation."""

    def __init__(self):
        """Initialize DBSCANClustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN is not available. Install with: pip install hdbscan")

    def cluster(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        dbscan_metric: str = 'euclidean'
    ) -> Tuple[Any, np.ndarray]:
        """
        Apply DBSCAN (HDBSCAN) clustering.

        Args:
            embeddings: Input embedding vectors
            min_cluster_size: Minimum samples in a cluster
            min_samples: Minimum samples threshold
            metric: Distance metric ('euclidean', 'cosine', etc.)

        Returns:
            Tuple of (model, labels)
        """
        if dbscan_metric == 'cosine':
            # Use precomputed distance for cosine metric
            distance_matrix = cosine_distances(embeddings)
            distance_matrix = distance_matrix.astype('float64')
            clustering_model = hdbscan.HDBSCAN(
                metric='precomputed',
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom',
                allow_single_cluster=False,
                prediction_data=True
            )
            labels = clustering_model.fit_predict(distance_matrix)
        else:
            clustering_model = hdbscan.HDBSCAN(
                metric=dbscan_metric,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom',
                allow_single_cluster=False,
                prediction_data=True
            )
            labels = clustering_model.fit_predict(embeddings)


        dbscan_metrics = {}
        valid_dbscan_mask = labels != -1
        dbscan_labels_clean = labels[valid_dbscan_mask]
        embeddings_filtered = embeddings[valid_dbscan_mask]
        dbscan_unique = np.unique(dbscan_labels_clean)
        dbscan_n_clusters = len(dbscan_unique)
        has_valid_dbscan_silhouette = len(embeddings_filtered) > 1 and dbscan_n_clusters > 1
        noise_count = np.sum(labels == -1)
        noise_percentage = (noise_count / len(labels)) * 100 if len(labels) > 0 else 0
        try:
            if has_valid_dbscan_silhouette:
                dbscan_silhouette = silhouette_score(embeddings_filtered, dbscan_labels_clean)
                # Normalize silhouette score from [-1, 1] to [0, 100]
                dbscan_metrics['silhouette_score'] = dbscan_silhouette
                dbscan_silhouette_normalized = ((dbscan_silhouette + 1) / 2) * 100
                dbscan_metrics['silhouette_score_normalized'] = dbscan_silhouette_normalized
            else:
                dbscan_silhouette_normalized = 0
                dbscan_metrics['silhouette_score'] = 0
                dbscan_metrics['silhouette_score_normalized'] =0
        except:
            dbscan_silhouette_normalized = 0
            dbscan_metrics['silhouette_score'] = 0
            dbscan_metrics['silhouette_score_normalized'] = 0
        
        if has_valid_dbscan_silhouette:
            try:
                dbscan_db_index = davies_bouldin_score(embeddings_filtered, dbscan_labels_clean)
                dbscan_metrics['davies_bouldin_normalized'] = 100 / (1 + dbscan_db_index)
            except:
                dbscan_metrics['davies_bouldin_normalized'] = 10
        else:
            dbscan_metrics['davies_bouldin_normalized'] = 10
        
        if has_valid_dbscan_silhouette:
            try:
                dbscan_ch_index = calinski_harabasz_score(embeddings_filtered, dbscan_labels_clean)
                dbscan_metrics['calinski_harabasz_normalized'] = min(dbscan_ch_index / 10, 100)
            except:
                dbscan_metrics['calinski_harabasz_normalized'] = 10
        else:
            dbscan_metrics['calinski_harabasz_normalized'] = 10
        
        if len(embeddings_filtered) > 0:
            dbscan_counts = np.bincount(dbscan_labels_clean)
            if len(dbscan_counts) > 1:
                dbscan_balance = 100 - (np.std(dbscan_counts) / np.mean(dbscan_counts) * 100)
                dbscan_balance = max(0, min(100, dbscan_balance))
            else:
                dbscan_balance = 10
        else:
            dbscan_balance = 10
        dbscan_metrics['cluster_balance_score'] = dbscan_balance
        
        dbscan_cluster_count_score = 100 - abs((dbscan_n_clusters - 5) * 10)
        dbscan_cluster_count_score = max(0, min(100, dbscan_cluster_count_score))
        dbscan_metrics['cluster_count_score'] = dbscan_cluster_count_score
        dbscan_metrics['n_clusters'] = dbscan_n_clusters
        
        noise_score = 100 - min(noise_percentage, 100)
        dbscan_metrics['noise_percentage'] = noise_percentage
        dbscan_metrics['noise_score'] = noise_score
        
        # silhouette_weight = 40 
        dbscan_composite = (
            (dbscan_metrics.get('silhouette_score_normalized', 0) * 40) +
            dbscan_metrics.get('davies_bouldin_normalized', 0) * 30 +
            dbscan_metrics.get('calinski_harabasz_normalized', 50) * 30 
            # dbscan_balance * 10 +
            # dbscan_cluster_count_score * 10 +
            # noise_score * (10 if noise_percentage < 30 else 5)
        ) / 100
        dbscan_metrics['composite_score'] = min(dbscan_composite, 100)

        return clustering_model, labels, dbscan_metrics