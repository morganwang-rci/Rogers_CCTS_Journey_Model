"""Leiden clustering implementation using kNN graph construction."""

import logging
from typing import Dict, Any, Tuple, Literal
import scipy.sparse as sp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datetime import datetime  # Added missing import

# Optional dependencies
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    ig = None
    leidenalg = None

logger = logging.getLogger(__name__)


class LeidenClustering:
    """Leiden clustering implementation with kNN graph construction."""

    def __init__(self):
        """Initialize LeidenClustering."""
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "Leiden algorithm is not available. "
                "Install with: pip install python-igraph leidenalg"
            )

    def cluster(
        self,
        embeddings: np.ndarray,
        k: int = 15,
        use_snn: bool = True,
        resolution: float = 1.0,
        leiden_metric: Literal["cosine", "euclidean"] = "cosine",
        return_graph: bool = False,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """
        Graph-based text clustering using kNN (+ optional SNN) and Leiden.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, dim). Recommended: L2-normalized embeddings.
        k : int
            Number of neighbors for kNN graph.
        use_snn : bool
            Whether to apply Shared Nearest Neighbor (SNN) weighting.
        resolution : float
            Leiden resolution parameter. Higher => more clusters.
        metric : {"cosine", "euclidean"}
            Distance metric for kNN.
        return_graph : bool
            If True, return adjacency matrix and igraph object.
        random_state : int
            Random seed for Leiden reproducibility.

        Returns
        -------
        result : dict
            {
                "labels": np.ndarray[int],
                "n_clusters": int,
                "modularity": float | None,
                "partition": leiden partition object,
                "graph": ig.Graph (optional),
                "adjacency": sparse matrix (optional),
                "metrics": {...},
                "info": {...}
            }
        """

        # -----------------------------
        # 0. Dependency check
        # -----------------------------
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "Leiden algorithm is not available. "
                "Install with: pip install python-igraph leidenalg"
            )

        # -----------------------------
        # 1. Input validation
        # -----------------------------
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array of shape (n_samples, n_features)")

        n_samples, n_features = embeddings.shape

        if n_samples < 2:
            raise ValueError("At least 2 samples are required")

        if k >= n_samples:
            k = n_samples - 1

        if k < 1:
            raise ValueError("k must be >= 1")

        if leiden_metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        # copy to avoid in-place modification
        X = embeddings.copy()

        # -----------------------------
        # 2. Normalize embeddings
        # -----------------------------
        # if leiden_metric == "cosine":
        #     X = normalize(X, norm="l2")

        # -----------------------------
        # 3. Build kNN graph
        # -----------------------------
        nn = NearestNeighbors(
            n_neighbors=k + 1,   # include self
            metric=leiden_metric,
            algorithm="auto"
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # remove self
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]

        # store neighbor sets for SNN
        knn_sets = [set(row) for row in neighbor_indices]

        # -----------------------------
        # 4. Build adjacency matrix
        # -----------------------------
        rows, cols, data = [], [], []

        if use_snn:
            # ---- SNN graph ----
            for i in range(n_samples):
                for pos, j in enumerate(neighbor_indices[i]):
                    j = int(j)
                    if j <= i:
                        continue

                    shared_count = len(knn_sets[i].intersection(knn_sets[j]))
                    if shared_count == 0:
                        continue

                    # base similarity from direct distance
                    d = float(neighbor_distances[i, pos])
                    if leiden_metric == "cosine":
                        base_sim = max(0.0, 1.0 - d)   # cosine similarity
                    else:
                        base_sim = float(np.exp(-d))   # convert distance -> similarity

                    # Recommended SNN weight:
                    # combine neighborhood overlap + direct similarity
                    weight = (shared_count / k) * base_sim

                    # If you want PURE SNN instead, replace above line with:
                    # weight = float(shared_count)

                    if weight > 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(weight)

        else:
            # ---- plain kNN graph ----
            for i in range(n_samples):
                for j, d in zip(neighbor_indices[i], neighbor_distances[i]):
                    j = int(j)
                    if j <= i:
                        continue

                    if leiden_metric == "cosine":
                        sim = max(0.0, 1.0 - float(d))
                    else:
                        sim = float(np.exp(-float(d)))

                    if sim > 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(sim)

        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        adj = adj + adj.T  # make undirected
        adj = sp.triu(adj, k=1).tocoo()  # keep only unique edges

        if adj.nnz == 0:
            raise ValueError(
                "Graph has no edges after construction. "
                "Try increasing k or disabling SNN, or check embedding quality."
            )

        # -----------------------------
        # 5. Convert to igraph
        # -----------------------------
        edge_list = list(zip(adj.row.tolist(), adj.col.tolist()))
        weights = adj.data.tolist()

        g = ig.Graph(n=n_samples, edges=edge_list, directed=False)
        g.es["weight"] = weights

        # -----------------------------
        # 6. Leiden clustering
        # -----------------------------
        np.random.seed(random_state)

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state
        )

        labels = np.array(partition.membership)
        n_clusters = len(np.unique(labels))

        try:
            modularity = partition.modularity
        except Exception:
            modularity = None

        # -----------------------------
        # 7. Evaluation metrics
        # -----------------------------
        metrics_dict: Dict[str, Any] = {}

        # silhouette
        try:
            if 1 < n_clusters < n_samples:
                sil_metric = "cosine" if leiden_metric == "cosine" else "euclidean"
                sil = silhouette_score(X, labels, metric=sil_metric)
                metrics_dict["silhouette_score"] = sil
                metrics_dict["silhouette_normalized"] = ((sil + 1) / 2) * 100
            else:
                metrics_dict["silhouette_score"] = 0.0
                metrics_dict["silhouette_normalized"] = 0.0
        except Exception as e:
            metrics_dict["silhouette_score"] = 0.0
            metrics_dict["silhouette_normalized"] = 0.0
            metrics_dict["silhouette_error"] = str(e)

        # davies-bouldin
        try:
            if 1 < n_clusters < n_samples:
                dbi = davies_bouldin_score(X, labels)
                metrics_dict["davies_bouldin_index"] = dbi
                metrics_dict["davies_bouldin_normalized"] = 100 / (1 + dbi)
            else:
                metrics_dict["davies_bouldin_index"] = float("inf")
                metrics_dict["davies_bouldin_normalized"] = 0.0
        except Exception as e:
            metrics_dict["davies_bouldin_index"] = float("inf")
            metrics_dict["davies_bouldin_normalized"] = 0.0
            metrics_dict["davies_bouldin_error"] = str(e)

        # calinski-harabasz
        try:
            if 1 < n_clusters < n_samples:
                chi = calinski_harabasz_score(X, labels)
                metrics_dict["calinski_harabasz_index"] = chi
                metrics_dict["calinski_harabasz_normalized"] = min(chi / 10, 100)
            else:
                metrics_dict["calinski_harabasz_index"] = 0.0
                metrics_dict["calinski_harabasz_normalized"] = 0.0
        except Exception as e:
            metrics_dict["calinski_harabasz_index"] = 0.0
            metrics_dict["calinski_harabasz_normalized"] = 0.0
            metrics_dict["calinski_harabasz_error"] = str(e)

        # cluster balance
        try:
            _, counts = np.unique(labels, return_counts=True)
            if len(counts) > 1 and np.mean(counts) > 0:
                balance = 100 - (np.std(counts) / np.mean(counts) * 100)
                cluster_balance_score = max(0, min(100, balance))
            else:
                cluster_balance_score = 50.0
            metrics_dict["cluster_balance_score"] = cluster_balance_score
        except Exception as e:
            metrics_dict["cluster_balance_score"] = 50.0
            metrics_dict["cluster_balance_error"] = str(e)

        # cluster count preference
        cluster_count_score = 100 - abs((n_clusters - 5) * 10)
        cluster_count_score = max(0, min(100, cluster_count_score))
        metrics_dict["cluster_count"] = n_clusters
        metrics_dict["cluster_count_score"] = cluster_count_score

        # modularity
        metrics_dict["modularity"] = modularity
        metrics_dict["modularity_normalized"] = (
            max(0.0, min(100.0, modularity * 100))
            if modularity is not None else 0.0
        )

        # composite score
        composite_score = (
            metrics_dict["silhouette_normalized"] * 40
            + metrics_dict["davies_bouldin_normalized"] * 30
            + metrics_dict["calinski_harabasz_normalized"] * 30
            # + metrics_dict["cluster_balance_score"] * 10
            # + metrics_dict["cluster_count_score"] * 10
            # + metrics_dict["modularity_normalized"] * 10
        ) / 100.0
        metrics_dict["composite_score"] = min(composite_score, 100.0)

        # -----------------------------
        # 8. Return
        # -----------------------------
        result: Dict[str, Any] = {
            "method": "KNN+Leiden",
            "n_clusters": n_clusters,
            "modularity": modularity,
            "partition": partition,
            "metrics": metrics_dict,
            "info": {
                "method": "kNN + SNN + Leiden" if use_snn else "kNN + Leiden",
                "k": k,
                "use_snn": use_snn,
                "resolution": resolution,
                "metric": leiden_metric,
                "random_state": random_state,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_edges": int(adj.nnz),
                "timestamp": datetime.now().isoformat()
            }
        }

        if return_graph:
            result["graph"] = g
            result["adjacency"] = adj

        return labels, result