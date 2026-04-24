"""Embedding utilities for text processing and dimension reduction."""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

# Optional dependencies
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Handles text embeddings and dimension reduction."""

    def __init__(self, client=None):
        """Initialize EmbeddingProcessor.

        Args:
            client: OpenAI client for generating embeddings
        """
        self.client = client

    def get_embeddings_in_batches(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings for texts in batches.

        Args:
            texts: List of text strings
            model: Embedding model to use
            batch_size: Number of texts per batch

        Returns:
            Array of embeddings
        """
        if not self.client:
            raise ValueError("OpenAI client not provided")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i} to {i + len(batch) - 1}")

            response = self.client.embeddings.create(
                model=model,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def normalize_embeddings(self, embeddings: np.ndarray, norm: str = 'l2') -> np.ndarray:
        """
        Normalize embeddings.

        Args:
            embeddings: Input embeddings
            norm: Normalization method ('l2', 'l1', etc.)

        Returns:
            Normalized embeddings
        """
        return normalize(embeddings, norm=norm)

    def apply_dimension_reduction(
        self,
        embeddings_array: np.ndarray,
        method: str = "auto",
        target_dim: Optional[int] = None,
        explained_variance_threshold: float = 0.7
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply dimension reduction to embeddings.

        Args:
            embeddings_array: Input embeddings
            method: 'pca', 'umap', or 'auto'
            target_dim: Target dimensions (if None, will be determined automatically)
            explained_variance_threshold: For PCA, minimum explained variance to retain

        Returns:
            Tuple of (reduced_embeddings, reduction_info)
        """
        print(f"Original embedding dimensions: {embeddings_array.shape}")

        reduction_info = {
            "original_dim": embeddings_array.shape[1],
            "method": method,
        }

        if method == "auto":
            # Choose method based on data size and availability
            if len(embeddings_array) > 1000 and UMAP_AVAILABLE:
                method = "umap"
            else:
                method = "pca"
            reduction_info["method"] = method

        if method == "pca":
            # Determine optimal number of components for PCA
            if target_dim is None:
                # First, fit PCA with all components to analyze explained variance
                pca_full = PCA()
                pca_full.fit(embeddings_array)

                # Find number of components needed for desired explained variance
                cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
                target_dim = np.argmax(cumsum_var >= explained_variance_threshold) + 1
                target_dim = min(target_dim, embeddings_array.shape[1])
                target_dim = max(target_dim, 5)  # Minimum 5 dimensions

            # Apply PCA with determined number of components
            pca = PCA(n_components=target_dim, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings_array)

            explained_variance = np.sum(pca.explained_variance_ratio_)
            reduction_info.update({
                "reduced_dim": target_dim,
                "explained_variance": explained_variance,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "reducer": pca,
            })

            print(f"PCA: Reduced to {target_dim} dimensions, explained variance: {explained_variance:.3f}")

        elif method == "umap" and UMAP_AVAILABLE:
            # Determine optimal number of components for UMAP
            if target_dim is None:
                # Use heuristic: sqrt of original dimensions, but at least 5 and at most 30
                target_dim = max(5, min(30, int(np.sqrt(embeddings_array.shape[1]))))

            # UMAP parameters
            n_neighbors = min(30, len(embeddings_array) - 1)
            min_dist = 0.3

            umap_reducer = umap.UMAP(
                n_components=target_dim,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                metric="cosine",
            )

            reduced_embeddings = umap_reducer.fit_transform(embeddings_array)

            reduction_info.update({
                "reduced_dim": target_dim,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "reducer": umap_reducer,
            })

            print(f"UMAP: Reduced to {target_dim} dimensions")

        else:
            # Fallback to PCA if UMAP not available
            print("Falling back to PCA...")
            return self.apply_dimension_reduction(
                embeddings_array,
                method="pca",
                target_dim=target_dim,
                explained_variance_threshold=explained_variance_threshold,
            )

        return reduced_embeddings, reduction_info