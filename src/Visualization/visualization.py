"""Visualization utilities for clustering results."""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional dependency
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """Handles visualization of clustering results."""

    @staticmethod
    def cluster_visual(
        embeddings: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        model=None,
        model_name: str = "Clustering",
        random_state: int = 42,
        tsne_perplexity: int = 30,
        figsize: Tuple[int, int] = (10, 7),
        alpha: float = 0.7,
        point_size: int = 40,
        show_percentage: bool = True
    ) -> Dict[str, Any]:
        """
        Plot clustering results using multiple projection methods.

        Parameters
        ----------
        embeddings : array-like, shape (n_samples, n_features)
        cluster_labels : array-like, optional
        model : clustering model, optional
        model_name : str
        random_state : int
        tsne_perplexity : int
        figsize : tuple
        alpha : float
        point_size : int
        show_percentage : bool

        Returns
        -------
        dict with projection data and cluster distribution
        """
        embeddings = np.asarray(embeddings)

        # Get cluster labels
        if cluster_labels is None:
            if model is None:
                raise ValueError("Provide either cluster_labels or a clustering model.")

            if hasattr(model, "fit_predict"):
                cluster_labels = model.fit_predict(embeddings)
            elif hasattr(model, "predict"):
                cluster_labels = model.predict(embeddings)
            else:
                raise ValueError("Model must have either fit_predict() or predict().")

        cluster_labels = np.asarray(cluster_labels)

        # t-SNE safety
        n_samples = embeddings.shape[0]
        if tsne_perplexity >= n_samples:
            tsne_perplexity = max(2, min(30, n_samples - 1))

        # Dimensionality reduction
        pca = PCA(n_components=2, random_state=random_state)
        embeddings_2d_pca = pca.fit_transform(embeddings)

        tsne = TSNE(n_components=2, random_state=random_state, perplexity=tsne_perplexity)
        embeddings_2d_tsne = tsne.fit_transform(embeddings)

        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, n_jobs = -1)
            embeddings_2d_umap = reducer.fit_transform(embeddings)
        else:
            # Fallback to PCA if UMAP not available
            embeddings_2d_umap = embeddings_2d_pca

        # Build dataframes
        df_pca = pd.DataFrame({
            "x": embeddings_2d_pca[:, 0],
            "y": embeddings_2d_pca[:, 1],
            "cluster": cluster_labels
        })

        df_tsne = pd.DataFrame({
            "x": embeddings_2d_tsne[:, 0],
            "y": embeddings_2d_tsne[:, 1],
            "cluster": cluster_labels
        })

        df_umap = pd.DataFrame({
            "x": embeddings_2d_umap[:, 0],
            "y": embeddings_2d_umap[:, 1],
            "cluster": cluster_labels
        })

        # Cluster distribution
        distribution = pd.Series(cluster_labels).value_counts().sort_index()
        distribution_df = pd.DataFrame({
            "cluster": distribution.index,
            "count": distribution.values
        })
        distribution_df["percentage"] = distribution_df["count"] / distribution_df["count"].sum() * 100

        # Colors
        unique_labels = sorted(np.unique(cluster_labels))
        non_noise_labels = [label for label in unique_labels if label != -1]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(non_noise_labels), 1)))

        label_to_color = {}
        for i, label in enumerate(non_noise_labels):
            label_to_color[label] = colors[i % len(colors)]

        if -1 in unique_labels:
            label_to_color[-1] = "black"   # noise for DBSCAN

        def plot_projection(proj_data: np.ndarray, title: str, xlabel: str, ylabel: str):
            plt.figure(figsize=figsize)

            for label in unique_labels:
                mask = cluster_labels == label
                color = label_to_color[label]
                label_name = "Noise" if label == -1 else f"Cluster {label}"

                plt.scatter(
                    proj_data[mask, 0],
                    proj_data[mask, 1],
                    c=[color],
                    label=label_name,
                    alpha=alpha,
                    s=point_size
                )

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # 1. PCA
        plot_projection(
            embeddings_2d_pca,
            f"{model_name} (PCA)",
            f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
            f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
        )

        # 2. t-SNE
        plot_projection(
            embeddings_2d_tsne,
            f"{model_name} (t-SNE)",
            "t-SNE 1",
            "t-SNE 2"
        )

        # 3. UMAP
        plot_projection(
            embeddings_2d_umap,
            f"{model_name} (UMAP)",
            "UMAP 1",
            "UMAP 2"
        )

        # 4. Distribution
        plt.figure(figsize=figsize)
        x_labels = ["Noise" if c == -1 else f"Cluster {c}" for c in distribution_df["cluster"]]
        bar_colors = [label_to_color[c] for c in distribution_df["cluster"]]

        bars = plt.bar(x_labels, distribution_df["count"], color=bar_colors, alpha=0.85)

        for bar, count, pct in zip(bars, distribution_df["count"], distribution_df["percentage"]):
            text = f"{count}"
            if show_percentage:
                text += f"\n({pct:.1f}%)"

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom"
            )

        plt.title(f"{model_name} Cluster Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return {
            "labels": cluster_labels,
            "pca": embeddings_2d_pca,
            "tsne": embeddings_2d_tsne,
            "umap": embeddings_2d_umap,
            "df_pca": df_pca,
            "df_tsne": df_tsne,
            "df_umap": df_umap,
            "cluster_distribution": distribution_df
        }