"""Main theme analyzer that orchestrates the complete theme analysis pipeline."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# from ..Visualization.visualization import ClusterVisualizer
from .evaluation import ClusterEvaluator
from .topic_analysis import TopicAnalyzer
from cluster_method import ClusteringAnalyzer
from data_processing.data_processing import DataProcessor
from data_processing.embeddings import EmbeddingProcessor
from Visualization.visualization import ClusterVisualizer
logger = logging.getLogger(__name__)


class ThemeAnalyzer:
    """Complete theme analysis pipeline for complaint data."""

    def __init__(self, client=None):
        """Initialize ThemeAnalyzer.

        Args:
            client: OpenAI client for embeddings and topic analysis
        """
        self.client = client
        self.data_processor = DataProcessor()
        self.embedding_processor = EmbeddingProcessor(client)
        self.cluster_analyzer = ClusteringAnalyzer()
        self.visualizer = ClusterVisualizer()
        self.evaluator = ClusterEvaluator()
        self.topic_analyzer = TopicAnalyzer(client)

        logger.info("ThemeAnalyzer initialized with all components")

    def process_complaint_data(self, folder_path: str, pattern: str = "*.json") -> pd.DataFrame:
        """
        Process complaint journey analysis files.

        Args:
            folder_path: Path to folder containing JSON files
            pattern: File pattern to match

        Returns:
            Processed DataFrame
        """
        return self.data_processor.process_case_journey_folder(folder_path, pattern)

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for text data.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Embedding array
        """
        return self.embedding_processor.get_embeddings_in_batches(texts, batch_size=batch_size)
    
    def normalize_embeddings(self, embeddings: np.ndarray, norm: str = 'l2'):
        return self.embedding_processor.normalize_embeddings(embeddings, norm)  

    def reduce_dimensions(self, embeddings: np.ndarray, method: str = "auto") -> tuple:
        """
        Apply dimension reduction to embeddings.

        Args:
            embeddings: Input embeddings
            method: Reduction method ('pca', 'umap', 'auto')

        Returns:
            Tuple of (reduced_embeddings, reduction_info)
        """
        return self.embedding_processor.apply_dimension_reduction(embeddings, method=method)

    def perform_clustering(self, embeddings: np.ndarray, method: str = "auto", **clustering_params) -> Dict[str, Any]:
        """
        Perform clustering analysis.

        Args:
            embeddings: Input embeddings
            method: Clustering method ('kmeans', 'dbscan', 'leiden', 'auto')

        Returns:
            Clustering results
        """
        if method == "auto":
            # Try different methods and select best
            kmeans_result = self.cluster_analyzer.apply_kmeans_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['n_clusters', 'auto_k']})
            dbscan_result = self.cluster_analyzer.apply_dbscan_clustering(embeddings,  **{k: v for k, v in clustering_params.items() if k in ['min_cluster_size', 'min_samples', 'dbscan_metric']})
            leiden_result = self.cluster_analyzer.apply_leiden_clustering(embeddings,  **{k: v for k, v in clustering_params.items() if k in ['k', 'use_snn', 'resolution_parameter', 'leiden_metric', 'return_graph']})
            selection = self.cluster_analyzer.select_best_clustering_method(kmeans_result, dbscan_result, leiden_result)

            if selection['best_method'] == 'kmeans':
                return kmeans_result
            
            elif selection['best_method'] == 'dbscan':
                return dbscan_result
            else: 
                return leiden_result
            
        elif method == "kmeans":
            return self.cluster_analyzer.apply_kmeans_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['n_clusters', 'auto_k']})
        elif method == "dbscan":
            return self.cluster_analyzer.apply_dbscan_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['min_cluster_size', 'min_samples', 'metric']})
        elif method == "leiden":
            return self.cluster_analyzer.apply_leiden_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['k', 'use_snn', 'resolution_parameter', 'metric', 'return_graph', 'random_state']})
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Visualize clustering results.

        Args:
            embeddings: Input embeddings
            cluster_labels: Cluster labels
            **kwargs: Additional visualization parameters

        Returns:
            Visualization data
        """
        return self.visualizer.cluster_visual(embeddings, cluster_labels=cluster_labels, **kwargs)

    def evaluate_clusters(self, embeddings: np.ndarray, labels: np.ndarray, model_name: str = "clusters") -> Dict[str, Any]:
        """
        Evaluate clustering results.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            model_name: Name for the model

        Returns:
            Evaluation metrics
        """
        return self.evaluator.evaluate_clustering_result(embeddings, labels, model_name)

    def extract_topics(self, embeddings: np.ndarray, cluster_labels: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """
        Extract topics from clusters.

        Args:
            embeddings: Input embeddings
            cluster_labels: Cluster labels
            texts: Original texts

        Returns:
            Topic analysis results
        """
        # Get representative points for each cluster
        cluster_payloads = self.topic_analyzer.get_top_n_closest_points_per_cluster(
            embeddings, cluster_labels, texts, top_n=15
        )

        # Extract topics using LLM
        raw_response = self.topic_analyzer.find_topics_all_clusters(cluster_payloads)

        return {
            'cluster_payloads': cluster_payloads,
            'raw_response': raw_response,
            'topics': self._parse_topic_response(raw_response)
        }

    def _parse_topic_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response for topics.

        Args:
            response: Raw LLM response

        Returns:
            Parsed topic data
        """
        try:
            # Try to parse as JSON
            import json
            from data_processing.utils import safe_json_loads
            return safe_json_loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse topic response as JSON: {response}")
            return []

    def run_theme_analysis(
        self,
        data_folder: str,
        text_column: str = "primary_complaint_issue",
        clustering_method: str = "auto",
        dim_reduction_method: str = "umap",
        norm: bool = True,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Run the complete theme analysis pipeline.

        Args:
            data_folder: Path to folder containing complaint data
            text_column: Column name containing text to analyze
            clustering_method: Clustering method to use
            reduce_dimensions: Whether to apply dimension reduction

        Returns:
            Complete analysis results
        """
        logger.info("Starting complete theme analysis pipeline")

        # 1. Process data
        logger.info("Processing complaint data...")
        df = self.process_complaint_data(data_folder)
        issue_clean = df[text_column].str.split('[').str[0].str.strip()
        texts = issue_clean.dropna().tolist()
        # df['primary_complaint_issue'].str.split('[').str[0].str.strip()
        # 2. Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)

        # 3. Optional dimension reduction and normalization
        if norm:
            logger.info("Embedding Normalization...")
            embeddings = self.normalize_embeddings(embeddings)
        else:
            logger.info("Skipping embedding normalization")
        if dim_reduction_method in ("umap", "pca", "auto"):
            logger.info("Reducing dimensions...")
            embeddings, reduction_info = self.reduce_dimensions(embeddings, method = dim_reduction_method)
        else:
            reduction_info = None
        
        # if norm:
        #     logger.info("Embedding Normalization...")
        #     embeddings = self.normalize_embeddings(embeddings)
        


        # 4. Perform clustering
        logger.info("Performing clustering...")
        clustering_result = self.perform_clustering(embeddings, method=clustering_method, **clustering_params)

        logger.info("Visualizing clustering results...")
        import os
        from pathlib import Path
        base_dir = Path("./output")
        output_dir = base_dir / "theme_clustering_plots"

        # Remove file conflicts
        for path_candidate in [base_dir, output_dir]:
            if path_candidate.exists() and path_candidate.is_file():
                logger.warning(f"Removing conflicting file: {path_candidate}")
                path_candidate.unlink()
            elif path_candidate.exists() and path_candidate.is_dir():
                # Directory exists, no action needed
                pass
        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plots
        import matplotlib.pyplot as plt
        vis_result = self.visualize_clusters(embeddings, clustering_result['labels'])
        
        # Save each plot
        for i, (method_name, df) in enumerate([("PCA", vis_result.get("df_pca")), ("t-SNE", vis_result.get("df_tsne")), ("UMAP", vis_result.get("df_umap"))]):
            if df is not None:
                plt.figure(figsize=(10, 7))
                unique_labels = sorted(df['cluster'].unique())
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for j, label in enumerate(unique_labels):
                    mask = df['cluster'] == label
                    plt.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                              c=[colors[j]], 
                              label=f"Cluster {label}" if label != -1 else "Noise",
                              alpha=0.7, s=40)
                plt.title(f"Clustering Results ({method_name}) - {clustering_result.get('method', 'Unknown')}")
                plt.xlabel(f"{method_name} 1")
                plt.ylabel(f"{method_name} 2")
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"ccts_theme_clustering_{method_name.lower()}_{clustering_result.get('method', 'unknown')}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved {method_name} plot to {plot_path}")
        
        distribution_df = vis_result.get('cluster_distribution')
        if distribution_df is not None:
            plt.figure(figsize=(10, 7))
            x_labels = ["Noise" if c == -1 else f"Cluster {c}" for c in distribution_df["cluster"]]
            bars = plt.bar(x_labels, distribution_df["count"], alpha=0.85)
            for bar, count, pct in zip(bars, distribution_df["count"], distribution_df["percentage"]):
                text = f"{count}\n({pct:.1f}%)"
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    text,
                    ha="center",
                    va="bottom"
                )
            plt.title(f"{clustering_result.get('method', 'unknown')} Cluster Distribution")
            plt.xlabel("Cluster")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = output_dir / f"theme_clustering_{clustering_result.get('method', 'unknown')}_clustering_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved clustering distribution plot to {plot_path}")


        # 5. Extract topics
        logger.info("Extracting topics...")
        topic_result = self.extract_topics(embeddings, clustering_result['labels'], texts)

        # 6. Evaluate results
        logger.info("visualization results...")
        # evaluation = self.evaluate_clusters(embeddings, clustering_result['labels'])
        # self.visualize_clusters(embeddings, clustering_result['labels'])
        return {
            # 'data': df,
            # 'texts': texts,
            'n_clusters': clustering_result.get('n_clusters', 'unknown'),
            'embeddings': embeddings,
            'reduction_info': reduction_info,
            'clustering': clustering_result,
            'clustering_method': clustering_result.get('method', 'unknown'),
            'topics': topic_result,
            'themes':topic_result.get('topics',[]),
            'cluster_payloads':topic_result.get('cluster_payloads',[])
        }