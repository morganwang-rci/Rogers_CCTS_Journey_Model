"""Agent Improvements Analyzer Module.

Main analyzer for agent performance improvement opportunities using clustering and LLM topic extraction.
"""

import os
import logging
import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from openai import AzureOpenAI
from scipy.spatial.distance import cdist

# Import standardized modules from the project structure
from data_processing.data_processing import DataProcessor
from data_processing.embeddings import EmbeddingProcessor
from cluster_method import ClusteringAnalyzer
from Visualization.visualization import ClusterVisualizer

# Import local specialized components
from .ai_topic_analysis import AITopicAnalyzer

logger = logging.getLogger(__name__)


class AgentImprovementAnalyzer:
    """
    End-to-end analyzer for Agent Performance Improvement Opportunities.
    
    Pipeline:
    1. Load JSONs -> Extract agent improvement areas
    2. Generate Embeddings (via EmbeddingProcessor)
    3. Reduce Dimensions & Cluster (via ClusteringAnalyzer)
    4. Extract Topics via LLM (via AITopicAnalyzer)
    5. Visualize results (via ClusterVisualizer)
    """

    def __init__(
        self, 
        azure_client: AzureOpenAI, 
        # llm_model: str = "gpt-4o",
        # embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize the Agent Improvement Analyzer.
        
        Args:
            azure_client: Azure OpenAI client instance
            llm_model: Model for topic extraction (default: gpt-4o)
            embedding_model: Model for embeddings (default: text-embedding-ada-002)
        """
        self.llm_client = azure_client
        # self.llm_model = llm_model
        # self.embedding_model = embedding_model
        
        # Initialize standardized processors
        self.embedding_processor = EmbeddingProcessor(azure_client)
        self.cluster_analyzer = ClusteringAnalyzer()
        self.data_processor = DataProcessor()
        
        # Initialize specialized topic analyzer
        self.topic_analyzer = AITopicAnalyzer(client=azure_client)
        self.visualizer = ClusterVisualizer()

        logger.info("Agent Improvement Analyzer initialized with all components")

    def load_agent_improvement_data(self, data_folder: str) -> pd.DataFrame:
        """
        Load and process agent improvement data from JSON files.
        
        Expects JSON files with nested structure containing agent evaluations
        with improvement_areas field.
        
        Args:
            data_folder: Path to folder containing JSON files
            
        Returns:
            DataFrame with exploded improvement_areas (one row per improvement)
        """
    
        data_folder = Path(data_folder)
        logger.info(f"Loading agent improvement data from: {data_folder}")
        
        # Use the comprehensive case interaction extraction
        df =  self.data_processor.process_case_interaction_folder(data_folder)
        
        if df.empty:
            raise logger.info("No valid records extracted from JSON files")
        
        logger.info(f"Loaded {len(df)} total agent evaluation records")
        
        # Extract and explode improvement_areas
        if 'improvement_areas' not in df.columns:
            raise logger.info("No 'improvement_areas' column found in data")
        
        # Drop rows without improvement areas
        df_with_improvements = df.dropna(subset=['improvement_areas'])
        logger.info(f"Records with improvements: {len(df_with_improvements)}")
        
        # Explode the list column
        df_exploded = df_with_improvements.explode('improvement_areas')
        df_exploded = df_exploded.dropna(subset=['improvement_areas'])
        df_exploded = df_exploded[df_exploded['improvement_areas'].str.strip().astype(bool)]
        
        logger.info(f"After explosion: {len(df_exploded)} improvement items")
        return df_exploded.reset_index(drop=True)
    
    
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


    def run_agentperformance_analysis(
        self,
        data_folder: str,
        clustering_method: str = "auto",
        # min_cluster_size: int = 10,
        # min_samples: int = 5,
        norm: bool = False,
        dim_reduction_method: str = "umap",
        target_dim: Optional[int] = None,
        top_n_representative: int = 15,
        output_path: Optional[str] = None,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Run complete agent improvement analysis pipeline.
        
        Args:
            data_folder: Path to folder with JSON files
            clustering_method: 'kmeans', 'dbscan', 'leiden', or 'auto'
            norm: Normalization type ('l2' for cosine)
            reduce_dimensions: Whether to apply dimension reduction
            dim_reduction_method: 'pca', 'umap', or 'auto'
            target_dim: Target dimensions (None for auto)
            top_n_representative: Number of representative points per cluster
            output_path: Optional path to save results
            
        Returns:
            Dictionary containing:
                - texts: List of improvement area texts
                - embeddings: Embedding vectors
                - clustering: Clustering results
                - topics: Extracted topics DataFrame
                - cluster_payloads: Representative points per cluster
                - visualization: Visualization results (if enabled)
        """
        logger.info("="*80)
        logger.info("Starting Agent Improvement Analysis Pipeline")
        logger.info("="*80)
        
        # Step 1: Load and preprocess data
        logger.info("\nInfo: Loading agent improvement data...")
        df_improvements = self.load_agent_improvement_data(data_folder)
        texts = df_improvements['improvement_areas'].tolist()
        logger.info(f"Extracted {len(texts)} improvement areas")
        
        # Step 2: Generate embeddings
        logger.info("\n[Step 2/6] Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 3: Normalize embeddings
        logger.info(f"\n[Step 3/6] Normalizing embeddings (norm={norm})...")
        if norm is not None:
            logger.info("Embedding Normalization...")
            embeddings = self.normalize_embeddings(embeddings)
        else:
            logger.info("Skipping embedding normalization")
        if dim_reduction_method in ("umap", "pca", "auto"):
            logger.info("Reducing dimensions...")
            embeddings, reduction_info = self.reduce_dimensions(embeddings, method = dim_reduction_method)
        else:
            reduction_info = None
        
        # 4. Perform clustering
        logger.info("Performing clustering...")
        clustering_result = self.perform_clustering(embeddings, method=clustering_method, **clustering_params)
        # 5. Visualizing
        logger.info("Visualizing clustering results...")
        import os
        from pathlib import Path
        if output_path is None:
            base_dir = Path("./output")
            output_dir = base_dir / "agent_improvement_clustering_plots"
        else:
            output_dir = output_path 

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


        