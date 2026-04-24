import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from pathlib import Path
# Import standardized modules from the project structure
from data_processing.data_processing import DataProcessor
from data_processing.embeddings import EmbeddingProcessor
from cluster_method import ClusteringAnalyzer

# Import local specialized components
from .RR_topic_analysis import  RRtopicAnalyzer
from .prompts import build_topic_extraction_prompt
from data_processing.utils import safe_json_loads
from Visualization.visualization import ClusterVisualizer

logger = logging.getLogger(__name__)

class ResolutionRecommendationAnalyzer:
    """
    End-to-end analyzer for Strategic Resolution Recommendations.
    
    Optimized Pipeline using standardized project modules:
    1. Load JSONs -> Extract Recommendations (via DataProcessor + Custom Extraction)
    2. Generate Embeddings (via EmbeddingProcessor)
    3. Reduce Dimensions & Cluster (via ClusteringAnalyzer)
    4. Extract Topics via LLM (via RRTopicAnalyzer)
    """

    def __init__(self, azure_client: AzureOpenAI, llm_model: str = "gpt-4o"):
        self.llm_client = azure_client
        self.llm_model = llm_model
        
        # Initialize standardized processors
        self.embedding_processor = EmbeddingProcessor(azure_client)
        self.cluster_analyzer = ClusteringAnalyzer()
        self.data_processor = DataProcessor()
        
        # Initialize specialized topic analyzer for Resolution Recommendations
        self.topic_analyzer = RRtopicAnalyzer(client=azure_client, model=llm_model) 
        self.visualizer = ClusterVisualizer()

        logger.info("Resolution Recommendation Analyzer initialized with all components")

    def _load_resolution_data(self, data_folder: str) -> pd.DataFrame:
        """
        Load and process resolution recommendation data from JSON files.
        
        Uses DataProcessor to load raw JSONs, then extracts and explodes 
        'strategic_recommendations' into a flat DataFrame suitable for clustering.
        """
        logger.info(f"Loading raw data from {data_folder}")
        # Load raw case journey data
        df_raw = self.data_processor.process_case_journey_folder(data_folder)
        
        if df_raw.empty:
            raise ValueError("No valid data found in data folder.")
        
        # Check if strategic_recommendations exists
        if 'strategic_recommendations' not in df_raw.columns:
            raise KeyError("Column 'strategic_recommendations' not found in loaded data.")
            
        # Explode the list column to get one recommendation per row
        df_exploded = df_raw.explode('strategic_recommendations')
        df_exploded = df_exploded.dropna(subset=['strategic_recommendations'])
        
        # Ensure text is string
        df_exploded['recommendation_text'] = df_exploded['strategic_recommendations'].astype(str)
        
        logger.info(f"Loaded and exploded {len(df_exploded)} recommendations.")
        return df_exploded

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
            selection = self.cluster_analyzer.select_best_clustering_method(
             kmeans_result, dbscan_result, leiden_result
            )

            if selection['best_method'] == 'kmeans':
                return kmeans_result
            
            elif selection['best_method'] == 'dbscan':
                return dbscan_result
            else: 
                return leiden_result
            
        elif method == "kmeans":
            return self.cluster_analyzer.apply_kmeans_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['n_clusters', 'auto_k']})
        elif method == "dbscan":
            return self.cluster_analyzer.apply_dbscan_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['min_cluster_size', 'min_samples', 'dbscan_metric']})
        elif method == "leiden":
            return self.cluster_analyzer.apply_leiden_clustering(embeddings, **{k: v for k, v in clustering_params.items() if k in ['k', 'use_snn', 'resolution_parameter', 'leiden_metric', 'return_graph']})
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
        raw_response = self.topic_analyzer.extract_topics_from_clusters(cluster_payloads)

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

    def run_recommendation_pipeline(
        self, 
        data_folder: str, 
        clustering_method: str = "auto",
        dim_reduction_method: str = "umap",
        normalize_embeddings: bool = True,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline using standardized components.
        """
        logger.info("Starting Resolution Recommendation Analysis Pipeline")
        
        # 1. Load Data
        df = self._load_resolution_data(data_folder)
        texts = df["recommendation_text"].dropna().tolist()
        
        if not texts:
            raise ValueError("No valid recommendation texts found after processing.")

        # 2. Embeddings
        logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # 3. Normalization (Optional but recommended for cosine similarity)
        if normalize_embeddings:
            logger.info("Normalizing embeddings...")
            embeddings = self.normalize_embeddings(embeddings)
        else:
            logger.info("Skipping embedding normalization")

        # 4. Dimension Reduction
        if dim_reduction_method in ("umap", "pca", "auto"):
                logger.info(f"Applying dimension reduction ({dim_reduction_method})...")
                # reduced_embeddings, reduction_info = (
                #     self.embedding_processor.apply_dimension_reduction(
                #         embeddings,
                #         method=dim_reduction_method
                #     )
                # )
                embeddings, reduction_info = self.reduce_dimensions(embeddings, method= dim_reduction_method)
        else:
            logger.info("Skipping dimension reduction")
            reduction_info = None
        # 4. Dimension Reduction
        # logger.info(f"Applying dimension reduction ({dim_reduction_method})...")
        # reduced_embeddings, reduction_info = self.embedding_processor.apply_dimension_reduction(
        #     embeddings, method=dim_reduction_method
        # )

        # 5. Clustering
        logger.info(f"Running clustering algorithm: {clustering_method}...")
        # logger.info("Performing clustering...")
        clustering_result = self.perform_clustering(embeddings, method=clustering_method, **clustering_params)
    
        # final_labels = clustering_result['labels']
        # logger.info(f"Clustering complete. Method: {clustering_result['method']}, Clusters: {clustering_result['n_clusters']}")
        # 6 Visualizing
        logger.info("Visualizing clustering results...")
        # output_dir = os.path.join("./output", "resolution_recommendation_clustering_plots")
        
        # # Ensure output directory exists (handles both file and directory conflicts)
        # try:
        #     os.makedirs(output_dir, exist_ok=True)
        # except NotADirectoryError:
        #     # If a file with the same name exists, remove it first
        #     if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        #         logger.warning(f"Removing conflicting file: {output_dir}")
        #         os.remove(output_dir)
        #         os.makedirs(output_dir, exist_ok=True)
        #     else:
        #         raise
        
        from pathlib import Path

        base_dir = Path("./output")
        output_dir = base_dir / "resolution_recommendation_clustering_plots"

        # Remove file conflicts
        for path_candidate in [base_dir, output_dir]:
            if path_candidate.exists() and path_candidate.is_file():
                logger.warning(f"Removing conflicting file: {path_candidate}")
                path_candidate.unlink()

        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)

        
        
        # Save plots
        import matplotlib.pyplot as plt
        vis_result = self.visualize_clusters(embeddings, clustering_result['labels'])
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
                # plot_path = os.path.join(output_dir, f"recommendation_clustering_{method_name.lower()}_{clustering_result.get('method', 'unknown')}.png")
                plot_path = output_dir / f"recommendation_clustering_{method_name.lower()}_{clustering_result.get('method', 'unknown')}.png"
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
            plot_path = output_dir / f"recommendation_clustering_{clustering_result.get('method', 'unknown')}_clustering_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved clustering distribution plot to {plot_path}")


        # 7. Topic Extraction using Specialized RR Analyzer
        logger.info("Extracting topics via LLM...")
        # We use the specialized analyzer which has the correct prompts for RR
        # topic_results = self.topic_analyzer.analyze_recommendations(
        #     embedding_vectors=reduced_embeddings,
        #     cluster_labels=final_labels,
        #     original_texts=texts,
        #     top_n=top_n_representatives
        # )
        # topics = topic_results['topics']
        logger.info("Extracting topics...")
        topic_result = self.extract_topics(embeddings, clustering_result['labels'], texts)
        return {
            # 'data': df,
            # 'texts': texts,
            'n_clusters': clustering_result.get('n_clusters', 'unknown'),
            'embeddings': embeddings,
            'reduction_info': reduction_info,
            'clustering': clustering_result,
            'clustering_method': clustering_result.get('method', 'unknown'),
            'topics': topic_result,
            'recommendations':topic_result.get('topics',[]),
            'cluster_payloads':topic_result.get('cluster_payloads',[])
        }



  

