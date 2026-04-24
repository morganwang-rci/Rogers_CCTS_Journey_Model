"""Topic analysis utilities for extracting resolution recommendation themes from clusters."""

import logging
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist

from .prompts import build_topic_extraction_prompt, build_breakdown_topic_extraction_prompt
from data_processing.utils import safe_json_loads

logger = logging.getLogger(__name__)


class RRtopicAnalyzer:
    """Handles topic extraction and analysis for resolution recommendations."""

    def __init__(self, client=None, model: str = "gpt-4o"):
        """Initialize TopicAnalyzer.

        Args:
            client: Azure OpenAI client for topic extraction
            model: LLM model to use (default: "gpt-4o")
        """
        self.client = client
        self.model = model
        logger.info(f"TopicAnalyzer initialized with model: {model}")

    def get_top_n_closest_points_per_cluster(
        self,
        embedding_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        original_texts: List[str],
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        For each cluster, select the top N points closest to the cluster centroid.

        Computes the centroid of each cluster and selects the N representative points
        closest to it based on cosine distance. These representatives are used for
        LLM-based topic extraction.

        Args:
            embedding_vectors: Embedding vectors for all points (n_samples, n_features)
            cluster_labels: Cluster labels for each point
            original_texts: Original resolution recommendation texts
            top_n: Number of closest points to select per cluster (default: 15)

        Returns:
            List of cluster payloads with structure:
            {
                "label": int,
                "cluster_size": int,
                "representative_points": List[str],
                "distances_to_centroid": List[float]
            }

        Raises:
            ValueError: If input dimensions don't match
        """
        if len(embedding_vectors) != len(cluster_labels) or len(cluster_labels) != len(original_texts):
            raise ValueError("Dimension mismatch: embeddings, labels, and texts must have same length")

        cluster_payloads = []
        unique_labels = sorted([x for x in np.unique(cluster_labels) if x != -1])

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = embedding_vectors[cluster_indices]
            cluster_texts = [original_texts[i] for i in cluster_indices]

            # Compute centroid from the cluster points
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

            # Distance to centroid using cosine distance
            distances = cdist(cluster_embeddings, centroid, metric="cosine").flatten()

            n_select = min(top_n, len(cluster_texts))
            closest_idx = np.argsort(distances)[:n_select]

            representative_texts = [cluster_texts[i] for i in closest_idx]
            representative_distances = [float(distances[i]) for i in closest_idx]

            cluster_payloads.append({
                "label": int(label),
                "cluster_size": int(len(cluster_texts)),
                "representative_points": representative_texts,
                "distances_to_centroid": representative_distances
            })

            logger.debug(
                f"Cluster {label}: {len(cluster_texts)} total points, "
                f"selected {n_select} representatives"
            )

        return cluster_payloads

    def extract_topics_from_clusters(
        self,
        cluster_payloads: List[Dict[str, Any]],
        temperature: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Extract resolution recommendation topics from cluster payloads using LLM.

        Sends all cluster representatives to LLM in one call for coherent,
        non-overlapping topic assignment. Uses specialized prompts designed for
        resolution recommendations rather than general complaint themes.

        Args:
            cluster_payloads: List of cluster dicts with representative points
            temperature: LLM temperature for sampling (default: 0.1, deterministic)

        Returns:
            List of topic dicts with keys: label, topic, description, reason, short_example

        Raises:
            ValueError: If cluster_payloads is empty or client not provided
            Exception: If LLM API call fails
        """
        if not self.client:
            raise ValueError("Azure OpenAI client not provided")

        if not cluster_payloads:
            raise ValueError("cluster_payloads cannot be empty")

        logger.info(f"Extracting topics from {len(cluster_payloads)} clusters")

        # Use specialized prompt builder for resolution recommendations
        prompt = build_topic_extraction_prompt(cluster_payloads)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior telecom customer service QA expert. "
                            "Your task is to identify precise and non-overlapping "
                            "resolution recommendation themes."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
            )

            # response_text = response.choices[0].message.content
            # topics = safe_json_loads(response_text)

            # logger.info(f"Successfully extracted {len(topics)} topics")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM for topic extraction: {e}")
            raise

    def extract_breakdown_topics_from_clusters(
        self,
        cluster_payloads: List[Dict[str, Any]],
        parent_topic: str,
        parent_description: str,
        temperature: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Extract sub-theme topics for clusters within a parent resolution theme.

        Useful for hierarchical analysis when large clusters need to be further
        subdivided into more specific operational recommendations.

        Args:
            cluster_payloads: List of sub-cluster dicts
            parent_topic: Parent cluster's resolution theme
            parent_description: Parent cluster's description
            temperature: LLM temperature for sampling (default: 0.1)

        Returns:
            List of sub-theme topic dicts with same structure as extract_topics_from_clusters

        Raises:
            ValueError: If cluster_payloads is empty or client not provided
            Exception: If LLM API call fails
        """
        if not self.client:
            raise ValueError("Azure OpenAI client not provided")

        if not cluster_payloads:
            raise ValueError("cluster_payloads cannot be empty")

        logger.info(
            f"Extracting {len(cluster_payloads)} sub-themes "
            f"under parent topic: {parent_topic}"
        )

        # Use specialized prompt builder for breakdown analysis
        prompt = build_breakdown_topic_extraction_prompt(
            cluster_payloads,
            parent_topic,
            parent_description
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior telecom customer service QA expert. "
                            "Your task is to identify precise and non-overlapping "
                            "sub-themes within a parent resolution recommendation."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
            )

            # response_text = response.choices[0].message.content
            # topics = safe_json_loads(response_text)

            logger.info(f"Successfully extracted sub-themes")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM for sub-theme extraction: {e}")
            raise
