"""Topic analysis utilities for extracting themes from clusters."""

import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from data_processing.utils import safe_json_loads
logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """Handles topic extraction and analysis from clustered data."""

    def __init__(self, client=None):
        """Initialize TopicAnalyzer.

        Args:
            client: OpenAI client for topic extraction
        """
        self.client = client

    def get_top_n_closest_points_per_cluster(
        self,
        embedding_vectors: np.ndarray,
        cluster_labels: np.ndarray,
        original_texts: List[str],
        top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        For each cluster, select the top N points closest to the cluster centroid.

        Args:
            embedding_vectors: Embedding vectors for all points
            cluster_labels: Cluster labels for each point
            original_texts: Original text for each point
            top_n: Number of closest points to select per cluster

        Returns:
            List of cluster payloads with representative points
        """
        cluster_payloads = []

        unique_labels = sorted([x for x in np.unique(cluster_labels) if x != -1])

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = embedding_vectors[cluster_indices]
            cluster_texts = [original_texts[i] for i in cluster_indices]

            # Compute centroid from the cluster points
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

            # Distance to centroid
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

        return cluster_payloads

    def build_all_clusters_prompt(self, cluster_payloads: List[Dict[str, Any]]) -> str:
        """
        Build one contrastive prompt containing all clusters together.
        This helps the LLM assign distinct, non-overlapping topics.

        Args:
            cluster_payloads: List of cluster payloads

        Returns:
            Formatted prompt string
        """
        cluster_sections = []

        for cluster in cluster_payloads:
            points = "\n".join(
                [f"{i+1}. {text}" for i, text in enumerate(cluster["representative_points"])]
            )

            cluster_sections.append(
                f"""
                Label {cluster['label']} (Cluster size: {cluster['cluster_size']}):
                Representative agent improvement points:
                {points}
                """.strip()
            )

        prompt = f"""
        You are a senior QA analyst for telecom customer service operations.

        You are given multiple clusters of main issue of CCTS Complaints
        Each label represents one cluster.
        For each cluster, 15 representative statements were selected to represent the cluster
        Your task:
        - For EACH label, identify the single strongest Complaint theme.
        - Topics MUST be unique across labels and clearly differentiated.
        - Avoid overlap across labels.
        - Avoid generic titles such as:
        - communication issue
        - poor service
        - customer dissatisfaction
        - expectation setting
        unless you make them operationally specific.
        - If issues are tied to telecom products, billing, returns, cancellations, promotions, plans, credits, device financing, policies, or service processes, explicitly include those details.


        Output requirements:
        - Return VALID JSON only
        - Return a JSON array
        - Each object must contain exactly these fields:
        - "label"
        - "topic"
        - "description"
        - "reason"
        - "short_example"

        Where:
        - "topic" = short, specific, unique title/theme
        - "description" = 2-3 concise sentences describing the dominant issue and what needs improvement
        - "reason" = 2-3 concise sentences explaining the dominant issue and why it matters operationally
        - "short_example" = one brief example

        Input clusters:
        {chr(10).join(cluster_sections)}

        Return format:
        [
        {{
            "label": 0,
            "topic": "Specific unique topic title",
            "description": "Concise explanation of the dominant issue and what needs improvement.",
            "reason": "explain the logic of the theme of this cluster",
            "short_example": "provide one example."
        }},
        {{
            "label": 1,
            "topic": "Another distinct topic title",
            "description": "Concise explanation.",
            "reason": "explain the logic of the theme of this cluster",
            "short_example": "provide one example."
        }}
        ]
        """
        return prompt

    def find_topics_all_clusters(
        self,
        cluster_payloads: List[Dict[str, Any]],
        model: str = "gpt-4o"
    ) -> str:
        """
        Send all cluster representatives together in one LLM call.

        Args:
            cluster_payloads: List of cluster payloads
            model: OpenAI model to use

        Returns:
            Raw LLM response content
        """
        if not self.client:
            raise ValueError("OpenAI client not provided")

        prompt = self.build_all_clusters_prompt(cluster_payloads)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior telecom customer service QA expert. "
                        "Your task is to identify precise and non-overlapping cluster topics."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
        # response_text = response.choices[0].message.content
        # topics = safe_json_loads(response_text)

        # logger.info(f"Successfully extracted {len(topics)} topics")

        return response.choices[0].message.content