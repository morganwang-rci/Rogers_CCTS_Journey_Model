"""Agent Improvements Topic Analysis Module.

Specialized topic extraction for agent performance improvement clusters.
"""

import logging
import numpy as np
from typing import List, Dict, Any
from openai import AzureOpenAI
from .prompts import build_agent_improvement_prompt
from data_processing.utils import safe_json_loads

logger = logging.getLogger(__name__)


class AITopicAnalyzer:
    """
    Analyzes agent improvement clusters to extract meaningful topics using LLM.
    
    This class handles the extraction of operational improvement themes from
    clustered agent performance data.
    """
    
    def __init__(self, client: AzureOpenAI, model: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the AI Topic Analyzer.
        
        Args:
            client: Azure OpenAI client instance
            model: LLM model to use for topic extraction
            temperature: Temperature setting for LLM generation
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        
        logger.info(f"AITopicAnalyzer initialized with model={model}, temperature={temperature}")
    
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
    
    
    def extract_topics_from_clusters(
        self, 
        cluster_payloads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract topics from all clusters in a single LLM call.
        
        Uses contrastive prompting to ensure distinct, non-overlapping topics
        across all clusters.
        
        Args:
            cluster_payloads: List of cluster dictionaries containing:
                - label: cluster identifier
                - cluster_size: number of items in cluster
                - representative_points: list of representative text samples
                
        Returns:
            List of topic dictionaries with fields:
                - label: cluster label
                - topic: topic title
                - description: detailed description
                - reason: operational rationale
                - short_example: concrete example
                
        Raises:
            Exception: If LLM call fails or JSON parsing fails
        """
        if not cluster_payloads:
            logger.warning("No cluster payloads provided for topic extraction")
            return []
        
        logger.info(f"Extracting topics from {len(cluster_payloads)} clusters")
        
        # Build the contrastive prompt
        prompt = self.build_agent_improvement_prompt(cluster_payloads)
        
        try:
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior telecom customer service QA expert. "
                            "Your task is to identify precise and non-overlapping agent improvement topics."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature
            )
            
            # Parse response
            raw_content = response.choices[0].message.content
            logger.debug(f"Raw LLM response length: {len(raw_content)} characters")
            
            # Safely parse JSON
            topics = safe_json_loads(raw_content)
            
            if not isinstance(topics, list):
                raise ValueError(f"Expected list from LLM, got {type(topics)}")
            
            logger.info(f"Successfully extracted {len(topics)} topics from clusters")
            
            # Validate topic structure
            validated_topics = self._validate_topics(topics, cluster_payloads)
            
            return validated_topics
            
        except Exception as e:
            logger.error(f"Failed to extract topics: {e}", exc_info=True)
            raise
    
    def _validate_topics(
        self, 
        topics: List[Dict[str, Any]], 
        cluster_payloads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and clean extracted topics.
        
        Args:
            topics: Raw topics from LLM
            cluster_payloads: Original cluster data for validation
            
        Returns:
            Validated and cleaned topics list
        """
        validated = []
        expected_labels = {cluster['label'] for cluster in cluster_payloads}
        
        for topic in topics:
            # Check required fields
            required_fields = ['label', 'topic', 'description', 'reason', 'short_example']
            if not all(field in topic for field in required_fields):
                logger.warning(f"Topic missing required fields: {topic}")
                continue
            
            # Validate label exists in clusters
            if topic['label'] not in expected_labels:
                logger.warning(f"Topic label {topic['label']} not found in clusters")
                continue
            
            validated.append(topic)
        
        # Check for missing labels
        validated_labels = {t['label'] for t in validated}
        missing_labels = expected_labels - validated_labels
        if missing_labels:
            logger.warning(f"Missing topics for labels: {missing_labels}")
        
        logger.info(f"Validated {len(validated)} topics out of {len(cluster_payloads)} clusters")
        
        return validated
    
    def build_agent_improvement_prompt(self, cluster_payloads: list) -> str:
        """
        Build a contrastive prompt containing all agent improvement clusters together.
        
        This helps the LLM assign distinct, non-overlapping improvement topics.
        
        Args:
            cluster_payloads: List of cluster data with representative points
            
        Returns:
            Formatted prompt string for LLM topic extraction
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
                You are a senior QA analyst specializing in telecom customer service operations, complaint analysis, and agent performance improvement.

                You are given multiple clusters of agent performance improvement opportunities.
                Each label represents one cluster.
                For each cluster, representative statements have been selected to reflect the most common patterns in that cluster.

                Your objective:
                For EACH label, identify the SINGLE strongest and most representative Agent Performance Improvement Topic.

                Critical instructions:
                1. Each topic MUST represent the dominant operational issue in that cluster.
                2. Topics MUST be UNIQUE across labels.
                3. Do NOT reuse similar themes across clusters unless the operational focus is clearly different.
                4. Avoid overlap between labels. If two clusters appear similar, differentiate them by:
                    - process stage
                    - product/policy type
                    - customer journey impact
                    - operational failure point
                    - resolution responsibility
                5. Avoid generic or vague topics such as:
                    - communication skills
                    - customer service
                    - agent training
                    - escalation issues
                    unless they are made operationally specific.
                6. If the cluster relates to telecom-specific issues, explicitly reference the relevant context where applicable
                7. Choose the topic based on the ROOT CAUSE or most actionable performance improvement area, not just the surface complaint.
                8. Ensure the topic sounds operational, specific, and suitable for executive review or QA reporting.
                9. Do not create duplicate themes with slightly different wording.
                10. Read all labels together before assigning topics so you can ensure clear differentiation across the full set.

                Output requirements:
                - Return VALID JSON only
                - Return a JSON array
                - Each object must contain exactly these fields:
                    - "label"
                    - "topic"
                    - "description"
                    - "reason"
                    - "short_example"

                Field definitions:
                - "label": the cluster label number
                - "topic": a short, specific, unique, operationally distinct improvement title
                - "description": 2–3 concise but detailed sentences describing the dominant issue, what is going wrong, and what needs to improve
                - "reason": 2–3 concise but detailed sentences explaining why this issue matters operationally, what impact it has on customer experience and resolution efficiency, and why improvement is needed
                - "short_example": one brief realistic telecom-related example showing how a customer experiences the issue

                Writing requirements:
                - Use professional business language
                - Make each topic actionable and specific
                - Ensure description and reason are not repetitive
                - Focus on operational improvement opportunities
                - Keep examples realistic and telecom-relevant
                - Do not include markdown
                - Do not include commentary outside the JSON
                - Do not include numbering outside the JSON objects

                Uniqueness check before finalizing:
                Before returning the JSON:
                - compare all topic titles across labels
                - confirm they are clearly distinct
                - remove overlap in meaning
                - refine wording so each topic reflects a different operational issue
                - ensure no two topics could reasonably be merged into one

                Input clusters:
                {chr(10).join(cluster_sections)}

                Return format:
                [
                    {{
                    "label": 0,
                    "topic": "Specific, unique, operational topic title",
                    "description": "2-3 concise sentences describing the dominant issue and the specific improvement needed.",
                    "reason": "2-3 concise sentences explaining why this issue matters operationally and why agents or processes need improvement in this area.",
                    "short_example": "When handling a billing or service issue, agents sometimes failed to clearly explain the process, resulting in customer confusion about next steps and timelines."
                    }}
                ]
                """
        return prompt

