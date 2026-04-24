"""LLM Prompt templates for resolution recommendation analysis."""

from typing import List, Dict, Any


def build_topic_extraction_prompt(cluster_payloads: List[Dict[str, Any]]) -> str:
    """
    Build prompt for extracting distinct resolution recommendation topics from clusters.

    Creates a contrastive prompt that helps the LLM assign distinct, non-overlapping
    topics to each cluster of resolution recommendations. Includes detailed instructions
    for producing operationally specific telecom-related topics.

    Args:
        cluster_payloads: List of cluster dicts with 'label', 'cluster_size',
                         'representative_points' keys

    Returns:
        Formatted prompt string for LLM

    Examples:
        >>> clusters = [
        ...     {
        ...         "label": 0,
        ...         "cluster_size": 150,
        ...         "representative_points": ["Improve billing transparency", ...]
        ...     }
        ... ]
        >>> prompt = build_topic_extraction_prompt(clusters)
    """
    cluster_sections = []

    for cluster in cluster_payloads:
        points = "\n".join(
            [f"{i+1}. {text}" for i, text in enumerate(cluster["representative_points"])]
        )

        cluster_sections.append(
            f"""Label {cluster['label']} (Cluster size: {cluster['cluster_size']}):
Representative agent improvement points:
{points}"""
        )

    prompt = f"""You are a senior QA analyst for telecom customer service operations.

You are given multiple clusters of customer complaint resolution recommendations intended to help prevent future CCTS complaints.
Each cluster is identified by a numeric label.
For each label, representative statements were selected to represent the cluster.

Your task:
For EACH label, identify the SINGLE strongest Resolution Recommendation Topic that best captures the main operational issue in that cluster.

Important rules:
1. Assign exactly ONE topic per label.
2. Topics MUST be unique across labels.
3. Topics MUST be clearly differentiated and non-overlapping.
4. Do NOT use vague or generic titles such as:
   - communication issue
   - poor service
   - customer dissatisfaction
   - expectation setting
   unless you make them operationally specific and telecom-related.
5. If the issue relates to telecom operations, explicitly include the relevant domain in the topic when applicable:
   - billing disputes
   - credits/refunds
   - device financing
   - returns and buyback
   - cancellations/disconnections
   - promotions and discounts
   - rate plan changes
   - contract terms
   - roaming charges
   - service activation/provisioning
   - technical support follow-up
   - account ownership/authorization
   - collection activity
   - callback / case handling process
6. Focus on the dominant root cause or strongest resolution theme, not minor secondary issues.
7. Prefer action-oriented, operational recommendation themes over emotional or generic complaint descriptions.
8. If two clusters appear similar, differentiate them using the most specific operational distinction.
9. Do not repeat the same wording or same operational theme across different labels.
10. Base your answer only on the content of each cluster.

How to choose the topic:
- Identify the most recurring and operationally actionable issue in the cluster.
- Summarize that issue as a resolution recommendation theme.
- Make the topic specific enough that a telecom operations team could assign ownership to a process area.

Output requirements:
- Return VALID JSON only.
- Return a JSON array.
- Do not include markdown, comments, explanations, or any text outside the JSON array.
- Each object must contain exactly these fields: "label", "topic", "description", "reason", "short_example"

Field definitions:
- "label": the cluster label exactly as provided
- "topic": a short, specific, unique title for the resolution recommendation theme
- "description": 2-3 concise sentences describing the recommended resolution focus and what should be improved operationally
- "reason": 2-3 concise sentences explaining why this is the dominant issue in the cluster and why this recommendation is needed
- "short_example": one brief example of a customer-facing issue represented by the cluster

Writing guidance:
- Write in clear business English.
- Keep each topic concise but specific.
- Make descriptions operational, not generic.
- Avoid repeating the same rationale across labels.
- Ensure each topic could realistically be used as a reporting category for telecom complaint prevention.

Input clusters:
{chr(10).join(cluster_sections)}

Return format:
[
{{
    "label": 0,
    "topic": "Specific telecom resolution recommendation topic",
    "description": "2-3 concise sentences describing the operational resolution recommendation.",
    "reason": "2-3 concise sentences explaining why this is the dominant issue and why this recommendation is needed.",
    "short_example": "One brief example of the customer issue."
}}
]"""

    return prompt


def build_breakdown_topic_extraction_prompt(
    cluster_payloads: List[Dict[str, Any]],
    parent_topic: str,
    parent_description: str
) -> str:
    """
    Build prompt for extracting topics from sub-clusters within a parent theme.

    Creates a contextualized prompt for analyzing sub-clusters that fall under
    a shared parent resolution recommendation theme.

    Args:
        cluster_payloads: List of sub-cluster dicts
        parent_topic: Parent cluster's topic name
        parent_description: Parent cluster's description

    Returns:
        Formatted prompt string for LLM

    Examples:
        >>> parent = "Billing Accuracy"
        >>> desc = "Improve billing calculation and transparency"
        >>> prompt = build_breakdown_topic_extraction_prompt(clusters, parent, desc)
    """
    cluster_sections = []

    for cluster in cluster_payloads:
        points = "\n".join(
            [f"{i+1}. {text}" for i, text in enumerate(cluster["representative_points"])]
        )

        cluster_sections.append(
            f"""Label {cluster['label']} (Cluster size: {cluster['cluster_size']}):
Representative agent improvement points:
{points}"""
        )

    context = f"Topic: {parent_topic}\nDescription: {parent_description}"

    prompt = f"""You are a senior QA analyst for telecom customer service operations.

You are given multiple clusters of customer complaint resolution recommendations intended to help prevent future CCTS complaints.
Each cluster is identified by a numeric label, with all clusters under the theme of {context}.
For each label, representative statements were selected to represent the cluster.

Your task:
For EACH label, identify the SINGLE strongest Resolution Recommendation Topic that best captures the main operational issue in that cluster.

Important rules:
1. Assign exactly ONE topic per label.
2. Topics MUST be unique across labels and different from the parent theme.
3. Topics MUST be clearly differentiated and non-overlapping.
4. Do NOT reuse the parent topic or create redundant variations.
5. Focus on sub-themes or specific operational distinctions within the parent category.
6. Topics should be more granular than the parent theme but still belong logically under it.
7. Include operational specificity: billing methods, system components, timeframes, etc.
8. Do not repeat the same wording or same operational theme across different labels.
9. Base your answer only on the content of each cluster.

How to choose the topic:
- Identify the most specific operational distinction within the parent theme.
- Differentiate sub-clusters by root cause, timing, process area, or stakeholder.
- Make the topic specific enough that a telecom operations team could assign ownership.

Output requirements:
- Return VALID JSON only.
- Return a JSON array.
- Do not include markdown, comments, explanations, or any text outside the JSON array.
- Each object must contain exactly these fields: "label", "topic", "description", "reason", "short_example"

Field definitions:
- "label": the cluster label exactly as provided
- "topic": a short, specific, unique title for the sub-theme resolution recommendation
- "description": 2-3 concise sentences describing the recommended focus within the parent theme
- "reason": 2-3 concise sentences explaining why this sub-theme is distinct and needs a separate recommendation
- "short_example": one brief example of a customer issue in this sub-cluster

Input clusters:
{chr(10).join(cluster_sections)}

Return format:
[
{{
    "label": 0,
    "topic": "Specific sub-theme within {parent_topic}",
    "description": "2-3 concise sentences.",
    "reason": "2-3 concise sentences.",
    "short_example": "One brief example."
}}
]"""

    return prompt
