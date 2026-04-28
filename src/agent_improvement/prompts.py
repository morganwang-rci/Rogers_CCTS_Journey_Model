"""Prompts for Agent Improvements Topic Analysis.

Contains prompt templates for extracting topics from agent improvement clusters.
"""


def build_agent_improvement_prompt(cluster_payloads: list) -> str:
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
