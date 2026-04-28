# Agent Improvements Analysis Module

## Overview

The Agent Improvements module provides production-level tools for analyzing agent performance improvement opportunities through clustering and LLM-powered topic extraction. It identifies actionable training and development themes from customer service interaction data.

## Features

- **Automated Data Extraction**: Parses nested JSON structures to extract agent evaluation data
- **Intelligent Clustering**: Supports multiple clustering algorithms (KMeans, DBSCAN/HDBSCAN, Leiden) with automatic method selection
- **LLM-Powered Topic Extraction**: Uses Azure OpenAI to generate operationally specific, non-overlapping improvement topics
- **Dimension Reduction**: PCA and UMAP support for high-dimensional embedding visualization
- **Comprehensive Visualization**: Multi-projection cluster visualization (PCA, t-SNE, UMAP)
- **Production-Ready**: Modular architecture with standardized interfaces

## Architecture

```
agent_improvements/
├── __init__.py              # Module exports
├── ai_analyzer.py           # Main analyzer orchestrator
├── ai_topic_analysis.py     # LLM-based topic extraction
├── prompts.py               # Prompt templates for LLM
├── example_usage.py         # Usage examples
├── MIGRATION_GUIDE.md       # Notebook to production conversion guide
└── README.md                # This file
```

**Note:** Data processing logic has been moved to the standardized `data_processing.DataProcessor` module for reusability across all analysis modules.

## Quick Start

### Basic Usage

```python
from openai import AzureOpenAI
from agent_improvements import AgentImprovementAnalyzer

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="your-api-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-endpoint.openai.azure.com/"
)

# Create analyzer
analyzer = AgentImprovementAnalyzer(
    azure_client=client,
    llm_model="gpt-4o",
    embedding_model="text-embedding-ada-002"
)

# Run complete analysis
results = analyzer.run_complete_analysis(
    data_folder="/path/to/json/files",
    clustering_method="auto",  # auto-selects best method
    min_cluster_size=10,
    norm='l2',  # cosine similarity
    reduce_dimensions=True,
    dim_reduction_method="umap",
    top_n_representative=15,
    visualize=True,
    output_path="/path/to/output"
)

# Access results
print(f"Found {results['metadata']['n_clusters']} clusters")
print(f"Extracted {results['metadata']['n_topics']} topics")
print(results['topics'])  # DataFrame with topics
```

### Advanced Configuration

```python
# Custom clustering parameters
results = analyzer.run_complete_analysis(
    data_folder="/path/to/data",
    clustering_method="dbscan",  # force specific method
    min_cluster_size=20,
    min_samples=8,
    norm='l2',
    reduce_dimensions=True,
    dim_reduction_method="pca",
    target_dim=10,  # explicit dimension count
    top_n_representative=30,
    visualize=False
)

# Access clustering details
clustering_info = results['clustering']
print(f"Method used: {clustering_info['method']}")
print(f"Labels: {clustering_info['labels']}")
```

## Input Data Format

The module expects JSON files with the following structure:

```json
{
  "interaction_identifier": "INT-001",
  "case_number": "CASE-12345",
  "calendar_date": "2024-11-25",
  "Agent Evaluation": {
    "agent_evaluations": [
      {
        "agent_identifier": "AGENT-001",
        "performance_evaluation": {
          "overall_performance": "Needs Improvement",
          "improvement_areas": [
            "Failed to explain billing process clearly",
            "Did not verify customer understanding",
            "Escalated too quickly without troubleshooting"
          ]
        }
      }
    ]
  }
}
```

**Key Requirements:**
- Files must be in `.json` format
- Each file can contain a single interaction or an array of interactions
- The `Agent Evaluation.agent_evaluations[].performance_evaluation.improvement_areas` field is required
- `improvement_areas` should be a list of strings

## Output Structure

The `run_complete_analysis()` method returns a dictionary containing:

```python
{
    "texts": List[str],                    # Original improvement area texts
    "embeddings": np.ndarray,              # Reduced-dimension embeddings
    "clustering": Dict,                    # Clustering results
        - "labels": cluster assignments
        - "method": clustering algorithm used
        - "metrics": quality metrics
    "topics": pd.DataFrame,                # Extracted topics
        - label: cluster identifier
        - topic: topic title
        - description: detailed explanation
        - reason: operational rationale
        - short_example: concrete example
    "cluster_payloads": List[Dict],        # Representative points per cluster
    "reduction_info": Dict,                # Dimension reduction metadata
    "visualization": Dict,                 # Visualization data (if enabled)
    "metadata": Dict                       # Summary statistics
}
```

## Configuration Parameters

### Clustering Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `"auto"` | Automatically selects best method based on data characteristics | General use |
| `"kmeans"` | K-Means clustering with automatic k selection | Well-separated, spherical clusters |
| `"dbscan"` | HDBSCAN density-based clustering | Variable-density clusters, noise handling |
| `"leiden"` | Leiden community detection on SNN graph | Complex relational structures |

### Dimension Reduction

| Method | Description | Use Case |
|--------|-------------|----------|
| `"auto"` | Auto-selects based on data size | General use |
| `"pca"` | Principal Component Analysis | Linear relationships, interpretability |
| `"umap"` | Uniform Manifold Approximation | Non-linear structures, visualization |

### Key Parameters

- **`min_cluster_size`**: Minimum samples per cluster (DBSCAN/Leiden). Default: 10
- **`min_samples`**: Minimum samples for core point (DBSCAN). Default: 5
- **`norm`**: Embedding normalization. `'l2'` for cosine similarity. Default: `'l2'`
- **`top_n_representative`**: Number of representative examples per cluster. Default: 15
- **`target_dim`**: Target dimensions for reduction. None = automatic. Default: None

## Topic Extraction Quality

The LLM prompt enforces:

1. **Uniqueness**: Topics must be distinct across clusters
2. **Operational Specificity**: Avoid generic terms like "communication skills"
3. **Actionability**: Topics should drive concrete improvements
4. **Telecom Context**: Examples should be telecom-relevant
5. **Root Cause Focus**: Identify underlying issues, not surface symptoms

Example output:
```json
{
  "label": 0,
  "topic": "Incomplete Billing Explanation During Account Transfers",
  "description": "Agents frequently fail to provide comprehensive billing breakdowns when customers transfer services, leading to confusion about prorated charges and effective dates.",
  "reason": "This gap creates customer dissatisfaction and increases follow-up contacts by 35%, impacting both CSAT scores and operational efficiency.",
  "short_example": "When a customer transferred their internet service to a new address, the agent didn't explain the prorated billing cycle, resulting in unexpected charges on the next statement."
}
```

## Error Handling

The module includes robust error handling:

- **File Processing**: Continues processing if individual files fail
- **JSON Parsing**: Validates structure and provides clear error messages
- **LLM Calls**: Retries on transient failures, validates response format
- **Clustering**: Falls back to alternative methods if primary fails
- **Visualization**: Gracefully handles plotting errors

## Dependencies

Required packages (already in project requirements):
- `openai`: Azure OpenAI API client
- `numpy`, `pandas`: Data processing
- `scikit-learn`: Clustering and dimension reduction
- `hdbscan`: Density-based clustering
- `umap-learn`: UMAP dimension reduction
- `scipy`: Distance calculations
- `matplotlib`: Visualization

## Integration with Project

This module follows the same architecture as other analysis modules:

```python
# Similar usage pattern to resolution_recommendation
from resolution_recommendation import ResolutionRecommendationAnalyzer
from agent_improvements import AgentImprovementAnalyzer

# Both use standardized components:
# - DataProcessor for loading
# - EmbeddingProcessor for embeddings
# - ClusteringAnalyzer for clustering
# - Specialized topic analyzers for LLM extraction
```

## Best Practices

1. **Data Quality**: Ensure improvement_areas are specific and actionable
2. **Cluster Size**: Adjust `min_cluster_size` based on dataset size
   - Small datasets (<500): Use 5-10
   - Medium datasets (500-5000): Use 10-20
   - Large datasets (>5000): Use 20-50
3. **Representative Points**: Increase `top_n_representative` for better topic quality
4. **Validation**: Review extracted topics for overlap and refine if needed
5. **Iteration**: Experiment with different clustering methods for optimal results

## Troubleshooting

### Common Issues

**Issue**: "No improvement_areas column found"
- **Solution**: Verify JSON structure matches expected format

**Issue**: "Too many noise points in DBSCAN"
- **Solution**: Reduce `min_cluster_size` or switch to KMeans

**Issue**: "Topics are too generic"
- **Solution**: Increase temperature slightly (0.1 → 0.2) or provide more specific input data

**Issue**: "Clustering produces too few clusters"
- **Solution**: Try different method or adjust parameters

## Performance Considerations

- **Embedding Generation**: Batched (100 per request) for API efficiency
- **Memory**: Dimension reduction recommended for large datasets (>10K samples)
- **LLM Calls**: Single call for all clusters (contrastive prompting) reduces cost
- **Parallelization**: File loading can be parallelized for very large datasets

## Future Enhancements

- [ ] Support for incremental learning (add new data without reprocessing)
- [ ] Topic evolution tracking over time
- [ ] Automated topic quality scoring
- [ ] Integration with training recommendation systems
- [ ] Multi-language support

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify input data format
3. Test with small sample first
4. Review parameter documentation

---

**Version**: 1.0.0  
**Author**: Morgan Wang  
**Last Updated**: 2026-04-28
