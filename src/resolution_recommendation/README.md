# Resolution Recommendations Analysis Module

This package provides tools for extracting, clustering, and analyzing customer resolution recommendations to identify actionable improvement themes from CCTS complaint journey data.

## Overview

The resolution recommendations pipeline processes strategic recommendations through the following stages:

1. **Data Extraction** - Extract strategic recommendations from CCTS journey analysis JSON files
2. **Text Processing** - Clean and prepare recommendation texts (explode list columns)
3. **Embedding Generation** - Convert recommendations to vector representations using Azure OpenAI
4. **Dimension Reduction** - Apply UMAP/PCA for efficiency and visualization
5. **Clustering** - Group similar recommendations using KMeans, DBSCAN, Leiden, or auto-selection
6. **Topic Extraction** - Identify operational improvement themes using LLM analysis
7. **Result Assembly** - Merge topics back with original data for actionable insights

## Quick Start

### Option 1: Using the Driver Script

```python
from resolution_recommendations import ResolutionRecommendationAnalyzer
from openai import AzureOpenAI

# Initialize OpenAI client
client = AzureOpenAI(
    api_key="your_key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com/"
)

# Initialize analyzer
analyzer = ResolutionRecommendationAnalyzer(azure_client=client, llm_model="gpt-4o")

# Run complete analysis
results = analyzer.run_pipeline(
    data_folder="/path/to/journey/jsons",
    clustering_method="auto",
    dim_reduction_method="umap",
    top_n_representatives=15,
    normalize_embeddings=True
)

# Access results
print(f"Found {len(results['topics'])} resolution themes")
print(f"Total recommendations analyzed: {len(results['dataframe'])}")
print("\nSample topics:")
for topic in results['topics'][:3]:
    print(f"- {topic['topic']}")
```

### Option 2: Command Line Execution

```bash
# From src/ directory
python resolution_recommendation_driver_analysis.py
```

Configure via environment variables:
```bash
export RR_CLUSTERING_METHOD="auto"
export RR_DIM_REDUCTION_METHOD="umap"
export RR_DATA_FOLDER="/path/to/data"
python resolution_recommendation_driver_analysis.py
```

## Components

### ResolutionRecommendationAnalyzer (Main Orchestrator)
End-to-end pipeline that coordinates all analysis steps.

```python
from resolution_recommendations import ResolutionRecommendationAnalyzer

analyzer = ResolutionRecommendationAnalyzer(azure_client=client)
results = analyzer.run_pipeline(
    data_folder="/path/to/jsons",
    clustering_method="auto",  # or "kmeans", "dbscan", "leiden"
    dim_reduction_method="umap",
    top_n_representatives=15,
    normalize_embeddings=True
)
```

**Returns:**
- `dataframe`: DataFrame with cluster labels and topics merged
- `topics`: List of extracted topics with descriptions
- `clustering_method`: Method selected (if auto)
- `embeddings`: Reduced-dimensional embeddings
- `labels`: Cluster assignments
- `reduction_info`: Dimension reduction metadata

### TopicAnalyzer
Specialized topic extraction for resolution recommendations using telecom-specific prompts.

```python
from resolution_recommendations import TopicAnalyzer

topic_analyzer = TopicAnalyzer(client=client, model="gpt-4o")

# Get representative points from clusters
cluster_payloads = topic_analyzer.get_top_n_closest_points_per_cluster(
    embedding_vectors=embeddings,
    cluster_labels=labels,
    original_texts=texts,
    top_n=15
)

# Extract main topics
topics = topic_analyzer.extract_topics_from_clusters(cluster_payloads)

# Or extract sub-themes for hierarchical analysis
sub_topics = topic_analyzer.extract_breakdown_topics_from_clusters(
    cluster_payloads=sub_cluster_payloads,
    parent_topic="Billing Transparency",
    parent_description="Improve billing accuracy and communication"
)
```

**Output Structure:**
```json
{
  "label": 0,
  "topic": "Proactive Credit Application Process",
  "description": "Implement automated credit application workflows...",
  "reason": "Multiple recommendations highlight delays in credit processing...",
  "short_example": "Customer waited 3 weeks for promised credit"
}
```

### Data Processing Utilities
Uses standardized [DataProcessor](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\data_processing\data_processing.py) from `data_processing` module with custom explosion logic for `strategic_recommendations`.

```python
from data_processing import DataProcessor

processor = DataProcessor()
df_raw = processor.process_case_journey_folder("/path/to/jsons")

# Explode strategic_recommendations (handled internally by ResolutionRecommendationAnalyzer)
df_exploded = df_raw.explode('strategic_recommendations')
```

### Embedding Generation
Uses standardized [EmbeddingProcessor](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\data_processing\embeddings.py) from `data_processing` module.

```python
from data_processing import EmbeddingProcessor

embedder = EmbeddingProcessor(client)
embeddings = embedder.get_embeddings_in_batches(texts, batch_size=100)
normalized = embedder.normalize_embeddings(embeddings, norm='l2')
reduced, info = embedder.apply_dimension_reduction(embeddings, method="umap")
```

### Clustering
Uses standardized [ClusteringAnalyzer](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\cluster_method\Clustering_analyzer.py) from `cluster_method` module.

```python
from cluster_method import ClusteringAnalyzer

clusterer = ClusteringAnalyzer()

# Auto method selection (recommended)
result = clusterer.apply_clustering(embeddings, method="auto")

# Or specific methods
kmeans_result = clusterer.apply_kmeans_clustering(embeddings, n_clusters=7)
dbscan_result = clusterer.apply_dbscan_clustering(embeddings, min_cluster_size=30)
leiden_result = clusterer.apply_leiden_clustering(embeddings, k=30, resolution_parameter=0.7)
```

## Pipeline Architecture

```
JSON Files → DataProcessor → Text Explosion → EmbeddingProcessor
    ↓
Embeddings → Normalization → Dimension Reduction → ClusteringAnalyzer
    ↓
Clusters → TopicAnalyzer → LLM Topic Extraction → Results Assembly
    ↓
DataFrame with Topics + Metadata
```

## Configuration Options

### Clustering Methods

| Method | Best For | Key Parameters |
|--------|----------|----------------|
| **auto** | Unknown data characteristics | Automatically selects best method |
| **kmeans** | Balanced, spherical clusters | `n_clusters` (default: 7) |
| **dbscan** | Arbitrary shapes, noise handling | `min_cluster_size`, `min_samples`, `metric` |
| **leiden** | Complex relationships, hierarchy | `k`, `resolution_parameter`, `use_snn` |

### Dimension Reduction

| Method | Best For | Notes |
|--------|----------|-------|
| **umap** | Large datasets, non-linear structures | Default, preserves local/global structure |
| **pca** | Linear relationships, interpretability | Fast, explains variance |
| **auto** | Unknown characteristics | Selects based on data size |

### Topic Extraction

- **top_n_representatives**: Number of closest points per cluster (default: 15)
- **Model**: GPT-4o recommended for best topic quality
- **Temperature**: 0.1 (deterministic, consistent topics)

## Environment Variables

For command-line execution via `resolution_recommendation_driver_analysis.py`:

```env
# Core Configuration
RR_DATA_FOLDER=/path/to/journey/jsons
RR_CLUSTERING_METHOD=auto
RR_DIM_REDUCTION_METHOD=umap
RR_NORMALIZE_EMBEDDINGS=True
RR_TOP_N_REPRESENTATIVES=15
RR_OUTPUT_PATH=./output/resolution_recommendations/

# KMeans Parameters
RR_KMEANS_N_CLUSTERS=7

# DBSCAN Parameters
RR_DBSCAN_MIN_CLUSTER_SIZE=30
RR_DBSCAN_MIN_SAMPLES=10
RR_DBSCAN_METRIC=euclidean

# Leiden Parameters
RR_LEIDEN_K=30
RR_LEIDEN_RESOLUTION=0.7
RR_LEIDEN_USE_SNN=True
RR_LEIDEN_METRIC=cosine
RR_LEIDEN_RANDOM_STATE=42

# Azure OpenAI (from ConfigManager)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_MODEL=gpt-4o
```

## Output Structure

### Results Dictionary
```python
{
    "dataframe": pd.DataFrame,  # Original data + cluster_label + topic fields
    "topics": List[Dict],       # Extracted topics with metadata
    "clustering_method": str,   # Method used (e.g., "kmeans", "auto")
    "clustering_metrics": Dict, # Quality metrics (if available)
    "embeddings": np.ndarray,   # Reduced-dimensional embeddings
    "labels": np.ndarray,       # Cluster assignments
    "reduction_info": Dict      # Dimension reduction metadata
}
```

### Topics Structure
Each topic includes:
- `label`: Cluster ID
- `topic`: Short, specific title (e.g., "Proactive Credit Application")
- `description`: 2-3 sentences on operational focus
- `reason`: Why this is the dominant issue
- `short_example`: Representative customer scenario

### Saved JSON (via driver script)
```json
{
  "clustering_method": "auto",
  "n_clusters": 7,
  "topics": [...],
  "total_recommendations": 1250
}
```

## Use Cases

### 1. Operational Improvement Identification
Identify recurring themes in resolution recommendations to prioritize process improvements:
```python
topics = results['topics']
for topic in topics:
    print(f"{topic['topic']}: {topic['description']}")
```

### 2. Root Cause Analysis
Understand why certain issues persist across cases:
```python
# Filter recommendations by topic
billing_issues = results['dataframe'][
    results['dataframe']['topic'].str.contains('billing', case=False)
]
print(f"Billing-related recommendations: {len(billing_issues)}")
```

### 3. Hierarchical Theme Exploration
Drill down into large clusters for sub-themes:
```python
# Get indices for a specific cluster
cluster_1_mask = results['labels'] == 1
cluster_1_embeddings = results['embeddings'][cluster_1_mask]
cluster_1_texts = results['dataframe'][cluster_1_mask]['recommendation_text'].tolist()

# Re-cluster within the cluster
sub_payloads = topic_analyzer.get_top_n_closest_points_per_cluster(
    cluster_1_embeddings, sub_labels, cluster_1_texts, top_n=10
)
sub_topics = topic_analyzer.extract_breakdown_topics_from_clusters(
    sub_payloads, 
    parent_topic=main_topics[1]['topic'],
    parent_description=main_topics[1]['description']
)
```

### 4. Trend Analysis Over Time
Track how resolution themes evolve:
```python
# Add time dimension if available in source data
results['dataframe']['month'] = pd.to_datetime(
    results['dataframe']['case_date']
).dt.to_period('M')

monthly_themes = results['dataframe'].groupby(['month', 'topic']).size().unstack(fill_value=0)
```

## Dependencies

Required packages:
- `openai` - Azure OpenAI integration
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Clustering algorithms
- `umap-learn` - UMAP dimension reduction
- `hdbscan` - HDBSCAN clustering
- `python-igraph` - Graph operations (Leiden)
- `leidenalg` - Leiden algorithm
- `scipy` - Distance calculations

Install with:
```bash
pip install openai numpy pandas scikit-learn umap-learn hdbscan python-igraph leidenalg scipy
```

## Integration with Project Ecosystem

This module integrates with the broader CCTS analysis pipeline:

- **Data Source**: Uses same JSON structure as [ThemeAnalyzer](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\ccts_theme_driver_analysis\analyzer.py)
- **Embeddings**: Shared [EmbeddingProcessor](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\data_processing\embeddings.py) for consistency
- **Clustering**: Unified [ClusteringAnalyzer](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\cluster_method\Clustering_analyzer.py) interface
- **Configuration**: Uses [ConfigManager](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\report_generation\config.py) from `report_generation`
- **Prompts**: Specialized telecom-focused prompts in [prompts.py](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\resolution_recommendations\prompts.py)

## Best Practices

1. **Use Auto Clustering**: Let the system select the best method unless you have specific requirements
2. **Normalize Embeddings**: Enable normalization for better cosine similarity performance
3. **Adjust Top-N**: Increase `top_n_representatives` for more diverse topic examples (15-30 recommended)
4. **Review Topics Manually**: LLM-generated topics should be validated by domain experts
5. **Iterate on Parameters**: Experiment with different clustering parameters to find optimal granularity
6. **Hierarchical Analysis**: For large clusters (>200 items), consider sub-clustering for finer themes

## Troubleshooting

### Common Issues

**Issue**: "No valid resolution recommendations found"
- **Solution**: Ensure JSON files contain `resolution_recommendations.strategic_recommendations` field

**Issue**: "Column 'strategic_recommendations' not found"
- **Solution**: Verify data structure matches expected CCTS journey analysis format

**Issue**: Low-quality or generic topics
- **Solution**: Increase `top_n_representatives`, try different clustering methods, or adjust dimension reduction

**Issue**: Too many/few clusters
- **Solution**: Adjust `n_clusters` for KMeans, `min_cluster_size` for DBSCAN, or `resolution_parameter` for Leiden

### Performance Tips

- **Batch Size**: Embedding generation uses batching (default: 100) to avoid API limits
- **Dimension Reduction**: UMAP is faster than t-SNE for large datasets
- **Caching**: Consider caching embeddings for repeated analysis
- **Parallel Processing**: For very large datasets, consider parallel embedding generation

## Examples

See `resolution_recommendation_driver_analysis.py` for complete production-ready usage example.

## Related Modules

- **[ccts_theme_driver_analysis](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\ccts_theme_driver_analysis)** - Complaint theme analysis (similar pipeline for complaint issues)
- **[cluster_method](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\cluster_method)** - Core clustering algorithms
- **[data_processing](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\data_processing)** - Data loading and embedding utilities
- **[report_generation](file://c:\Users\Morgan.Wang\Desktop\CCTS_Journey_Model-main\src\report_generation)** - Traditional report generation pipeline
