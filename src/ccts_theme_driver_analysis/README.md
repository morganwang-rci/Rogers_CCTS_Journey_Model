# Theme Analysis Pipeline

This package provides a complete end-to-end pipeline for analyzing complaint themes from CCTS journey analysis data.

## Overview

The theme analysis pipeline processes complaint data through the following stages:

1. **Data Processing** - Extract and clean complaint journey data from JSON files
2. **Embedding Generation** - Convert text to vector representations
3. **Dimension Reduction** - Reduce embedding dimensions for efficiency
4. **Clustering** - Group similar complaints using multiple algorithms
5. **Topic Extraction** - Identify themes using LLM analysis
6(option).  **Evaluation** - Assess quality of results

## Quick Start

```python
from theme_analysis import ThemeAnalyzer
from openai import AzureOpenAI

# Initialize OpenAI client
client = AzureOpenAI(
    api_key="your_key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com/"
)

# Initialize analyzer
analyzer = ThemeAnalyzer(client)

# Run complete analysis
results = analyzer.run_complete_analysis(
    data_folder="/path/to/complaint/jsons",
    text_column="primary_complaint_issue_clean",
    clustering_method="auto"
)

# Access results
print(f"Found {results['clustering']['n_clusters']} themes")
print("Topics:", results['topics'])
```

## Components

### DataProcessor
Handles extraction of complaint data from JSON files.

```python
from data_processing import DataProcessor

processor = DataProcessor()
df = processor.process_case_journey_folder("/path/to/jsons")
```

### EmbeddingProcessor
Generates embeddings and applies dimension reduction.

```python
from data_processing import EmbeddingProcessor

embedder = EmbeddingProcessor(client)
embeddings = embedder.get_embeddings_in_batches(texts)
reduced_embeddings, info = embedder.apply_dimension_reduction(embeddings)
```

### ClusteringAnalyzer (from cluster_method)
Provides multiple clustering algorithms.

```python
from cluster_method import ClusteringAnalyzer

clusterer = ClusteringAnalyzer()
kmeans_result = clusterer.apply_kmeans_clustering(embeddings)
dbscan_result = clusterer.apply_dbscan_clustering(embeddings)
leiden_result = clusterer.apply_leiden_clustering(embeddings)
```

### ClusterVisualizer
Creates visualizations of clustering results.

```python
from theme_analysis import ClusterVisualizer

visualizer = ClusterVisualizer()
plots = visualizer.cluster_visual(embeddings, cluster_labels=labels)
```

### ClusterEvaluator
Evaluates clustering quality.

```python
from theme_analysis import ClusterEvaluator

evaluator = ClusterEvaluator()
metrics = evaluator.evaluate_clustering_result(embeddings, labels)
```

### TopicAnalyzer
Extracts themes using LLM analysis.

```python
from theme_analysis import TopicAnalyzer

topic_analyzer = TopicAnalyzer(client)
cluster_payloads = topic_analyzer.get_top_n_closest_points_per_cluster(
    embeddings, labels, texts
)
topics_raw = topic_analyzer.find_topics_all_clusters(cluster_payloads)
```

## Dependencies

- openai
- numpy
- pandas
- scikit-learn
- matplotlib
- hdbscan
- python-igraph
- leidenalg
- umap-learn
- scipy

## Configuration

The pipeline expects complaint journey analysis JSON files with the following structure:

```json
{
  "ccts_complaint_journey_analysis": {
    "case_number": "...",
    "customer_complaint_genesis": {
      "primary_complaint_issue": "..."
    }
  }
}
```

## Output

The complete analysis returns a dictionary with:

- `data`: Processed DataFrame
- `texts`: Cleaned text data
- `embeddings`: Generated embeddings
- `reduction_info`: Dimension reduction details
- `clustering`: Clustering results and metadata
- `topics`: Extracted themes with descriptions
- `evaluation`: Quality metrics

## Customization

Each component can be used independently for custom analysis workflows:

```python
# Custom workflow
processor = DataProcessor()
embedder = EmbeddingProcessor(client)
clusterer = ClusteringAnalyzer()
topic_analyzer = TopicAnalyzer(client)

# Process data
df = processor.process_case_journey_folder(data_folder)
texts = df['primary_complaint_issue_clean'].tolist()

# Generate embeddings
embeddings = embedder.get_embeddings_in_batches(texts)

# Cluster
clustering_result = clusterer.apply_leiden_clustering(embeddings)

# Extract topics
topics = topic_analyzer.extract_topics(embeddings, clustering_result['labels'], texts)
```