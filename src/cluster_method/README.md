# Clustering Methods

This package provides three clustering methods for text embeddings analysis:

## Available Methods

### 1. K-Means Clustering (`kmeans.py`)
- **Class**: `KMeansClustering`
- **Features**:
  - Automatic optimal k determination using elbow method and silhouette scores
  - Standard K-Means implementation with configurable parameters

### 2. DBSCAN Clustering (`dbscan.py`)
- **Class**: `DBSCANClustering`
- **Features**:
  - HDBSCAN implementation for density-based clustering
  - Handles noise points automatically
  - Supports different distance metrics (euclidean, cosine)

### 3. Leiden Clustering (`leiden.py`)
- **Class**: `LeidenClustering`
- **Features**:
  - Graph-based clustering using kNN + Leiden algorithm
  - More efficient than Louvain for large networks
  - Includes comprehensive evaluation metrics

### 4. Clustering Analyzer (`analyzer.py`)
- **Class**: `ClusteringAnalyzer`
- **Features**:
  - Combines all three methods
  - Includes method selection logic to choose the best clustering approach
  - Provides unified interface for all clustering operations

## Usage Example

```python
from cluster_method import ClusteringAnalyzer
import numpy as np

# Initialize analyzer
analyzer = ClusteringAnalyzer()

# Sample embeddings (replace with your actual embeddings)
embeddings = np.random.rand(100, 384)  # 100 samples, 384 dimensions

# Apply K-Means clustering
kmeans_result = analyzer.apply_kmeans_clustering(embeddings, auto_k=True)
print(f"K-Means found {kmeans_result['n_clusters']} clusters")

# Apply DBSCAN clustering
dbscan_result = analyzer.apply_dbscan_clustering(embeddings, min_cluster_size=10)
print(f"DBSCAN found {dbscan_result['n_clusters']} clusters with {dbscan_result['noise_percentage']:.1f}% noise")

# Apply Leiden clustering
leiden_result = analyzer.apply_leiden_clustering(embeddings, k=15)
print(f"Leiden found {leiden_result['n_clusters']} clusters with modularity {leiden_result['modularity']:.3f}")

# Select best method (K-Means vs DBSCAN)
selection = analyzer.select_best_clustering_method(embeddings, kmeans_result, dbscan_result)
print(f"Best method: {selection['best_method']} with score {selection['explainability_score']}")
```

## Dependencies
- numpy
- scikit-learn
- matplotlib
- kneed
- hdbscan
- python-igraph
- leidenalg

Install with:
```bash
pip install numpy scikit-learn matplotlib kneed hdbscan python-igraph leidenalg
```

## Method Selection Logic

The `select_best_clustering_method` uses the following decision rules:

1. **Noise Threshold**: If DBSCAN has ≥30% noise, prefer K-Means
2. **Insufficient Clusters**: Select method with n_cluster >= 3
3. **Composite Score**: Compare overall quality metrics between methods

Each method is evaluated on:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Cluster Balance
- Cluster Count (preference for 3-8 clusters)
- Method-specific metrics (modularity for Leiden, noise for DBSCAN)