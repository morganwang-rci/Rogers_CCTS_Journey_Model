# Solution Architecture - CCTS Complaint Analysis Pipeline

## Overview

This is a comprehensive, multi-step solution for CCTS complaint analysis that combines traditional report generation with advanced AI-powered theme analysis. The solution is organized into specialized modules under the `src/` directory, enabling modular development and seamless integration of analysis capabilities.


### ✅ **Complete Modular Architecture**
- **Clustering Methods**: Production-ready clustering framework with K-Means, DBSCAN, and Leiden algorithms
- **Theme Analysis Pipeline**: End-to-end automated theme discovery from raw data to insights
- **Unified Interfaces**: Consistent APIs across all modules for easy integration

### 🎯 **Key Improvements**
- **Automatic Method Selection**: Intelligent clustering algorithm recommendation
- **LLM-Powered Topic Extraction**: GPT-based theme identification and description
- **Comprehensive Clustering Evaluation**: Multiple quality metrics and visualization options
- **Production-Ready Code**: Error handling, logging, and configuration management

### 📊 **Enhanced Capabilities**
- Multi-projection visualization (PCA, t-SNE, UMAP)
- Batch processing for large datasets
- Noise handling and outlier detection
- Hierarchical clustering with Leiden algorithm

## Architecture Overview

```
src/
├── __init__.py                    # Package marker
├── levels_report_generation.py    # Main entry point for report generation
├── theme_driver_analysis.py       # Theme analysis driver script
├── requirements.txt               # Dependencies
| 
├── report_generation/             # 📋 Traditional Report Generation
│   ├── __init__.py                # Package marker
│   ├── analyzer.py                # Core analysis engine
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Utility functions
│   ├── prompts.py                 # AI prompt templates
│   └── README.md                  # Module documentation
│
├── cluster_method/                # 🔍 Core Clustering Algorithms
│   ├── __init__.py                # Package marker
│   ├── Clustering_analyzer.py     # Unified clustering interface
│   ├── kmeans.py                 # K-Means with auto k-determination
│   ├── dbscan.py                 # DBSCAN/HDBSCAN implementation
│   ├── leiden.py                 # Graph-based Leiden clustering
│   └── README.md                 # Detailed clustering docs
│
|── ccts_theme_driver_analysis/    # 🎯 Theme Analysis Pipeline
|    ├── __init__.py               # Package marker
|    ├── analyzer.py               # Main pipeline orchestrator
|    ├── evaluation.py             # Clustering quality metrics
|    ├── topic_analysis.py         # LLM-based theme extraction
|    └── README.md                 # Pipeline documentation
|
├── data_processing/               # 🧹 Data extraction and embedding preprocessing
│   ├── __init__.py                # Package marker
│   ├── data_processing.py         # Data extraction and cleaning
│   ├── embeddings.py              # Embedding generation and helpers
│
└── Visualization/                 # 📊 Visualization Tools
    ├── __init__.py                # Package maker
    └── visualization.py           # Visualization helpers: PCA, UMAP, t-SNE
```

## 📁 Root Directory Files

| File | Purpose | Usage |
|------|---------|-------|
| `__init__.py` | Package marker | Enables Python package imports |
| `levels_report_generation.py` | Main entry point for report generation | `python levels_report_generation.py` |
| `theme_driver_analysis.py` | Theme analysis driver script | `python theme_driver_analysis.py` |
| `main.ipynb` | Interactive Jupyter notebook | Development and testing environment |
| `requirements.txt` | Python dependencies | `pip install -r requirements.txt` |
| `README.md` | This documentation | Comprehensive project guide |
| `output/` | Generated output artifacts | Stores report and analysis files |


### Key Entry Points

**`levels_report_generation.py`** - Production script for end-to-end report generation:
- Loads environment configuration
- Processes customer interaction data
- Generates AI-powered analysis reports
- Handles errors and logging

**`main.ipynb`** - Development and interactive environment:
- Environment setup and validation
- Step-by-step execution
- Component testing and debugging
- Databricks notebook compatibility

## 📋 Report Generation Module (Step 1)

### Purpose
Generates comprehensive complaint reports by analyzing customer interactions and creating detailed journeys.

### Key Components

| Component | File | Purpose | Key Functions |
|-----------|------|---------|---------------|
| **Analyzer** | `analyzer.py` | Core analysis engine with API integration | `analyze_all_interactions()`, `analyze_single_interaction()` |
| **Configuration** | `config.py` | Environment-based configuration management | `load_azure_config()`, `load_data_config()` |
| **Utilities** | `utils.py` | JSON parsing, validation, helpers | `load_pickle_data()`, `save_json_results()` |
| **Prompts** | `prompts.py` | AI prompt templates for analysis | Interaction analysis templates, journey summary templates |

### Analysis Output

**Interaction Analysis Files** (`interaction_analysis_{file_number}.json`):
- Detailed interaction-by-interaction analysis
- Customer sentiment and intent identification
- Issue categorization and severity assessment
- Resolution effectiveness evaluation

**Journey Summary Files** (`journey_summary_{file_number}.json`):
- Aggregated journey analysis
- Root cause identification
- Process improvement recommendations
- Customer experience insights

### Usage Example

```python
from report_generation import (
    ConfigManager,
    InteractionAnalyzer,
    save_analysis_results,
)

# Load configurations
azure_config = ConfigManager.load_azure_config()
data_config = ConfigManager.load_data_config()
processing_config = ConfigManager.load_processing_config()

# Initialize analyzer
analyzer = InteractionAnalyzer(azure_config, processing_config)

# Process all interactions
results = analyzer.analyze_all_interactions(data_config.data_pickle_path)

# Save results
save_analysis_results(results, data_config.conversation_output_dir, data_config.journey_output_dir)
```

### Configuration Requirements

The module requires these environment variables:
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI resource endpoint
- `AZURE_OPENAI_API_VERSION` - API version (default: 2024-02-01)
- `DATA_PICKLE_PATH` - Path to interactions pickle file
- `CONVERSATION_OUTPUT_DIR` - Output directory for interaction analysis
- `JOURNEY_OUTPUT_DIR` - Output directory for journey summaries

## 🔍 Clustering Methods Module

### Purpose
Provides multiple clustering algorithms specifically optimized for text embeddings analysis and complaint theme identification.

### Available Algorithms

| Algorithm | File | Best For | Key Features |
|-----------|------|----------|--------------|
| **K-Means** | `kmeans.py` | Balanced datasets | Auto k-determination, centroid-based |
| **DBSCAN** | `dbscan.py` | Noise handling | Density-based, handles outliers |
| **Leiden** | `leiden.py` | Complex relationships | Graph-based, high modularity |

### Key Features
- **Automatic Method Selection**: Intelligent algorithm recommendation based on data characteristics
- **Parameter Optimization**: Auto-tuning of clustering parameters (k, eps, resolution, etc.)
- **Comprehensive Evaluation**: Multiple metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Noise Handling**: Built-in support for outlier detection and noise filtering
- **Distance Metrics**: Support for cosine, euclidean, and other distance measures

### Unified Interface

```python
from cluster_method import ClusteringAnalyzer

# Initialize analyzer
analyzer = ClusteringAnalyzer()

# Apply clustering with automatic method selection
result = analyzer.apply_clustering(
    embeddings, 
    method="auto",  # or "kmeans", "dbscan", "leiden"
    **method_params
)

# Access results
labels = result['labels']
n_clusters = result['n_clusters']
method_used = result['method']
```

### Individual Algorithm Usage

```python
# K-Means clustering
kmeans_result = analyzer.apply_kmeans_clustering(
    embeddings, 
    n_clusters=5, 
    random_state=42
)

# DBSCAN clustering
dbscan_result = analyzer.apply_dbscan_clustering(
    embeddings, 
    eps=0.5, 
    min_samples=5
)

# Leiden clustering
leiden_result = analyzer.apply_leiden_clustering(
    embeddings, 
    resolution=1.0, 
    random_state=42
)
```

### Method Selection Guide

- **K-Means**: When you know the number of clusters, data is spherical
- **DBSCAN**: When clusters have arbitrary shapes, need noise detection
- **Leiden**: When you want hierarchical community detection
- **Auto**: Let the system choose based on data characteristics

## 🎯 Theme Analysis Module

### Purpose
Complete end-to-end pipeline for automated complaint theme discovery and analysis, from raw JSON files to actionable insights.

### Pipeline Components

| Component | File | Purpose | Key Functions |
|-----------|------|---------|---------------|
| **Data Processing** | `data_processing.py` | Extract & clean complaint data | JSON parsing, text cleaning |
| **Embeddings** | `embeddings.py` | Generate vector representations | OpenAI embeddings, dimension reduction |
| **Visualization** | `visualization.py` | Plot clustering results | PCA, t-SNE, UMAP projections |
| **Evaluation** | `evaluation.py` | Assess clustering quality | Multiple quality metrics |
| **Topic Analysis** | `topic_analysis.py` | Extract themes via LLM | GPT-powered theme identification |
| **Main Analyzer** | `analyzer.py` | Orchestrate complete pipeline | End-to-end automation |

### Complete Analysis Pipeline

1. **📂 Data Extraction**: Process complaint journey JSON files
2. **🧹 Text Processing**: Clean and prepare complaint issue text
3. **🧠 Embedding Generation**: Create vector representations using OpenAI
4. **📏 Dimension Reduction**: Apply PCA/UMAP for efficiency and visualization
5. **🔗 Clustering**: Apply optimal algorithm (K-Means/DBSCAN/Leiden)
6. **🎭 Topic Extraction**: Use LLM to identify and describe complaint themes
7. **📊 Visualization**: Plot results in multiple 2D projections
8. **📈 Evaluation**: Assess clustering quality and theme coherence

### Quick Start Example

```python
from ccts_theme_driver_analysis import ThemeAnalyzer
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
    clustering_method="auto",
    reduce_dimensions=True
)

# Access results
print(f"📊 Found {results['clustering']['n_clusters']} complaint themes")
print(f"🎯 Top theme: {results['topics']['topics'][0]['topic']}")
print(f"📈 Clustering quality: {results['evaluation']['silhouette']:.3f}")
```

### Advanced Usage

```python
# Individual component usage
from ccts_theme_driver_analysis import (
    DataProcessor, EmbeddingProcessor,
    ClusterVisualizer, TopicAnalyzer
)

# Process data
processor = DataProcessor()
df = processor.process_case_journey_folder("/data/folder")

# Generate embeddings
embedder = EmbeddingProcessor(client)
texts = df['primary_complaint_issue_clean'].dropna().tolist()
embeddings = embedder.get_embeddings_in_batches(texts)

# Apply clustering
from cluster_method import ClusteringAnalyzer
clusterer = ClusteringAnalyzer()
clustering_result = clusterer.apply_leiden_clustering(embeddings)

# Extract topics
topic_analyzer = TopicAnalyzer(client)
topics = topic_analyzer.extract_topics(
    embeddings, clustering_result['labels'], texts
)

# Visualize results
visualizer = ClusterVisualizer()
plots = visualizer.cluster_visual(embeddings, clustering_result['labels'])
```


## 🧪 Testing Suite

Located in `../unit_test/` directory with comprehensive test coverage:

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `test_clustering.py` | Core clustering algorithm validation | K-Means, DBSCAN, Leiden |
| `test_clustering_methods.py` | Method comparison and selection | Performance metrics, selection logic |
| `test_report_genneration modular.py` | Report generation pipeline | API integration, data processing |

```bash
# Run all tests
python -m pytest ../unit_test/

# Run specific test
python ../unit_test/test_clustering.py
```

## Getting Started

### Prerequisites
- Python 3.8+
- Azure OpenAI access with API key
- Customer interaction data in pickle format



### Notes and Known Issues

- The actual theme analysis package in this repo is `ccts_theme_driver_analysis`.
- `ccts_theme_driver_analysis/analyzer.py` defaults to `text_column="primary_complaint_issue_clean"`, but the data processor currently extracts `primary_complaint_issue`.
- `report_generation/analyzer.py` sorts by `conversation_start` but does not include that field in the required column list, which may cause failures if the field is missing.
- `levels_report_generation.py` imports `subprocess` but does not use it.



### Configuration

Create `.env` file in project root with all required variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_MODEL=gpt-4o

# Data Paths (all directory configurations together)
DATA_PICKLE_PATH=/path/to/interactions.pkl
CONVERSATION_OUTPUT_DIR=/path/to/output/conversations
JOURNEY_OUTPUT_DIR=/path/to/output/journeys

# Processing Configuration
MAX_TOKENS_INTERACTION=10240
MAX_TOKENS_SUMMARY=8192
TEMPERATURE=0.1
API_RETRY_ATTEMPTS=3
API_RETRY_DELAY=2
REQUEST_DELAY=2.0
```

## Module Design

### Separation of Concerns

Each step module is self-contained with:
- **config.py** - All configuration/settings
- **\*.py** - Step-specific business logic
- **utils.py** - Shared utilities
- **prompts.py** - AI prompt templates (if applicable)

### Import Pattern

```python
# From levels_report_generation.py
from report_generation import (
    ConfigManager,
    InteractionAnalyzer,
    save_analysis_results,
)

# To add new step
from analysis_step_2 import SomeAnalyzer
```

## Adding New Steps

To add Step 2 (or another step):

1. Create new folder: `src/analysis_step_2/`
2. Add core files:
   - `__init__.py` - Package exports
   - `analyzer.py` - Main logic
   - `config.py` - Configuration (if needed)
   - `utils.py` - Utilities

3. Update `levels_report_generation.py` to orchestrate the new step

Example:
```python
from analysis_step_2 import PatternAnalyzer

# In main():
pattern_analyzer = PatternAnalyzer()
patterns = pattern_analyzer.analyze(interaction_results)
```

## Dependencies

Complete list of Python packages in `requirements.txt`:

### Core Dependencies
- `openai>=1.3.0` - Azure OpenAI API integration
- `pandas>=1.5.0` - Data manipulation and analysis
- `python-dotenv>=1.0.0` - Environment variable management

### Machine Learning & Clustering
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `hdbscan>=0.8.0` - Hierarchical density-based clustering
- `umap-learn>=0.5.0` - Dimensionality reduction
- `python-igraph>=0.10.0` - Graph algorithms
- `leidenalg>=0.9.0` - Leiden community detection
- `scipy>=1.7.0` - Scientific computing

### Visualization & Utilities
- `matplotlib>=3.5.0` - Plotting and visualization
- `kneed>=0.8.0` - Knee point detection for K-Means

### Installation
```bash
pip install -r requirements.txt
```

### Optional Dependencies
Some packages may require system-level installations:
- `python-igraph` may need system graph libraries
- `leidenalg` requires igraph as dependency

## Logging

All modules log to:
- Console (stdout)
- `level_reports.log` file

Log levels:
- DEBUG - Detailed diagnostic info
- INFO - Progress/operations
- WARNING - Warning conditions
- ERROR - Error conditions

## Error Handling

Each step module includes:
- **Retry logic** - Automatic retry with backoff
- **Validation** - Input/output validation
- **Error recovery** - Graceful degradation
- **Logging** - Detailed error context

## Configuration Hierarchy

1. Environment variables (`.env` file)
2. Default values in `config.py`
3. Command-line arguments (future enhancement)

## Testing

Modules are designed to be independently testable:

```python
from report_generation import ConfigManager, InteractionAnalyzer

# Test configuration
config = ConfigManager.load_processing_config()
assert config.temperature == 0.1

# Test analyzer
analyzer = InteractionAnalyzer(mock_client, config)
results = analyzer.analyze_all_interactions(test_df)
```

## Performance Considerations

- **API Rate Limiting** - Configurable delays between requests
- **Retry Strategy** - Exponential backoff for failed requests
- **Memory Usage** - Processes data in chunks
- **Logging Overhead** - Can be adjusted via log levels



### API Connection
```bash
# Test Azure OpenAI connection
python -c "from openai import AzureOpenAI; ..."
```

---

**Version**: 1.0  
**Status**: Production-Ready with Complete Modular Architecture  
**Documentation**: Comprehensive coverage of all src/ components  
**Last Updated**: April 2026

