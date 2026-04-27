# CCTS Statistical Analysis Pipeline

This package runs statistical analysis for CCTS complaints and prior customer interactions inside Databricks.

## What it does

- Loads CCTS complaints from a local CSV file in the Databricks workspace
- Fetches prior chat and voice interactions via Spark SQL from Databricks Unity Catalog
- Merges interactions with complaints by account number and complaint date
- Uses Azure OpenAI to label interactions as relevant or irrelevant
- Computes complaint-level statistics and interaction history metrics
- Analyzes department and tier distributions for matched interactions
- Builds Plotly-ready plots for interactive exploration

## Inputs

1. `CCTS_CSV_PATH` pointing to the complaint CSV in Databricks
2. Prior chat and voice interactions fetched from Databricks Spark tables

> There is no external database connection needed outside the Databricks environment.

## Running the Pipeline

```python
from statistical_analysis import CCTSAnalysisPipeline

pipeline = CCTSAnalysisPipeline()
results = pipeline.run()

print(results["statistics"])
```

### Notebook execution

In `src/main.ipynb`, run the statistical analysis pipeline before theme driver analysis:

```python
from statistical_analysis import CCTSAnalysisPipeline

pipeline = CCTSAnalysisPipeline()
results = pipeline.run()
print(results["statistics"])
```

## Required environment variables

- `CCTS_CSV_PATH` — path to the CSV file with CCTS complaints
- `ANALYSIS_PERIOD` — analysis label used for logging
- `CCTS_START` — start date for CCTS filtering
- `CCTS_END` — end date for CCTS filtering
- `DATABRICKS_CATALOG` — Unity Catalog catalog name
- `DATABRICKS_SCHEMA` — schema for Genesys transcript and call detail tables
- `DATABRICKS_REPORTING_SCHEMA` — schema for reporting tables
- `AZURE_OPENAI_API_KEY` — Azure OpenAI key
- `AZURE_OPENAI_API_VERSION` — Azure OpenAI API version
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI endpoint
- `RELEVANCY_SLEEP_TIME` — seconds to wait between relevance API calls

## Package structure

- `config.py` — environment configuration
- `data_loader.py` — load CCTS and interaction data from Databricks
- `relevancy.py` — Azure OpenAI relevancy labeling
- `analyzer.py` — compute statistics and buckets
- `pipeline.py` — orchestrates the full analysis flow
- `README.md` — this file

## Notes

- The pipeline is designed to run inside Databricks with an active Spark session.
- Interactions are matched to complaints by account number and transcript date.
- If no interactions are found or no complaints match, the pipeline returns an empty result structure instead of failing.


## Contributing

1. Follow the modular class structure
2. Add comprehensive logging
3. Include docstrings for all methods
4. Add error handling for new functionality
5. Update tests for new features


## Usage

### Basic Usage

```python
from statistical_analysis_pipeline import CCTSAnalysisPipeline

# Run full analysis
pipeline = CCTSAnalysisPipeline()
results = pipeline.run_full_analysis()

# Access results
stats = results['stats']
plots = results['plots']
interaction_history = results['interaction_history']
```

### Command Line

```bash
python statistical_analysis_pipeline.py
```

### Advanced Usage

```python
from statistical_analysis_pipeline import Config, CCTSAnalysisPipeline

# Custom configuration
config = Config()
config.analysis_period = "Dec 2025"
config.ccts_start = "2025/12/01"
config.ccts_end = "2025/12/31"

pipeline = CCTSAnalysisPipeline(config)
results = pipeline.run_full_analysis()
```

## Architecture

### Core Classes

- **`Config`**: Configuration management from environment variables
- **`DataLoader`**: Handles data loading from Salesforce CSV and database queries
- **`RelevancyChecker`**: Uses Azure OpenAI to check interaction relevancy
- **`StatisticalAnalyzer`**: Performs statistical calculations and creates visualizations
- **`CCTSAnalysisPipeline`**: Main orchestrator class

### Key Metrics

The pipeline calculates:

- **Basic Statistics**:
  - Total CCTS complaints
  - Percentage with prior interactions
  - Percentage with relevant interactions
  - Average interactions per complaint

- **Interaction Patterns**:
  - Time between first/last interaction and CCTS
  - Interaction count distributions
  - Total interaction time buckets

- **Department Analysis**:
  - Department distribution for CCTS complaints
  - Tier 1 vs Tier 2 escalation patterns

## Output

The pipeline generates:

1. **Pickle Files**: Processed interaction data
2. **Statistical Summary**: Printed to console
3. **Interactive Plots**: Plotly visualizations
4. **DataFrames**: Structured analysis results

## Database Setup

The pipeline expects access to:

- **Salesforce Data**: CSV export with CCTS complaints
- **Interaction Data**: Chat and voice transcripts from Genesys
- **Queue Hierarchy**: Department and tier mappings

Database queries are currently written for Databricks/Spark SQL syntax. Modify `DataLoader` methods for other database systems.

## Configuration Details

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANALYSIS_PERIOD` | Analysis period label | Nov 2025 |
| `CCTS_START` | Start date for CCTS filtering | 2025/11/01 |
| `CCTS_END` | End date for CCTS filtering | 2025/11/30 |
| `SALESFORCE_CSV_PATH` | Path to Salesforce CSV | /Workspace/... |
| `DB_SERVER` | Database server | your_server |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `RELEVANCY_SLEEP_TIME` | Sleep between API calls | 0.5 |

## Error Handling

The pipeline includes comprehensive error handling:

- Database connection failures
- API rate limiting
- Data processing errors
- File I/O issues

All errors are logged with detailed information.

## Logging

The pipeline uses Python's logging module with the following levels:

- **INFO**: General progress and results
- **DEBUG**: Detailed processing information
- **ERROR**: Error conditions
- **WARNING**: Warning conditions

## Dependencies

- `openai`: Azure OpenAI integration
- `pandas`: Data manipulation
- `plotly`: Interactive visualizations
- `pyodbc`: Database connectivity
- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations
- `scipy`: Statistical functions
