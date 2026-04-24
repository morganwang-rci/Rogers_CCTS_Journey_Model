# Report Generation Module (Step 1)

## Overview

This module handles the first step of the CCTS complaint analysis solution: generating comprehensive complaint reports from customer interaction data.

## What It Does

1. **Interaction Analysis** - Analyzes each customer-agent interaction
   - Correlates interactions to CCTS complaints
   - Identifies key topics, moments, and issues
   - Assesses escalation risk
   
2. **Agent Evaluation** - Evaluates agent performance
   - Communication and empathy assessment
   - Professionalism and efficiency rating
   - Identifies infractions and educational gaps
   
3. **Journey Mapping** - Creates customer journey summary
   - Traces complaint evolution
   - Identifies critical breakdowns
   - Root cause analysis
   - Prevention recommendations

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Package exports and metadata | 30 |
| `analyzer.py` | Core analysis engine | 350+ |
| `config.py` | Configuration management | 90 |
| `utils.py` | Utility functions | 100 |
| `prompts.py` | AI prompt templates | 400+ |
| `Level_reports_generation.py` | Original script (reference) | 726 |

## Quick Start

### Installation

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Azure credentials and data paths
```

### Basic Usage

```python
from report_generation import ConfigManager, InteractionAnalyzer
from openai import AzureOpenAI
import pandas as pd

# Load configuration
azure_config = ConfigManager.load_azure_config()
processing_config = ConfigManager.load_processing_config()

# Initialize client and analyzer
client = AzureOpenAI(
    api_key=azure_config.api_key,
    api_version=azure_config.api_version,
    azure_endpoint=azure_config.azure_endpoint,
)
analyzer = InteractionAnalyzer(client, processing_config)

# Load and analyze data
df = pd.read_pickle("data.pkl")
results = analyzer.analyze_all_interactions(df)
summary = analyzer.generate_summary_report(results, "file_123")
```

### Command Line

```bash
# From project root
python src/main.py
```

## Output Format

### Interaction Analysis
File: `interaction_analysis_{file_number}.json`

```json
{
  "interaction_sequence": 1,
  "interaction_identifier": "Case 12345_Interaction 1",
  "calendar_date": "2024-01-15",
  "case_number": "12345",
  "Conversational analysis": {
    "interaction_metadata": {...},
    "interaction_summary": {...},
    "escalation_factors": {...},
    "journey_insights": {...}
  },
  "Agent Evaluation": {
    "agent_evaluations": [...]
  }
}
```

### Journey Summary
File: `journey_summary_{file_number}.json`

```json
{
  "ccts_complaint_journey_analysis": {
    "case_number": "FN12345",
    "customer_complaint_genesis": {...},
    "response_assessment": {...},
    "journey_failure_points": {...}
  },
  "value_gap_analysis": {...},
  "prevention_opportunity_analysis": {...},
  "resolution_recommendations": {...}
}
```

## Configuration

### Environment Variables

```env
# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_MODEL=gpt-4o

# Data Paths (Required)
DATA_PICKLE_PATH=/path/to/interactions.pkl

# Output Directories (Optional)
CONVERSATION_OUTPUT_DIR=./output/conversations
JOURNEY_OUTPUT_DIR=./output/journeys

# Processing (Optional)
MAX_TOKENS_INTERACTION=10240
MAX_TOKENS_SUMMARY=8192
TEMPERATURE=0.1
API_RETRY_ATTEMPTS=3
API_RETRY_DELAY=2
REQUEST_DELAY=2.0
```

## API Reference

### InteractionAnalyzer

```python
class InteractionAnalyzer:
    def __init__(self, client: AzureOpenAI, processing_config: ProcessingConfig)
    def analyze_interaction(self, row: pd.Series) -> Tuple[str, str]
    def analyze_all_interactions(self, df: pd.DataFrame) -> List[Dict]
    def generate_summary_report(self, results: List, file_number: str) -> Dict
```

### ConfigManager

```python
class ConfigManager:
    @staticmethod
    def load_azure_config() -> AzureOpenAIConfig
    @staticmethod
    def load_data_config() -> DataConfig
    @staticmethod
    def load_processing_config() -> ProcessingConfig
```

### Utility Functions

```python
def safe_json_loads(json_str: str) -> Dict
def clean_openai_response(response: str) -> str
def validate_dataframe_columns(df, required_columns)
def create_interaction_identifier(case_number: str, seq: int) -> str
```

## Required DataFrame Columns

The input DataFrame must contain:

- `attr_account_number` - Account identifier
- `calendar_date` - Interaction date
- `Case Number` - Case identifier
- `File Number` - File identifier
- `Brand` - Product brand
- `Product Line` - Product line name
- `Product` - Specific product
- `media_type` - Contact channel (phone, chat, etc.)
- `Customer Issue` - Issue description
- `Root Cause` - Identified root cause
- `Notes` - Additional notes
- `full_transcript` - Interaction transcript
- `emp_id` - Employee identifier
- `conversation_start` - Conversation start time (for sorting)
- `relevancy` - Relevancy filter ('yes'/'no')

## Performance Tips

### Optimize Speed
- Increase `REQUEST_DELAY` if hitting rate limits (default: 2.0s)
- Reduce `MAX_TOKENS_*` if costs are high
- Process large files in batches

### Optimize Quality
- Lower `TEMPERATURE` for more consistent results (default: 0.1)
- Increase `MAX_TOKENS_*` for detailed analysis
- Use `API_RETRY_ATTEMPTS` for reliability

### Monitor Progress
- Check logs: `tail -f level_reports.log`
- Look for INFO level messages for progress
- WARNING level indicates recoverable issues
- ERROR level indicates case-specific failures

## Logging

All operations logged to:
- Console (INFO and above)
- `level_reports.log` (ALL levels)

Example log entries:
```
2024-01-15 10:30:45,123 - analyzer - INFO - Processing case 12345
2024-01-15 10:31:12,456 - analyzer - DEBUG - Analysis completed for case 12345
2024-01-15 10:32:00,789 - utils - WARNING - JSON parsing attempt 2
```

## Error Handling

### Common Errors

**Configuration Error**
```
ValueError: AZURE_OPENAI_API_KEY environment variable not set
Solution: Add to .env file and reload
```

**DataFrame Column Error**
```
ValueError: Missing required columns in dataframe: col_name
Solution: Verify input data has all required columns
```

**JSON Parsing Error**
```
JSONProcessingError: Failed to parse JSON
Solution: Check logs for raw content, may indicate API issue
```

### Retry Logic

- Automatic retry for API failures
- Exponential backoff: 2s, 4s, 6s (configurable)
- Graceful degradation on final failure
- Detailed error logging

## Testing

### Unit Test Example

```python
import pytest
from report_generation import safe_json_loads

def test_json_parsing():
    # Test markdown cleanup
    result = safe_json_loads('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}
    
def test_config_loading():
    from report_generation import ConfigManager
    config = ConfigManager.load_processing_config()
    assert config.temperature == 0.1
```

## Troubleshooting

### Issue: "No relevant interactions found"
```
Solution:
1. Verify 'relevancy' column exists in data
2. Check that some rows have relevancy == 'yes'
3. Confirm pickle file is correctly loaded
```

### Issue: "API rate limited"
```
Solution:
1. Increase REQUEST_DELAY in .env
2. Reduce number of simultaneous processes
3. Contact Azure support for higher limits
```





## Future Improvements

- [ ] Streaming API for large datasets
- [ ] Cache Azure OpenAI responses
- [ ] Parallel processing of cases
- [ ] Database backend for results
- [ ] Real-time progress tracking
- [ ] Structured output format options
- [ ] Custom prompt templates



## Support

For issues or questions:
1. Check logs in `level_reports.log`
2. Review configuration in `.env`
3. Verify input data format
4. Check Azure OpenAI service status
5. Review documentation files

---

**Version**: 1.0
**Last Updated**: April 2026
