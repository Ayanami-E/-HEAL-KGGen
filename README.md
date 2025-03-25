# HEAL-KGGen
# Knowledge Graph Enhanced Medical Question Answering System

This is an evaluation framework for enhancing Large Language Models (LLMs) with knowledge graphs to improve performance on medical domain question answering. The system integrates structured medical knowledge to significantly increase the accuracy of LLMs on medical QA tasks.

## Features

- **Knowledge Graph Enhancement**: Leverages structured medical knowledge graphs to boost LLM performance
- **Entity Extraction**: Automatically extracts medical entities from questions
- **Relationship Queries**: Retrieves entity relationships to support answer generation
- **Dual System Evaluation**: Evaluates both baseline LLM and knowledge graph enhanced system
- **Detailed Analysis**: Generates comprehensive evaluation reports and comparative analysis
- **Multi-model Support**: Supports various LLM models through an OpenAI-compatible interface

## System Architecture

The system consists of several core modules:

1. **KG Handler**: Manages knowledge graph loading and querying
2. **Entity Extraction**: Identifies medical entities in text
3. **LLM Interface**: Provides a unified interface for large language model interaction
4. **Knowledge Graph Enhancement**: Enriches questions with graph information
5. **Evaluation Framework**: Executes tests and analyzes results

## File Structure

- `main.py` - Main entry point for running evaluations
- `config.py` - Configuration settings for API keys and paths
- `kg_handler.py` - Knowledge graph processing and querying
- `llm_interface.py` - Interface for interacting with language models
- `medqa_core.py` - Core evaluation framework implementation
- `entity_extraction.py` - Medical entity extraction functionality
- `kg_enhancement.py` - Knowledge graph enhancement logic
- `kg_evaluation.py` - Evaluation with knowledge graph enhancement
- `baseline_evaluation.py` - Baseline model evaluation functionality
- `analysis_utils.py` - Utilities for analyzing and comparing results
- `data_utils.py` - Data loading and preprocessing utilities

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
```

### Configure API

Set up your API key in `config.py`:

```python
PANDALLA_API_KEY = "your_api_key_here"
PANDALLA_API_BASE = "your_api_base_url_here"
```

### Prepare Knowledge Graph

Set your knowledge graph JSON file path in the configuration:

```python
KG_JSON_PATH = "path/to/your/knowledge_graph.json"
```

### Run Tests

```bash
python main.py --test_file ./your_test_data.json --sample_limit 100 --mode both
```

Parameters:
- `--test_file`: Test data file path
- `--kg_path`: Knowledge graph file path (defaults to path in config)
- `--sample_limit`: Limit of samples to process
- `--output_dir`: Results output directory
- `--mode`: Evaluation mode (kg=knowledge graph only, baseline=baseline only, both=evaluate both)
- `--entity_log`: Entity extraction log file path

## Data Format

Test data should be provided in JSON format, with each sample containing:
- `question`: Medical question text
- `options`: Answer options (as dictionary or list)
- `answer` or `answer_idx`: Correct answer

Knowledge graph should be in a specific JSON format containing nodes and edges:
```json
{
  "nodes": [
    {"id": "disease: Diabetes", "category": "disease", ...},
    {"id": "gene: BRCA1", "category": "gene", ...},
    ...
  ],
  "edges": [
    {"source": "gene: BRCA1", "target": "disease: Breast Cancer", "type": "associated_with", ...},
    ...
  ]
}
```

## Output Results

The system generates the following output files:
1. Evaluation results for the knowledge graph enhanced system
2. Evaluation results for the baseline system
3. Comparative analysis report
4. Medical entity extraction log
5. Detailed knowledge graph contribution analysis report


