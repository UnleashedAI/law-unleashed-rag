# RAG Evaluation Framework

This directory contains a simplified evaluation framework for comparing different RAG approaches using LLM-based metrics.

## Overview

The evaluation framework provides a streamlined way to test and compare RAG systems by:
- **Project-specific evaluation suites** - Each project gets its own evaluation suite
- **LLM-as-a-Judge metrics** - Uses GPT-4o-mini to evaluate relevance, completeness, and coherence
- **Simplified case definitions** - Just prompt and expected output strings
- **Detailed explanations** - LLM provides explanations for each metric score
- **Performance tracking** - Monitors processing time and memory usage

## Requirements

Install the evaluation framework dependencies:

```bash
# Install evaluation framework dependencies
pip install -r evals/requirements.txt

# Or install individually
pip install httpx sentence-transformers scikit-learn openai pydantic psutil
```

### Dependencies

- **httpx**: HTTP client for API calls to the RAG service
- **sentence-transformers**: For semantic similarity calculations
- **scikit-learn**: For machine learning metrics and similarity calculations
- **openai**: For LLM-based evaluation metrics
- **pydantic**: For data validation and serialization
- **psutil**: For system resource monitoring during evaluations

## Directory Structure

```
evals/
├── __init__.py                 # Main evals module
├── README.md                   # This file
├── eval_models.py             # Evaluation data models
├── eval_service.py            # Evaluation execution service
├── eval_cli.py                # Command-line interface
├── project_registry.py        # RAG project configuration management
├── requirements.txt           # Evaluation framework dependencies
├── suites/                    # Project-specific test suite definitions
│   ├── __init__.py
│   └── [projectId]_suite.py   # Project-specific evaluation suites
└── results/                   # Evaluation results storage
    └── [timestamp]_[projectId]_[approach].json  # Results files
```

## Evaluation Suites

### Project-Specific Suites

Each RAG project has its own evaluation suite file named `[projectId]_suite.py`:

- **NbbabGQy3gkCJIDzkSoE_suite.py**: Chiropractic medical records evaluation
- **Future projects**: Each will get its own `[projectId]_suite.py` file

### Suite Structure

Each suite contains:
- **User ID**: Stored in the suite for convenience
- **Evaluation Cases**: Simple prompt + expected output pairs
- **Project Configuration**: Default RAG approaches and settings

## Usage

### Command Line Interface

```bash
# List available project suites
python evals/eval_cli.py list-project-suites

# Run evaluation for a specific project and RAG approach
python evals/eval_cli.py run-project-suite [projectId] --run-type [approach]

# Examples:
python evals/eval_cli.py run-project-suite NbbabGQy3gkCJIDzkSoE --run-type raganything
python evals/eval_cli.py run-project-suite NbbabGQy3gkCJIDzkSoE --run-type rag_vertex
```

### Available RAG Approaches

- **raganything**: Uses the raganything RAG service
- **rag_vertex**: Uses Google Vertex AI RAG service

## Results Organization

Evaluation results are saved with the naming convention `[timestamp]_[projectId]_[approach].json`:

```
results/
├── 20251027_114930_NbbabGQy3gkCJIDzkSoE_rag_vertex.json
├── 20251027_115141_NbbabGQy3gkCJIDzkSoE_raganything.json
└── ...
```

### Results Format

Each results file contains:
- **Metadata**: Timestamp, project ID, approach
- **Evaluation Results**: For each test case:
  - Prompt and expected output
  - Actual LLM response
  - Metrics with explanations:
    - **Relevance**: How well the response addresses the prompt
    - **Completeness**: How much expected content was included
    - **Coherence**: How well-structured and logical the response is
  - Performance data: Processing time and memory usage
  - Overall weighted score

## Creating Custom Evaluation Suites

1. **Create a new project suite file** in `suites/`:

```python
from ..eval_models import EvaluationCase, EvaluationSuite

def create_[projectId]_suite() -> EvaluationSuite:
    """Create evaluation suite for [projectId] project"""
    
    evaluation_cases = [
        EvaluationCase(
            id="test_case_1",
            name="Test Case 1",
            project_id="[projectId]",
            prompt="Your test prompt here...",
            expected_output="Expected response string here..."
        ),
        # Add more test cases...
    ]
    
    return EvaluationSuite(
        id="[projectId]",
        name="[Project Name] Evaluation Suite",
        description="Evaluation suite for [project]",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "rag_vertex"],
        tags=["your", "tags", "here"],
        user_id="your_user_id_here"
    )
```

2. **Add to the main module** by updating `__init__.py`:

```python
from .suites.[projectId]_suite import create_[projectId]_suite

def get_all_evaluation_suites():
    # ... existing suites ...
    return {
        # ... existing suites ...
        "[projectId]": create_[projectId]_suite()
    }
```

## Evaluation Metrics

The framework uses LLM-based evaluation with three key metrics:

### Relevance (0.0 - 1.0)
- **High (0.8-1.0)**: Response directly addresses the prompt
- **Medium (0.5-0.7)**: Partially relevant but missing key elements
- **Low (0.0-0.4)**: Off-topic, refusal, or error responses

### Completeness (0.0 - 1.0)
- **High (0.8-1.0)**: Contains all expected information
- **Medium (0.5-0.7)**: Missing some expected elements
- **Low (0.0-0.4)**: Missing most or all expected content

### Coherence (0.0 - 1.0)
- **High (0.8-1.0)**: Well-structured, logical flow
- **Medium (0.5-0.7)**: Somewhat organized but unclear in places
- **Low (0.0-0.4)**: Poorly structured or confusing

### Performance Metrics
- **Processing Time**: Total time for evaluation (seconds)
- **Memory Usage**: Additional memory consumed during evaluation (MB)
- **Overall Score**: Weighted combination of all metrics

## Key Features

### Simplified Case Definition
- Just `prompt` and `expected_output` strings
- No complex document paths or criteria
- Easy to create and maintain

### LLM-as-a-Judge
- Uses GPT-4o-mini for consistent evaluation
- Provides detailed explanations for each score
- Handles edge cases like empty responses and refusals

### Project Integration
- Automatically resolves project IDs from registry
- Uses correct models and configurations per approach
- Handles corpus info for Vertex AI

### Detailed Logging
- Step-by-step progress tracking
- API call monitoring
- Performance metrics

## Best Practices

1. **Start with simple test cases** to understand the framework
2. **Use realistic prompts** that match your actual use cases
3. **Define clear expected outputs** with specific details
4. **Review LLM explanations** to understand scoring decisions
5. **Compare multiple approaches** to see relative performance
6. **Monitor performance metrics** for optimization opportunities

## Troubleshooting

### Common Issues

1. **Service not running**: Start the RAG service with `./start_rag_service.sh`
2. **Project not found**: Check that the project exists in `rag_projects.json`
3. **Empty responses**: Verify the RAG service is working and documents are processed
4. **Model errors**: Check that the correct model is configured for each approach

### Getting Help

- Check the main project README for setup instructions
- Review the API documentation at `http://localhost:8000/docs`
- Look at existing suite files for reference implementations
- Check the evaluation results JSON files for detailed diagnostics