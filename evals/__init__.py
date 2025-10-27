"""
Evaluation framework for RAG approach comparison
"""

from .eval_service import EvaluationService
from .eval_models import (
    EvaluationCase, EvaluationSuite, EvaluationRun, EvaluationCaseResult,
    EvaluationRequest, EvaluationResponse, EvaluationStatusResponse,
    EvaluationResultsResponse, EvaluationStatus, MetricType, EvaluationType
)

def get_all_evaluation_suites():
    """Get all available evaluation suites"""
    # Import only the suites that exist
    suites = {}
    
    try:
        from .suites.real_world_test_suites import get_real_world_evaluation_suites
        suites.update(get_real_world_evaluation_suites())
    except ImportError:
        pass
    
    try:
        from .suites.unstructured_test_suites import get_unstructured_evaluation_suites
        suites.update(get_unstructured_evaluation_suites())
    except ImportError:
        pass
    
    return suites

__all__ = [
    'get_all_evaluation_suites',
    'EvaluationService',
    'EvaluationCase', 'EvaluationSuite', 'EvaluationRun', 'EvaluationCaseResult',
    'EvaluationRequest', 'EvaluationResponse', 'EvaluationStatusResponse',
    'EvaluationResultsResponse', 'EvaluationStatus', 'MetricType', 'EvaluationType'
]