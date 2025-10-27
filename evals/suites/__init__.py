"""
Test suite definitions for RAG evaluation
"""

# Import only the suites that exist
try:
    from .real_world_test_suites import get_real_world_evaluation_suites
except ImportError:
    def get_real_world_evaluation_suites():
        return {}

try:
    from .unstructured_test_suites import get_unstructured_evaluation_suites
except ImportError:
    def get_unstructured_evaluation_suites():
        return {}

__all__ = [
    'get_real_world_evaluation_suites',
    'get_unstructured_evaluation_suites'
]

