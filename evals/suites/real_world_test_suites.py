"""
Real-world evaluation suites based on actual use cases
"""

from typing import Dict
from ..eval_models import (
    EvaluationCase, EvaluationSuite, ExpectedOutput, EvaluationCriteria,
    EvaluationType, MetricType
)
from ..project_registry import project_registry


def _populate_document_paths(evaluation_case: EvaluationCase) -> EvaluationCase:
    """Populate document paths from project registry"""
    if evaluation_case.project_id:
        document_paths = project_registry.get_project_document_paths(evaluation_case.project_id)
        evaluation_case.document_paths = document_paths
    return evaluation_case


def create_chiropractic_records_evaluation_suite() -> EvaluationSuite:
    """Create a test suite for chiropractic medical records processing"""
    
    evaluation_cases = [
        EvaluationCase(
            id="chiropractic_records_analysis",
            name="Chiropractic Medical Records Analysis",
            description="Test ability to extract key information from chiropractic medical records",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",
            document_paths=[],  # Will be populated from project registry
            expected_outputs=[
                ExpectedOutput(
                    type="patient_info",
                    content=["patient_name", "date_of_birth", "patient_id", "contact_info"],
                    weight=1.0,
                    description="Basic patient identification information"
                ),
                ExpectedOutput(
                    type="medical_history",
                    content=["chief_complaint", "symptoms", "pain_level", "duration"],
                    weight=1.0,
                    description="Primary medical complaints and symptoms"
                ),
                ExpectedOutput(
                    type="treatment_info",
                    content=["diagnosis", "treatment_plan", "medications", "follow_up"],
                    weight=1.0,
                    description="Diagnosis and treatment information"
                ),
                ExpectedOutput(
                    type="chiropractic_specific",
                    content=["spinal_assessment", "adjustments", "range_of_motion", "posture_analysis"],
                    weight=0.8,
                    description="Chiropractic-specific assessment and treatment details"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.PROCESSING_TIME],
                thresholds={
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.7,
                    MetricType.PROCESSING_TIME: 60.0
                },
                weights={
                    MetricType.COMPLETENESS: 0.6,
                    MetricType.SEMANTIC_SIMILARITY: 0.3,
                    MetricType.PROCESSING_TIME: 0.1
                }
            ),
            tags=["medical", "chiropractic", "records", "real-world"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="chiropractic_query_answering",
            name="Chiropractic Records Query Answering",
            description="Test ability to answer specific questions about chiropractic treatment",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",
            document_paths=[],  # Will be populated from project registry
            query="What was the patient's chief complaint and what treatment was recommended?",
            expected_outputs=[
                ExpectedOutput(
                    type="query_answer",
                    content="Patient's primary complaint and recommended treatment plan",
                    weight=1.0,
                    description="Comprehensive answer about patient's condition and treatment"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY],
                thresholds={
                    MetricType.RELEVANCE: 0.8,
                    MetricType.COMPLETENESS: 0.7,
                    MetricType.SEMANTIC_SIMILARITY: 0.7
                },
                weights={
                    MetricType.RELEVANCE: 0.4,
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.3
                }
            ),
            tags=["medical", "chiropractic", "query-answering", "real-world"],
            difficulty="medium"
        ),
        
        EvaluationCase(
            id="chiropractic_evidence_extraction",
            name="Chiropractic Evidence Extraction",
            description="Test ability to extract evidence for legal/insurance purposes",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",
            document_paths=[],  # Will be populated from project registry
            expected_outputs=[
                ExpectedOutput(
                    type="legal_evidence",
                    content=["treatment_dates", "provider_credentials", "billing_codes", "medical_necessity"],
                    weight=1.0,
                    description="Information relevant for legal or insurance claims"
                ),
                ExpectedOutput(
                    type="clinical_evidence",
                    content=["objective_findings", "subjective_symptoms", "functional_limitations", "prognosis"],
                    weight=1.0,
                    description="Clinical evidence supporting treatment decisions"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.RELEVANCE],
                thresholds={
                    MetricType.COMPLETENESS: 0.85,
                    MetricType.SEMANTIC_SIMILARITY: 0.75,
                    MetricType.RELEVANCE: 0.8
                },
                weights={
                    MetricType.COMPLETENESS: 0.4,
                    MetricType.SEMANTIC_SIMILARITY: 0.3,
                    MetricType.RELEVANCE: 0.3
                }
            ),
            tags=["medical", "chiropractic", "legal", "evidence", "real-world"],
            difficulty="hard"
        )
    ]
    
    # Populate document paths from project registry
    evaluation_cases = [_populate_document_paths(case) for case in evaluation_cases]
    
    return EvaluationSuite(
        id="chiropractic_records",
        name="Chiropractic Medical Records",
        description="Test suite for evaluating RAG approaches on chiropractic medical records",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["medical", "chiropractic", "real-world", "legal"]
    )


def create_law_unleashed_evaluation_suite() -> EvaluationSuite:
    """Create a comprehensive test suite for law-unleashed use cases"""
    
    evaluation_cases = [
        EvaluationCase(
            id="medical_records_comprehensive",
            name="Comprehensive Medical Records Analysis",
            description="Test comprehensive analysis of medical records for legal purposes",
            evaluation_type=EvaluationType.DOCUMENT_PROCESSING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",
            document_paths=[],  # Will be populated from project registry
            expected_outputs=[
                ExpectedOutput(
                    type="patient_summary",
                    content=["demographics", "medical_history", "current_condition", "treatment_timeline"],
                    weight=1.0,
                    description="Comprehensive patient summary"
                ),
                ExpectedOutput(
                    type="legal_relevant_info",
                    content=["injuries", "causation", "damages", "treatment_costs", "prognosis"],
                    weight=1.0,
                    description="Information relevant for legal proceedings"
                ),
                ExpectedOutput(
                    type="medical_entities",
                    content=["diagnoses", "procedures", "medications", "providers", "facilities"],
                    weight=0.8,
                    description="Medical entities and terminology"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.RELEVANCE, MetricType.PROCESSING_TIME],
                thresholds={
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.7,
                    MetricType.RELEVANCE: 0.8,
                    MetricType.PROCESSING_TIME: 90.0
                },
                weights={
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.2,
                    MetricType.RELEVANCE: 0.3,
                    MetricType.PROCESSING_TIME: 0.2
                }
            ),
            tags=["legal", "medical", "comprehensive", "real-world"],
            difficulty="hard"
        ),
        
        EvaluationCase(
            id="legal_case_analysis",
            name="Legal Case Analysis",
            description="Test ability to analyze medical records for legal case preparation",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",
            document_paths=[],  # Will be populated from project registry
            query="What evidence supports the patient's claim of injury and what are the key facts for the legal case?",
            expected_outputs=[
                ExpectedOutput(
                    type="legal_analysis",
                    content="Comprehensive analysis of injury evidence, causation, damages, and key legal facts",
                    weight=1.0,
                    description="Legal analysis suitable for case preparation"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.COHERENCE],
                thresholds={
                    MetricType.RELEVANCE: 0.85,
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.75,
                    MetricType.COHERENCE: 0.8
                },
                weights={
                    MetricType.RELEVANCE: 0.3,
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.2,
                    MetricType.COHERENCE: 0.2
                }
            ),
            tags=["legal", "case-analysis", "real-world"],
            difficulty="hard"
        )
    ]
    
    # Populate document paths from project registry
    evaluation_cases = [_populate_document_paths(case) for case in evaluation_cases]
    
    return EvaluationSuite(
        id="law_unleashed_comprehensive",
        name="Law Unleashed Comprehensive",
        description="Comprehensive test suite for law-unleashed legal document processing",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "evidence_sweep", "rag_vertex"],
        tags=["legal", "medical", "comprehensive", "real-world", "law-unleashed"]
    )


def create_rag_approach_comparison_suite() -> EvaluationSuite:
    """Create a test suite for comparing different RAG approaches on the same project"""
    
    evaluation_cases = [
        EvaluationCase(
            id="rag_approach_comparison",
            name="RAG Approach Comparison",
            description="Compare RAGAnything vs Vertex RAG on the same chiropractic records",
            evaluation_type=EvaluationType.QUERY_ANSWERING,
            project_id="NbbabGQy3gkCJIDzkSoE_raganything_1",  # Will be overridden per approach
            document_paths=[],  # Will be populated from project registry
            query="What are the main findings and treatment recommendations for this patient?",
            expected_outputs=[
                ExpectedOutput(
                    type="comprehensive_analysis",
                    content="Detailed analysis of patient condition, treatment plan, and key medical findings",
                    weight=1.0,
                    description="Comprehensive medical analysis suitable for legal or insurance purposes"
                )
            ],
            evaluation_criteria=EvaluationCriteria(
                metrics=[MetricType.RELEVANCE, MetricType.COMPLETENESS, MetricType.SEMANTIC_SIMILARITY, MetricType.PROCESSING_TIME],
                thresholds={
                    MetricType.RELEVANCE: 0.8,
                    MetricType.COMPLETENESS: 0.8,
                    MetricType.SEMANTIC_SIMILARITY: 0.7,
                    MetricType.PROCESSING_TIME: 60.0
                },
                weights={
                    MetricType.RELEVANCE: 0.3,
                    MetricType.COMPLETENESS: 0.3,
                    MetricType.SEMANTIC_SIMILARITY: 0.2,
                    MetricType.PROCESSING_TIME: 0.2
                }
            ),
            tags=["comparison", "rag-approaches", "real-world"],
            difficulty="medium"
        )
    ]
    
    # Populate document paths from project registry
    evaluation_cases = [_populate_document_paths(case) for case in evaluation_cases]
    
    return EvaluationSuite(
        id="rag_approach_comparison",
        name="RAG Approach Comparison",
        description="Compare different RAG approaches on the same real-world project",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "rag_vertex"],  # Compare these two approaches
        tags=["comparison", "rag-approaches", "real-world", "chiropractic"]
    )


def get_real_world_evaluation_suites() -> dict:
    """Get all real-world evaluation suites"""
    
    return {
        "chiropractic_records": create_chiropractic_records_evaluation_suite(),
        "law_unleashed_comprehensive": create_law_unleashed_evaluation_suite(),
        "rag_approach_comparison": create_rag_approach_comparison_suite()
    }
