"""
NbbabGQy3gkCJIDzkSoE Project Evaluation Suite
- Chiropractic medical records evaluation
- Runnable with any RAG approach
"""

from ..eval_models import EvaluationCase, EvaluationSuite


def create_nbbabgqy3gkcjidzksoe_suite() -> EvaluationSuite:
    """Create evaluation suite for NbbabGQy3gkCJIDzkSoE project"""
    
    evaluation_cases = [
        EvaluationCase(
            id="insureds_liability",
            name="Insured's Liability",
            project_id="NbbabGQy3gkCJIDzkSoE",
            prompt="""You are the plaintiff's personal injury attorney. Write a few paragraphs describing why the insured is liable. This information generally comes from a wreck report or a police report describing the incident. Start with something like "The wreck report establishes that the acts or omissions of your insured driver..." Clearly establish the at-fault driver and if there are violations present detail those out e.g. "he violated O.C.G.A. Section 40-6-72(b) (failure to stop at a stop sign)". Finish with a paragraph starting with "If forced to litigate, it is quite clear that..." Do not include an headers, just paragraphs.""",
            expected_output="""The wreck report establishes that the acts or omissions of your insured driver, Marion Donald Cantrell, were the sole proximate cause of the collision that occurred on January 7, 2023. According to the report, Mr. Cantrell, operating Vehicle 1, was traveling on Old Atlanta Road and failed to stop at a clearly marked stop sign at the intersection with Birdie Road. This egregious failure resulted in his vehicle striking Mr. Colby A. Nuce's truck (Vehicle 2) in the front driver corner and side, pushing Mr. Nuce's vehicle to the side. By running the stop sign, your insured driver violated O.C.G.A. Section 40-6-72(b), which requires drivers to stop at stop signs. The police report further corroborates that Mr. Nuce, operating Vehicle 2, was stopped at the 4-way intersection when your insured driver ran the stop sign. A witness also confirmed that Vehicle 2 was stopped at the stop sign, while Vehicle 1 ran the stop sign and did not stop, causing the collision as Vehicle 2 attempted to proceed through the intersection. Your insured's actions directly led to the injuries sustained by Mr. Nuce. If forced to litigate, it is quite clear that a jury would find your insured driver entirely at fault for this collision, given the explicit statements in the wreck report and the direct violation of traffic law."""
        ),
    ]
    
    return EvaluationSuite(
        id="NbbabGQy3gkCJIDzkSoE",
        name="Colby Nuce v. Marion Cantrell",
        description="Comprehensive evaluation suite for Colby Nuce v. Marion Cantrell",
        evaluation_cases=evaluation_cases,
        default_rag_approaches=["raganything", "rag_vertex"],
        tags=["chiropractic", "medical_records", "legal_analysis"],
        user_id="7CtdhckRcxOIjU3Dh7Ao3jvigg13"
    )


def get_project_suites():
    """Get all project-specific evaluation suites"""
    return {
        "NbbabGQy3gkCJIDzkSoE": create_nbbabgqy3gkcjidzksoe_suite()
    }
