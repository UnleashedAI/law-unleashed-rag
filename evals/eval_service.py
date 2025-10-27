"""
Evaluation service for comparing RAG approaches
"""

import logging
import uuid
import asyncio
import time
import psutil
import os
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .eval_models import (
    EvaluationCase, EvaluationSuite, EvaluationCaseResult, EvaluationRun, MetricResult,
    EvaluationStatus, MetricType, EvaluationType
)
from src.utils.firebase_utils import FirebaseManager

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for running evaluations of RAG approaches"""
    
    def __init__(self, firebase_manager: FirebaseManager, gcs_manager, auth_service):
        self.firebase_manager = firebase_manager
        self.gcs_manager = gcs_manager
        self.auth_service = auth_service
        
        # API configuration
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_timeout = 300  # 5 minutes timeout for API calls
        
        # Initialize evaluation components
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_transformer = None
    
    async def run_evaluation(
        self,
        evaluation_run: EvaluationRun,
        evaluation_suite: EvaluationSuite
    ) -> EvaluationRun:
        """Run a complete evaluation across all evaluation cases and RAG approaches"""
        
        logger.info(f"Starting evaluation run: {evaluation_run.id}")
        
        try:
            # Update evaluation status
            evaluation_run.status = EvaluationStatus.RUNNING
            evaluation_run.started_at = datetime.utcnow()
            evaluation_run.total_evaluation_cases = len(evaluation_suite.evaluation_cases) * len(evaluation_run.rag_approaches)
            
            await self._update_evaluation_run(evaluation_run)
            
            # Run evaluation cases for each RAG approach
            for rag_approach in evaluation_run.rag_approaches:
                logger.info(f"Evaluating RAG approach: {rag_approach}")
                
                for evaluation_case in evaluation_suite.evaluation_cases:
                    # Skip if evaluation case specifies a different RAG approach
                    if evaluation_case.rag_approach and evaluation_case.rag_approach != rag_approach:
                        continue
                    
                    try:
                        result = await self._run_evaluation_case(evaluation_case, rag_approach, evaluation_run.user_id)
                        evaluation_run.evaluation_case_results.append(result)
                        
                        # Update progress
                        evaluation_run.completed_evaluation_cases += 1
                        if result.status == EvaluationStatus.FAILED:
                            evaluation_run.failed_evaluation_cases += 1
                        
                        await self._update_evaluation_run(evaluation_run)
                        
                    except Exception as e:
                        logger.error(f"Error running evaluation case {evaluation_case.id} with {rag_approach}: {e}")
                        
                        # Create failed result
                        failed_result = EvaluationCaseResult(
                            evaluation_case_id=evaluation_case.id,
                            rag_approach=rag_approach,
                            status=EvaluationStatus.FAILED,
                            error_message=str(e),
                            executed_by=evaluation_run.user_id
                        )
                        evaluation_run.evaluation_case_results.append(failed_result)
                        evaluation_run.failed_evaluation_cases += 1
                        evaluation_run.completed_evaluation_cases += 1
                        
                        await self._update_evaluation_run(evaluation_run)
            
            # Calculate summary statistics
            await self._calculate_summary_statistics(evaluation_run)
            
            # Mark as completed
            evaluation_run.status = EvaluationStatus.COMPLETED
            evaluation_run.completed_at = datetime.utcnow()
            
            await self._update_evaluation_run(evaluation_run)
            
            logger.info(f"Completed evaluation run: {evaluation_run.id}")
            return evaluation_run
            
        except Exception as e:
            logger.error(f"Error in evaluation run {evaluation_run.id}: {e}")
            evaluation_run.status = EvaluationStatus.FAILED
            evaluation_run.completed_at = datetime.utcnow()
            await self._update_evaluation_run(evaluation_run)
            raise
    
    async def _run_evaluation_case(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> EvaluationCaseResult:
        """Run a single evaluation case with a specific RAG approach"""
        
        logger.info(f"Running evaluation case {evaluation_case.id} with {rag_approach}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = EvaluationCaseResult(
            evaluation_case_id=evaluation_case.id,
            rag_approach=rag_approach,
            status=EvaluationStatus.RUNNING,
            executed_by=user_id
        )
        
        try:
            # Process documents based on evaluation case type using API calls
            if evaluation_case.evaluation_type == EvaluationType.DOCUMENT_PROCESSING:
                actual_outputs = await self._evaluate_document_processing(
                    evaluation_case, rag_approach, user_id
                )
            elif evaluation_case.evaluation_type == EvaluationType.QUERY_ANSWERING:
                actual_outputs = await self._evaluate_query_answering(
                    evaluation_case, rag_approach, user_id
                )
            elif evaluation_case.evaluation_type == EvaluationType.CROSS_DOCUMENT_ANALYSIS:
                actual_outputs = await self._evaluate_cross_document_analysis(
                    evaluation_case, rag_approach, user_id
                )
            else:
                raise ValueError(f"Unsupported evaluation type: {evaluation_case.evaluation_type}")
            
            result.actual_outputs = actual_outputs
            
            # Calculate metrics
            metrics = await self._calculate_metrics(evaluation_case, actual_outputs)
            result.metrics = metrics
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(metrics, evaluation_case.evaluation_criteria)
            
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result.processing_time = end_time - start_time
            result.memory_usage = end_memory - start_memory
            result.status = EvaluationStatus.COMPLETED
            
            logger.info(f"Completed evaluation case {evaluation_case.id} with score: {result.overall_score}")
            
        except Exception as e:
            logger.error(f"Error in evaluation case {evaluation_case.id}: {e}")
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
        
        return result
    
    async def _evaluate_document_processing(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate document processing capabilities using API calls"""
        
        # Process each document using API calls
        processing_results = []
        for doc_path in evaluation_case.document_paths:
            # Process the document via API
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=doc_path,
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
            
            processing_results.append({
                'document_path': doc_path,
                'result': result
            })
        
        return {
            'processing_results': processing_results,
            'total_documents': len(evaluation_case.document_paths),
            'successful_documents': len([r for r in processing_results if r['result'].get('success', False)])
        }
    
    async def _evaluate_query_answering(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate query answering capabilities using API calls"""
        
        if not evaluation_case.query:
            raise ValueError("Query is required for query answering evaluation")
        
        # First process documents via API
        processing_results = []
        for doc_path in evaluation_case.document_paths:
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=doc_path,
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
            processing_results.append(result)
        
        # Query the processed documents via API
        query_result = await self._query_documents_via_api(
            user_id=user_id,
            project_id=evaluation_case.project_id,
            query=evaluation_case.query,
            rag_approach=rag_approach,
            model=evaluation_case.model,
            config=evaluation_case.config
        )
        
        return {
            'query': evaluation_case.query,
            'processing_results': processing_results,
            'answer': query_result.get('answer', ''),
            'sources': query_result.get('sources', []),
            'metadata': query_result.get('metadata', {})
        }
    
    async def _evaluate_cross_document_analysis(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Evaluate cross-document analysis capabilities"""
        
        # Process all documents using API calls
        if len(evaluation_case.document_paths) > 1:
            # Use folder processing for multiple documents via API
            result = await self._process_folder_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_folder_path=os.path.dirname(evaluation_case.document_paths[0]) + "/",
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
        else:
            # Single document processing via API
            result = await self._process_document_via_api(
                user_id=user_id,
                project_id=evaluation_case.project_id,
                workspace_id=evaluation_case.project_id,
                gcs_path=evaluation_case.document_paths[0],
                rag_approach=rag_approach,
                parser=evaluation_case.parser,
                model=evaluation_case.model,
                config=evaluation_case.config
            )
        
        return {
            'cross_document_analysis': result,
            'document_count': len(evaluation_case.document_paths)
        }
    
    async def _calculate_metrics(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> List[MetricResult]:
        """Calculate metrics for an evaluation case"""
        
        metrics = []
        
        for metric_type in evaluation_case.evaluation_criteria.metrics:
            try:
                if metric_type == MetricType.PROCESSING_TIME:
                    # This is calculated in the evaluation case execution
                    continue
                elif metric_type == MetricType.MEMORY_USAGE:
                    # This is calculated in the evaluation case execution
                    continue
                elif metric_type == MetricType.SEMANTIC_SIMILARITY:
                    value = await self._calculate_semantic_similarity(evaluation_case, actual_outputs)
                elif metric_type == MetricType.COMPLETENESS:
                    value = await self._calculate_completeness(evaluation_case, actual_outputs)
                elif metric_type == MetricType.RELEVANCE:
                    value = await self._calculate_relevance(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_COMPLETENESS:
                    value = await self._calculate_llm_completeness(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_ACCURACY:
                    value = await self._calculate_llm_accuracy(evaluation_case, actual_outputs)
                elif metric_type == MetricType.LLM_COHERENCE:
                    value = await self._calculate_llm_coherence(evaluation_case, actual_outputs)
                else:
                    # Default to 0 for unimplemented metrics
                    value = 0.0
                
                threshold = evaluation_case.evaluation_criteria.thresholds.get(metric_type)
                passed = value >= threshold if threshold is not None else None
                
                metrics.append(MetricResult(
                    metric_type=metric_type,
                    value=value,
                    threshold=threshold,
                    passed=passed
                ))
                
            except Exception as e:
                logger.warning(f"Error calculating metric {metric_type}: {e}")
                metrics.append(MetricResult(
                    metric_type=metric_type,
                    value=0.0,
                    threshold=evaluation_case.evaluation_criteria.thresholds.get(metric_type),
                    passed=False
                ))
        
        return metrics
    
    async def _calculate_semantic_similarity(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between expected and actual outputs"""
        
        if not self.sentence_transformer:
            return 0.0
        
        try:
            # Extract text from expected outputs
            expected_texts = []
            for expected in evaluation_case.expected_outputs:
                if isinstance(expected.content, str):
                    expected_texts.append(expected.content)
                elif isinstance(expected.content, list):
                    expected_texts.extend([str(item) for item in expected.content])
            
            # Extract text from actual outputs
            actual_texts = []
            if 'processing_results' in actual_outputs:
                for result in actual_outputs['processing_results']:
                    if 'result' in result and 'rag_result' in result['result']:
                        rag_result = result['result']['rag_result']
                        if 'synthesis' in rag_result:
                            actual_texts.append(str(rag_result['synthesis']))
            
            if not expected_texts or not actual_texts:
                return 0.0
            
            # Calculate embeddings
            expected_embeddings = self.sentence_transformer.encode(expected_texts)
            actual_embeddings = self.sentence_transformer.encode(actual_texts)
            
            # Calculate similarity
            similarities = cosine_similarity(expected_embeddings, actual_embeddings)
            return float(np.max(similarities))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _calculate_completeness(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate completeness of actual outputs compared to expected"""
        
        try:
            expected_count = len(evaluation_case.expected_outputs)
            if expected_count == 0:
                return 1.0
            
            # Count how many expected outputs have corresponding actual outputs
            found_count = 0
            
            for expected in evaluation_case.expected_outputs:
                if self._find_expected_in_actual(expected, actual_outputs):
                    found_count += 1
            
            return found_count / expected_count
            
        except Exception as e:
            logger.warning(f"Error calculating completeness: {e}")
            return 0.0
    
    async def _calculate_relevance(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate relevance of actual outputs to the evaluation case"""
        
        try:
            if not evaluation_case.query:
                return 1.0  # No query to compare against
            
            # Use LLM to evaluate relevance
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the relevance of the actual output to the given query on a scale of 0-1, where 1 is highly relevant. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {evaluation_case.query}\n\nActual Output: {actual_text[:1000]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            relevance_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, relevance_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.0
    
    async def _calculate_llm_completeness(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate completeness using LLM-as-a-Judge for unstructured content"""
        
        try:
            if not evaluation_case.expected_outputs:
                return 1.0
            
            # Prepare expected outputs for LLM evaluation
            expected_descriptions = []
            for expected in evaluation_case.expected_outputs:
                expected_descriptions.append(f"- {expected.description or expected.type}: {expected.content}")
            
            expected_text = "\n".join(expected_descriptions)
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate how completely the actual output covers the expected information on a scale of 0-1, where 1 means all expected information is present. Consider that the actual output may express the same information in different words. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Expected Information:\n{expected_text}\n\nActual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            completeness_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, completeness_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM completeness: {e}")
            return 0.0
    
    async def _calculate_llm_accuracy(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate accuracy using LLM-as-a-Judge for unstructured content"""
        
        try:
            if not evaluation_case.expected_outputs:
                return 1.0
            
            # Prepare expected outputs for LLM evaluation
            expected_descriptions = []
            for expected in evaluation_case.expected_outputs:
                expected_descriptions.append(f"- {expected.description or expected.type}: {expected.content}")
            
            expected_text = "\n".join(expected_descriptions)
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the accuracy of the actual output compared to the expected information on a scale of 0-1, where 1 means the information is completely accurate. Consider that the actual output may express the same information in different words. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Expected Information:\n{expected_text}\n\nActual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            accuracy_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, accuracy_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM accuracy: {e}")
            return 0.0
    
    async def _calculate_llm_coherence(
        self,
        evaluation_case: EvaluationCase,
        actual_outputs: Dict[str, Any]
    ) -> float:
        """Calculate coherence using LLM-as-a-Judge"""
        
        try:
            actual_text = str(actual_outputs)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Rate the coherence and logical flow of the actual output on a scale of 0-1, where 1 means the output is well-structured, logical, and easy to follow. Respond with only a number."
                    },
                    {
                        "role": "user",
                        "content": f"Actual Output: {actual_text[:1500]}..."
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            coherence_score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, coherence_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating LLM coherence: {e}")
            return 0.0
    
    def _find_expected_in_actual(
        self,
        expected: Any,
        actual_outputs: Dict[str, Any]
    ) -> bool:
        """Check if expected output is found in actual outputs"""
        
        try:
            expected_str = str(expected.content).lower()
            actual_str = str(actual_outputs).lower()
            
            # Simple substring matching for now
            return expected_str in actual_str
            
        except Exception:
            return False
    
    def _calculate_overall_score(
        self,
        metrics: List[MetricResult],
        criteria
    ) -> float:
        """Calculate overall weighted score from individual metrics"""
        
        if not metrics:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in metrics:
            weight = criteria.weights.get(metric.metric_type, 1.0)
            weighted_sum += metric.value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _calculate_summary_statistics(self, evaluation_run: EvaluationRun):
        """Calculate summary statistics for the evaluation run"""
        
        # Group results by RAG approach
        approach_results = {}
        for result in evaluation_run.evaluation_case_results:
            if result.rag_approach not in approach_results:
                approach_results[result.rag_approach] = []
            approach_results[result.rag_approach].append(result)
        
        # Calculate average scores by approach
        for approach, results in approach_results.items():
            scores = [r.overall_score for r in results if r.overall_score is not None]
            if scores:
                evaluation_run.average_scores[approach] = sum(scores) / len(scores)
    
    async def _update_evaluation_run(self, evaluation_run: EvaluationRun):
        """Update evaluation run in storage"""
        
        try:
            # Store in Firebase or local storage
            # For now, we'll just log the update
            logger.info(f"Updated evaluation run {evaluation_run.id}: {evaluation_run.status}")
            
        except Exception as e:
            logger.error(f"Error updating evaluation run: {e}")
    
    async def get_evaluation_run(self, evaluation_run_id: str) -> Optional[EvaluationRun]:
        """Get an evaluation run by ID"""
        
        # TODO: Implement retrieval from storage
        return None
    
    async def get_evaluation_suite(self, evaluation_suite_id: str) -> Optional[EvaluationSuite]:
        """Get an evaluation suite by ID"""
        
        # TODO: Implement retrieval from storage
        return None
    
    async def run_evaluation_with_approach(
        self,
        evaluation_suite: EvaluationSuite,
        rag_approach: str,
        user_id: str,
        project_id: str
    ) -> List[EvaluationCaseResult]:
        """Run an evaluation suite with a specific RAG approach"""
        
        logger.info(f"🚀 Starting evaluation suite: {evaluation_suite.name}")
        logger.info(f"📋 RAG Approach: {rag_approach}")
        logger.info(f"📊 Total cases: {len(evaluation_suite.evaluation_cases)}")
        logger.info(f"👤 User ID: {user_id}")
        logger.info(f"📁 Project ID: {project_id}")
        print()
        
        results = []
        for i, evaluation_case in enumerate(evaluation_suite.evaluation_cases, 1):
            logger.info(f"[{i}/{len(evaluation_suite.evaluation_cases)}] 🔄 Running: {evaluation_case.name}")
            logger.info(f"📝 Prompt: {evaluation_case.prompt[:100]}...")
            logger.info(f"🎯 Expected: {evaluation_case.expected_output}")
            
            try:
                result = await self._run_simplified_evaluation_case(evaluation_case, rag_approach, user_id)
                results.append(result)
                
                if result.status == EvaluationStatus.COMPLETED:
                    logger.info(f"✅ Completed {evaluation_case.id}: Score {result.overall_score:.3f}")
                    logger.info(f"🤖 Response: {result.actual_response[:200]}...")
                else:
                    logger.error(f"❌ Failed {evaluation_case.id}: {result.error_message}")
                
                print()
                
            except Exception as e:
                logger.error(f"💥 Error running {evaluation_case.id}: {e}")
                # Create failed result
                failed_result = EvaluationCaseResult(
                    evaluation_case_id=evaluation_case.id,
                    rag_approach=rag_approach,
                    prompt=evaluation_case.prompt,
                    expected_output=evaluation_case.expected_output,
                    actual_response="",
                    status=EvaluationStatus.FAILED,
                    error_message=str(e),
                    executed_by=user_id
                )
                results.append(failed_result)
                print()
        
        logger.info(f"🎉 Evaluation suite completed! Processed {len(results)} cases")
        return results
    
    async def _run_simplified_evaluation_case(
        self,
        evaluation_case: EvaluationCase,
        rag_approach: str,
        user_id: str
    ) -> EvaluationCaseResult:
        """Run a simplified evaluation case with just prompt and expected output"""
        
        logger.info(f"🔍 Processing evaluation case: {evaluation_case.id}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = EvaluationCaseResult(
            evaluation_case_id=evaluation_case.id,
            rag_approach=rag_approach,
            prompt=evaluation_case.prompt,
            expected_output=evaluation_case.expected_output,
            actual_response="",
            status=EvaluationStatus.RUNNING,
            executed_by=user_id
        )
        
        try:
            # Get the actual project ID and corpus info from the registry
            from .project_registry import project_registry
            actual_project_id = project_registry.get_actual_project_id(evaluation_case.project_id, rag_approach)
            if not actual_project_id:
                logger.error(f"❌ No project found for base ID '{evaluation_case.project_id}' with approach '{rag_approach}'")
                raise Exception(f"No project found for base ID '{evaluation_case.project_id}' with approach '{rag_approach}'")
            
            logger.info(f"🔍 Using actual project ID: {actual_project_id}")
            
            # Get corpus info and model for rag_vertex approach
            corpus_info = None
            model = evaluation_case.model
            
            if rag_approach == "rag_vertex":
                corpus_info = project_registry.get_project_corpus_info(actual_project_id)
                if corpus_info:
                    logger.info(f"📋 Found corpus info: {corpus_info.get('corpus_name', 'N/A')}")
                else:
                    logger.warning(f"⚠️ No corpus info found for {actual_project_id}")
                
                # Use the model from the project registry for rag_vertex
                project = project_registry.get_project(actual_project_id)
                if project and project.get("model"):
                    model = project["model"]
                    logger.info(f"🔧 Using model from registry: {model}")
            
            # Query the RAG system directly
            logger.info(f"🌐 Making API call to RAG system...")
            actual_response = await self._query_documents_via_api(
                user_id=user_id,
                project_id=actual_project_id,
                query=evaluation_case.prompt,
                rag_approach=rag_approach,
                model=model,
                corpus_info=corpus_info
            )
            
            # Extract the response text
            response_text = actual_response.get("answer", "")
            result.actual_response = response_text
            
            if not response_text or response_text.strip() == "":
                logger.warning(f"⚠️ Empty response received from RAG system")
                logger.warning(f"🔍 Full API response: {actual_response}")
            else:
                logger.info(f"📥 Received response ({len(response_text)} chars)")
            
            # Calculate metrics using LLM
            logger.info(f"🧠 Calculating metrics with LLM...")
            metrics = await self._calculate_simplified_metrics(
                evaluation_case.prompt,
                response_text,
                evaluation_case.expected_output
            )
            result.metrics = metrics
            
            # Calculate overall score
            result.overall_score = self._calculate_simplified_overall_score(metrics)
            
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result.processing_time = end_time - start_time
            result.memory_usage = end_memory - start_memory
            result.status = EvaluationStatus.COMPLETED
            
            logger.info(f"✅ Evaluation case completed in {result.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"💥 Error in evaluation case {evaluation_case.id}: {e}")
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
        
        return result
    
    async def _calculate_simplified_metrics(
        self,
        prompt: str,
        response: str,
        expected_output: str
    ) -> List[MetricResult]:
        """Calculate metrics for simplified evaluation using LLM with separate calls for each metric"""
        
        logger.info(f"🔍 Calculating relevance, completeness, and coherence with separate LLM calls...")
        
        # Check for empty or invalid responses
        if not response or response.strip() == "":
            logger.warning(f"⚠️ Empty response detected, assigning zero scores")
            return [
                MetricResult(
                    metric_type=MetricType.RELEVANCE, 
                    value=0.0,
                    details={"explanation": "Empty response - no content to evaluate"}
                ),
                MetricResult(
                    metric_type=MetricType.COMPLETENESS, 
                    value=0.0,
                    details={"explanation": "Empty response - no content to evaluate"}
                ),
                MetricResult(
                    metric_type=MetricType.COHERENCE, 
                    value=0.0,
                    details={"explanation": "Empty response - no content to evaluate"}
                )
            ]
        
        metrics = []
        
        # Evaluate RELEVANCE
        try:
            logger.info(f"🧠 Evaluating relevance...")
            relevance_result = await self._evaluate_relevance(prompt, response, expected_output)
            metrics.append(relevance_result)
        except Exception as e:
            logger.warning(f"⚠️ Relevance evaluation failed: {e}")
            metrics.append(MetricResult(
                metric_type=MetricType.RELEVANCE, 
                value=0.5,
                details={"explanation": f"Evaluation failed: {str(e)}"}
            ))
        
        # Evaluate COMPLETENESS
        try:
            logger.info(f"🧠 Evaluating completeness...")
            completeness_result = await self._evaluate_completeness(prompt, response, expected_output)
            metrics.append(completeness_result)
        except Exception as e:
            logger.warning(f"⚠️ Completeness evaluation failed: {e}")
            metrics.append(MetricResult(
                metric_type=MetricType.COMPLETENESS, 
                value=0.5,
                details={"explanation": f"Evaluation failed: {str(e)}"}
            ))
        
        # Evaluate COHERENCE
        try:
            logger.info(f"🧠 Evaluating coherence...")
            coherence_result = await self._evaluate_coherence(prompt, response, expected_output)
            metrics.append(coherence_result)
        except Exception as e:
            logger.warning(f"⚠️ Coherence evaluation failed: {e}")
            metrics.append(MetricResult(
                metric_type=MetricType.COHERENCE, 
                value=0.5,
                details={"explanation": f"Evaluation failed: {str(e)}"}
            ))
        
        logger.info(f"📊 Metrics calculated: Relevance={metrics[0].value:.3f}, Completeness={metrics[1].value:.3f}, Coherence={metrics[2].value:.3f}")
        return metrics
    
    async def _evaluate_relevance(self, prompt: str, response: str, expected_output: str) -> MetricResult:
        """Evaluate how well the response addresses the prompt"""
        eval_prompt = f"""
You are evaluating a RAG system response for RELEVANCE.

PROMPT: {prompt}

RESPONSE: {response}

EXPECTED OUTPUT: {expected_output}

IMPORTANT: If the response is a refusal to help, an error message, or indicates it cannot assist (like "I can't assist", "I'm sorry but I can't", etc.), it should receive a very low relevance score (0.0-0.2) because it does not address the actual prompt.

Rate the RELEVANCE of the response (0.0 to 1.0):
- 1.0: Perfectly addresses the prompt, directly answers what was asked
- 0.8-0.9: Mostly relevant, addresses most aspects of the prompt
- 0.6-0.7: Somewhat relevant, addresses some aspects but misses others
- 0.4-0.5: Partially relevant, addresses some parts but largely off-topic
- 0.0-0.3: Not relevant, doesn't address the prompt, is a refusal, or is completely off-topic

Respond in JSON format:
{{"score": <score>, "explanation": "<explanation>"}}
"""
        
        llm_response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )
        
        content = llm_response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        import json
        data = json.loads(content)
        
        return MetricResult(
            metric_type=MetricType.RELEVANCE,
            value=float(data.get("score", 0.5)),
            details={"explanation": data.get("explanation", "No explanation provided")}
        )
    
    async def _evaluate_completeness(self, prompt: str, response: str, expected_output: str) -> MetricResult:
        """Evaluate how complete the response is compared to expected output"""
        eval_prompt = f"""
You are evaluating a RAG system response for COMPLETENESS.

PROMPT: {prompt}

RESPONSE: {response}

EXPECTED OUTPUT: {expected_output}

IMPORTANT: If the response is a refusal to help, an error message, or indicates it cannot assist (like "I can't assist", "I'm sorry but I can't", etc.), it should receive a very low completeness score (0.0-0.2) because it provides none of the expected content.

Rate the COMPLETENESS of the response (0.0 to 1.0):
- 1.0: Contains all expected elements and information
- 0.8-0.9: Contains most expected elements, minor gaps
- 0.6-0.7: Contains some expected elements, moderate gaps
- 0.4-0.5: Contains few expected elements, major gaps
- 0.0-0.3: Contains almost no expected elements, is incomplete, or is a refusal to help

Respond in JSON format:
{{"score": <score>, "explanation": "<explanation>"}}
"""
        
        llm_response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )
        
        content = llm_response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        import json
        data = json.loads(content)
        
        return MetricResult(
            metric_type=MetricType.COMPLETENESS,
            value=float(data.get("score", 0.5)),
            details={"explanation": data.get("explanation", "No explanation provided")}
        )
    
    async def _evaluate_coherence(self, prompt: str, response: str, expected_output: str) -> MetricResult:
        """Evaluate how well-structured and coherent the response is"""
        eval_prompt = f"""
You are evaluating a RAG system response for COHERENCE.

PROMPT: {prompt}

RESPONSE: {response}

EXPECTED OUTPUT: {expected_output}

IMPORTANT: If the response is a refusal to help, an error message, or indicates it cannot assist (like "I can't assist", "I'm sorry but I can't", etc.), it should receive a low coherence score (0.0-0.3) because it doesn't provide the expected structured content.

Rate the COHERENCE of the response (0.0 to 1.0):
- 1.0: Well-structured, logical flow, clear and coherent
- 0.8-0.9: Mostly coherent with minor structural issues
- 0.6-0.7: Somewhat coherent but has noticeable structural problems
- 0.4-0.5: Partially coherent but difficult to follow
- 0.0-0.3: Incoherent, poorly structured, difficult to understand, or is a refusal to help

Respond in JSON format:
{{"score": <score>, "explanation": "<explanation>"}}
"""
        
        llm_response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )
        
        content = llm_response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        import json
        data = json.loads(content)
        
        return MetricResult(
            metric_type=MetricType.COHERENCE,
            value=float(data.get("score", 0.5)),
            details={"explanation": data.get("explanation", "No explanation provided")}
        )
    
    def _calculate_simplified_overall_score(self, metrics: List[MetricResult]) -> float:
        """Calculate overall score from simplified metrics"""
        if not metrics:
            return 0.0
        
        # Equal weights for the three metrics
        weights = {
            MetricType.RELEVANCE: 0.4,
            MetricType.COMPLETENESS: 0.3,
            MetricType.COHERENCE: 0.3
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.metric_type, 0.0)
            weighted_sum += metric.value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _process_document_via_api(
        self,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        rag_approach: str,
        parser: str = "mineru",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "workspace_id": workspace_id,
                "gcs_path": gcs_path,
                "rag_approach": rag_approach,
                "parser": parser,
                "model": model,
                "config": config or {}
            }
            
            response = await client.post(
                f"{self.api_base_url}/process-document",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Wait for processing to complete
            job_id = result["job_id"]
            return await self._wait_for_job_completion(client, job_id, user_id)
    
    async def _process_folder_via_api(
        self,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        rag_approach: str,
        parser: str = "mineru",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a folder via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "workspace_id": workspace_id,
                "gcs_folder_path": gcs_folder_path,
                "rag_approach": rag_approach,
                "parser": parser,
                "model": model,
                "config": config or {}
            }
            
            response = await client.post(
                f"{self.api_base_url}/process-folder",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Wait for processing to complete
            job_id = result["job_id"]
            return await self._wait_for_job_completion(client, job_id, user_id)
    
    async def _query_documents_via_api(
        self,
        user_id: str,
        project_id: str,
        query: str,
        rag_approach: str,
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None,
        corpus_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents via API call"""
        
        async with httpx.AsyncClient(timeout=self.api_timeout) as client:
            payload = {
                "user_id": user_id,
                "project_id": project_id,
                "query": query,
                "rag_approach": rag_approach,
                "model": model,
                "config": config or {}
            }
            
            # Add corpus_info for rag_vertex approach
            if corpus_info:
                payload["corpus_info"] = corpus_info
            
            response = await client.post(
                f"{self.api_base_url}/query",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            return response.json()
    
    async def _wait_for_job_completion(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        user_id: str,
        max_wait_time: int = 300
    ) -> Dict[str, Any]:
        """Wait for a processing job to complete"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"{self.api_base_url}/processing-status/{job_id}",
                params={"user_id": user_id}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get job status: {response.status_code} - {response.text}")
            
            status_data = response.json()
            status = status_data["status"]
            
            if status == "completed":
                return status_data.get("result", {})
            elif status == "failed":
                error_msg = status_data.get("error_message", "Unknown error")
                raise Exception(f"Job failed: {error_msg}")
            
            # Wait before checking again
            await asyncio.sleep(5)
        
        raise Exception(f"Job {job_id} did not complete within {max_wait_time} seconds")
