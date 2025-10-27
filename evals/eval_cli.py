#!/usr/bin/env python3
"""
Command-line interface for RAG evaluation framework
"""

import asyncio
import json
import sys
import argparse
import logging
from typing import Dict, Any, List
import httpx
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from evals.results_manager import EvaluationResultsManager
from evals import get_all_evaluation_suites
from evals.project_registry import project_registry
from evals.eval_service import EvaluationService
from evals.suites.NbbabGQy3gkCJIDzkSoE_suite import get_project_suites


class RAGEvaluationCLI:
    """CLI for interacting with the RAG evaluation framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.results_manager = EvaluationResultsManager()
        # Initialize evaluation service with proper API base URL
        self.eval_service = EvaluationService(None, None, None)
        self.eval_service.api_base_url = base_url
    
    async def list_evaluation_suites(self):
        """List all available evaluation suites"""
        try:
            # Get local evaluation suites
            suites = get_all_evaluation_suites()
            
            print("Available Evaluation Suites:")
            print("=" * 50)
            for suite_id, suite in suites.items():
                print(f"ID: {suite.id}")
                print(f"Name: {suite.name}")
                print(f"Description: {suite.description}")
                print(f"Evaluation Cases: {len(suite.evaluation_cases)}")
                print(f"Tags: {', '.join(suite.tags)}")
                print(f"Default RAG Approaches: {', '.join(suite.default_rag_approaches)}")
                print("-" * 30)
            
        except Exception as e:
            print(f"Error loading evaluation suites: {e}")
    
    def list_projects(self):
        """List all available RAG projects from the registry"""
        try:
            projects = project_registry.list_available_projects()
            
            print("Available RAG Projects:")
            print("=" * 50)
            for project_id, project in projects.items():
                print(f"ID: {project_id}")
                print(f"Name: {project.get('name', 'N/A')}")
                print(f"RAG Approach: {project.get('rag_approach', 'N/A')}")
                print(f"User ID: {project.get('user_id', 'N/A')}")
                print(f"Project ID: {project.get('project_id', 'N/A')}")
                print(f"Status: {project.get('status', 'N/A')}")
                print(f"Document Count: {project.get('document_count', 0)}")
                print("-" * 30)
            
        except Exception as e:
            print(f"Error loading projects: {e}")
    
    async def get_evaluation_suite(self, evaluation_suite_id: str):
        """Get detailed information about an evaluation suite"""
        try:
            # Get local evaluation suites
            suites = get_all_evaluation_suites()
            
            if evaluation_suite_id not in suites:
                print(f"Evaluation suite '{evaluation_suite_id}' not found")
                return
            
            suite = suites[evaluation_suite_id]
            
            print(f"Evaluation Suite: {suite.name}")
            print("=" * 50)
            print(f"Description: {suite.description}")
            print(f"Evaluation Cases: {len(suite.evaluation_cases)}")
            print(f"Tags: {', '.join(suite.tags)}")
            print()
            
            print("Evaluation Cases:")
            for i, evaluation_case in enumerate(suite.evaluation_cases, 1):
                print(f"{i}. {evaluation_case.name}")
                print(f"   ID: {evaluation_case.id}")
                print(f"   Type: {evaluation_case.evaluation_type}")
                print(f"   Difficulty: {evaluation_case.difficulty}")
                print(f"   Documents: {len(evaluation_case.document_paths)}")
                if evaluation_case.query:
                    print(f"   Query: {evaluation_case.query}")
                print(f"   Expected Outputs: {len(evaluation_case.expected_outputs)}")
                print()
            
        except Exception as e:
            print(f"Error loading evaluation suite: {e}")
    
    async def start_evaluation(
        self,
        evaluation_suite_id: str,
        rag_approaches: list,
        user_id: str,
        project_id: str,
        name: str,
        description: str = None
    ):
        """Start a new evaluation run"""
        try:
            # Get the evaluation suite
            suites = get_all_evaluation_suites()
            if evaluation_suite_id not in suites:
                print(f"Evaluation suite '{evaluation_suite_id}' not found")
                return None
            
            suite = suites[evaluation_suite_id]
            
            # Run the actual evaluation
            evaluation_result = await self.eval_runner.run_evaluation(
                suite, rag_approaches, user_id, project_id, name, description
            )
            
            # Save the results
            results_path = self.results_manager.save_evaluation_run(evaluation_result["evaluation_run"]["id"], evaluation_result)
            print(f"📁 Results saved to: {results_path}")
            
            return evaluation_result["evaluation_run"]["id"]
            
        except Exception as e:
            print(f"Error starting evaluation: {e}")
            return None
    
    async def get_evaluation_status(self, evaluation_run_id: str, user_id: str):
        """Get the status of an evaluation run"""
        try:
            response = await self.client.get(
                f"{self.base_url}/evaluations/{evaluation_run_id}/status",
                params={"user_id": user_id}
            )
            response.raise_for_status()
            status = response.json()
            
            print(f"Evaluation Status: {status['status']}")
            print(f"Progress: {status['progress']['progress_percentage']:.1f}%")
            print(f"Completed: {status['progress']['completed_evaluation_cases']}/{status['progress']['total_evaluation_cases']}")
            print(f"Failed: {status['progress']['failed_evaluation_cases']}")
            
            if status.get('results_summary'):
                print("\nResults Summary:")
                for approach, score in status['results_summary']['average_scores'].items():
                    print(f"  {approach}: {score:.3f}")
            
            if status.get('error_message'):
                print(f"Error: {status['error_message']}")
            
            return status['status']
            
        except httpx.HTTPError as e:
            print(f"Error fetching evaluation status: {e}")
            return None
    
    async def get_evaluation_results(self, evaluation_run_id: str, user_id: str, save_locally: bool = True):
        """Get detailed results of a completed evaluation"""
        try:
            # Load results from local storage
            results = self.results_manager.load_evaluation_run(evaluation_run_id)
            
            if not results:
                print(f"Evaluation results not found for ID: {evaluation_run_id}")
                return
            
            print("Evaluation Results")
            print("=" * 50)
            
            # Summary
            summary = results.get('comparison_summary', {})
            print(f"Best Approach: {summary.get('best_approach', 'N/A')}")
            print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
            print()
            
            print("Average Scores by Approach:")
            for approach, score in summary.get('average_scores', {}).items():
                print(f"  {approach}: {score:.3f}")
            print()
            
            print("Recommendations:")
            for rec in results.get('recommendations', []):
                print(f"  • {rec}")
            print()
            
            # Detailed results
            print("Detailed Results:")
            for result in results.get('detailed_results', []):
                print(f"Evaluation Case: {result.get('evaluation_case_id', 'N/A')}")
                print(f"RAG Approach: {result.get('rag_approach', 'N/A')}")
                print(f"Status: {result.get('status', 'N/A')}")
                print(f"Overall Score: {result.get('overall_score', 0):.3f}" if result.get('overall_score') else "Overall Score: N/A")
                print(f"Processing Time: {result.get('processing_time', 0):.2f}s" if result.get('processing_time') else "Processing Time: N/A")
                
                if result.get('metrics'):
                    print("Metrics:")
                    for metric in result['metrics']:
                        print(f"  {metric.get('metric_type', 'N/A')}: {metric.get('value', 0):.3f}")
                print("-" * 30)
            
            print(f"\n📁 Results loaded from local storage")
            
        except Exception as e:
            print(f"Error loading evaluation results: {e}")
    
    async def monitor_evaluation(self, evaluation_run_id: str, user_id: str, poll_interval: int = 10):
        """Monitor an evaluation run until completion"""
        print(f"Monitoring evaluation {evaluation_run_id}...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                status = await self.get_evaluation_status(evaluation_run_id, user_id)
                
                if status in ['completed', 'failed']:
                    print(f"\nEvaluation {status}!")
                    if status == 'completed':
                        await self.get_evaluation_results(evaluation_run_id, user_id)
                    break
                
                print(f"Waiting {poll_interval} seconds...")
                await asyncio.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
    
    def list_local_results(self, date: str = None):
        """List locally saved evaluation results"""
        runs = self.results_manager.list_evaluation_runs(date)
        
        if not runs:
            print("No evaluation results found locally")
            return
        
        print("Local Evaluation Results:")
        print("=" * 50)
        
        for run in runs:
            print(f"ID: {run['evaluation_run_id']}")
            print(f"Name: {run['name']}")
            print(f"Status: {run['status']}")
            print(f"Completed: {run['completed_at']}")
            if run.get('best_approach'):
                print(f"Best Approach: {run['best_approach']}")
            print("-" * 30)
    
    def compare_results(self, evaluation_run_ids: List[str]):
        """Compare multiple evaluation runs"""
        report_path = self.results_manager.create_comparison_report(evaluation_run_ids)
        print(f"Comparison report created: {report_path}")
        
        # Also display the report
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    # Project evaluation methods
    async def list_project_suites(self):
        """List all project-specific evaluation suites"""
        try:
            # Get project-specific suites
            project_suites = get_project_suites()
            
            print("Project Evaluation Suites:")
            print("=" * 50)
            
            for suite_id, suite in project_suites.items():
                print(f"ID: {suite.id}")
                print(f"Name: {suite.name}")
                print(f"Description: {suite.description}")
                print(f"Evaluation Cases: {len(suite.evaluation_cases)}")
                print(f"Default RAG Approaches: {', '.join(suite.default_rag_approaches)}")
                print(f"Tags: {', '.join(suite.tags)}")
                print("-" * 30)
            
        except Exception as e:
            print(f"Error loading project suites: {e}")
    
    async def run_project_suite(self, suite_id: str, run_type: str):
        """Run a project evaluation suite with specified RAG approach"""
        try:
            project_suites = get_project_suites()
            if suite_id not in project_suites:
                print(f"Project suite '{suite_id}' not found")
                return
            
            suite = project_suites[suite_id]
            
            print(f"🚀 Running suite: {suite.name} ({run_type})")
            print(f"📋 {len(suite.evaluation_cases)} evaluation cases")
            print(f"👤 User ID: {suite.user_id}")
            print()
            
            # Run the evaluation using the user_id from the suite
            results = await self.eval_service.run_evaluation_with_approach(
                suite, run_type, suite.user_id, suite_id
            )
            
            # Print summary
            self._print_evaluation_summary(results, run_type)
            
            # Save results with new naming convention
            self._save_evaluation_results(results, suite_id, run_type)
            
        except Exception as e:
            print(f"Error running project suite: {e}")
    
    def _print_evaluation_summary(self, results, run_type: str):
        """Print evaluation summary with detailed results"""
        if not results:
            return
        
        completed = [r for r in results if r.status.value == "completed"]
        failed = [r for r in results if r.status.value == "failed"]
        
        print("🎉 Evaluation completed!")
        print("=" * 50)
        print(f"RAG Approach: {run_type}")
        print(f"Total Cases: {len(results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        
        if completed:
            scores = [r.overall_score for r in completed if r.overall_score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"Average Score: {avg_score:.3f}")
                
                # Show detailed results
                print("\n" + "=" * 80)
                print("DETAILED RESULTS")
                print("=" * 80)
                
                for result in completed:
                    print(f"\n📋 {result.evaluation_case_id.upper()}")
                    print("-" * 40)
                    print(f"Score: {result.overall_score:.3f}")
                    print(f"Time: {result.processing_time:.2f}s")
                    
                    # Show metrics with explanations
                    if result.metrics:
                        print("Metrics:")
                        for metric in result.metrics:
                            explanation = metric.details.get("explanation", "No explanation") if metric.details else "No explanation"
                            print(f"  {metric.metric_type.value}: {metric.value:.3f}")
                            print(f"    Explanation: {explanation}")
                    
                    print(f"\n📝 Prompt:")
                    print(f"  {result.prompt}")
                    
                    print(f"\n🎯 Expected:")
                    print(f"  {result.expected_output}")
                    
                    print(f"\n🤖 Response:")
                    print(f"  {result.actual_response}")
                    print()
        
        if failed:
            print("\n❌ FAILED CASES:")
            for result in failed:
                print(f"  {result.evaluation_case_id}: {result.error_message}")
    
    def _save_evaluation_results(self, results, project_id: str, approach: str):
        """Save evaluation results with naming convention [dts]_[projectId]_[approach].json"""
        import time
        import json
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evals/results/{timestamp}_{project_id}_{approach}.json"
        
        # Convert results to serializable format
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_id": project_id,
            "approach": approach,
            "results": [
                {
                    "evaluation_case_id": r.evaluation_case_id,
                    "rag_approach": r.rag_approach,
                    "status": r.status.value,
                    "prompt": r.prompt,
                    "expected_output": r.expected_output,
                    "actual_response": r.actual_response,
                    "overall_score": r.overall_score,
                    "processing_time": r.processing_time,
                    "memory_usage": r.memory_usage,
                    "error_message": r.error_message,
                    "metrics": [
                        {
                            "metric_type": m.metric_type.value,
                            "value": m.value,
                            "explanation": m.details.get("explanation", "No explanation provided") if m.details else "No explanation provided"
                        }
                        for m in r.metrics
                    ],
                    "executed_at": r.executed_at.isoformat(),
                    "executed_by": r.executed_by
                }
                for r in results
            ]
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the RAG service")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List evaluation suites
    subparsers.add_parser("list-suites", help="List all available evaluation suites")
    
    # List projects
    subparsers.add_parser("list-projects", help="List all available RAG projects from registry")
    
    # Project evaluation commands
    subparsers.add_parser("list-project-suites", help="List all project-specific evaluation suites")
    
    # Run project suite
    project_suite_parser = subparsers.add_parser("run-project-suite", help="Run a project evaluation suite")
    project_suite_parser.add_argument("suite_id", help="Project suite ID")
    project_suite_parser.add_argument("--run-type", required=True, choices=["raganything", "rag_vertex"], help="RAG approach to use")
    
    # Get evaluation suite details
    suite_parser = subparsers.add_parser("get-suite", help="Get detailed information about an evaluation suite")
    suite_parser.add_argument("suite_id", help="Evaluation suite ID")
    
    # Start evaluation
    eval_parser = subparsers.add_parser("start-eval", help="Start a new evaluation")
    eval_parser.add_argument("suite_id", help="Evaluation suite ID")
    eval_parser.add_argument("--approaches", nargs="+", default=["raganything", "evidence_sweep", "rag_vertex"], help="RAG approaches to evaluate")
    eval_parser.add_argument("--user-id", required=True, help="User ID")
    eval_parser.add_argument("--project-id", required=True, help="Project ID")
    eval_parser.add_argument("--name", required=True, help="Evaluation name")
    eval_parser.add_argument("--description", help="Evaluation description")
    
    # Get evaluation status
    status_parser = subparsers.add_parser("status", help="Get evaluation status")
    status_parser.add_argument("eval_id", help="Evaluation run ID")
    status_parser.add_argument("--user-id", required=True, help="User ID")
    
    # Get evaluation results
    results_parser = subparsers.add_parser("results", help="Get evaluation results")
    results_parser.add_argument("eval_id", help="Evaluation run ID")
    results_parser.add_argument("--user-id", required=True, help="User ID")
    results_parser.add_argument("--no-save", action="store_true", help="Don't save results locally")
    
    # Monitor evaluation
    monitor_parser = subparsers.add_parser("monitor", help="Monitor evaluation until completion")
    monitor_parser.add_argument("eval_id", help="Evaluation run ID")
    monitor_parser.add_argument("--user-id", required=True, help="User ID")
    monitor_parser.add_argument("--poll-interval", type=int, default=10, help="Polling interval in seconds")
    
    # List local results
    local_parser = subparsers.add_parser("list-results", help="List locally saved evaluation results")
    local_parser.add_argument("--date", help="Filter by date (YYYY-MM-DD)")
    
    # Compare results
    compare_parser = subparsers.add_parser("compare", help="Compare multiple evaluation runs")
    compare_parser.add_argument("eval_ids", nargs="+", help="Evaluation run IDs to compare")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = RAGEvaluationCLI(args.base_url)
    
    try:
        if args.command == "list-suites":
            await cli.list_evaluation_suites()
        
        elif args.command == "list-projects":
            cli.list_projects()
        
        elif args.command == "list-project-suites":
            await cli.list_project_suites()
        
        elif args.command == "run-project-suite":
            await cli.run_project_suite(args.suite_id, args.run_type)
        
        elif args.command == "get-suite":
            await cli.get_evaluation_suite(args.suite_id)
        
        elif args.command == "start-eval":
            eval_id = await cli.start_evaluation(
                args.suite_id,
                args.approaches,
                args.user_id,
                args.project_id,
                args.name,
                args.description
            )
            if eval_id:
                print(f"\nTo monitor this evaluation, run:")
                print(f"python evals/evals_cli.py monitor {eval_id} --user-id {args.user_id}")
        
        elif args.command == "status":
            await cli.get_evaluation_status(args.eval_id, args.user_id)
        
        elif args.command == "results":
            await cli.get_evaluation_results(args.eval_id, args.user_id, not args.no_save)
        
        elif args.command == "monitor":
            await cli.monitor_evaluation(args.eval_id, args.user_id, args.poll_interval)
        
        elif args.command == "list-results":
            cli.list_local_results(args.date)
        
        elif args.command == "compare":
            cli.compare_results(args.eval_ids)
    
    finally:
        await cli.close()


if __name__ == "__main__":
    asyncio.run(main())
