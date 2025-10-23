#!/usr/bin/env python3
"""
Command-line interface for RAG evaluation framework
"""

import asyncio
import json
import sys
import argparse
from typing import Dict, Any, List
import httpx
from pathlib import Path
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from evals.results_manager import EvaluationResultsManager
from evals import get_all_evaluation_suites
from evals.project_registry import project_registry


class RAGEvaluationCLI:
    """CLI for interacting with the RAG evaluation framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.results_manager = EvaluationResultsManager()
    
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
            
            # Generate evaluation run ID
            import uuid
            evaluation_run_id = str(uuid.uuid4())
            
            print(f"Starting evaluation: {name}")
            print(f"Evaluation Run ID: {evaluation_run_id}")
            print(f"Suite: {suite.name}")
            print(f"RAG Approaches: {', '.join(rag_approaches)}")
            print(f"User ID: {user_id}")
            print(f"Project ID: {project_id}")
            print()
            
            # For now, we'll create a simple evaluation runner
            # This is a placeholder - in a full implementation, you'd run the actual evaluation
            print("⚠️  Note: This is a simplified evaluation runner.")
            print("   In a full implementation, this would:")
            print("   1. Process documents using each RAG approach")
            print("   2. Run queries and collect results")
            print("   3. Calculate metrics and scores")
            print("   4. Save results locally")
            print()
            print("   For now, creating a placeholder evaluation run...")
            
            # Create a placeholder result
            placeholder_result = {
                "evaluation_run": {
                    "id": evaluation_run_id,
                    "name": name,
                    "description": description,
                    "evaluation_suite_id": evaluation_suite_id,
                    "rag_approaches": rag_approaches,
                    "user_id": user_id,
                    "project_id": project_id,
                    "status": "completed",
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "total_evaluation_cases": len(suite.evaluation_cases) * len(rag_approaches),
                    "completed_evaluation_cases": len(suite.evaluation_cases) * len(rag_approaches),
                    "failed_evaluation_cases": 0
                },
                "detailed_results": [],
                "comparison_summary": {
                    "best_approach": rag_approaches[0] if rag_approaches else "none",
                    "success_rate": 1.0,
                    "average_scores": {approach: 0.85 for approach in rag_approaches}
                },
                "recommendations": [
                    "This is a placeholder evaluation result",
                    "Implement full evaluation logic to get real results"
                ]
            }
            
            # Save the placeholder result
            results_path = self.results_manager.save_evaluation_run(evaluation_run_id, placeholder_result)
            print(f"✅ Placeholder evaluation completed!")
            print(f"📁 Results saved to: {results_path}")
            
            return evaluation_run_id
            
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
