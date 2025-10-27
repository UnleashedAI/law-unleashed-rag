"""
Project registry integration for evaluation system
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ProjectRegistry:
    """Manages integration with RAG project registry"""
    
    def __init__(self, registry_file: str = "projects/rag_projects.json"):
        self.registry_file = registry_file
        self.projects = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load RAG project registry from JSON file"""
        try:
            registry_path = Path(self.registry_file)
            if not registry_path.exists():
                # Try relative to current working directory
                registry_path = Path.cwd() / self.registry_file
                if not registry_path.exists():
                    return {"rag_databases": {}}
            
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load project registry: {e}")
            return {"rag_databases": {}}
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project configuration by ID"""
        return self.projects.get("rag_databases", {}).get(project_id)
    
    def get_project_document_paths(self, project_id: str) -> list:
        """Get document paths for a project"""
        project = self.get_project(project_id)
        if not project:
            return []
        
        # For now, return the GCS path as a single document path
        # In a full implementation, you might list all files in the GCS folder
        gcs_path = project.get("gcs_path", "")
        if gcs_path:
            return [gcs_path]
        return []
    
    def get_project_user_id(self, project_id: str) -> Optional[str]:
        """Get user ID for a project"""
        project = self.get_project(project_id)
        return project.get("user_id") if project else None
    
    def get_project_workspace_id(self, project_id: str) -> Optional[str]:
        """Get workspace ID for a project"""
        project = self.get_project(project_id)
        return project.get("workspace_id") if project else None
    
    def get_project_rag_approach(self, project_id: str) -> Optional[str]:
        """Get RAG approach for a project"""
        project = self.get_project(project_id)
        return project.get("rag_approach") if project else None
    
    def get_project_corpus_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get corpus info for a project (for Vertex RAG)"""
        project = self.get_project(project_id)
        return project.get("corpus_info") if project else None
    
    def list_available_projects(self) -> Dict[str, Dict[str, Any]]:
        """List all available projects"""
        return self.projects.get("rag_databases", {})
    
    def get_projects_by_rag_approach(self, rag_approach: str) -> Dict[str, Dict[str, Any]]:
        """Get all projects using a specific RAG approach"""
        projects = {}
        for project_id, project in self.projects.get("rag_databases", {}).items():
            if project.get("rag_approach") == rag_approach:
                projects[project_id] = project
        return projects
    
    def find_project_by_base_id_and_approach(self, base_project_id: str, rag_approach: str) -> Optional[Dict[str, Any]]:
        """Find project by base project ID and RAG approach"""
        for project_id, project in self.projects.get("rag_databases", {}).items():
            if (project.get("project_id") == base_project_id and 
                project.get("rag_approach") == rag_approach):
                return project
        return None
    
    def get_actual_project_id(self, base_project_id: str, rag_approach: str) -> Optional[str]:
        """Get the actual project ID from registry for a base project ID and RAG approach"""
        project = self.find_project_by_base_id_and_approach(base_project_id, rag_approach)
        if project:
            # Find the key that corresponds to this project
            for project_id, proj in self.projects.get("rag_databases", {}).items():
                if proj == project:
                    return project_id
        return None


# Global instance
project_registry = ProjectRegistry()
