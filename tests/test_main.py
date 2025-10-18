"""
Tests for the main FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import os

# Set test environment variables
os.environ["FIREBASE_PROJECT_ID"] = "test-project"
os.environ["GCS_BUCKET_NAME"] = "test-bucket"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.main import app

client = TestClient(app)


@pytest.fixture
def mock_services():
    """Mock all external services"""
    with patch('src.main.firebase_manager') as mock_firebase, \
         patch('src.main.gcs_manager') as mock_gcs, \
         patch('src.main.auth_service') as mock_auth, \
         patch('src.main.rag_service') as mock_rag:
        
        # Configure mocks
        mock_firebase.test_connection = AsyncMock(return_value=True)
        mock_gcs.test_connection = AsyncMock(return_value=True)
        mock_auth.verify_workspace_access = AsyncMock(return_value=True)
        mock_auth.verify_project_access = AsyncMock(return_value=True)
        mock_auth.verify_user_exists = AsyncMock(return_value=True)
        
        yield {
            'firebase': mock_firebase,
            'gcs': mock_gcs,
            'auth': mock_auth,
            'rag': mock_rag
        }


def test_health_check():
    """Test basic health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-anything"


def test_detailed_health_check(mock_services):
    """Test detailed health check endpoint"""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-anything"
    assert "services" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


def test_process_document_success(mock_services):
    """Test successful document processing request"""
    mock_services['firebase'].create_rag_processing_job = AsyncMock(return_value="test-job-id")
    
    request_data = {
        "user_id": "user123",
        "project_id": "project456",
        "workspace_id": "workspace789",
        "gcs_path": "gs://bucket/document.pdf",
        "parser": "mineru",
        "parse_method": "auto",
        "model": "gpt-4"
    }
    
    response = client.post("/process-document", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["message"] == "Document processing job created successfully"


def test_process_document_unauthorized(mock_services):
    """Test document processing with unauthorized access"""
    mock_services['auth'].verify_workspace_access = AsyncMock(return_value=False)
    
    request_data = {
        "user_id": "user123",
        "project_id": "project456",
        "workspace_id": "workspace789",
        "gcs_path": "gs://bucket/document.pdf"
    }
    
    response = client.post("/process-document", json=request_data)
    assert response.status_code == 403
    assert "does not have access" in response.json()["detail"]


def test_process_folder_success(mock_services):
    """Test successful folder processing request"""
    mock_services['firebase'].create_rag_processing_job = AsyncMock(return_value="test-job-id")
    
    request_data = {
        "user_id": "user123",
        "project_id": "project456",
        "workspace_id": "workspace789",
        "gcs_folder_path": "gs://bucket/folder/",
        "file_extensions": [".pdf", ".doc"],
        "parser": "mineru",
        "parse_method": "auto",
        "model": "gpt-4"
    }
    
    response = client.post("/process-folder", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["message"] == "Folder processing job created successfully"


def test_get_processing_status_success(mock_services):
    """Test getting processing status"""
    mock_job_data = {
        "id": "test-job-id",
        "status": "processing",
        "progress": {"overallProgress": 50},
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:01:00Z",
        "userId": "user123"
    }
    
    mock_services['rag'].get_processing_status = AsyncMock(return_value=mock_job_data)
    
    response = client.get("/processing-status/test-job-id?user_id=user123")
    assert response.status_code == 200
    
    data = response.json()
    assert data["job_id"] == "test-job-id"
    assert data["status"] == "processing"
    assert data["progress"]["overallProgress"] == 50


def test_get_processing_status_not_found(mock_services):
    """Test getting processing status for non-existent job"""
    mock_services['rag'].get_processing_status = AsyncMock(return_value=None)
    
    response = client.get("/processing-status/non-existent?user_id=user123")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_processing_status_unauthorized(mock_services):
    """Test getting processing status with unauthorized access"""
    mock_job_data = {
        "id": "test-job-id",
        "userId": "different-user"
    }
    
    mock_services['rag'].get_processing_status = AsyncMock(return_value=mock_job_data)
    
    response = client.get("/processing-status/test-job-id?user_id=user123")
    assert response.status_code == 403
    assert "does not have access" in response.json()["detail"]


def test_get_user_jobs_success(mock_services):
    """Test getting user jobs"""
    mock_jobs = [
        {"id": "job1", "status": "completed"},
        {"id": "job2", "status": "processing"}
    ]
    
    mock_services['firebase'].get_user_rag_processing_jobs = AsyncMock(return_value=mock_jobs)
    
    response = client.get("/user-jobs/user123?limit=50")
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == "user123"
    assert len(data["jobs"]) == 2
    assert data["total"] == 2


def test_get_user_jobs_user_not_found(mock_services):
    """Test getting jobs for non-existent user"""
    mock_services['auth'].verify_user_exists = AsyncMock(return_value=False)
    
    response = client.get("/user-jobs/non-existent?limit=50")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_project_jobs_success(mock_services):
    """Test getting project jobs"""
    mock_jobs = [
        {"id": "job1", "projectId": "project456", "status": "completed"}
    ]
    
    mock_services['firebase'].get_project_rag_processing_jobs = AsyncMock(return_value=mock_jobs)
    
    response = client.get("/project-jobs/project456?user_id=user123")
    assert response.status_code == 200
    
    data = response.json()
    assert data["project_id"] == "project456"
    assert len(data["jobs"]) == 1
    assert data["total"] == 1


def test_get_project_jobs_unauthorized(mock_services):
    """Test getting project jobs with unauthorized access"""
    mock_services['auth'].verify_project_access = AsyncMock(return_value=False)
    
    response = client.get("/project-jobs/project456?user_id=user123")
    assert response.status_code == 403
    assert "does not have access" in response.json()["detail"]
