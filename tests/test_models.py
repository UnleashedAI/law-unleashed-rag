"""
Tests for Pydantic models
"""

import pytest
from datetime import datetime
from src.models.api_models import (
    ProcessDocumentRequest,
    ProcessFolderRequest,
    ProcessingJobResponse,
    ProcessingStatusResponse,
    HealthResponse,
    ErrorResponse
)
from src.models.firebase_models import (
    User,
    Workspace,
    Project,
    RAGProcessingJob,
    RAGProcessingMetadata
)


class TestAPIModels:
    """Test API request/response models"""
    
    def test_process_document_request(self):
        """Test ProcessDocumentRequest model"""
        request = ProcessDocumentRequest(
            user_id="user123",
            project_id="project456",
            workspace_id="workspace789",
            gcs_path="gs://bucket/document.pdf"
        )
        
        assert request.user_id == "user123"
        assert request.project_id == "project456"
        assert request.workspace_id == "workspace789"
        assert request.gcs_path == "gs://bucket/document.pdf"
        assert request.parser == "mineru"  # default
        assert request.parse_method == "auto"  # default
        assert request.model == "gpt-4"  # default
    
    def test_process_document_request_with_config(self):
        """Test ProcessDocumentRequest with custom configuration"""
        config = {"custom_param": "value"}
        request = ProcessDocumentRequest(
            user_id="user123",
            project_id="project456",
            workspace_id="workspace789",
            gcs_path="gs://bucket/document.pdf",
            parser="docling",
            parse_method="ocr",
            model="gpt-3.5-turbo",
            config=config
        )
        
        assert request.parser == "docling"
        assert request.parse_method == "ocr"
        assert request.model == "gpt-3.5-turbo"
        assert request.config == config
    
    def test_process_folder_request(self):
        """Test ProcessFolderRequest model"""
        request = ProcessFolderRequest(
            user_id="user123",
            project_id="project456",
            workspace_id="workspace789",
            gcs_folder_path="gs://bucket/folder/"
        )
        
        assert request.user_id == "user123"
        assert request.project_id == "project456"
        assert request.workspace_id == "workspace789"
        assert request.gcs_folder_path == "gs://bucket/folder/"
        assert request.parser == "mineru"  # default
        assert request.parse_method == "auto"  # default
        assert request.model == "gpt-4"  # default
        assert ".pdf" in request.file_extensions  # default
    
    def test_process_folder_request_custom_extensions(self):
        """Test ProcessFolderRequest with custom file extensions"""
        custom_extensions = [".txt", ".md"]
        request = ProcessFolderRequest(
            user_id="user123",
            project_id="project456",
            workspace_id="workspace789",
            gcs_folder_path="gs://bucket/folder/",
            file_extensions=custom_extensions
        )
        
        assert request.file_extensions == custom_extensions
    
    def test_processing_job_response(self):
        """Test ProcessingJobResponse model"""
        now = datetime.utcnow()
        response = ProcessingJobResponse(
            job_id="job123",
            status="pending",
            message="Job created",
            created_at=now
        )
        
        assert response.job_id == "job123"
        assert response.status == "pending"
        assert response.message == "Job created"
        assert response.created_at == now
    
    def test_processing_status_response(self):
        """Test ProcessingStatusResponse model"""
        now = datetime.utcnow()
        progress = {"overallProgress": 50}
        
        response = ProcessingStatusResponse(
            job_id="job123",
            status="processing",
            message="Processing...",
            progress=progress,
            created_at=now,
            updated_at=now
        )
        
        assert response.job_id == "job123"
        assert response.status == "processing"
        assert response.message == "Processing..."
        assert response.progress == progress
        assert response.created_at == now
        assert response.updated_at == now
    
    def test_health_response(self):
        """Test HealthResponse model"""
        response = HealthResponse(
            status="healthy",
            service="rag-anything"
        )
        
        assert response.status == "healthy"
        assert response.service == "rag-anything"
        assert response.services is None
    
    def test_health_response_with_services(self):
        """Test HealthResponse with services status"""
        services = {"firebase": "healthy", "gcs": "healthy"}
        response = HealthResponse(
            status="healthy",
            service="rag-anything",
            services=services
        )
        
        assert response.status == "healthy"
        assert response.service == "rag-anything"
        assert response.services == services
    
    def test_error_response(self):
        """Test ErrorResponse model"""
        response = ErrorResponse(
            error="Test error",
            detail="Detailed error message"
        )
        
        assert response.error == "Test error"
        assert response.detail == "Detailed error message"
        assert isinstance(response.timestamp, datetime)


class TestFirebaseModels:
    """Test Firebase data models"""
    
    def test_user_model(self):
        """Test User model"""
        user = User(
            uid="user123",
            email="test@example.com",
            displayName="Test User"
        )
        
        assert user.uid == "user123"
        assert user.email == "test@example.com"
        assert user.displayName == "Test User"
        assert user.tier == "PILOT"  # default
        assert user.premiumStatus == "none"  # default
        assert user.isAdmin is False  # default
        assert user.isDisabled is False  # default
        assert isinstance(user.createdAt, datetime)
        assert isinstance(user.updatedAt, datetime)
    
    def test_workspace_model(self):
        """Test Workspace model"""
        workspace = Workspace(
            id="workspace123",
            name="Test Workspace",
            ownerId="user123"
        )
        
        assert workspace.id == "workspace123"
        assert workspace.name == "Test Workspace"
        assert workspace.ownerId == "user123"
        assert workspace.description is None
        assert isinstance(workspace.createdAt, datetime)
        assert isinstance(workspace.updatedAt, datetime)
    
    def test_project_model(self):
        """Test Project model"""
        project = Project(
            id="project123",
            workspaceId="workspace123",
            title="Test Project",
            description="Test Description",
            userId="user123"
        )
        
        assert project.id == "project123"
        assert project.workspaceId == "workspace123"
        assert project.title == "Test Project"
        assert project.description == "Test Description"
        assert project.userId == "user123"
        assert project.reportsCount == 0  # default
        assert project.caseFilesCount == 0  # default
        assert project.graphProcessingStatus is None
        assert isinstance(project.createdAt, datetime)
        assert isinstance(project.updatedAt, datetime)
    
    def test_rag_processing_job_model(self):
        """Test RAGProcessingJob model"""
        job = RAGProcessingJob(
            id="job123",
            projectId="project123",
            userId="user123",
            workspaceId="workspace123",
            jobType="document",
            gcsPath="gs://bucket/document.pdf"
        )
        
        assert job.id == "job123"
        assert job.projectId == "project123"
        assert job.userId == "user123"
        assert job.workspaceId == "workspace123"
        assert job.jobType == "document"
        assert job.gcsPath == "gs://bucket/document.pdf"
        assert job.status == "pending"  # default
        assert isinstance(job.createdAt, datetime)
        assert isinstance(job.updatedAt, datetime)
        assert isinstance(job.progress, dict)
        assert job.errorMessage is None
        assert job.result is None
        assert job.config is None
    
    def test_rag_processing_metadata_model(self):
        """Test RAGProcessingMetadata model"""
        metadata = RAGProcessingMetadata(
            totalDocuments=10,
            processedDocuments=5,
            failedDocuments=1,
            processingTime=120.5,
            model="gpt-4",
            parser="mineru",
            parseMethod="auto"
        )
        
        assert metadata.totalDocuments == 10
        assert metadata.processedDocuments == 5
        assert metadata.failedDocuments == 1
        assert metadata.processingTime == 120.5
        assert metadata.model == "gpt-4"
        assert metadata.parser == "mineru"
        assert metadata.parseMethod == "auto"
        assert metadata.config is None
        assert isinstance(metadata.lastProcessedAt, datetime)
        assert metadata.storageInfo is None
