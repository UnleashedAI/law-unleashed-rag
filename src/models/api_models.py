"""
API request and response models for RAG-Anything service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProcessDocumentRequest(BaseModel):
    """Request model for processing a single document"""
    user_id: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    workspace_id: str = Field(..., description="Workspace ID")
    gcs_path: str = Field(..., description="GCS path to the document")
    parser: str = Field(default="mineru", description="Parser to use: mineru or docling")
    parse_method: str = Field(default="auto", description="Parse method: auto, ocr, or txt")
    model: str = Field(default="gpt-4", description="LLM model to use")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")


class ProcessFolderRequest(BaseModel):
    """Request model for processing all documents in a folder"""
    user_id: str = Field(..., description="User ID")
    project_id: str = Field(..., description="Project ID")
    workspace_id: str = Field(..., description="Workspace ID")
    gcs_folder_path: str = Field(..., description="GCS path to the folder")
    parser: str = Field(default="mineru", description="Parser to use: mineru or docling")
    parse_method: str = Field(default="auto", description="Parse method: auto, ocr, or txt")
    model: str = Field(default="gpt-4", description="LLM model to use")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    file_extensions: List[str] = Field(
        default=[".pdf", ".doc", ".docx", ".txt", ".md", ".jpg", ".png", ".bmp", ".tiff"],
        description="File extensions to process"
    )


class ProcessingJobResponse(BaseModel):
    """Response model for processing job creation"""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Job creation time")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status"""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Current status")
    message: Optional[str] = Field(default=None, description="Status message")
    progress: Dict[str, Any] = Field(default_factory=dict, description="Processing progress")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Processing result")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    services: Optional[Dict[str, str]] = Field(default=None, description="Dependent services status")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
