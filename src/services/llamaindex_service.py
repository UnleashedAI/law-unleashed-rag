"""
LlamaIndex RAG service implementation
"""

import logging
import os
import tempfile
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from llama_index.vector_stores.google import GoogleVectorStore
from llama_index.core import Settings

from src.services.rag_interface import RAGInterface

logger = logging.getLogger(__name__)


class LlamaIndexService(RAGInterface):
    """Service for processing documents with LlamaIndex"""
    
    def __init__(self, firebase_manager, gcs_manager, auth_service):
        super().__init__(firebase_manager, gcs_manager, auth_service)
        
        # Storage configuration
        self.storage_dir = os.getenv("LLAMAINDEX_STORAGE_DIR", "./storage/llamaindex")
        self.temp_dir = os.getenv("RAG_TEMP_DIR", "./temp")
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize LlamaIndex components
        self._initialize_llamaindex()
    
    def _initialize_llamaindex(self):
        """Initialize LlamaIndex components"""
        try:
            # Configure LLM
            self.llm = OpenAI(
                model=os.getenv("MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            
            # Configure embeddings
            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            
            # Initialize LlamaParse if API key is available
            self.llama_parse = None
            if os.getenv("LLAMA_CLOUD_API_KEY"):
                self.llama_parse = LlamaParse(
                    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                    result_type="markdown",
                    verbose=True
                )
            
            # Initialize Google Vector Store if configured
            self.vector_store = None
            if os.getenv("GOOGLE_PROJECT_ID") and os.getenv("GOOGLE_VECTOR_STORE_ID"):
                self.vector_store = GoogleVectorStore(
                    project_id=os.getenv("GOOGLE_PROJECT_ID"),
                    vector_store_id=os.getenv("GOOGLE_VECTOR_STORE_ID")
                )
            
            logger.info("LlamaIndex initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {e}")
            raise
    
    async def process_document(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        parser: str = "llamaparse",
        parse_method: str = "auto",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a single document with LlamaIndex"""
        start_time = datetime.utcnow()
        local_file_path = None
        
        try:
            # Update job status to processing
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Download file from GCS
            logger.info(f"Downloading document from GCS: {gcs_path}")
            local_file_path = await self.gcs_manager.download_file(gcs_path)
            
            # Process document with LlamaIndex
            logger.info(f"Processing document with LlamaIndex: {local_file_path}")
            
            # Create documents
            if self.llama_parse and parser == "llamaparse":
                # Use LlamaParse for advanced parsing
                documents = await self.llama_parse.aload_data(local_file_path)
            else:
                # Use simple directory reader
                reader = SimpleDirectoryReader(input_files=[local_file_path])
                documents = reader.load_data()
            
            # Create or load index
            if self.vector_store:
                # Use Google Vector Store
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                index = VectorStoreIndex.from_documents(
                    documents, 
                    storage_context=storage_context
                )
            else:
                # Use local storage
                index_path = os.path.join(self.storage_dir, f"index_{project_id}")
                if os.path.exists(index_path):
                    storage_context = StorageContext.from_defaults(persist_dir=index_path)
                    index = load_index_from_storage(storage_context)
                    # Add new documents to existing index
                    for doc in documents:
                        index.insert(doc)
                else:
                    index = VectorStoreIndex.from_documents(documents)
                    index.storage_context.persist(persist_dir=index_path)
            
            result = {
                'success': True,
                'source': 'llamaindex_processed',
                'index_created': True,
                'document_count': len(documents)
            }
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 1,
                    'failedDocuments': 0,
                    'overallProgress': 100,
                    'currentDocument': gcs_path
                },
                'result': {
                    'processing_time': processing_time,
                    'document_path': gcs_path,
                    'local_path': local_file_path,
                    'rag_result': result,
                    'storage_info': {
                        'index_path': os.path.join(self.storage_dir, f"index_{project_id}"),
                        'vector_store_type': 'google' if self.vector_store else 'local'
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Document processed successfully with LlamaIndex: {Path(gcs_path).name}',
                {
                    'processing_time': processing_time,
                    'document_count': 1,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method
                }
            )
            
            logger.info(f"Successfully processed document with LlamaIndex: {gcs_path}")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'document_path': gcs_path,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error processing document {gcs_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 1,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Document processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # Clean up temporary files
            if local_file_path and os.path.exists(local_file_path):
                try:
                    await self.gcs_manager.cleanup_temp_files([local_file_path])
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {local_file_path}: {e}")
    
    async def process_folder(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        file_extensions: List[str] = None,
        parser: str = "llamaparse",
        parse_method: str = "auto",
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process all documents in a folder with LlamaIndex"""
        start_time = datetime.utcnow()
        downloaded_files = {}
        
        try:
            # Set default file extensions
            if file_extensions is None:
                file_extensions = [".pdf", ".doc", ".docx", ".txt", ".md"]
            
            # List files in folder
            logger.info(f"Listing files in folder: {gcs_folder_path}")
            file_paths = await self.gcs_manager.list_files_in_folder(gcs_folder_path, file_extensions)
            
            if not file_paths:
                raise ValueError(f"No files found in folder {gcs_folder_path} with extensions {file_extensions}")
            
            total_files = len(file_paths)
            logger.info(f"Found {total_files} files to process")
            
            # Update job status with total file count
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Download all files
            logger.info("Downloading files from GCS")
            downloaded_files = await self.gcs_manager.download_files(file_paths)
            
            if not downloaded_files:
                raise ValueError("Failed to download any files from GCS")
            
            # Process all documents
            all_documents = []
            processed_count = 0
            failed_count = 0
            processing_results = []
            
            for i, (gcs_path, local_path) in enumerate(downloaded_files.items()):
                try:
                    logger.info(f"Processing document {i+1}/{total_files}: {gcs_path}")
                    
                    # Update progress
                    progress = int((i / total_files) * 100)
                    await self.firebase_manager.update_rag_processing_job(job_id, {
                        'progress': {
                            'totalDocuments': total_files,
                            'processedDocuments': processed_count,
                            'failedDocuments': failed_count,
                            'overallProgress': progress,
                            'currentDocument': gcs_path
                        }
                    })
                    
                    # Create documents
                    if self.llama_parse and parser == "llamaparse":
                        documents = await self.llama_parse.aload_data(local_path)
                    else:
                        reader = SimpleDirectoryReader(input_files=[local_path])
                        documents = reader.load_data()
                    
                    all_documents.extend(documents)
                    processed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': True,
                        'document_count': len(documents)
                    })
                    
                    logger.info(f"Successfully processed: {gcs_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {gcs_path}: {e}")
                    failed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            # Create index from all documents
            if all_documents:
                if self.vector_store:
                    storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    index = VectorStoreIndex.from_documents(
                        all_documents, 
                        storage_context=storage_context
                    )
                else:
                    index_path = os.path.join(self.storage_dir, f"index_{project_id}")
                    if os.path.exists(index_path):
                        storage_context = StorageContext.from_defaults(persist_dir=index_path)
                        index = load_index_from_storage(storage_context)
                        for doc in all_documents:
                            index.insert(doc)
                    else:
                        index = VectorStoreIndex.from_documents(all_documents)
                        index.storage_context.persist(persist_dir=index_path)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': processed_count,
                    'failedDocuments': failed_count,
                    'overallProgress': 100,
                    'currentDocument': None
                },
                'result': {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'folder_path': gcs_folder_path,
                    'processing_results': processing_results,
                    'storage_info': {
                        'index_path': os.path.join(self.storage_dir, f"index_{project_id}"),
                        'vector_store_type': 'google' if self.vector_store else 'local'
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Folder processed successfully with LlamaIndex: {processed_count}/{total_files} documents processed',
                {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method
                }
            )
            
            logger.info(f"Successfully processed folder with LlamaIndex: {gcs_folder_path} ({processed_count}/{total_files} documents)")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'total_files': total_files,
                'processed_files': processed_count,
                'failed_files': failed_count,
                'folder_path': gcs_folder_path,
                'results': processing_results
            }
            
        except Exception as e:
            logger.error(f"Error processing folder {gcs_folder_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': len(downloaded_files) if downloaded_files else 0,
                    'processedDocuments': 0,
                    'failedDocuments': len(downloaded_files) if downloaded_files else 0,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Folder processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # Clean up temporary files
            if downloaded_files:
                try:
                    local_paths = list(downloaded_files.values())
                    await self.gcs_manager.cleanup_temp_files(local_paths)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary files: {e}")
    
    async def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a job"""
        try:
            return await self.firebase_manager.get_rag_processing_job(job_id)
        except Exception as e:
            logger.error(f"Error getting processing status for job {job_id}: {e}")
            return None
    
    def get_approach_name(self) -> str:
        """Get the name of this RAG approach"""
        return "LlamaIndex"
    
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported parsers for this approach"""
        parsers = ["simple"]
        if self.llama_parse:
            parsers.append("llamaparse")
        return parsers
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this approach"""
        return ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
