"""
Factory for creating RAG implementations
"""

import logging
from typing import Dict, Type, Optional
from src.services.rag_interface import RAGInterface
from src.services.rag_anything_service import RAGAnythingService

logger = logging.getLogger(__name__)

# Conditionally import other RAG services
LlamaIndexService = None
EvidenceSweepService = None

try:
    from src.services.llamaindex_service import LlamaIndexService
except ImportError as e:
    logger.warning(f"LlamaIndex service not available: {e}")

try:
    from src.services.evidence_sweep_service import EvidenceSweepService
except ImportError as e:
    logger.warning(f"EvidenceSweep service not available: {e}")


class RAGFactory:
    """Factory for creating RAG implementations"""
    
    def __init__(self):
        # Build implementations dict dynamically based on what's available
        self._implementations: Dict[str, Type[RAGInterface]] = {
            "raganything": RAGAnythingService,
        }
        
        if LlamaIndexService is not None:
            self._implementations["llamaindex"] = LlamaIndexService
            
        if EvidenceSweepService is not None:
            self._implementations["evidence_sweep"] = EvidenceSweepService
    
    def create_rag_service(
        self, 
        approach: str, 
        firebase_manager, 
        gcs_manager, 
        auth_service
    ) -> RAGInterface:
        """
        Create a RAG service instance
        
        Args:
            approach: RAG approach name (raganything, llamaindex, evidence_sweep)
            firebase_manager: Firebase manager instance
            gcs_manager: GCS manager instance
            auth_service: Auth service instance
            
        Returns:
            RAG service instance
            
        Raises:
            ValueError: If approach is not supported
        """
        if approach not in self._implementations:
            available = ", ".join(self._implementations.keys())
            raise ValueError(f"Unsupported RAG approach: {approach}. Available: {available}")
        
        implementation_class = self._implementations[approach]
        logger.info(f"Creating RAG service with approach: {approach}")
        
        return implementation_class(firebase_manager, gcs_manager, auth_service)
    
    def get_available_approaches(self) -> list[str]:
        """Get list of available RAG approaches"""
        return list(self._implementations.keys())
    
    def get_approach_info(self, approach: str) -> Dict[str, any]:
        """
        Get information about a specific RAG approach
        
        Args:
            approach: RAG approach name
            
        Returns:
            Dictionary with approach information
        """
        if approach not in self._implementations:
            raise ValueError(f"Unsupported RAG approach: {approach}")
        
        # Create a temporary instance to get approach info
        # Note: This is a bit hacky, but allows us to get dynamic info
        # In a real implementation, you might want to store this info statically
        try:
            temp_instance = self._implementations[approach](None, None, None)
            return {
                "name": temp_instance.get_approach_name(),
                "supported_parsers": temp_instance.get_supported_parsers(),
                "supported_models": temp_instance.get_supported_models(),
            }
        except Exception as e:
            logger.warning(f"Could not get info for approach {approach}: {e}")
            return {
                "name": approach,
                "supported_parsers": [],
                "supported_models": [],
            }
