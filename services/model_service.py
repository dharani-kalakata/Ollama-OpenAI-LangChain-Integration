from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional, Union

class ModelService(ABC):
    """Abstract base class for model services."""
    
    @abstractmethod
    async def generate(self, 
                       prompt: str, 
                       model: str, 
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """Generate a complete response."""
        pass
    
    @abstractmethod
    async def stream_generate(self, 
                             prompt: str, 
                             model: str,
                             **kwargs) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass
    
    @abstractmethod
    async def get_formatted_response_non_streaming(self, 
                                                  prompt: str, 
                                                  model: str,
                                                  **kwargs) -> Dict[str, Any]:
        """Get a formatted non-streaming response."""
        pass
    
    @abstractmethod
    async def get_formatted_response_streaming(self, 
                                              prompt: str, 
                                              model: str,
                                              **kwargs) -> AsyncIterator[str]:
        """Get a formatted streaming response."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> list:
        """Get a list of available models."""
        pass
