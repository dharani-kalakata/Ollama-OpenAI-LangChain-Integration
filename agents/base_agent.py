from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAgent(ABC):
    """Base class for all agents."""
    
    @abstractmethod
    async def run(self, 
                 input_data: Dict[str, Any], 
                 **kwargs) -> Dict[str, Any]:
        """Run the agent with the given input data."""
        pass
    
    @abstractmethod
    async def stream_run(self, 
                        input_data: Dict[str, Any], 
                        **kwargs):
        """Run the agent with streaming output."""
        pass
