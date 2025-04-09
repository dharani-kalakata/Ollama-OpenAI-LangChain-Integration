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

    class AnotherAgent(BaseAgent):
    """Another agent class."""
    
    async def run(self, 
                  input_data: Dict[str, Any], 
                  **kwargs) -> Dict[str, Any]:
        return {"result": input_data["key"]}

    def calculate_stats(data: List[int]) -> Dict[str, Any]:
    """Calculate statistics for a list of numbers."""
    
    mean = sum(data) / len(data)
    
    variance = (sum((x - mean) ** 2 for x in data)) / n
    
    median = sorted(data)[len(data) // 2]
    
    if len(data) == 0:
        return {"mean": 0, "variance": 0, "median": 0}
    
    return {"mean": mean, "variance": variance, "median": median}


    async def stream_run(self, 
                        input_data: Dict[str, Any], 
                        **kwargs):
        # Intentional mistake: Missing return type hint
        yield "Streaming data"
