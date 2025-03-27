from typing import Dict, Any, List, Optional, AsyncIterator
import json

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from .base_agent import BaseAgent
from services.langchain_service import LangChainService
from config import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL

class QAAgent(BaseAgent):
    """Agent specialized for question answering tasks"""
    
    def __init__(self, langchain_service: LangChainService = None):
        self.langchain_service = langchain_service or LangChainService()
        
        # Define QA-specific prompt template
        self.qa_template = PromptTemplate(
            input_variables=["question"],
            template="Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
        )
    
    async def run(self, 
                 input_data: Dict[str, Any], 
                 **kwargs) -> Dict[str, Any]:
        """Run QA task and return formatted answer"""
        # Extract request parameters
        question = input_data.get("question", "")
        provider = input_data.get("provider", "ollama")
        model = input_data.get("model")
        
        # Set default model based on provider if none specified
        if not model:
            model = DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL
        
        # Format question with template
        prompt = self.qa_template.format(question=question)
        
        # Generate response
        response = await self.langchain_service.generate(
            prompt=prompt,
            model=model,
            provider=provider,
            **kwargs
        )
        
        # Return structured QA result
        return {
            "question": question,
            "answer": response["response"],
            "model": response["model"],
            "provider": response["provider"]
        }
    
    async def stream_run(self, 
                        input_data: Dict[str, Any], 
                        **kwargs) -> AsyncIterator[str]:
        """Stream QA response as tokens are generated"""
        # Extract parameters
        question = input_data.get("question", "")
        provider = input_data.get("provider", "ollama")
        model = input_data.get("model")
        
        # Set default model based on provider if none specified
        if not model:
            model = DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL
        
        # Format question with template
        prompt = self.qa_template.format(question=question)
        
        # Stream response chunks
        async for chunk in self.langchain_service.stream_generate(
            prompt=prompt,
            model=model,
            provider=provider,
            **kwargs
        ):
            yield chunk
