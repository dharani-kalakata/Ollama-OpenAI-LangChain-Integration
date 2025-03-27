from typing import AsyncIterator, Dict, Any, List, Literal, Union
import json
import requests
import openai
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from .model_service import ModelService
from config import (
    OPENAI_API_KEY, 
    DEFAULT_OPENAI_MODEL, 
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL
)

class StreamingCallback(BaseCallbackHandler):
    """Handles token streaming from LLM responses"""
    
    def __init__(self):
        self.text = ""
        self.streaming_generator = None
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.streaming_generator:
            self.streaming_generator.send(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if self.streaming_generator:
            self.streaming_generator.close()

class LangChainService(ModelService):
    """Unified service for LLM interactions using LangChain"""
    
    def __init__(self, 
                 openai_api_key: str = OPENAI_API_KEY,
                 ollama_base_url: str = OLLAMA_BASE_URL):
        self.openai_api_key = openai_api_key
        self.ollama_base_url = ollama_base_url
    
    def _get_model(self, 
                  provider: Literal["openai", "ollama"],
                  model_name: str,
                  streaming_callback = None,
                  **kwargs):
        """Configure and return appropriate LangChain model based on provider"""
        if provider == "openai":
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=self.openai_api_key,
                streaming=streaming_callback is not None,
                callbacks=[streaming_callback] if streaming_callback else None,
                **kwargs
            )
        elif provider == "ollama":
            # Remove streaming parameter as it's not directly supported
            ollama_kwargs = {k: v for k, v in kwargs.items() if k != 'streaming'}
            return OllamaLLM(
                model=model_name,
                base_url=self.ollama_base_url,
                callbacks=[streaming_callback] if streaming_callback else None,
                **ollama_kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate(self, 
                      prompt: str, 
                      model: str = DEFAULT_OLLAMA_MODEL,
                      provider: str = "ollama",
                      stream: bool = False, 
                      **kwargs) -> Dict[str, Any]:
        """Generate a complete LLM response"""
        if provider not in ["openai", "ollama"]:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Apply default model if needed
        if model is None:
            model = DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL
        
        llm = self._get_model(provider, model, **kwargs)
        response = llm.invoke(prompt)
        
        # Handle different return types based on provider
        content = response.content if provider == "openai" else response
            
        return {
            "response": content,
            "model": model,
            "provider": provider
        }
    
    async def stream_generate(self, 
                             prompt: str, 
                             model: str = DEFAULT_OLLAMA_MODEL,
                             provider: str = "ollama",
                             **kwargs) -> AsyncIterator[str]:
        """Generate a streaming response with progress updates"""
        if provider not in ["openai", "ollama"]:
            raise ValueError(f"Unsupported provider: {provider}")
            
        if model is None:
            model = DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL
        
        streaming_callback = StreamingCallback()
        llm = self._get_model(provider, model, streaming_callback=streaming_callback, **kwargs)
        
        async def token_generator():
            try:
                with ThreadPoolExecutor() as executor:
                    # Run LLM in a separate thread to avoid blocking
                    future = executor.submit(llm.invoke, prompt)
                    
                    # Stream tokens while generating
                    while not future.done():
                        if streaming_callback.text:
                            yield json.dumps({
                                "response": streaming_callback.text,
                                "model": model,
                                "provider": provider,
                                "done": False
                            })
                            streaming_callback.text = ""
                        await asyncio.sleep(0.01)
                    
                    # Get result and send final completion message
                    future.result()
                    yield json.dumps({
                        "response": "",
                        "model": model,
                        "provider": provider,
                        "done": True
                    })
            except Exception as e:
                yield json.dumps({
                    "error": str(e),
                    "model": model,
                    "provider": provider,
                    "done": True
                })
        
        async for token in token_generator():
            yield token
    
    async def get_formatted_response_non_streaming(self, 
                                                 prompt: str, 
                                                 model: str = DEFAULT_OLLAMA_MODEL,
                                                 provider: str = "ollama",
                                                 **kwargs) -> Dict[str, Any]:
        """Get a complete response formatted for client consumption"""
        result = await self.generate(prompt, model, provider, **kwargs)
        return {"response": result["response"]}
    
    async def get_formatted_response_streaming(self, 
                                              prompt: str, 
                                              model: str = DEFAULT_OLLAMA_MODEL,
                                              provider: str = "ollama",
                                              **kwargs) -> AsyncIterator[str]:
        """Stream response chunks formatted for client consumption"""
        async for chunk in self.stream_generate(prompt, model, provider, **kwargs):
            try:
                json_chunk = json.loads(chunk)
                if "response" in json_chunk and json_chunk["response"]:
                    yield json_chunk["response"]
            except json.JSONDecodeError:
                continue
    
    async def get_formatted_response(self, 
                                    prompt: str, 
                                    model: str = DEFAULT_OLLAMA_MODEL,
                                    provider: str = "ollama", 
                                    stream: bool = False,
                                    **kwargs) -> Union[Dict[str, Any], AsyncIterator[str]]: 
        """Legacy method - use specialized streaming/non-streaming methods instead"""
        raise NotImplementedError(
            "Please use get_formatted_response_streaming or get_formatted_response_non_streaming directly"
        )
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Retrieve available models from all configured providers"""
        # Get Ollama models
        ollama_models = []
        try:
            models_url = f"{self.ollama_base_url}/api/tags"
            response = requests.get(models_url)
            response.raise_for_status()
            data = response.json()
            ollama_models = [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
        
        # Get OpenAI models
        openai_models = []
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            models = client.models.list()
            openai_models = [model.id for model in models.data]
        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
        
        return {
            "ollama": ollama_models,
            "openai": openai_models
        }
