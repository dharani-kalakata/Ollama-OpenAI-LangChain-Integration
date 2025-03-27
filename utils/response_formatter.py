import json
from typing import Dict, Any, AsyncIterator

async def format_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Format a complete response for API output"""
    if "response" in response:
        return {"response": response["response"].strip()}
    return response

async def stream_format_response(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Format streaming response chunks for consistent API output"""
    async for chunk in stream:
        try:
            if isinstance(chunk, str):
                json_chunk = json.loads(chunk)
                if "response" in json_chunk:
                    yield json_chunk["response"]
        except json.JSONDecodeError:
            # For non-JSON chunks, pass through unchanged
            yield chunk
