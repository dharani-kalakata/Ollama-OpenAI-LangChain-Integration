import requests
import json
import sys
from typing import Optional

def send_request(prompt, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Send a text generation request to the API"""
    url = "http://localhost:8000/generate"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "stream": stream,
        "provider": provider
    }
    
    # Only include optional parameters when provided
    if model:
        data["model"] = model
    if temperature != 0.7:
        data["temperature"] = temperature
    if max_tokens:
        data["max_tokens"] = max_tokens

    print(f"Sending request to {url} with stream={stream}, provider={provider}, model={model or 'default'}")
    
    # Send request
    response = requests.post(url, headers=headers, json=data, stream=stream)

    # Process response
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        try:
                            json_data = json.loads(line_text)
                            if "response" in json_data:
                                print(json_data["response"], end="", flush=True)
                            if json_data.get("done", False):
                                print()
                        except json.JSONDecodeError:
                            print(line_text, end="", flush=True)
                    except Exception as e:
                        print(f"Error processing streaming response: {e}")
        else:
            # Handle complete response
            try:
                response_json = response.json()
                if isinstance(response_json, dict) and "response" in response_json:
                    print(response_json["response"])
                else:
                    print(response_json)
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def send_qa_request(question, model=None, stream=False, provider="ollama", temperature=0.7, max_tokens=None):
    """Send a question to the QA endpoint"""
    url = "http://localhost:8000/qa"
    
    # Build request payload
    headers = {"Content-Type": "application/json"}
    data = {
        "question": question,
        "stream": stream,
        "provider": provider
    }
    
    # Only include optional parameters when provided
    if model:
        data["model"] = model
    if temperature != 0.7:
        data["temperature"] = temperature
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    print(f"Sending QA request to {url} with stream={stream}, provider={provider}, model={model or 'default'}")
    
    # Send request and process response
    response = requests.post(url, headers=headers, json=data, stream=stream)
    
    if response.status_code == 200:
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        try:
                            json_data = json.loads(line_text)
                            if "response" in json_data:
                                print(json_data["response"], end="", flush=True)
                            if json_data.get("done", False):
                                print()
                        except json.JSONDecodeError:
                            print(line_text, end="", flush=True)
                    except Exception as e:
                        print(f"Error processing streaming response: {e}")
        else:
            # Handle complete response with structured output
            try:
                response_json = response.json()
                print(f"Question: {response_json.get('question', '')}")
                print(f"Answer: {response_json.get('answer', '')}")
                print(f"Model: {response_json.get('model', '')}")
                print(f"Provider: {response_json.get('provider', '')}")
            except json.JSONDecodeError:
                print(response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_models():
    """Retrieve and display available models from all providers"""
    url = "http://localhost:8000/models"
    response = requests.get(url)
    
    if response.status_code == 200:
        models = response.json()
        print("Available models:")
        print("\nOllama models:")
        for model in models.get("ollama", []):
            print(f"  - {model}")
        print("\nOpenAI models:")
        for model in models.get("openai", []):
            print(f"  - {model}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# CLI command handling
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python send_request.py generate <prompt> [model] [stream] [provider]")
        print("  python send_request.py qa <question> [model] [stream] [provider]")
        print("  python send_request.py models")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "generate":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py generate <prompt> [model] [stream] [provider]")
            sys.exit(1)
            
        # Parse arguments
        prompt = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        send_request(prompt, model, stream, provider)
        
    elif command == "qa":
        if len(sys.argv) < 3:
            print("Usage: python send_request.py qa <question> [model] [stream] [provider]")
            sys.exit(1)
            
        # Parse arguments
        question = sys.argv[2]
        model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() not in ["true", "false"] else None
        stream = len(sys.argv) > 3 and sys.argv[3].lower() == "true" or len(sys.argv) > 4 and sys.argv[4].lower() == "true"
        provider = sys.argv[5] if len(sys.argv) > 5 else "ollama"
        
        send_qa_request(question, model, stream, provider)
        
    elif command == "models":
        get_models()
        
    else:
        print("Unknown command. Available commands: generate, qa, models")
        sys.exit(1)