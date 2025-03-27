# FastAPI, LangChain, and LLM Integration

This project demonstrates how to integrate FastAPI with LangChain to provide a unified interface for different LLM providers. It uses LangChain as an abstraction layer to interact with Ollama and OpenAI.

## Main Functionalities

1. **Unified LLM API**: A consistent interface for interacting with multiple LLM providers.
2. **Streaming and Non-streaming Responses**: Support for both streaming and complete responses.
3. **Question-Answering Agent**: Specialized endpoint for question-answering tasks.
4. **Provider Selection**: Switch between Ollama (local models) and OpenAI with simple parameters.

## Dependencies

### Required Software
- **Python**: Python 3.7 or later
- **FastAPI**: Modern web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **Ollama**: Tool for running AI models locally (for the Ollama provider)
- **OpenAI API Key**: Required for the OpenAI provider

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/fastapi-langchain-llm.git
    cd fastapi-langchain-llm
    ```

2. **Set up environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Configure environment variables**:
    Copy `.env.example` to `.env` and update with your OpenAI API key:
    ```sh
    cp .env.example .env
    # Edit .env file with your actual API keys
    ```

## Usage

### Running the Server

Start the FastAPI server:
```sh
uvicorn app:app --reload
```

The server will be available at `http://localhost:8000`.

### API Endpoints

- **`/generate`**: Generate text with a specified LLM
- **`/qa`**: Use a QA agent to answer questions
- **`/models`**: List available models from all providers

### Sending Requests

You can use the provided utility scripts:

1. **Using send_request.py**:
    ```sh
    python send_request.py generate "Write a poem about AI." llama3.1 false ollama
    python send_request.py qa "What is the capital of France?" none false openai
    python send_request.py models
    ```

2. **Using demo_script.py**:
    ```sh
    python demo_script.py
    ```

3. **Using cURL**:
    ```sh
    curl -X POST "http://localhost:8000/generate" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Write a poem.", "model": "llama3.1", "provider": "ollama"}'
    ```

## Architecture

This project uses a layered architecture:

1. **API Layer**: FastAPI endpoints for user interaction
2. **Service Layer**: LangChain integration for different LLM providers
3. **Agent Layer**: Specialized components for specific tasks (like QA)
4. **Utility Layer**: Helper functions and formatters

LangChain acts as a unified interface to different LLM providers, allowing the application to interact with both local models (via Ollama) and cloud-based models (via OpenAI) through the same code paths.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
