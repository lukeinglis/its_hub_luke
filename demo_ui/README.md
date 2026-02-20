# ITS Demo UI

A simple web interface for comparing baseline LLM inference vs Inference-Time Scaling (ITS) side by side.

## Overview

This demo provides:

- **Backend**: FastAPI server exposing `/health` and `/compare` endpoints
- **Frontend**: Simple HTML/JS interface with side-by-side comparison
- **Algorithms**: Best-of-N and Self-Consistency
- **Security**: API keys stored server-side only

## Project Structure

```
demo_ui/
├── README.md              # This file
├── .gitignore            # Git ignore patterns
├── .env.example          # Environment variables template
├── backend/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Model registry
│   ├── models.py         # Pydantic models
│   └── requirements.txt  # Backend dependencies
└── frontend/
    └── index.html        # Web interface
```

## Setup

### Prerequisites

- Python 3.10+
- `its_hub` library installed (from parent directory)
- API keys for your chosen model provider

### 1. Install its_hub

From the repository root:

```bash
cd ..  # Go to repo root if you're in demo_ui/
uv sync --extra dev
# OR using pip:
# pip install -e .
```

### 2. Install backend dependencies

```bash
cd demo_ui
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for OpenAI models
OPENAI_API_KEY=your-openai-api-key-here

# Optional: for local vLLM server
# VLLM_BASE_URL=http://localhost:8100/v1
# VLLM_API_KEY=NO_API_KEY
# VLLM_MODEL_NAME=your-model-name
```

## Running the Demo

### Start the backend server

From the `demo_ui` directory:

```bash
# Using uvicorn directly
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# OR using python
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000`

### Open the frontend

Open `frontend/index.html` in your web browser:

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

## Using the Demo

1. **Select a model** from the dropdown (e.g., GPT-4o Mini)
2. **Choose an algorithm**:
   - **Best-of-N**: Generates N responses and selects the best using an LLM judge
   - **Self-Consistency**: Generates N responses and selects the most common answer
3. **Set the budget** (1-32): Number of generations to produce
4. **Enter a question** in the text area
5. **Click "Run Comparison"**

The demo will show:
- **Left pane (ITS Off)**: Single baseline inference
- **Right pane (ITS On)**: ITS algorithm result
- **Latency badges**: Time taken for each approach

## API Reference

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "ITS Demo API is running"
}
```

### GET /models

List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "gpt-4o-mini",
      "description": "OpenAI GPT-4o Mini",
      "model_name": "gpt-4o-mini"
    }
  ]
}
```

### POST /compare

Compare baseline vs ITS inference.

**Request:**
```json
{
  "question": "Explain quantum entanglement",
  "model_id": "gpt-4o-mini",
  "algorithm": "best_of_n",
  "budget": 4
}
```

**Response:**
```json
{
  "baseline": {
    "answer": "...",
    "latency_ms": 1234,
    "log_preview": ""
  },
  "its": {
    "answer": "...",
    "latency_ms": 2345,
    "log_preview": ""
  },
  "meta": {
    "model_id": "gpt-4o-mini",
    "algorithm": "best_of_n",
    "budget": 4,
    "run_id": "uuid-here"
  }
}
```

## Adding Custom Models

Edit `backend/config.py` and add your model to the `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    "my-custom-model": {
        "base_url": "https://api.example.com/v1",
        "api_key_env_var": "MY_CUSTOM_API_KEY",
        "model_name": "my-model-name",
        "description": "My Custom Model",
    },
    # ... existing models
}
```

Then add the corresponding API key to your `.env` file.

## Future Enhancements

The demo includes placeholder support for:

- **Step-by-step logs**: `log_preview` field in responses (currently empty)
- **Process reward models**: For algorithms like Particle Filtering
- **More algorithms**: Beam Search, Particle Gibbs, etc.

## Troubleshooting

### "API key not found" error

Make sure:
1. You created a `.env` file from `.env.example`
2. You added your API key to the `.env` file
3. The backend server was restarted after adding the key

### CORS errors in browser console

The backend is configured to allow all origins for local development. If you're running in production, update the CORS configuration in `backend/main.py`.

### Module import errors

Make sure `its_hub` is installed:
```bash
cd ..  # Go to repo root
pip install -e .
```

## Development

The backend uses:
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation
- **its_hub**: ITS algorithms library

The frontend uses:
- Vanilla HTML/CSS/JavaScript (no build step required)
- Fetch API for HTTP requests

To modify the backend, edit files in `backend/` and the server will auto-reload (if started with `--reload` flag).

To modify the frontend, edit `frontend/index.html` and refresh your browser.
