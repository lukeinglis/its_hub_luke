# ITS Demo UI

A simple web interface for comparing baseline LLM inference vs Inference-Time Scaling (ITS) side by side.

## Overview

This demo provides:

- **Backend**: FastAPI server exposing `/health` and `/compare` endpoints
- **Frontend**: Simple HTML/JS interface with side-by-side comparison
- **Models**: OpenAI and Vertex AI models
  - **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
  - **Vertex AI Claude**: Sonnet, Opus, Haiku
  - **Vertex AI Gemini**: Pro, Flash
  - **Local**: vLLM server (any self-hosted model)
- **Algorithms**:
  - Outcome-based: Best-of-N, Self-Consistency
  - Process-based: Beam Search, Particle Filtering, Entropic Particle Filtering, Particle Gibbs
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

Edit `.env` and add your credentials:

```bash
# Required: OpenAI API (for GPT models + LLM judge)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Google Cloud Vertex AI (for Claude and Gemini)
# Setup: gcloud auth application-default login
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1

# Optional: Local vLLM server
# VLLM_BASE_URL=http://localhost:8100/v1
# VLLM_API_KEY=NO_API_KEY
# VLLM_MODEL_NAME=your-model-name
```

**Note**: OpenAI models use native OpenAI API. Vertex AI models (Claude, Gemini) require Google Cloud authentication.

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
2. **Choose an algorithm** (descriptions appear below the dropdown):

   **Outcome-Based Algorithms** (evaluate final responses):
   - **Best-of-N**: Generates N responses and selects the best using an LLM judge
   - **Self-Consistency**: Generates N responses and selects the most common answer

   **Process-Based Algorithms** (evaluate step-by-step reasoning):
   - **Beam Search**: Tree search maintaining top-k paths at each step
   - **Particle Filtering**: Sequential Monte Carlo sampling with resampling
   - **Entropic Particle Filtering**: Particle filtering with temperature annealing
   - **Particle Gibbs**: Iterative particle filtering with Gibbs sampling

3. **Set the budget** (1-32): Computational resources allocated
   - For outcome-based: Number of complete generations
   - For process-based: Total inference calls (divided across steps/particles)
4. **Enter a question** or **select an example**:
   - Type your own question in the text area
   - Or choose from pre-populated examples optimized for each algorithm
   - Examples are organized by difficulty (Easy, Medium, Hard)
   - Examples automatically filter based on the selected algorithm
5. **Click "Run Comparison"**

The demo will show:
- **Left pane (ITS Off)**: Single baseline inference
- **Right pane (ITS On)**: ITS algorithm result
- **Latency badges**: Time taken for each approach
- **Expected Answer** (for example questions): The correct answer shown below the results for easy verification

**Note**: Process-based algorithms work best with step-by-step reasoning tasks (e.g., math problems, logic puzzles) where intermediate steps can be evaluated.

### Example Questions

The demo includes 16 curated example questions inspired by the **MATH500** and **AIME-2024** benchmark datasets used in its_hub research:

**Easy Questions** (3):
- Basic arithmetic and algebra
- Simple logical reasoning
- Good for testing all algorithms quickly

**Medium Questions** (8):
- Quadratic equations, calculus, number theory
- Word problems and geometry
- Demonstrate clear ITS benefits

**Hard Questions** (5):
- Advanced algebra and optimization
- Complex combinatorics
- Best for showcasing advanced algorithms (Particle Gibbs, Entropic PF)

Each example includes:
- **Category**: Type of problem (Algebra, Calculus, etc.)
- **Difficulty**: Easy, Medium, or Hard
- **Expected Answer**: The correct answer for verification
- **Best for**: Recommended algorithms
- **Why**: Explanation of why this question suits those algorithms

When you select an example question and run a comparison, the expected answer will be displayed in a green box below the results, making it easy to verify which approach (baseline vs ITS) produced the correct answer.

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

### GET /examples

Get example questions, optionally filtered by algorithm.

**Query Parameters:**
- `algorithm` (optional): Filter questions by algorithm (e.g., `beam_search`)

**Response:**
```json
{
  "examples": [
    {
      "question": "Solve the quadratic equation: x^2 + 5x + 6 = 0",
      "category": "Algebra",
      "difficulty": "Medium",
      "expected_answer": "x = -2 or x = -3 (factoring: (x+2)(x+3) = 0)",
      "best_for": ["beam_search", "particle_filtering"],
      "why": "Multi-step solution with verification"
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

## Algorithm Details

### Outcome-Based vs Process-Based

- **Outcome-based algorithms** generate complete responses and evaluate only the final output
- **Process-based algorithms** generate responses step-by-step and evaluate intermediate reasoning

The demo uses an **LLM-based Process Reward Model** for process-based algorithms, which:
- Scores partial responses during generation
- Enables step-by-step reasoning evaluation
- Works without requiring a separate reward model server
- Uses GPT-4o Mini as the judge (configurable in code)

For production use with process-based algorithms, consider using a dedicated process reward model server (e.g., via vLLM) for better performance and cost efficiency.

## Future Enhancements

Potential improvements:

- **Step-by-step logs**: Visualize intermediate reasoning steps in the UI
- **Dedicated PRM server**: Use vLLM with specialized process reward models
- **Performance metrics**: Token usage, cost estimates, quality scores
- **Batch testing**: Evaluate multiple questions for benchmarking

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
