# ITS Demo UI

A simple web interface for comparing baseline LLM inference vs Inference-Time Scaling (ITS) side by side.

## 🆕 Recent Updates (February 2026)

**Guided Demo Experience (Latest):**

- 🎯 **New Guided Demo Flow**: Completely redesigned step-by-step guided experience with 6-stage progressive disclosure
- 📈 **Goal-first navigation**: Users choose their goal (Improve Performance or Match Frontier) before anything else
- 🗳️ **ITS Method selection**: Dedicated step for choosing Self-Consistency or Best-of-N
- 🔀 **Branching scenarios**: Model options change based on the selected goal
- ⚡ **Trace animation**: Staged 3-phase visualization (Generate → Evaluate → Select) shows how ITS works
- 📊 **Performance page**: Dedicated bar charts comparing Cost, Quality, Latency, and Tokens
- 🧩 **Modular architecture**: New `guided-demo.js` and `guided-demo.css` files with clearly marked placeholder data

**Previous: Answer Extraction & Tool Consensus:**

- ✨ **Answer Extraction**: Math questions now extract `\boxed{}` answers for proper consensus voting (fixes Self-Consistency on mathematical reasoning)
- 🤖 **Tool Consensus**: New demo showing agent reliability through tool voting
- 🧠 **Auto-Detection**: Automatically recognizes question types (math, tool_calling, general)
- 📝 **System Prompts**: QWEN prompt applied to math questions for consistent formatting
- 📊 **Algorithm Traces**: Expandable UI showing vote counts, tool distributions, and candidate responses
- 📚 **Documentation**: Complete demo guides (`DEMO_CHEAT_SHEET.md`, `IDEAL_DEMO_CONFIGURATIONS.md`)

## Overview

This demo provides two entry points from the landing page:

1. **Guided Demo** — A step-by-step walkthrough that walks the user through the ITS experience with pre-populated scenarios and mock data (no backend required)
2. **Interactive Demo** — Full live experience with real API calls, model selection, and custom questions (requires backend)

### Guided Demo Flow

The Guided Demo follows a 6-step progressive disclosure:

| Step | Screen | What the user does |
|------|--------|--------------------|
| 1 | **Goal** | Choose "Improve Model Performance" or "Improve Small Model to Match Frontier" |
| 2 | **Method** | Choose ITS algorithm: Self-Consistency or Best-of-N |
| 3 | **Scenario** | Pick a model scenario (options depend on goal selected in step 1) |
| 4 | **Run** | Review pre-populated question and click Submit |
| 5 | **Trace** | View responses side-by-side, then trigger a staged trace animation |
| 6 | **Performance** | Bar charts comparing Cost, Quality, Latency, and Tokens |

**Branching logic in Step 3:**
- If goal = "Improve Model Performance" → Frontier Model or Open Source Model
- If goal = "Match Frontier" → Same Family (e.g. GPT-4o mini → GPT-4o) or Cross-Family (e.g. Llama 3 8B → GPT-4o)

All guided demo data uses placeholders. See [Plugging in Real Data](#plugging-in-real-data-guided-demo) below for where to replace them.

### Interactive Demo Use Cases

- **Three Use Cases**:
  1. **Improve Model**: Compare same model with/without ITS (2-column view)
  2. **Match Frontier**: Show small model + ITS matching large frontier model performance (3-column view)
  3. **Tool Consensus**: Demonstrate agent reliability through tool voting (2-column view)
- **Smart Features**:
  - **Answer Extraction**: Automatically extracts `\boxed{}` answers from math responses for proper consensus voting
  - **Auto-Detection**: Recognizes question types (math, tool_calling, general) and applies optimal configuration
  - **System Prompts**: Applies QWEN system prompt for math questions to ensure consistent answer formatting
- **Backend**: FastAPI server with comprehensive API endpoints
- **Frontend**: Modern HTML/JS interface with expandable sections, algorithm traces, and real-time metrics
- **Models**: OpenAI, OpenRouter, and Vertex AI models
  - **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
  - **OpenRouter**:
    - Weak models (3B-9B): Llama 3.1/3.2, Mistral 7B, Qwen 2.5, Gemma 2 - **great for ITS demos**
    - Medium models (27B-70B): Llama 3.1/3.3 70B, Qwen 2.5 72B, Gemma 2 27B
    - Frontier models: DeepSeek R1 (reasoning specialist)
    - Code specialists: Qwen 2.5 Coder, DeepSeek Coder V2
  - **Vertex AI Claude**: Sonnet, Opus, Haiku
  - **Vertex AI Gemini**: Pro, Flash
  - **Local**: vLLM server (any self-hosted model)
- **Algorithms**:
  - Outcome-based: Best-of-N, Self-Consistency (with answer extraction and tool voting)
  - Process-based: Beam Search, Particle Filtering, Entropic Particle Filtering, Particle Gibbs
- **Cost Tracking**: Real-time token usage and cost calculation
- **Security**: API keys stored server-side only

## Key Features

### 🎯 Three Powerful Use Cases

1. **Improve Any Model's Performance**
   - See how ITS enhances any model's capabilities
   - 2-column comparison: Baseline vs ITS
   - **NEW**: Answer extraction for math questions ensures proper consensus voting
   - Example: GPT-3.5 achieves 30-60% accuracy improvement on probability questions

2. **Match Frontier at Lower Cost**
   - Show small model + ITS matching large frontier model
   - 3-column comparison: Small, Small+ITS, Frontier
   - **Cost savings: 64-73%** while maintaining quality
   - Example: GPT-4o Mini + ITS saves 64% vs GPT-4o with same accuracy

3. **Tool Consensus for Agent Reliability** 🆕
   - Demonstrate reliable agent decision-making through tool voting
   - 2-column comparison: Single tool call vs Consensus voting
   - Shows distribution of tool selections (e.g., `{'calculate': 6}`)
   - Perfect for showcasing agent reliability in production scenarios

### 🧠 Intelligent Answer Extraction & Auto-Detection 🆕

**Answer Extraction for Math Questions:**
- Automatically extracts `\boxed{}` answers from mathematical responses
- Enables proper consensus voting (votes on "3/4" not "Therefore, the answer is 3/4...")
- Fixes Self-Consistency voting on full text (which was previously broken for math)
- Shows vote counts in algorithm trace: e.g., `{"('\\frac{3',)": 8}` (8/8 consensus)

**Auto-Detection:**
- **Math questions**: Detects LaTeX symbols (`$`, `\frac`, `\boxed`), math keywords → applies QWEN system prompt + answer extraction
- **Tool calling**: Detects `enable_tools=True` → applies tool voting configuration
- **General**: Default full-text matching for other questions
- No manual configuration needed - just paste your question!

**System Prompts:**
- Math questions automatically get: "Please reason step by step, and put your final answer within \boxed{}."
- Ensures consistent answer formatting across all models
- Improves consensus accuracy by 30-60% on mathematical reasoning tasks

### 💰 Real-Time Cost Tracking

- Accurate token counting (input/output)
- Per-request cost calculation in USD
- Cost comparison across models and configurations
- Demonstrates ROI of ITS approach

### 🎨 Modern, Clean UI

- **Response Section**: Complete chatbot answer (always visible)
- **Expandable Reasoning**: Step-by-step trace (click to reveal)
- **Expandable Performance**: All metrics in one place
- **LaTeX Support**: Beautiful math rendering
- **Responsive Design**: Works on desktop and mobile

### 🧮 Comprehensive Algorithm Support

All 6 ITS algorithms tested and working:
- **Outcome-based**: Best-of-N, Self-Consistency
- **Process-based**: Beam Search, Particle Filtering, Entropic Particle Filtering, Particle Gibbs

### 📚 Example Questions Library

16 curated questions inspired by MATH500 and AIME-2024:
- Easy, Medium, and Hard difficulty levels
- Organized by category (Algebra, Calculus, Logic, etc.)
- Each includes expected answer and algorithm recommendations
- Auto-filter by selected algorithm

## Project Structure

```
demo_ui/
├── README.md                                    # This file
├── DEMO_CHEAT_SHEET.md                         # Quick reference for demos (NEW)
├── IDEAL_DEMO_CONFIGURATIONS.md                # Detailed demo configurations (NEW)
├── ANSWER_EXTRACTION_FIX_VERIFICATION.md       # Verification report (NEW)
├── .gitignore                                  # Git ignore patterns
├── .env.example                                # Environment variables template
├── backend/
│   ├── __init__.py
│   ├── main.py                                 # FastAPI application
│   ├── config.py                               # Model registry
│   ├── models.py                               # Pydantic models
│   ├── tools.py                                # Tool definitions for agent demos (NEW)
│   ├── example_questions.py                    # Example questions library
│   ├── vertex_lm.py                            # Vertex AI model implementations
│   ├── llm_prm.py                              # LLM-based process reward model
│   └── requirements.txt                        # Backend dependencies
└── frontend/
    ├── index.html                              # Main web interface (landing page + interactive + guided shell)
    ├── guided-demo.js                          # Guided Demo flow logic, state, and mock data
    ├── guided-demo.css                         # Guided Demo styles
    ├── performance-viz-v2.js                   # Performance visualization charts
    └── performance-viz-v2.css                  # Performance visualization styles
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

# Optional: OpenRouter (for Llama, Mistral, Qwen, Gemma, DeepSeek models)
# Get your key from: https://openrouter.ai/keys
# Provides access to 15+ open-source and specialized models
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Optional: Google Cloud Vertex AI (for Claude and Gemini)
# Setup: gcloud auth application-default login
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1

# Optional: Local vLLM server
# VLLM_BASE_URL=http://localhost:8100/v1
# VLLM_API_KEY=NO_API_KEY
# VLLM_MODEL_NAME=your-model-name
```

**Provider Notes**:
- **OpenAI**: Native OpenAI API - most reliable, best for production demos
- **OpenRouter**: Unified API for open-source models - great for cost-effective demos and model diversity
  - **Note**: OpenRouter does NOT support function/tool calling - use OpenAI models for Tool Consensus demos
- **Vertex AI**: Claude and Gemini models via Google Cloud - requires `gcloud auth application-default login`

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

## Quick Start Guides 📚

For detailed demo configurations and talking points, see:

- **`DEMO_CHEAT_SHEET.md`** - Quick reference with exact settings, questions, and a 5-minute demo flow
- **`IDEAL_DEMO_CONFIGURATIONS.md`** - Comprehensive guide with recommended configurations for each use case
- **`ANSWER_EXTRACTION_FIX_VERIFICATION.md`** - Technical verification of answer extraction implementation

## Using the Demo

### Selecting a Use Case

Choose between three use cases at the top of the page:

1. **Improve Model Performance** (default):
   - Compare the same model with and without ITS
   - Shows 2-column comparison: Baseline vs ITS
   - **Best for**: Mathematical reasoning, probability questions
   - **Recommended models**: GPT-3.5 Turbo, Mistral 7B, Llama 3.1 8B (weak models show dramatic improvement)
   - **Example**: GPT-3.5 on "Alice & Bob dice probability" - baseline may get wrong answer, ITS achieves consensus on correct answer

2. **Match Frontier Performance**:
   - Compare small model + ITS vs large frontier model
   - Shows 3-column comparison: Small Baseline, Small + ITS, Frontier
   - **Best for**: Algebra, complex math problems
   - **Recommended combos**:
     - GPT-4o Mini + ITS vs GPT-4o (64% cost savings)
     - Mistral 7B + ITS vs GPT-4o (73% cost savings via OpenRouter)
   - **Example**: GPT-4o Mini + ITS matches GPT-4o at 64% lower cost
   - Demonstrates cost savings (64-73% cheaper) while matching quality

3. **Agent Tool Consensus** 🆕:
   - Compare single tool call vs consensus voting
   - Shows 2-column comparison: Baseline vs ITS with tool voting
   - **Best for**: Agent scenarios, financial calculations, data analysis
   - **Recommended models**: GPT-4o Mini, GPT-3.5 Turbo (OpenAI models only)
   - **⚠️ Note**: OpenRouter models do NOT support function/tool calling - use OpenAI models for this demo
   - **Example**: CAGR calculation shows tool consensus (e.g., `{'calculate': 6}`)
   - Demonstrates agent reliability through democratic voting

### Running a Comparison

1. **Select your model(s)**:
   - For "Improve Model": Choose one model (e.g., GPT-4o Mini)
   - For "Match Frontier": Choose small model (e.g., GPT-4o Mini) and frontier model (e.g., GPT-4o)

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

### Understanding the Results

Each result card shows:

- **Response Section** (always visible):
  - Complete chatbot response - exactly what the user would see
  - Clean, readable formatting with LaTeX math support

- **View Detailed Reasoning** (expandable):
  - Click to reveal step-by-step reasoning trace
  - Shows how the model arrived at the answer
  - Only appears for responses with structured reasoning

- **Performance Details** (expandable):
  - **Latency**: Time taken in milliseconds
  - **Model Size**: Small or Large
  - **Input Tokens**: Number of tokens in the prompt
  - **Output Tokens**: Number of tokens in the response
  - **Cost**: Calculated cost in USD

### Cost Comparison (Match Frontier Use Case)

When using the "Match Frontier" use case, the demo shows:
- **Small Baseline**: Cost of small model alone (e.g., $0.000007)
- **Small + ITS**: Cost of small model with ITS (e.g., $0.000021)
- **Frontier**: Cost of large frontier model (e.g., $0.000117)
- **Savings**: ITS typically saves 40-95% vs frontier while matching quality

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

## Plugging in Real Data (Guided Demo)

The Guided Demo uses placeholder/mock data throughout. All data lives in `frontend/guided-demo.js` in clearly marked sections:

| What to replace | Where in `guided-demo.js` | Format |
|-----------------|---------------------------|--------|
| **Scenario definitions** | `GUIDED_SCENARIOS` object | Add/remove/edit scenario cards (title, icon, model names) |
| **Demo questions** | `GUIDED_MOCK_QUESTIONS` object | Key = `${scenarioId}_${method}`, value = question string |
| **Model responses** | `getMockResponse()` function | Return `{ baseline: {...}, its: {...}, frontier?: {...}, trace: {...} }` |
| **Performance metrics** | `getMockPerformance()` function | Return `{ columns, cost, latency, tokens, quality }` arrays |
| **Trace data** | Inside `getMockResponse()` return | Self-Consistency: `vote_counts` object; Best-of-N: `scores` array |

### Example: replacing a question

```javascript
// In GUIDED_MOCK_QUESTIONS, change:
'improve_frontier_self_consistency': 'What is 144 / 12 + 7 * 3 - 5?',
// To your real demo question:
'improve_frontier_self_consistency': 'Your real question here',
```

### Example: replacing mock responses with captured data

```javascript
// In getMockResponse(), add a specific key check at the top:
function getMockResponse(scenarioId, method) {
    const key = `${scenarioId}_${method}`;
    // Add your real captured data here:
    if (key === 'improve_frontier_self_consistency') {
        return {
            baseline: { response: '...', latency_ms: 523, input_tokens: 24, output_tokens: 42, cost_usd: 0.000021 },
            its:      { response: '...', latency_ms: 1022, input_tokens: 192, output_tokens: 48, cost_usd: 0.000168 },
            trace:    { algorithm: 'self_consistency', candidates: [...], vote_counts: {...}, total_votes: 8 },
        };
    }
    // ... fallback to placeholder data
}
```

### Adding a new scenario

1. Add an entry to `GUIDED_SCENARIOS` with a unique `id`
2. Add a question to `GUIDED_MOCK_QUESTIONS` for each method (`${id}_self_consistency`, `${id}_best_of_n`)
3. Optionally add specific response data in `getMockResponse()`
4. The scenario will automatically appear in Step 3 based on its `goal` field

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

**Request (Improve Model Use Case):**
```json
{
  "question": "What is the derivative of x^3 + 2x^2?",
  "model_id": "gpt-4o-mini",
  "algorithm": "best_of_n",
  "budget": 4,
  "use_case": "improve_model"
}
```

**Request (Match Frontier Use Case):**
```json
{
  "question": "What is the derivative of x^3 + 2x^2?",
  "model_id": "gpt-4o-mini",
  "frontier_model_id": "gpt-4o",
  "algorithm": "best_of_n",
  "budget": 4,
  "use_case": "match_frontier"
}
```

**Request (Tool Consensus Use Case):** 🆕
```json
{
  "question": "What's the compound annual growth rate if I invest $1000 and it grows to $2000 in 5 years?",
  "model_id": "gpt-4o-mini",
  "algorithm": "self_consistency",
  "budget": 6,
  "use_case": "tool_consensus",
  "enable_tools": true,
  "tool_vote": "tool_name"
}
```

**Parameters:**
- `question` (string, required): The question to answer
- `model_id` (string, required): Model to use for ITS
- `algorithm` (string, required): ITS algorithm (best_of_n, self_consistency, beam_search, particle_filtering, entropic_particle_filtering, particle_gibbs)
- `budget` (integer, required): Computation budget (1-32)
- `use_case` (string, optional): "improve_model" (default), "match_frontier", or "tool_consensus"
- `frontier_model_id` (string, optional): Large model to compare against (required for match_frontier)
- `question_type` (string, optional): "auto" (default), "math", "tool_calling", or "general" - auto-detects if not specified
- `enable_tools` (boolean, optional): Enable tool calling for agent scenarios (default: false)
- `tool_vote` (string, optional): Tool voting strategy - "tool_name", "tool_args", or "tool_hierarchical"
- `exclude_args` (array, optional): Argument names to exclude from tool voting (e.g., `["timestamp", "id"]`)

**Response (Improve Model):**
```json
{
  "baseline": {
    "answer": "...",
    "latency_ms": 1234,
    "log_preview": "",
    "model_size": "Small",
    "cost_usd": 0.000007,
    "input_tokens": 15,
    "output_tokens": 50
  },
  "its": {
    "answer": "...",
    "latency_ms": 2345,
    "log_preview": "",
    "model_size": "Small",
    "cost_usd": 0.000021,
    "input_tokens": 45,
    "output_tokens": 150
  },
  "meta": {
    "model_id": "gpt-4o-mini",
    "algorithm": "best_of_n",
    "budget": 4,
    "run_id": "uuid-here",
    "use_case": "improve_model"
  },
  "small_baseline": null
}
```

**Response (Match Frontier):**
```json
{
  "baseline": {
    "answer": "...",
    "latency_ms": 1000,
    "model_size": "Large",
    "cost_usd": 0.000117,
    "input_tokens": 15,
    "output_tokens": 50
  },
  "its": {
    "answer": "...",
    "latency_ms": 3500,
    "model_size": "Small",
    "cost_usd": 0.000021,
    "input_tokens": 45,
    "output_tokens": 150
  },
  "small_baseline": {
    "answer": "...",
    "latency_ms": 500,
    "model_size": "Small",
    "cost_usd": 0.000007,
    "input_tokens": 15,
    "output_tokens": 50
  },
  "meta": {
    "model_id": "gpt-4o-mini",
    "frontier_model_id": "gpt-4o",
    "algorithm": "best_of_n",
    "budget": 4,
    "run_id": "uuid-here",
    "use_case": "match_frontier"
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

## OpenRouter Models

OpenRouter provides access to 15+ open-source and specialized models through a unified API. Get your API key at https://openrouter.ai/keys.

### Available Model Categories

**🎯 Weak Models (3B-9B) - Great for ITS Demos:**
- **Llama 3.1 8B** - Good balance of performance and cost
- **Llama 3.2 3B** - Very weak, shows dramatic ITS improvement
- **Mistral 7B** - Popular open-source model
- **Qwen 2.5 7B** - Strong reasoning for its size
- **Gemma 2 9B** - Google's open model

**Why use weak models?** They make mistakes on baseline but ITS corrects them, creating impressive before/after demos.

**⚖️ Medium Models (27B-70B) - Good Balance:**
- **Llama 3.1/3.3 70B** - Latest Meta models
- **Qwen 2.5 72B** - Strong reasoning capabilities
- **Gemma 2 27B** - Good mid-size option

**🏆 Frontier Models:**
- **DeepSeek R1** - Reasoning specialist

**💻 Code Specialists:**
- **Qwen 2.5 Coder 32B** - Code-focused model
- **DeepSeek Coder V2** - Advanced coding model

### OpenRouter Best Practices

**✅ Use OpenRouter for:**
- **Improve Model** demos - weak models show dramatic improvement
- **Match Frontier** demos - Mistral 7B + ITS vs GPT-4o saves 73%
- **Cost-effective testing** - open-source models are much cheaper
- **Model diversity** - test across different architectures

**❌ Don't use OpenRouter for:**
- **Tool Consensus** demos - OpenRouter doesn't support function calling
- Production tool calling scenarios - use OpenAI models instead

**⚠️ Important Notes:**
- Model availability changes - check https://openrouter.ai/models for current list
- Some models may have rate limits or timeouts
- For live demos, test your chosen model first
- If you get "No endpoints found", the model may no longer be available

### Recommended Demo Configurations

**Best cost/impact ratio:**
```
Use Case: Match Frontier
Small Model: mistral-7b (via OpenRouter - $0.06/1M)
Frontier Model: gpt-4o (via OpenAI - $10/1M)
Cost Savings: 73%
```

**Most dramatic improvement:**
```
Use Case: Improve Model
Model: llama-3.2-3b (very weak - makes many mistakes)
Algorithm: self_consistency
Budget: 8
Expected: Baseline often wrong, ITS corrects through consensus
```

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

## Features

### ✅ Implemented

- ✅ **Guided Demo Flow**: 6-step progressive disclosure with goal → method → scenario → submit → trace → performance
- ✅ **Trace Animation**: Staged 3-phase visualization showing Generate → Evaluate → Select
- ✅ **Performance Page**: Bar charts for Cost, Quality, Latency, and Tokens with best-value highlighting
- ✅ **Branching Scenarios**: Model options change based on selected goal (Improve vs Match Frontier)
- ✅ **Three Use Cases**: Improve model performance, match frontier at lower cost, or demonstrate tool consensus
- ✅ **Answer Extraction**: Extracts `\boxed{}` answers from math responses for proper consensus voting
- ✅ **Auto-Detection**: Recognizes math, tool_calling, and general question types automatically
- ✅ **Tool Consensus**: Agent reliability through democratic tool voting
- ✅ **Algorithm Traces**: Expandable visualization of vote counts, tool distributions, and candidate responses
- ✅ **6 ITS Algorithms**: Outcome-based and process-based algorithms
- ✅ **Cost Tracking**: Real-time token counting and cost calculation
- ✅ **Performance Metrics**: Latency, model size, tokens, and cost per request
- ✅ **Expandable UI**: Clean response view with collapsible reasoning and metrics
- ✅ **Example Questions**: Curated problems across difficulty levels (math and tool calling)
- ✅ **Multi-Provider Support**: OpenAI and Vertex AI (Claude, Gemini)
- ✅ **LaTeX Math Rendering**: Proper formatting for mathematical content

### 🔧 Potential Enhancements

- **Step-by-step visualization**: Show intermediate reasoning steps during generation
- **Dedicated PRM server**: Use vLLM with specialized process reward models for better performance
- **Quality scoring**: Automatic evaluation of answer correctness
- **Batch testing**: Evaluate multiple questions for systematic benchmarking
- **Response comparison**: Side-by-side diff view for answer comparison
- **Export results**: Download comparison results as JSON/CSV

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

Frontend file responsibilities:
- `index.html` — Landing page, interactive demo, inline styles/scripts, shared utility functions
- `guided-demo.js` — Guided Demo flow: state management, step navigation, mock data, trace animation, performance charts
- `guided-demo.css` — Guided Demo styles: progress bar, cards, response panes, trace phases, performance charts
- `performance-viz-v2.js` / `.css` — Performance visualization component (used by interactive mode)

To modify the backend, edit files in `backend/` and the server will auto-reload (if started with `--reload` flag).

To modify the frontend, edit the relevant file and refresh your browser. The guided demo logic is self-contained in `guided-demo.js` — it overrides the `initGuidedWizard()` function from `index.html` at load time.
