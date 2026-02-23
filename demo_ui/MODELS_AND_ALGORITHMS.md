# Available Models and Algorithms

## ✅ Supported Models

All models use either OpenAI API or Google Cloud Vertex AI.

### OpenAI Models (Native OpenAI API)

**Frontier Models 🏆**
- **gpt-4o** - GPT-4o (Latest, most capable)
- **gpt-4-turbo** - GPT-4 Turbo

**Small/Fast Models ⚡**
- **gpt-4o-mini** - GPT-4o Mini (Fast & affordable)
- **gpt-3.5-turbo** - GPT-3.5 Turbo

### Vertex AI Models (Google Cloud)

**Claude Models (via Vertex AI)**
- **claude-sonnet-vertex** 🏆 - Claude 3.5 Sonnet (Frontier)
- **claude-opus-vertex** 🏆 - Claude 3 Opus (Frontier)
- **claude-haiku-vertex** ⚡ - Claude 3.5 Haiku (Small, fast)

**Gemini Models (via Vertex AI)**
- **gemini-pro-vertex** 🏆 - Gemini 1.5 Pro (Frontier)
- **gemini-flash-vertex** ⚡ - Gemini 1.5 Flash (Small, fast)

### IBM Granite Models (Self-Hosted)

**IBM's Open-Source Enterprise Models 🏢**

Latest Granite models available via Hugging Face:
- **granite-4.0-8b** - Granite 4.0 8B Instruct (Latest, Dec 2024)
- **granite-4.0-3b** ⚡ - Granite 4.0 3B Instruct (Small, fast)
- **granite-3.3-8b** - Granite 3.3 8B Instruct (Open-source)

Run with vLLM: `vllm serve ibm-granite/granite-4.0-8b-instruct --port 8100`

## ✅ Working Algorithms

### Outcome-Based Algorithms
These evaluate the final response only:

1. **best_of_n**
   - Generates N responses and selects best using LLM judge
   - Requires: OpenAI API key for judge
   - Parameters: None (budget passed to ainfer)

2. **self_consistency**
   - Generates N responses and selects most common answer
   - No judge required
   - Parameters: None (budget passed to ainfer)

### Process-Based Algorithms
These evaluate step-by-step reasoning:

3. **beam_search**
   - Tree search maintaining top-k paths at each step
   - Requires: StepGeneration, ProcessRewardModel
   - Parameters: `beam_width` (configured in backend)

4. **particle_filtering**
   - Sequential Monte Carlo sampling with resampling
   - Requires: StepGeneration, ProcessRewardModel
   - Parameters: None (budget passed to ainfer)

5. **entropic_particle_filtering**
   - Particle filtering with temperature annealing
   - Requires: StepGeneration, ProcessRewardModel
   - Parameters: None (budget passed to ainfer)

6. **particle_gibbs**
   - Iterative particle filtering with Gibbs sampling
   - Requires: StepGeneration, ProcessRewardModel
   - Parameters: `num_iterations` (configured in backend)

## Environment Variables Needed

Create a `.env` file in `demo_ui/` with:

```bash
# Required: OpenAI API (for GPT models + LLM judge)
OPENAI_API_KEY=your-openai-api-key

# Optional: Google Cloud Vertex AI (for Claude and Gemini)
# Setup with: gcloud auth application-default login
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1

# Optional: For IBM Granite models (self-hosted)
GRANITE_BASE_URL=http://localhost:8100/v1
GRANITE_API_KEY=NO_API_KEY

# Optional: For other local vLLM models
VLLM_BASE_URL=http://localhost:8200/v1
VLLM_API_KEY=NO_API_KEY
VLLM_MODEL_NAME=your-model-name
```

## Model Provider Summary

| Provider | Models | Authentication | Notes |
|----------|--------|----------------|-------|
| **OpenAI** | GPT-4o, GPT-4o Mini, GPT-3.5 | OPENAI_API_KEY | Native OpenAI API |
| **Vertex AI** | Claude (Sonnet, Opus, Haiku) | GCP credentials | Anthropic on Vertex |
| **Vertex AI** | Gemini (Pro, Flash) | GCP credentials | Native Google models |
| **Self-Hosted** | IBM Granite 4.0, 3.3 | N/A | vLLM via Hugging Face |
| **Local vLLM** | Any open-source model | N/A | Self-hosted |

## Vertex AI Setup

To use Vertex AI models (Claude and Gemini):

1. **Set up GCP Project**: https://console.cloud.google.com/
2. **Enable Vertex AI API**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```
3. **For Claude models**, request access:
   - Visit: https://console.cloud.google.com/vertex-ai/publishers/anthropic
   - Request access to Claude models
4. **Authenticate**:
   ```bash
   # Option A: Application Default Credentials (recommended)
   gcloud auth application-default login

   # Option B: Service Account
   # Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```
5. **Set environment variables** in `.env`:
   ```bash
   VERTEX_PROJECT=your-gcp-project-id
   VERTEX_LOCATION=us-central1
   ```

## IBM Granite Setup

To use IBM Granite 4.0 or 3.3 models:

1. **Install vLLM** (if not already installed):
   ```bash
   pip install vllm
   ```

2. **Start vLLM server** with Granite model:
   ```bash
   # Granite 4.0 8B (Latest)
   vllm serve ibm-granite/granite-4.0-8b-instruct --port 8100

   # Or Granite 4.0 3B (Faster)
   vllm serve ibm-granite/granite-4.0-3b-instruct --port 8100

   # Or Granite 3.3 8B
   vllm serve ibm-granite/granite-3.3-8b-instruct --port 8100
   ```

3. **Set environment variables** in `.env`:
   ```bash
   GRANITE_BASE_URL=http://localhost:8100/v1
   GRANITE_API_KEY=NO_API_KEY
   ```

4. **Available Granite models**:
   - `granite-4.0-8b` - Latest 8B instruct model (Dec 2024)
   - `granite-4.0-3b` - Smaller 3B instruct model
   - `granite-3.3-8b` - Previous generation 8B model

**Hugging Face Hub**: https://huggingface.co/ibm-granite

## Testing

To test if models are accessible:
1. Start backend: `uvicorn backend.main:app --reload`
2. Check models endpoint: `http://localhost:8000/models`
3. Try a comparison with gpt-4o-mini and best_of_n algorithm
