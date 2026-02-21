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

# Optional: For local vLLM
VLLM_BASE_URL=http://localhost:8100/v1
VLLM_API_KEY=NO_API_KEY
VLLM_MODEL_NAME=your-model-name
```

## Model Provider Summary

| Provider | Models | Authentication | Notes |
|----------|--------|----------------|-------|
| **OpenAI** | GPT-4o, GPT-4o Mini, GPT-3.5 | OPENAI_API_KEY | Native OpenAI API |
| **Vertex AI** | Claude (Sonnet, Opus, Haiku) | GCP credentials | Anthropic on Vertex |
| **Vertex AI** | Gemini (Pro, Flash) | GCP credentials | Native Google models |
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

## Testing

To test if models are accessible:
1. Start backend: `uvicorn backend.main:app --reload`
2. Check models endpoint: `http://localhost:8000/models`
3. Try a comparison with gpt-4o-mini and best_of_n algorithm
