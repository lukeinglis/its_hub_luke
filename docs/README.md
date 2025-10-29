# its-hub

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)

**its-hub** provides inference-time scaling for LLMs through multiple approaches:

1. **Direct Library Usage** - For Python integration
2. **Inference-as-a-Service (IaaS) API** - OpenAI-compatible HTTP API (⚠️ Alpha)

## What is Inference-Time Scaling?

Inference-time scaling improves LLM performance by using computational resources during inference to generate better responses. Unlike training-time scaling which requires more parameters or training data, inference-time scaling algorithms can improve any pre-trained model's performance by:

- **Generating multiple candidate responses** and selecting the best one
- **Using step-by-step reasoning** with reward models to guide generation
- **Applying probabilistic methods** like particle filtering for better exploration

## Key Features

- 🔬 **Multiple Algorithms**: Self-Consistency, Best-of-N, Beam Search, Particle Filtering, Planning Enhancement
- 🚀 **OpenAI-Compatible API**: Easy integration with existing applications
- 🧮 **Math-Optimized**: Built for mathematical reasoning with specialized prompts and evaluation
- 📊 **Benchmarking Tools**: Compare algorithms on standard datasets like MATH500 and AIME-2024
- ⚡ **Async Support**: Concurrent generation with limits and error handling

## Supported Algorithms

| Algorithm | Budget Interpretation | Reward Model Needed | Validated Use Cases | Snippet |
|-----------|----------------------|---------------------|---------------------|---------|
| **Self-Consistency** | Number of parallel generations | No | Mathematical reasoning (MATH500, AIME-2024) | `SelfConsistency()` |
| **Best-of-N** | Number of candidates to generate | Yes (Outcome) | Mathematical reasoning | `BestOfN(rm)` |
| **Beam Search** | Total generations ÷ beam width | Yes (Process) | Mathematical reasoning (MATH500, AIME-2024) | `BeamSearch(sg, prm, beam_width=4)` |
| **Particle Filtering** | Number of particles to maintain | Yes (Process) | Mathematical reasoning (MATH500, AIME-2024) | `ParticleFiltering(sg, prm)` |
| **Planning Enhancement** | Enhances any algorithm with planning | Depends on base algorithm | Mathematical reasoning | `PlanningWrapper(base_algorithm)` |

### Planning Enhancement

The **PlanningWrapper** can enhance any ITS algorithm with a planning phase that generates multiple solution approaches before execution. See [PLANNING_WRAPPER.md](PLANNING_WRAPPER.md) for detailed documentation.