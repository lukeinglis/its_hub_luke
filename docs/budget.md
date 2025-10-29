# Budget & Inference-Time Scaling

## Overview

The `budget` parameter is a core concept in inference-time scaling (ITS) that controls how much computational effort is allocated to generating a response. Unlike traditional LLM inference which generates a single response, ITS algorithms use the budget to create multiple candidate responses or reasoning paths, then select the best one.

**Key Concept:** Higher budget = More computation = Better quality (but slower and more expensive)

## What is Budget?

Budget represents the computational resources allocated to an inference request. Different algorithms interpret budget in different ways:

- **Self-Consistency**: Number of complete responses to generate in parallel
- **Best-of-N**: Number of candidate responses to generate and score
- **Beam Search**: Total generations divided by beam width (controls search depth)
- **Particle Filtering**: Number of particles (reasoning paths) to maintain

### Budget vs Traditional Inference

```python
# Traditional inference (budget = 1 implicitly)
response = lm.generate(prompt)

# Inference-time scaling (budget = 8)
result = algorithm.infer(lm, prompt, budget=8)
# - Generates 8 responses/paths
# - Selects the best one
# - Returns higher quality result
```

## Budget Interpretation by Algorithm

| Algorithm | Budget Interpretation | Reward Model Needed | Example |
|-----------|----------------------|---------------------|---------|
| **Self-Consistency** | Number of parallel generations | No | `budget=8` → Generate 8 responses, return most common answer |
| **Best-of-N** | Number of candidates to generate | Yes (Outcome) | `budget=16` → Generate 16 responses, return highest-scored |
| **Beam Search** | Total generations ÷ beam width | Yes (Process) | `budget=32, beam_width=4` → 8 steps deep |
| **Particle Filtering** | Number of particles to maintain | Yes (Process) | `budget=8` → Maintain 8 reasoning paths |
| **Planning Enhancement** | Enhanced budget for base algorithm | Depends on base | `budget=16` → 1 for planning, 15 for execution |

## Choosing the Right Budget

### Guidelines by Use Case

#### Quick Experimentation (budget: 1-4)
- Fast iteration during development
- Testing algorithm behavior
- Prototyping with limited resources

```python
# Quick test with minimal overhead
result = algorithm.infer(lm, prompt, budget=2)
```

#### Production Use Cases (budget: 4-16)
- Balance between quality and latency
- Good for most mathematical reasoning tasks
- Recommended starting point

```python
# Production balance
result = algorithm.infer(lm, prompt, budget=8)
```

#### High-Stakes Tasks (budget: 16-64)
- Maximum quality requirements
- Competition-level problems (AIME, IMO)
- When accuracy is more important than speed

```python
# High-quality results
result = algorithm.infer(lm, prompt, budget=32)
```

#### Research & Benchmarking (budget: 64+)
- Exploring scaling limits
- Academic research
- Performance benchmarking

```python
# Research-level scaling
result = algorithm.infer(lm, prompt, budget=128)
```

## Budget vs Performance Trade-offs

### Quality vs Speed

As budget increases:
- ✅ **Quality improves**: More chances to find correct answer
- ✅ **Coverage increases**: Explores more solution strategies
- ❌ **Latency increases**: Takes longer to complete
- ❌ **Cost increases**: More API calls or GPU time

### Scaling Behavior

Different algorithms show different scaling characteristics:

**Self-Consistency:**
- Linear cost scaling: `budget=8` takes ~8x longer than `budget=1`
- Diminishing returns after certain point
- Best for: Fast scaling with minimal infrastructure

**Best-of-N:**
- Linear generation cost, plus reward model scoring
- Quality improves steadily with budget
- Best for: When you have a reliable reward model

**Beam Search:**
- Non-linear scaling due to beam width interaction
- `budget = beam_width × steps`
- Best for: Step-by-step problems with clear intermediate states

**Particle Filtering:**
- Balanced exploration and exploitation
- More efficient than pure random sampling
- Best for: Complex problems with uncertain solution paths

## Practical Budget Examples

### Example 1: Simple Math Problem

```python
from its_hub.algorithms import SelfConsistency

# Problem: "What is 15 × 23?"
prompt = "What is 15 × 23? Show your work."

# Low budget - might work for simple arithmetic
result = sc.infer(lm, prompt, budget=3)

# Medium budget - more reliable
result = sc.infer(lm, prompt, budget=8)

# High budget - overkill for this problem
result = sc.infer(lm, prompt, budget=32)
```

### Example 2: Complex Reasoning

```python
from its_hub.algorithms import ParticleFiltering

# AIME-level problem
prompt = "Let a be a positive real number such that all the roots of x^3 + ax^2 + ax + 1 = 0 are real. Find the smallest possible value of a."

# Too low - likely to fail
result = pf.infer(lm, prompt, budget=2)

# Better chance
result = pf.infer(lm, prompt, budget=8)

# High-quality result
result = pf.infer(lm, prompt, budget=32)
```

### Example 3: Budget with IaaS API

```bash
# Low budget request
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}],
    "budget": 4
  }'

# High budget request
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}],
    "budget": 16
  }'
```

## Advanced Topics

### Budget Allocation in PlanningWrapper

The PlanningWrapper automatically manages budget across multiple solution approaches:

```python
from its_hub.algorithms.planning_wrapper import PlanningWrapper

# Total budget: 16
# - 1 used for generating plan
# - 15 divided across 3 approaches (5 each)
result = planning_algorithm.infer(lm, prompt, budget=16)
```

**Budget breakdown:**
1. Planning phase: 1 generation
2. Approach 1: ⌊15/3⌋ = 5 generations
3. Approach 2: ⌊15/3⌋ = 5 generations
4. Approach 3: ⌊15/3⌋ = 5 generations
5. Remainder distributed to first approach(es)

### Budget Scaling Analysis

When benchmarking, test multiple budget values to understand scaling behavior:

```bash
# Benchmark with multiple budgets
python scripts/benchmark.py \
    --benchmark math-500 \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --alg particle-filtering \
    --budgets 1,2,4,8,16,32,64 \
    --does_eval
```

This produces data showing accuracy vs budget trade-offs:
- Identify diminishing returns point
- Optimize for your latency/quality requirements
- Understand cost-effectiveness

### Dynamic Budget Selection

For production systems, consider dynamic budget selection based on:

```python
def select_budget(prompt: str, difficulty_estimate: float) -> int:
    """Dynamically choose budget based on problem difficulty"""
    if difficulty_estimate < 0.3:
        return 4  # Easy problem
    elif difficulty_estimate < 0.7:
        return 8  # Medium problem
    else:
        return 16  # Hard problem

# Use it
budget = select_budget(prompt, estimate_difficulty(prompt))
result = algorithm.infer(lm, prompt, budget=budget)
```

## Best Practices

### 1. Start Small, Scale Up
Begin with `budget=4` and increase if quality is insufficient:
```python
budgets_to_try = [4, 8, 16, 32]
for budget in budgets_to_try:
    result = algorithm.infer(lm, prompt, budget=budget)
    if is_satisfactory(result):
        break
```

### 2. Monitor Costs
Track budget usage in production:
```python
total_generations = budget * num_requests
cost_estimate = total_generations * cost_per_generation
```

### 3. Profile Your Workload
Different problem types have different optimal budgets:
- **Arithmetic**: budget=2-4
- **Algebra**: budget=4-8
- **Geometry**: budget=8-16
- **Competition math**: budget=16-64

### 4. Consider Latency Requirements
```python
# Real-time applications
budget = 2  # ~2x latency of single generation

# Batch processing
budget = 16  # Quality over speed

# Critical decisions
budget = 32  # Maximum quality
```

### 5. Use Benchmarking Data
Refer to published benchmarks to choose appropriate budgets:
- MATH500: budget=8-16 recommended
- AIME-2024: budget=16-32 recommended

## Common Pitfalls

### ❌ Using Budget=1
```python
# This defeats the purpose of ITS
result = algorithm.infer(lm, prompt, budget=1)
```
**Why it's bad:** No benefit from inference-time scaling

### ❌ Excessive Budget Without Analysis
```python
# Wasteful without measuring benefit
result = algorithm.infer(lm, prompt, budget=256)
```
**Why it's bad:** Diminishing returns, excessive cost

### ❌ Ignoring Algorithm-Specific Interpretation
```python
# Budget=32 with beam_width=8 → only 4 steps!
beam_search.infer(lm, prompt, budget=32)
```
**Why it's bad:** May not be enough depth for the problem

## Summary

- **Budget controls computational investment** in inference-time scaling
- **Each algorithm interprets budget differently** - understand the specifics
- **Higher budget = better quality** but with trade-offs in speed and cost
- **Start with budget=8** for most applications and adjust based on results
- **Benchmark your specific use case** to find optimal budget values

For algorithm-specific details, see [Algorithms](algorithms.md).

For benchmarking budget scaling, see [Benchmarking](benchmarking.md).
