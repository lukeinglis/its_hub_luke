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
| **Self-Consistency** | Number of parallel generations | No | `budget=x` → Generate x responses, return most common answer |
| **Best-of-N** | Number of candidates to generate | Yes (Outcome) | `budget=x` → Generate x responses, return highest-scored |
| **Beam Search** | Total generations ÷ beam width | Yes (Process) | `budget=x, beam_width=x` → x steps deep |
| **Particle Filtering** | Number of particles to maintain | Yes (Process) | `budget=x` → Maintain x reasoning paths |
| **Planning Enhancement** | Enhanced budget for base algorithm | Depends on base | `budget=x` → 1 for planning,  for execution |

## Choosing the Right Budget

### Guidelines by Use Case

#### Quick Experimentation (budget: 1-4)

#### Production Use Cases (budget: 4-8)

#### High-Stakes Tasks (budget: 8-64)

#### Research & Benchmarking (budget: 64+)

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

## Advanced Topics

### Budget Allocation in PlanningWrapper


### Budget Scaling Analysis


### Budget Selection


## Best Practices

## Summary

- **Budget controls computational investment** in inference-time scaling
- **Each algorithm interprets budget differently** - understand the specifics
- **Higher budget = better quality** but with trade-offs in speed and cost
- **Start with budget=8** for most applications and adjust based on results
- **Benchmark your specific use case** to find optimal budget values

For algorithm-specific details, see [Algorithms](algorithms.md).

For benchmarking budget scaling, see [Benchmarking](benchmarking.md).
