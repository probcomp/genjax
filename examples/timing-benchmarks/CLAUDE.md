# CLAUDE.md - Timing Benchmarks

Framework performance comparison showing GenJAX's competitive performance across importance sampling and HMC inference.

## Overview

This directory contains systematic performance benchmarks comparing GenJAX against other probabilistic programming frameworks (NumPyro, Pyro, Gen.jl) and handcoded implementations in JAX/PyTorch/TensorFlow.

## Benchmark Structure

The benchmarks evaluate two inference algorithms across multiple frameworks:

### Importance Sampling (IS)
- Particle counts: 1000, 5000, 10000
- Measures wall-clock time for full inference
- Includes particle initialization, likelihood computation, and resampling

### Hamiltonian Monte Carlo (HMC)
- Chain lengths: 100, 500, 1000, 5000
- Includes warm-up phase timing
- Measures complete sampling process

## Figures Generated

1. **IS Performance Comparison** (`benchmark_timings_is_all_frameworks.pdf`)
   - Horizontal bar chart showing relative performance
   - GenJAX and handcoded JAX as baseline (~1x)
   - Framework overhead clearly visible

2. **HMC Performance Comparison** (`benchmark_timings_hmc_all_frameworks.pdf`)
   - Similar layout for HMC methods
   - Comparison across different chain lengths

## Key Results

### Importance Sampling
- **GenJAX**: Matches handcoded JAX performance (1.0x)
- **NumPyro**: ~1.3-1.6x overhead
- **Pyro**: ~20-40x overhead (Python loop overhead)
- **Gen.jl**: ~2-3x overhead

### HMC
- **GenJAX**: Near-optimal performance
- **NumPyro**: Highly optimized, often matches GenJAX
- **Pyro**: Significant overhead from PyTorch backend

## Technical Details

- **Model**: Curvefit polynomial regression (same as curvefit case study)
- **Hardware**: Results vary between CPU and GPU
- **JIT**: All JAX-based frameworks use XLA compilation
- **Data**: Archived timing results available for reproducibility

## Summary

Demonstrates that GenJAX achieves its design goal of zero-overhead abstractions, matching handcoded performance while providing high-level probabilistic programming constructs. The benchmarks validate GenJAX's position as a performance-focused PPL.