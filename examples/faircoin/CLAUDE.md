# CLAUDE.md - Fair Coin Case Study

Demonstrates GenJAX's competitive performance through a Beta-Bernoulli model, comparing against NumPyro and handcoded JAX implementations.

## Overview

This case study validates GenJAX's zero-overhead abstraction claim by implementing identical importance sampling across three frameworks for a simple conjugate model where the exact posterior is known analytically.

## Key Model and Inference Patterns

### Beta-Bernoulli Model
```python
@gen
def beta_ber_multi(num_obs: Const[int]):
    """Multiple coin flips with Beta prior."""
    alpha0, beta0 = jnp.array(10.0), jnp.array(10.0)
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip.repeat(n=num_obs.value)(f) @ "obs"
```

**Key patterns:**
- **Const wrapper**: Static parameters like `num_obs` wrapped in `Const[int]`
- **Address operator**: `@` assigns addresses to random choices for inference
- **Vectorized observations**: `flip.repeat()` for efficient multiple coin flips

### Importance Sampling Implementation
All three frameworks implement identical algorithms:
1. Sample from Beta(10, 10) prior
2. Weight by likelihood of observed data (80% heads)
3. Return weighted posterior samples

```python
# GenJAX
traces = seed(beta_ber_multi.importance)(key, Const(num_obs), obs_data, n_samples)
weights = traces.log_weights

# NumPyro (similar pattern)
samples = numpyro.infer.Importance(model, guide, num_samples=n_samples).run(...)

# Handcoded JAX (direct implementation)
prior_samples = jax.random.beta(key, a=10.0, b=10.0, shape=(n_samples,))
log_weights = compute_likelihood(prior_samples, obs_data)
```

### Performance Results
- **GenJAX**: ~100% of handcoded baseline (zero overhead)
- **Handcoded JAX**: 100% baseline (theoretical optimum)
- **NumPyro**: ~130-400% of baseline (framework overhead varies by platform)

## Figure Generated

**Combined Posterior and Timing** - 3x2 layout showing:
- Top row: Posterior histograms for each framework vs exact Beta posterior
- Bottom row: Horizontal bar chart comparing execution times

The figure demonstrates both correctness (all frameworks recover the true posterior) and efficiency (GenJAX matches handcoded performance).

## Usage

```bash
# Generate the figure (default: 2000 samples)
pixi run -e faircoin python -m examples.faircoin.main

# Custom sample size
pixi run -e faircoin python -m examples.faircoin.main --num-samples 5000

# GPU acceleration
pixi run -e faircoin-cuda python -m examples.faircoin.main
```

## Technical Notes

- **Data pattern**: 80% heads, 20% tails for clear posterior shift
- **Exact posterior**: Beta(50, 20) for 50 observations
- **JIT compilation**: All frameworks use JAX JIT for fair comparison
- **Timing methodology**: Multiple repeats with warm-up runs

## Summary

Validates GenJAX's core value proposition: high-level probabilistic programming abstractions with zero performance penalty compared to handcoded implementations, while providing cleaner syntax than manual implementations.