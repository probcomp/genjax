# CLAUDE.md - Curve Fitting Case Study

Demonstrates Bayesian polynomial regression with hierarchical modeling, GPU-accelerated importance sampling, and robust inference with outlier detection.

## Overview

This case study showcases Bayesian curve fitting using degree-2 polynomials, comparing standard models with robust variants that handle outliers through GenJAX's Cond combinator for natural mixture modeling.

## Key Model and Inference Patterns

### Hierarchical Polynomial Model
```python
@gen
def polynomial():
    """Degree 2 polynomial with hierarchical priors."""
    a = normal(0.0, 1.0) @ "a"      # Constant
    b = normal(0.0, 0.5) @ "b"      # Linear  
    c = normal(0.0, 0.2) @ "c"      # Quadratic
    return Lambda(Const(polyfn), jnp.array([a, b, c]))

@gen
def npoint_curve(xs):
    """Multi-point observations with Gaussian noise."""
    curve = polynomial() @ "curve"
    ys = point.vmap(in_axes=(0, None))(xs, curve) @ "ys"
    return curve, (xs, ys)
```

### Outlier Detection with Cond
```python
@gen
def point_with_outliers(x, curve, outlier_rate, outlier_std):
    """Mixture model using Cond combinator."""
    y_det = curve(x)
    is_outlier = flip(outlier_rate) @ "is_outlier"
    
    # Natural mixture expression
    cond_model = Cond(outlier_branch, inlier_branch)
    y_observed = cond_model(is_outlier, y_det, outlier_std) @ "y"
    return y_observed
```

### Inference Patterns
```python
# Importance sampling with SMC
samples, log_weights = seed(infer_latents)(
    key, xs, ys, Const(n_particles)
)

# Mixed Gibbs/HMC for outliers
def kernel(key, trace):
    # Gibbs for discrete outlier indicators
    trace = enumerative_gibbs_outliers(trace, xs, ys, outlier_rate)
    # HMC for continuous parameters
    return mixed_gibbs_hmc_kernel(...)(key, trace)
```

**Key patterns:**
- **Direct input pattern**: Models take `xs` as input to avoid static array issues
- **Const wrapper**: Static parameters like `n_particles` use `Const[int]`
- **Cond combinator**: Natural mixture modeling syntax compiles to efficient `jnp.where`
- **Mixed inference**: Combine discrete (Gibbs) and continuous (HMC) updates seamlessly

### Performance Characteristics
- **Flat scaling**: GPU vectorization shows constant runtime with increasing particles
- **Up to 100k particles**: Efficient memory usage enables large-scale inference
- **Zero abstraction cost**: High-level constructs compile to optimized JAX code

## Figures Generated

1. **Prior/Posterior Traces** - Visualizations with log density annotations
2. **Scaling Performance** - 3-panel analysis of runtime, LML, and ESS vs particles
3. **Posterior Scaling** - Effect of particle count on posterior quality
4. **Outlier Detection** - Comparison of standard vs robust models with different inference

Note: The vectorization illustration figure is stored in the `images/` directory.

## Usage

```bash
# Generate all figures (default)
pixi run -e curvefit python -m examples.curvefit.main

# Specific analyses
pixi run -e curvefit python -m examples.curvefit.main traces    # Trace visualizations
pixi run -e curvefit python -m examples.curvefit.main scaling   # Performance analysis
pixi run -e curvefit python -m examples.curvefit.main outlier   # Outlier detection

# Custom parameters
pixi run -e curvefit python -m examples.curvefit.main scaling --n-trials 200
```

## Summary

Illustrates GenJAX's capabilities for continuous probabilistic models, demonstrating zero-overhead abstractions, GPU acceleration benefits, and sophisticated inference combining importance sampling with mixed discrete-continuous MCMC.