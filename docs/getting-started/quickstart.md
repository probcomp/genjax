# Quick Start

Get up and running with GenJAX in minutes! This guide covers the essential concepts through practical examples.

## Installation

```bash
pip install genjax
```

## Your First Model

Let's start with a simple Bayesian coin flipping model that follows JAX best practices:

```python
import jax
import jax.numpy as jnp
from genjax import gen, distributions, Const

@gen
def coin_model(n_flips: Const[int]):
    # Prior belief about coin fairness
    bias = distributions.beta(2.0, 2.0) @ "bias"
    
    # Generate predictions using JAX-friendly vectorized operations
    # This uses GenJAX's built-in vectorization support
    flips = distributions.bernoulli(bias).vmap().apply(jnp.arange(n_flips))
    
    return flips
```

!!! tip "JAX Best Practice"
    Notice we use `Const[int]` for `n_flips` to indicate it's a static value.
    This allows JAX to compile efficiently without creating tracers for loop bounds.

## Running Inference

### Forward Sampling

Sample from the prior:

```python
key = jax.random.PRNGKey(0)

# Simulate without any observations
trace = coin_model.simulate(key, (4,))  # 4 flips

# Extract the sampled bias
bias_sample = trace["bias"]
print(f"Sampled bias: {bias_sample:.3f}")

# Get the predictions
predictions = trace.retval
print(f"Predicted flips: {predictions}")
```

### Conditioning on Data

Observe some coin flips and infer the bias:

```python
# Observed data: 1 = Heads, 0 = Tails
observed_flips = jnp.array([1, 1, 0, 1])

# Create constraints using JAX-friendly dictionary comprehension
constraints = {f"vmap/flip_{i}": observed_flips[i] for i in range(4)}

# Generate a trace with these constraints
trace = coin_model.generate(key, constraints, (4,))

# Extract posterior sample
posterior_bias = trace["bias"]
print(f"Posterior bias sample: {posterior_bias:.3f}")
```

!!! warning "Address Format"
    When using `vmap`, addresses are prefixed with `vmap/`. This is important
    for correctly targeting vectorized random choices.

## Using MCMC

For more complex models, use Markov Chain Monte Carlo with JAX-friendly patterns:

```python
from genjax.inference.mcmc import metropolis_hastings
from genjax import select

# Initialize with constrained trace
key, subkey = jax.random.split(key)
trace = coin_model.generate(subkey, constraints, (4,))

# Define MCMC kernel
selection = select("bias")  # Only update the bias

# Run chain using JAX scan for efficiency
def mcmc_step(carry, key):
    trace = carry
    new_trace = metropolis_hastings(trace, selection, key)
    return new_trace, new_trace["bias"]  # Return trace and save bias

# Generate keys for each MCMC step
keys = jax.random.split(key, 1000)

# Run MCMC chain
final_trace, bias_samples = jax.lax.scan(mcmc_step, trace, keys)

# Analyze posterior (thin by taking every 100th sample)
thinned_samples = bias_samples[::100]
posterior_mean = jnp.mean(thinned_samples)
print(f"Posterior mean bias: {posterior_mean:.3f}")
```

!!! success "JAX Best Practice"
    Using `jax.lax.scan` instead of Python loops allows:
    - JIT compilation of the entire MCMC chain
    - Efficient memory usage
    - GPU/TPU acceleration

## A More Complex Example: Linear Regression

```python
@gen
def linear_regression(x: jnp.ndarray):
    # Priors
    slope = distributions.normal(0.0, 10.0) @ "slope"
    intercept = distributions.normal(0.0, 10.0) @ "intercept"
    noise = distributions.gamma(1.0, 1.0) @ "noise"
    
    # Vectorized likelihood - no Python loops!
    mu = intercept + slope * x
    y = distributions.normal(mu, noise).vmap().apply(jnp.arange(len(x)))
    
    return y

# Generate synthetic data
true_slope = 2.0
true_intercept = 1.0
x_data = jnp.linspace(-2, 2, 20)

key, noise_key = jax.random.split(key)
y_data = true_intercept + true_slope * x_data + jax.random.normal(noise_key, shape=(20,)) * 0.5

# Create constraints for observed y values
constraints = {f"vmap/y_{i}": y_data[i] for i in range(len(y_data))}

# Use SMC for inference
from genjax.inference.smc import importance_sampling

# Run importance sampling with multiple particles
keys = jax.random.split(key, 100)
traces = jax.vmap(lambda k: linear_regression.generate(k, constraints, (x_data,)))(keys)

# Extract and analyze posterior samples
slopes = traces["slope"]  # Shape: (100,)
intercepts = traces["intercept"]  # Shape: (100,)

print(f"Posterior mean slope: {jnp.mean(slopes):.3f} (true: {true_slope})")
print(f"Posterior mean intercept: {jnp.mean(intercepts):.3f} (true: {true_intercept})")
```

!!! info "Vectorized Operations"
    GenJAX's `vmap()` method on distributions allows us to vectorize random
    choices across array dimensions, avoiding Python loops entirely.

## JAX & GenJAX Best Practices

### ✅ DO:
- Use `jax.lax.scan` for loops with accumulation
- Use `jax.lax.fori_loop` for simple iterations
- Use `jax.lax.cond` for conditionals
- Use `Const[T]` for static values in generative functions
- Use `vmap()` for vectorizing operations
- Use JAX's functional random number generation

### ❌ DON'T:
- Use Python `for` loops in JIT-compiled code
- Use Python `if/else` statements with traced values
- Build Python lists and convert to arrays
- Use mutable state or side effects

## Key Concepts Summary

1. **`@gen` decorator**: Transforms functions into generative functions
2. **`@` operator**: Assigns addresses to random choices
3. **Traces**: Immutable records of model execution
4. **Constraints**: Fix random choices for conditioning
5. **Vectorization**: Use `vmap()` for efficient batched operations
6. **Static values**: Use `Const[T]` for compile-time constants

## Next Steps

- Explore the [Tutorial](tutorial.md) for in-depth examples
- Learn about [Inference Algorithms](../inference/overview.md)
- Check out [Advanced Examples](../examples/overview.md)
- Read the [API Reference](../api/overview.md)

## Getting Help

- GitHub Issues: [github.com/femtomc/genjax/issues](https://github.com/femtomc/genjax/issues)
- Documentation: [femtomc.github.io/genjax](https://femtomc.github.io/genjax)