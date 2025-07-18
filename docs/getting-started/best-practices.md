# Best Practices

This guide covers essential best practices for writing efficient and idiomatic GenJAX code.

## JAX Control Flow

GenJAX is built on JAX, which requires special handling of control flow for JIT compilation.

### ❌ Avoid Python Control Flow

```python
# BAD: Python for loop
@gen
def bad_model(n):
    values = []
    for i in range(n):
        x = distributions.normal(0, 1) @ f"x_{i}"
        values.append(x)
    return jnp.array(values)
```

### ✅ Use JAX Control Flow

```python
# GOOD: Vectorized operations
@gen
def good_model(n: Const[int]):
    # Use vmap for vectorized sampling
    values = distributions.normal(0, 1).vmap().apply(jnp.arange(n))
    return values
```

### Control Flow Patterns

#### Conditionals

```python
# BAD: Python if/else
if condition:
    result = model_a()
else:
    result = model_b()

# GOOD: jax.lax.cond
result = jax.lax.cond(
    condition,
    lambda: model_a(),
    lambda: model_b()
)
```

#### Loops with State

```python
# BAD: Python loop with accumulation
total = 0
for i in range(n):
    total += compute(i)

# GOOD: jax.lax.scan
def step(carry, i):
    return carry + compute(i), None

total, _ = jax.lax.scan(step, 0, jnp.arange(n))
```

## Static vs Dynamic Values

Use `Const[T]` to mark static values that should not become JAX tracers:

```python
from genjax import Const

@gen
def model(n_samples: Const[int], scale: float):
    # n_samples is static - safe to use in Python control
    # scale is dynamic - will be traced by JAX
    return distributions.normal(0, scale).vmap().apply(jnp.arange(n_samples))
```

## Random Number Generation

Always use JAX's functional RNG pattern:

```python
# Split keys for multiple uses
key, subkey1, subkey2 = jax.random.split(key, 3)

# Pass keys explicitly
trace1 = model.simulate(subkey1, args)
trace2 = model.simulate(subkey2, args)
```

## Vectorization Best Practices

### Use Built-in Vectorization

```python
# Vectorize distributions
@gen
def vectorized_model(data):
    # Vectorized prior
    mu = distributions.normal(0, 1) @ "mu"
    
    # Vectorized likelihood
    obs = distributions.normal(mu, 0.1).vmap().apply(jnp.arange(len(data)))
    return obs
```

### Batch Operations

```python
# Process multiple traces efficiently
keys = jax.random.split(key, n_chains)
traces = jax.vmap(lambda k: model.simulate(k, args))(keys)
```

## Memory Efficiency

### Avoid Materializing Large Intermediate Arrays

```python
# BAD: Creates large intermediate array
@gen
def inefficient(n):
    all_samples = distributions.normal(0, 1).vmap().apply(jnp.arange(n))
    return jnp.mean(all_samples)

# GOOD: Use scan for memory efficiency
@gen
def efficient(n: Const[int]):
    def step(carry, i):
        sample = distributions.normal(0, 1) @ f"sample_{i}"
        return carry + sample, None
    
    total, _ = jax.lax.scan(step, 0.0, jnp.arange(n))
    return total / n
```

## Type Annotations

Use type hints for better code clarity and IDE support:

```python
from typing import Tuple
import jax.numpy as jnp
from genjax import Trace

@gen
def typed_model(x: jnp.ndarray) -> jnp.ndarray:
    mu = distributions.normal(0.0, 1.0) @ "mu"
    y = distributions.normal(mu * x, 0.1) @ "y"
    return y

def inference_step(trace: Trace, key: jax.random.PRNGKey) -> Tuple[Trace, float]:
    new_trace = metropolis_hastings(trace, select("mu"), key)
    return new_trace, new_trace["mu"]
```

## Common Pitfalls

### 1. Forgetting vmap Address Prefixes

```python
# When using vmap, addresses are prefixed
constraints = {
    "vmap/x_0": 1.0,  # Correct
    "x_0": 1.0,       # Wrong - won't match vmapped choice
}
```

### 2. Using Python Randomness

```python
# BAD: Python's random module
import random
x = random.normal()

# GOOD: JAX random
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey)
```

### 3. Modifying Arrays In-Place

```python
# BAD: In-place modification
arr[0] = 1.0

# GOOD: Functional update
arr = arr.at[0].set(1.0)
```

## Performance Tips

1. **JIT Compile Inference Loops**: Wrap your inference code in `jax.jit`
2. **Batch Operations**: Use `vmap` instead of loops when possible
3. **Reuse Compiled Functions**: JIT compilation has overhead, reuse compiled functions
4. **Profile Your Code**: Use JAX's profiling tools to identify bottlenecks

## Testing Best Practices

```python
def test_model_deterministic():
    \"\"\"Test with fixed random seed for reproducibility\"\"\"
    key = jax.random.PRNGKey(42)
    trace = model.simulate(key, args)
    
    # Test should be deterministic
    expected_value = 1.234
    assert jnp.allclose(trace.retval, expected_value)

def test_gradients():
    \"\"\"Ensure gradients flow correctly\"\"\"
    def loss(params):
        trace = model.generate(key, constraints, params)
        return -trace.score  # Negative log probability
    
    grad_fn = jax.grad(loss)
    grads = grad_fn(params)
    
    # Check gradients are finite
    assert jnp.all(jnp.isfinite(grads))
```

## Next Steps

- Review the [API Reference](../api/overview.md) for detailed documentation
- Explore [Advanced Topics](../advanced/pjax.md) for deeper JAX integration
- Check out [Examples](../examples/overview.md) for real-world patterns