# GenJAX

GenJAX is a JAX-based probabilistic programming language that provides a **Generative Function Interface (GFI)** for writing and composing probabilistic models with programmable inference.

## Quick Example

```python
import jax
import jax.numpy as jnp
from genjax import gen, distributions

@gen
def coin_flips(n):
    p = distributions.beta(1.0, 1.0) @ "bias"
    for i in range(n):
        distributions.bernoulli(p) @ f"flip_{i}"
    return p

# Run inference
key = jax.random.PRNGKey(0)
trace = coin_flips.simulate(key, (10,))
print(f"Inferred bias: {trace.retval}")
```

## Key Features

- **Composable Models**: Build complex models from simple components
- **JAX Integration**: Leverage JAX's JIT compilation and automatic differentiation
- **Programmable Inference**: Combine different inference algorithms seamlessly
- **Pytree Compatible**: All GenJAX types work with JAX transformations

## Getting Started

- [Installation](getting-started/installation.md) - Install GenJAX and its dependencies
- [Quick Start](getting-started/quickstart.md) - Dive into your first GenJAX program
- [Tutorial](getting-started/tutorial.md) - Learn GenJAX concepts step by step

## Learn More

- [Core API](api/overview.md) - Understand the Generative Function Interface
- [Inference Algorithms](inference/overview.md) - Explore MCMC, SMC, and VI
- [Examples](examples/overview.md) - See GenJAX in action