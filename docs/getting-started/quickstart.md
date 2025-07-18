# Quick Start

Get up and running with GenJAX in minutes! This guide covers the essential concepts through practical examples.

## Installation

```bash
pip install genjax
```

## Your First Model

Let's start with a simple Bayesian coin flipping model:

```python
import jax.numpy as jnp
from genjax import gen, beta, bernoulli

@gen
def coin_model(flips):
    # Prior belief about coin fairness
    bias = beta(2, 2) @ "bias"
    
    # Generate predictions for each flip
    predictions = []
    for i, flip in enumerate(flips):
        # Each flip is Bernoulli distributed
        pred = bernoulli(bias) @ f"flip_{i}"
        predictions.append(pred)
    
    return jnp.array(predictions)
```

## Running Inference

### Forward Sampling

Sample from the prior:

```python
# Simulate without any observations
trace = coin_model.simulate(flips=[None, None, None])

# Extract the sampled bias
bias_sample = trace.get_choices()["bias"]
print(f"Sampled bias: {bias_sample:.3f}")

# Get the predictions
predictions = trace.get_retval()
print(f"Predicted flips: {predictions}")
```

### Conditioning on Data

Observe some coin flips and infer the bias:

```python
# Observed data: True = Heads, False = Tails
observed_flips = [True, True, False, True]

# Condition on observations
constraints = {f"flip_{i}": flip for i, flip in enumerate(observed_flips)}

# Generate a trace with these constraints
trace, weight = coin_model.generate(constraints, flips=observed_flips)

# Extract posterior sample
posterior_bias = trace.get_choices()["bias"]
print(f"Posterior bias sample: {posterior_bias:.3f}")
```

## Using MCMC

For more complex models, use Markov Chain Monte Carlo:

```python
from genjax import mh, sel
import jax.random as random

# Initialize with prior sample
key = random.PRNGKey(0)
trace = coin_model.simulate(flips=observed_flips)

# Run Metropolis-Hastings
selection = sel("bias")  # Only update the bias
chain = []

for i in range(1000):
    key, subkey = random.split(key)
    trace, accepted = mh(trace, selection, key=subkey)
    
    if i % 100 == 0:  # Thin the chain
        chain.append(trace.get_choices()["bias"])

# Analyze posterior
posterior_mean = jnp.mean(jnp.array(chain))
print(f"Posterior mean bias: {posterior_mean:.3f}")
```

## A More Complex Example: Linear Regression

```python
@gen
def linear_regression(x):
    # Priors
    slope = normal(0, 10) @ "slope"
    intercept = normal(0, 10) @ "intercept"
    noise = gamma(1, 1) @ "noise"
    
    # Likelihood
    y_values = []
    for i in range(len(x)):
        y = normal(intercept + slope * x[i], noise) @ f"y_{i}"
        y_values.append(y)
    
    return jnp.array(y_values)

# Generate synthetic data
true_slope = 2.0
true_intercept = 1.0
x_data = jnp.linspace(-2, 2, 20)
y_data = true_intercept + true_slope * x_data + random.normal(key, shape=(20,)) * 0.5

# Condition on observed y values
constraints = {f"y_{i}": y_data[i] for i in range(len(y_data))}

# Use SMC for inference
from genjax import init, resample

# Initialize particles
particles = init(linear_regression, (x_data,), n_particles=100, constraints=constraints)

# Resample based on weights
particles = resample(particles)

# Extract posterior samples
slopes = [p.trace.get_choices()["slope"] for p in particles]
print(f"Posterior mean slope: {jnp.mean(jnp.array(slopes)):.3f}")
print(f"True slope: {true_slope}")
```

## Key Concepts Summary

1. **`@gen` decorator**: Transforms functions into generative functions
2. **`@` operator**: Assigns addresses to random choices
3. **Traces**: Record executions including choices and return values
4. **Constraints**: Fix certain random choices for conditioning
5. **Inference**: Various algorithms (MH, SMC, VI) for posterior inference

## Next Steps

- Explore the [Tutorial](tutorial.md) for in-depth examples
- Learn about [Inference Algorithms](../inference/overview.md)
- Check out [Advanced Examples](../examples/overview.md)
- Read the [API Reference](../api/overview.md)

## Getting Help

- GitHub Issues: [github.com/femtomc/genjax/issues](https://github.com/femtomc/genjax/issues)
- Documentation: [femtomc.github.io/genjax](https://femtomc.github.io/genjax)