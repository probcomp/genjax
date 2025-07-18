# Distributions

GenJAX provides a comprehensive set of probability distributions that implement the Generative Function Interface. All distributions can be used directly in `@gen` functions with the `@` addressing operator.

## Continuous Distributions

### Normal (Gaussian)

```python
from genjax import normal

# Standard normal
x = normal(0, 1) @ "x"

# With parameters
y = normal(mu=5.0, sigma=2.0) @ "y"

# In a model
@gen
def model():
    mean = normal(0, 10) @ "mean"
    data = normal(mean, 1) @ "data"
    return data
```

**Parameters:**
- `mu`: Mean (location parameter)
- `sigma`: Standard deviation (scale parameter, must be positive)

### Beta

```python
from genjax import beta

# Beta(2, 5)
p = beta(2, 5) @ "probability"

# Uniform prior (Beta(1, 1))
uniform_p = beta(1, 1) @ "uniform"
```

**Parameters:**
- `alpha`: First shape parameter (must be positive)
- `beta`: Second shape parameter (must be positive)

**Support:** [0, 1]

### Gamma

```python
from genjax import gamma

# Gamma(shape=2, rate=1)
x = gamma(2, 1) @ "x"

# For inverse scale parameterization
# Gamma(shape=α, scale=1/β) has mean α/β
precision = gamma(1, 1) @ "precision"
```

**Parameters:**
- `shape`: Shape parameter α (must be positive)
- `rate`: Rate parameter β (must be positive)

**Support:** (0, ∞)

### Exponential

```python
from genjax import exponential

# Exponential with rate 2.0
waiting_time = exponential(2.0) @ "wait"

# Mean = 1/rate, so rate=0.1 gives mean=10
long_wait = exponential(0.1) @ "long_wait"
```

**Parameters:**
- `rate`: Rate parameter λ (must be positive)

**Support:** [0, ∞)

### Uniform

```python
from genjax import uniform

# Uniform on [0, 1]
u = uniform(0, 1) @ "u"

# Uniform on [-5, 5]  
x = uniform(-5, 5) @ "x"
```

**Parameters:**
- `low`: Lower bound
- `high`: Upper bound (must be greater than low)

**Support:** [low, high]

### Dirichlet

```python
from genjax import dirichlet
import jax.numpy as jnp

# Symmetric Dirichlet
probs = dirichlet(jnp.ones(3)) @ "probs"

# Asymmetric Dirichlet
alphas = jnp.array([1.0, 2.0, 3.0])
weights = dirichlet(alphas) @ "weights"
```

**Parameters:**
- `alpha`: Concentration parameters (array, all elements must be positive)

**Support:** Simplex (sums to 1)

### Multivariate Normal

```python
from genjax import multivariate_normal
import jax.numpy as jnp

# 2D standard normal
x = multivariate_normal(
    jnp.zeros(2), 
    jnp.eye(2)
) @ "x"

# With correlation
mean = jnp.array([1.0, 2.0])
cov = jnp.array([[1.0, 0.5], 
                 [0.5, 2.0]])
y = multivariate_normal(mean, cov) @ "y"
```

**Parameters:**
- `mean`: Mean vector
- `cov`: Covariance matrix (must be positive definite)

## Discrete Distributions

### Bernoulli

```python
from genjax import bernoulli

# Fair coin
coin = bernoulli(0.5) @ "coin"

# Biased coin
biased = bernoulli(0.7) @ "biased"

# In a model
@gen
def coin_flips(n):
    p = beta(1, 1) @ "bias"
    flips = []
    for i in range(n):
        flip = bernoulli(p) @ f"flip_{i}"
        flips.append(flip)
    return flips
```

**Parameters:**
- `p`: Probability of success (must be in [0, 1])

**Support:** {0, 1} (False, True)

### Categorical

```python
from genjax import categorical
import jax.numpy as jnp

# Three categories with equal probability
x = categorical(jnp.ones(3) / 3) @ "x"

# With specified probabilities
probs = jnp.array([0.1, 0.3, 0.6])
category = categorical(probs) @ "category"

# In a mixture model
@gen
def mixture(n_components):
    weights = dirichlet(jnp.ones(n_components)) @ "weights"
    
    # Assign to components
    assignments = []
    for i in range(n_data):
        z = categorical(weights) @ f"z_{i}"
        assignments.append(z)
    return assignments
```

**Parameters:**
- `probs`: Probability vector (must sum to 1)

**Support:** {0, 1, ..., len(probs)-1}

### Poisson

```python
from genjax import poisson

# Poisson with rate 3.0
count = poisson(3.0) @ "count"

# Modeling count data
@gen
def count_model(exposure):
    rate = gamma(2, 1) @ "rate"
    counts = []
    for i in range(len(exposure)):
        count = poisson(rate * exposure[i]) @ f"count_{i}"
        counts.append(count)
    return counts
```

**Parameters:**
- `rate`: Rate parameter λ (must be positive)

**Support:** {0, 1, 2, ...}

### Flip

Alias for Bernoulli with boolean output:

```python
from genjax import flip

# Equivalent to bernoulli but more intuitive for booleans
if flip(0.8) @ "success":
    reward = normal(10, 1) @ "reward"
else:
    reward = normal(0, 1) @ "reward"
```

## Using Distributions Outside `@gen` Functions

All distributions implement the full GFI and can be used directly:

```python
# Direct sampling (requires explicit key)
from genjax import seed
import jax.random as random

key = random.PRNGKey(0)
sample = seed(normal.simulate)(key, mu=0, sigma=1)

# Log probability
log_prob, _ = normal.assess({"value": 1.5}, mu=0, sigma=1)

# Generate with constraints  
trace, weight = normal.generate({"value": 2.0}, mu=0, sigma=1)
```

## Custom Distributions

You can create custom distributions by implementing the GFI:

```python
from genjax import Distribution
import jax.numpy as jnp

class Laplace(Distribution):
    """Laplace (double exponential) distribution."""
    
    def sample(self, key, loc, scale):
        u = random.uniform(key, minval=-0.5, maxval=0.5)
        return loc - scale * jnp.sign(u) * jnp.log(1 - 2 * jnp.abs(u))
    
    def log_density(self, value, loc, scale):
        return -jnp.log(2 * scale) - jnp.abs(value - loc) / scale

# Use in a model
@gen
def robust_regression(x):
    # Laplace errors for robust regression
    intercept = normal(0, 10) @ "intercept"
    slope = normal(0, 5) @ "slope"
    
    errors = []
    for i in range(len(x)):
        # Would need to register as GenJAX distribution
        error = custom_laplace(0, 1) @ f"error_{i}"
        errors.append(error)
    
    return intercept + slope * x + jnp.array(errors)
```

## Distribution Parameters

### Shape Conventions

- **Scalar parameters**: Single values (e.g., `normal(0, 1)`)
- **Vector parameters**: Use JAX arrays (e.g., `dirichlet(jnp.ones(3))`)
- **Matrix parameters**: For multivariate distributions (e.g., `multivariate_normal(mean, cov)`)

### Broadcasting

GenJAX distributions support JAX broadcasting:

```python
# Sample multiple values with different means
means = jnp.array([0.0, 1.0, 2.0])
x = normal(means, 1.0) @ "x"  # Shape: (3,)

# Different means and sigmas
sigmas = jnp.array([0.5, 1.0, 2.0])  
y = normal(means, sigmas) @ "y"  # Shape: (3,)
```

## Common Patterns

### Hierarchical Models

```python
@gen
def hierarchical():
    # Global parameters
    global_mean = normal(0, 10) @ "global_mean"
    global_std = gamma(1, 1) @ "global_std"
    
    # Group-level parameters
    group_means = []
    for g in range(n_groups):
        group_mean = normal(global_mean, global_std) @ f"group_{g}_mean"
        group_means.append(group_mean)
    
    # Observations
    for g in range(n_groups):
        for i in range(n_obs_per_group):
            obs = normal(group_means[g], 1.0) @ f"obs_{g}_{i}"
```

### Prior Predictive Sampling

```python
@gen
def model():
    # Priors
    theta = beta(2, 2) @ "theta"
    
    # Likelihood
    successes = 0
    for i in range(n_trials):
        if bernoulli(theta) @ f"trial_{i}":
            successes += 1
    
    return successes

# Sample from prior predictive
trace = model.simulate()
prior_predictive_sample = trace.get_retval()
```

### Posterior Predictive

```python
# After inference, use posterior samples
posterior_trace = inference_algorithm(model, data)
theta_posterior = posterior_trace.get_choices()["theta"]

# Generate new predictions
@gen
def predictive(theta):
    predictions = []
    for i in range(n_future):
        pred = bernoulli(theta) @ f"pred_{i}"
        predictions.append(pred)
    return predictions

pred_trace = predictive.simulate(theta=theta_posterior)
```