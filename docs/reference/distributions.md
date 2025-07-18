# genjax.distributions

Built-in probability distributions that implement the Generative Function Interface.

::: genjax.distributions
    options:
      show_source: true
      show_bases: true
      members_order: source
      filters:
        - "!^_"
      docstring_section_style: google

## Live Examples

### Continuous Distributions

```python exec="true" source="material-block"
import jax.numpy as jnp
from genjax import distributions

# Assess log probability under various distributions
x = 1.5

# Normal distribution
log_prob_normal, _ = distributions.normal.assess(x, 0.0, 1.0)
print(f"Log prob of {x} under Normal(0, 1): {log_prob_normal:.3f}")

# Beta distribution (x must be in [0, 1])
x_beta = 0.7
log_prob_beta, _ = distributions.beta.assess(x_beta, 2.0, 2.0)
print(f"Log prob of {x_beta} under Beta(2, 2): {log_prob_beta:.3f}")

# Exponential distribution
log_prob_exp, _ = distributions.exponential.assess(x, 1.0)
print(f"Log prob of {x} under Exponential(1): {log_prob_exp:.3f}")
```

### Discrete Distributions

```python exec="true" source="material-block"
import jax.numpy as jnp
from genjax import distributions

# Bernoulli distribution
log_prob_bern, _ = distributions.bernoulli.assess(1, 0.7)
print(f"Log prob of 1 under Bernoulli(0.7): {log_prob_bern:.3f}")

# Categorical distribution
probs = jnp.array([0.2, 0.3, 0.5])
log_prob_cat, _ = distributions.categorical.assess(2, probs)
print(f"Log prob of category 2 under Categorical({probs}): {log_prob_cat:.3f}")

# Poisson distribution
log_prob_pois, _ = distributions.poisson.assess(4, 3.0)
print(f"Log prob of 4 under Poisson(3.0): {log_prob_pois:.3f}")
```

### Distribution Parameters

```python exec="true" source="material-block"
# Distributions are parameterized by their standard parameters
print("Common distribution parameterizations:")
print("- normal(mu, sigma)")
print("- beta(alpha, beta)")
print("- exponential(rate)")
print("- bernoulli(p)")
print("- categorical(probs)")
print("- poisson(rate)")
print("- gamma(concentration, rate)")
print("- uniform(low, high)")
```