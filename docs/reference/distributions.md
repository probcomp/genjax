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

```pycon exec="true" source="material-block"
>>> import jax
>>> import jax.numpy as jnp
>>> from genjax import distributions
>>> 
>>> key = jax.random.PRNGKey(42)
>>> 
>>> # Normal distribution
>>> normal_sample = distributions.normal.simulate(key, (0.0, 1.0))
>>> print(f"Normal sample: {normal_sample.retval:.3f}")
>>> 
>>> # Beta distribution
>>> key, subkey = jax.random.split(key)
>>> beta_sample = distributions.beta.simulate(subkey, (2.0, 2.0))
>>> print(f"Beta sample: {beta_sample.retval:.3f}")
>>> 
>>> # Assess log probability
>>> log_prob, _ = distributions.normal.assess(1.5, (0.0, 1.0))
>>> print(f"Log prob of 1.5 under N(0,1): {log_prob:.3f}")
```

### Discrete Distributions

```pycon exec="true" source="material-block"
>>> # Bernoulli distribution
>>> key, subkey = jax.random.split(key)
>>> coin_flip = distributions.bernoulli.simulate(subkey, (0.7,))
>>> print(f"Coin flip (p=0.7): {coin_flip.retval}")
>>> 
>>> # Categorical distribution
>>> key, subkey = jax.random.split(key)
>>> probs = jnp.array([0.2, 0.3, 0.5])
>>> category = distributions.categorical.simulate(subkey, (probs,))
>>> print(f"Category (probs={probs}): {category.retval}")
>>> 
>>> # Poisson distribution
>>> key, subkey = jax.random.split(key)
>>> count = distributions.poisson.simulate(subkey, (3.0,))
>>> print(f"Poisson count (λ=3): {count.retval}")
```

### Vectorized Operations

```pycon exec="true" source="material-block"
>>> # Vectorized sampling with vmap
>>> from genjax import gen
>>> 
>>> @gen
... def vectorized_normal(n):
...     # Sample n values from standard normal
...     samples = distributions.normal(0.0, 1.0).vmap().apply(jnp.arange(n))
...     return samples
... 
>>> key, subkey = jax.random.split(key)
>>> trace = vectorized_normal.simulate(subkey, (5,))
>>> print(f"Vectorized samples shape: {trace.retval.shape}")
>>> print(f"Samples: {trace.retval}")
>>> 
>>> # Access individual vectorized choices
>>> for i in range(5):
...     print(f"  vmap/sample_{i}: {trace[f'vmap/sample_{i}']:.3f}")
```