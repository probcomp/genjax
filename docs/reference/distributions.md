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

## Available Distributions

### Continuous Distributions

- **normal**: Normal (Gaussian) distribution
- **uniform**: Uniform distribution over an interval
- **beta**: Beta distribution
- **gamma**: Gamma distribution
- **exponential**: Exponential distribution
- **cauchy**: Cauchy distribution
- **student_t**: Student's t-distribution

### Discrete Distributions

- **bernoulli**: Bernoulli distribution (binary outcomes)
- **categorical**: Categorical distribution over finite outcomes
- **poisson**: Poisson distribution
- **binomial**: Binomial distribution
- **geometric**: Geometric distribution

## Usage Examples

```python
from genjax import distributions

# Continuous distributions
x = distributions.normal(0.0, 1.0) @ "x"
p = distributions.beta(2.0, 2.0) @ "p"
rate = distributions.gamma(1.0, 1.0) @ "rate"

# Discrete distributions
coin = distributions.bernoulli(0.5) @ "coin"
category = distributions.categorical(jnp.array([0.2, 0.3, 0.5])) @ "category"
count = distributions.poisson(3.0) @ "count"
```

## Vectorized Sampling

All distributions support vectorization via `vmap()`:

```python
# Sample 100 values from a normal distribution
samples = distributions.normal(0, 1).vmap().apply(jnp.arange(100))
```