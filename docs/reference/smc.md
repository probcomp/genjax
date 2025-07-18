# genjax.inference.smc

Sequential Monte Carlo methods for particle-based inference.

::: genjax.inference.smc
    options:
      show_source: true
      show_bases: true
      members_order: source
      filters:
        - "!^_"
      docstring_section_style: google

## Core Functions

### importance_sampling
Basic importance sampling with multiple particles.

### particle_filter
Sequential Monte Carlo for state-space models.

### rejuvenation_smc
SMC with MCMC rejuvenation steps for better particle diversity.

## Usage Examples

### Importance Sampling

```python
from genjax.inference.smc import importance_sampling

# Run with 1000 particles
keys = jax.random.split(key, 1000)
traces = jax.vmap(lambda k: model.generate(k, constraints, args))(keys)

# Extract weights
log_weights = traces.score
weights = jax.nn.softmax(log_weights)

# Weighted posterior mean
posterior_mean = jnp.sum(weights * traces["parameter"])
```

### Particle Filter

```python
from genjax.inference.smc import particle_filter

# For sequential data
@gen
def transition(prev_state, t):
    return distributions.normal(prev_state, 0.1) @ f"state_{t}"

@gen
def observation(state, t):
    return distributions.normal(state, 0.5) @ f"obs_{t}"

# Run particle filter
particles = particle_filter(
    initial_model,
    transition,
    observation,
    observations,
    n_particles=100,
    key=key
)
```

### SMC with Rejuvenation

```python
from genjax.inference.smc import rejuvenation_smc

# SMC with optional MCMC moves
result = rejuvenation_smc(
    model,
    observations,
    n_particles=100,
    n_mcmc_steps=5,  # Optional: rejuvenation steps
    key=key
)
```

## Best Practices

1. **Particle Count**: Use enough particles (typically 100-10000)
2. **Resampling**: Monitor effective sample size for resampling
3. **Proposal Design**: Use good proposal distributions
4. **Rejuvenation**: Add MCMC steps to maintain diversity