# genjax.inference.mcmc

Markov Chain Monte Carlo algorithms for probabilistic inference.

::: genjax.inference.mcmc
    options:
      show_source: true
      show_bases: true
      members_order: source
      filters:
        - "!^_"
      docstring_section_style: google

## Available Algorithms

### metropolis_hastings
Basic Metropolis-Hastings algorithm with custom proposals.

### hmc
Hamiltonian Monte Carlo for efficient exploration of continuous spaces.

### mala
Metropolis-Adjusted Langevin Algorithm for gradient-informed proposals.

## Usage Examples

### Metropolis-Hastings

```python
from genjax.inference.mcmc import metropolis_hastings
from genjax import select

# Define selection of variables to update
selection = select("mu", "sigma")

# Single MH step
new_trace = metropolis_hastings(trace, selection, key)

# Run MCMC chain
def mcmc_step(carry, key):
    trace = carry
    new_trace = metropolis_hastings(trace, selection, key)
    return new_trace, new_trace["mu"]

keys = jax.random.split(key, 1000)
final_trace, samples = jax.lax.scan(mcmc_step, initial_trace, keys)
```

### Hamiltonian Monte Carlo

```python
from genjax.inference.mcmc import hmc

# HMC with custom parameters
new_trace = hmc(
    trace, 
    selection,
    key,
    step_size=0.01,
    num_leapfrog_steps=10
)
```

## Best Practices

1. **Warm-up Period**: Discard initial samples during burn-in
2. **Thinning**: Keep every nth sample to reduce autocorrelation
3. **Multiple Chains**: Run parallel chains for convergence diagnostics
4. **Adaptive Step Size**: Tune step sizes during warm-up for HMC