# CLAUDE.md - Localization Case Study

Demonstrates robot localization using Sequential Monte Carlo (SMC) with different proposal and rejuvenation strategies.

## Overview

This case study shows how to track a robot's position over time using noisy sensor measurements from fixed beacons. It compares three SMC variants: bootstrap filter, optimal proposal, and optimal proposal with HMC rejuvenation.

## Key Model and Inference Patterns

### Model Architecture
```python
@gen
def step(state: Const[ParticleState], action: Const[Action]):
    """Single timestep dynamics with Gaussian noise."""
    x, y, theta = state.x, state.y, state.theta
    new_x = normal(x + action.dx, 0.1) @ "x"
    new_y = normal(y + action.dy, 0.1) @ "y" 
    new_theta = normal(theta + action.dtheta, 0.01) @ "theta"
    return ParticleState(new_x, new_y, new_theta)

@gen
def observe(state: Const[ParticleState], beacons):
    """Observation model with distance + bearing to beacons."""
    distances = dist_map.vmap(...)(state, beacons) @ "dists"
    bearings = bearing_map.vmap(...)(state, beacons) @ "bearings"
    return distances, bearings
```

### SMC Inference Pattern
```python
# Simplified rejuvenation_smc API (new in GenJAX)
traces, log_weights = rejuvenation_smc(
    initial_particles, initial_log_weights,
    model=trajectory,
    args=(beacons, actions),
    observations=observations,
    key=key,
    n_timesteps=n_timesteps,
    ess_threshold=n_particles // 2,
    # Optional: custom proposals and MCMC
    transition_proposal=custom_proposal,
    mcmc_kernel=custom_kernel
)
```

**Key patterns:**
- **Pytree dataclasses**: Clean state representation with automatic vectorization
- **Vectorized observations**: `vmap` for efficient beacon processing
- **Simplified SMC API**: Optional proposals/kernels with sensible defaults
- **ESS-based rejuvenation**: Automatic resampling when effective samples drop

### Performance Characteristics
- **Scales linearly** with particles and timesteps
- **GPU-friendly** through JAX vectorization
- **50 timesteps, 200 particles** typical for demos

## Figures Generated

1. **Localization Problem Explanation** - 1x4 panel showing setup
2. **SMC Methods Comparison** - 4-panel comparing bootstrap, optimal, and HMC variants

## Usage

```bash
# Generate both figures (default)
pixi run -e cuda python -m examples.localization.main

# Generate only problem explanation
pixi run -e cuda python -m examples.localization.main --mode problem

# Generate only method comparison
pixi run -e cuda python -m examples.localization.main --mode comparison
```

**Note**: Requires cuda environment for matplotlib dependencies.

## Summary

Demonstrates GenJAX's SMC capabilities for state-space models, showcasing the benefits of custom proposals and MCMC rejuvenation for maintaining particle diversity in sequential inference problems.