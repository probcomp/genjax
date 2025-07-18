# CLAUDE.md - Game of Life Case Study

Demonstrates inverse dynamics in Conway's Game of Life using Gibbs sampling with probabilistic rule violations.

## Overview

This case study solves the inverse problem: given an observed Game of Life grid state, infer the most likely previous state that could have generated it. This is achieved using a probabilistic variant of Conway's rules that allows soft violations.

## Key Model and Inference Patterns

### Probabilistic Game of Life Model
```python
@gen
def game_of_life_step(prev_grid, flip_prob: Const[float]):
    """Single GOL step with probabilistic rule violations."""
    H, W = prev_grid.shape
    
    # Count live neighbors efficiently
    neighbor_counts = count_neighbors_vectorized(prev_grid)
    
    # Standard Conway rules  
    survive = (prev_grid == 1) & ((neighbor_counts == 2) | (neighbor_counts == 3))
    birth = (prev_grid == 0) & (neighbor_counts == 3)
    next_grid_det = survive | birth
    
    # Probabilistic deviations
    violations = flip.vmap(shape=(H, W))(flip_prob.value) @ "violations"
    next_grid = jnp.where(violations, 1 - next_grid_det, next_grid_det)
    
    return next_grid @ "grid"
```

### Gibbs Sampling with 9-Coloring
```python
def gibbs_sweep(trace, observed, flip_prob, num_sweeps):
    """Efficient Gibbs sampling using 9-coloring for parallelization."""
    for sweep in range(num_sweeps):
        for color in range(9):
            # Update all cells of same color in parallel
            mask = (color_grid == color)
            trace = update_cells_vectorized(trace, mask, observed, flip_prob)
    return trace
```

**Key patterns:**
- **Soft constraints**: `flip_prob` parameter controls probability of rule violations
- **O(1) cell updates**: Each cell update only considers 3×3 neighborhood
- **9-coloring parallelization**: Update non-neighboring cells simultaneously
- **Inverse dynamics**: Infer past states from future observations

### Performance Characteristics
- **4×4 grids**: ~2.3s for 250 Gibbs steps
- **1024×1024 grids**: Scales O(n²) with grid size
- **GPU acceleration**: Available through JAX/CUDA backend

## Figures Generated

1. **Integrated Showcase** - 2-row visualization showing inverse problem and Gibbs chain
2. **Timing Bar Plot** - Performance comparison across different grid sizes

## Usage

```bash
# Generate both figures (default)
pixi run -e gol python -m examples.gol.main

# Generate only showcase
pixi run -e gol python -m examples.gol.main --mode showcase

# Generate only timing
pixi run -e gol python -m examples.gol.main --mode timing

# Custom parameters
pixi run -e gol python -m examples.gol.main --mode showcase --size 512 --chain-length 1000
```

## Summary

Showcases GenJAX's ability to handle discrete, structured probabilistic models through efficient Gibbs sampling with parallelization strategies, demonstrating inverse dynamics in a classic cellular automaton.