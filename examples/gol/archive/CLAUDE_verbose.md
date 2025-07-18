# CLAUDE.md - Game of Life Case Study (CLEANED VERSION)

This cleaned version of the Game of Life case study only generates the two figures used in the POPL paper.

## ⚠️ CLEANUP STATUS

This is the **CLEANED VERSION** of the GOL case study. Changes from original:

1. **Removed 16+ unused figure types** - Only kept the 2 figures used in the paper
2. **Simplified main.py** - Removed all unused figure generation modes  
3. **Cleaned figs.py** - Only includes necessary plotting functions
4. **Kept core functionality** - All Gibbs sampling and GOL model code unchanged

**Figures generated (ONLY 2):**
- `gol_integrated_showcase_wizards_1024.pdf` - Main showcase figure with schematic
- `gol_gibbs_timing_bar_plot.pdf` - Performance timing comparison

## Overview

The Game of Life case study demonstrates probabilistic inference for Conway's Game of Life using Gibbs sampling with GenJAX. It showcases inverse dynamics - given an observed future state, infer the most likely previous state that would generate it.

## Directory Structure (Cleaned)

```
examples/gol/
├── core.py             # Game of Life model and Gibbs sampler (UNCHANGED)
├── data.py             # Pattern loading utilities (UNCHANGED)
├── figs.py             # Only 2 figure functions (CLEANED)
├── main.py             # Simplified CLI for 2 figures only (CLEANED)
├── CLAUDE.md           # This file - cleaned version documentation
├── assets/             # Image assets (UNCHANGED)
│   └── wizards.jpg     # Wizards pattern for showcase
├── data/               # Timing data (UNCHANGED)
├── figs/               # Only 2 PDFs generated (CLEANED)
└── archive/            # Original versions before cleanup
    ├── main_original.py
    ├── figs_original.py
    └── CLAUDE_original.md
```

## Core Implementation (UNCHANGED)

All core GOL and Gibbs sampling implementations remain exactly as in the original:

### Game of Life Model with Softness
- **Probabilistic Rule Violations**: `flip_prob` controls probability of violating GoL rules
- **Deterministic Core**: Standard Conway's rules as baseline behavior
- **Probabilistic Deviations**: Allow model flexibility for noisy observations

### Efficient Gibbs Sampling
- **O(1) Single Cell Updates**: Only consider 3×3 neighborhood
- **Full Grid Sweep Strategy**: 9-coloring pattern for parallel updates
- **Vectorized Operations**: Exploit JAX for efficient computation

## Visualization (CLEANED)

### Only 2 Figures Generated

1. **Integrated Showcase Figure** (`save_integrated_showcase_figure()`)
   - 2-row layout: Schematic (top) + Inference results (bottom)
   - Shows inverse problem: ? → GoL → observed state
   - Gibbs chain visualization with 2x4 grid of samples
   - Reconstruction quality plot
   - Default: wizards pattern at 1024x1024

2. **Timing Bar Plot** (`create_timing_bar_plot()`)
   - Horizontal bar chart of execution times
   - Shows performance across different grid sizes
   - Includes error bars and annotations
   - Reads from `data/gibbs_sweep_timing.json`

### Removed Visualizations (16+ functions deleted)
- ❌ Blinker pattern demonstrations
- ❌ Logo pattern experiments (MIT, POPL, Hermes)
- ❌ Performance scaling plots
- ❌ Nested vectorization illustrations
- ❌ Generative conditional demonstrations
- ❌ Separate monitoring/samples figures
- ❌ Forward/inverse problem schematics
- ❌ Animation functions
- ❌ And many other experimental visualizations

## CLI Usage (SIMPLIFIED)

```bash
# Generate both figures (default)
pixi run -e gol python -m examples.gol.main

# Generate only showcase figure
pixi run -e gol python -m examples.gol.main --mode showcase

# Generate only timing plot
pixi run -e gol python -m examples.gol.main --mode timing

# Custom showcase parameters
pixi run -e gol python -m examples.gol.main --mode showcase --size 512 --chain-length 1000
```

### CLI Arguments
- `--mode`: Which figures to generate (showcase, timing, all)
- `--pattern-type`: Pattern for showcase (wizards, mit, popl, blinker)
- `--size`: Grid size (default: 1024 for paper figure)
- `--chain-length`: Gibbs steps (default: 500)
- `--flip-prob`: Rule violation probability (default: 0.03)

## Key Implementation Details

### Pattern Loading
The showcase figure uses the wizards pattern at 1024x1024 by default:
- Full resolution wizards image for paper quality
- Downsampled versions available for faster testing
- Other patterns (MIT, POPL logos) available but not used in paper

### Timing Data
The timing bar plot requires pre-generated data:
- Run timing experiments to generate `data/gibbs_sweep_timing.json`
- If data not found, shows placeholder message
- Timing includes JIT compilation overhead

## Performance Characteristics (UNCHANGED)

Same performance as original:
- **Small Patterns (4×4)**: ~2.3 seconds for 250 steps
- **Large Patterns (1024×1024)**: Scales with O(n²) complexity
- **GPU Acceleration**: Available via CUDA environment

## Archive Notes

The original versions before cleanup are preserved in the `archive/` directory:
- `archive/main_original.py` - Original main with all figure modes
- `archive/figs_original.py` - Original figs with 30+ visualization functions
- `archive/CLAUDE_original.md` - Original comprehensive documentation

The cleaned versions are now the default `main.py` and `figs.py`.

## Summary

This cleaned version maintains all core GOL functionality while removing 16+ unused visualization functions. It generates only the 2 figures actually used in the POPL paper, making the codebase cleaner and more focused on the essential demonstrations.