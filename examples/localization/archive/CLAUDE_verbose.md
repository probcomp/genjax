# CLAUDE.md - Localization Case Study (CLEANED VERSION)

Probabilistic robot localization using particle filtering with GenJAX. This cleaned version only generates the two figures used in the POPL paper.

## ⚠️ CLEANUP STATUS

This is the **CLEANED VERSION** of the localization case study. Changes from original:

1. **Removed 30 unused figure types** - Only kept the 2 figures used in the paper
2. **Simplified main.py** - Removed all unused plotting code  
3. **Cleaned figs.py** - Only includes necessary plotting functions
4. **Kept core functionality** - All SMC methods and models remain unchanged

**Figures generated (ONLY 2):**
- `localization_r8_p200_basic_localization_problem_1x4_explanation.pdf` - Problem explanation
- `localization_r8_p200_basic_comprehensive_4panel_smc_methods_analysis.pdf` - SMC comparison

## Environment Setup

**IMPORTANT**: This case study requires the CUDA environment for proper execution:

```bash
# Always use the cuda environment for localization
pixi run -e cuda python -m examples.localization.main [command]

# Or use the predefined tasks (which automatically use cuda environment):
pixi run cuda-localization-generate-data
pixi run cuda-localization-plot-figures
pixi run cuda-localization  # Full pipeline
```

## Directory Structure (Cleaned)

```
examples/localization/
├── core.py             # SMC methods, drift-only model, world geometry (UNCHANGED)
├── data.py             # Trajectory generation (UNCHANGED)
├── figs.py             # Only 2 plot functions + dependencies (CLEANED)
├── main.py             # Simplified CLI for 2 figures only (CLEANED)
├── export.py           # CSV data export/import system (UNCHANGED)
├── data/               # Experimental data (UNCHANGED)
├── figs/               # Only 2 PDFs generated (CLEANED)
└── archive/            # Original versions before cleanup
    ├── main_original.py
    ├── figs_original.py
    └── CLAUDE_original.md
```

## Core Implementation (UNCHANGED)

All core SMC and model implementations remain exactly as in the original:

### Drift-Only Model Design
- **State space**: Only (x, y, θ) - no velocity variables
- **Dynamics**: Simple positional drift `x_t ~ Normal(x_{t-1}, σ)`
- **Benefits**: Better particle diversity, stable convergence

### SMC Methods
- **`run_smc_basic()`**: Bootstrap filter (no rejuvenation)
- **`run_smc_with_mh()`**: SMC + Metropolis-Hastings rejuvenation
- **`run_smc_with_hmc()`**: SMC + Hamiltonian Monte Carlo rejuvenation
- **`run_smc_with_locally_optimal_big_grid()`**: SMC + Locally optimal proposal

## Visualization (CLEANED)

### Only 2 Figures Generated

1. **Localization Problem Explanation** (`plot_localization_problem_explanation()`)
   - 1x4 row showing robot at different timesteps
   - LIDAR rays and noisy measurements
   - Clean room visualization without axes
   - Shows how robot moves and senses environment

2. **4-Panel SMC Comparison** (`plot_smc_method_comparison()`)
   - Row 1: Initial particle distributions with method titles
   - Row 2: Particle evolution showing trajectory blending  
   - Row 3: ESS raincloud plots
   - Row 4: Timing comparison horizontal bars

### Removed Visualizations (30 functions deleted)
- ❌ Individual particle evolution timelines (4 PDFs)
- ❌ Trajectory types comparison
- ❌ LIDAR demo visualization  
- ❌ Temporal particle evolution grid
- ❌ Final particle distribution
- ❌ Position/heading error plots
- ❌ Weight evolution plots
- ❌ Sensor observation plots
- ❌ Multi-method error comparison
- ❌ All p50 configuration figures
- ❌ And 20+ other unused plots

## CLI Usage (SIMPLIFIED)

### Two-Step Workflow (Same as Original)
```bash
# Step 1: Generate experimental data
pixi run cuda-localization-generate-data

# Step 2: Plot the 2 figures from saved data
pixi run cuda-localization-plot-figures

# Or run full pipeline:
pixi run cuda-localization
```

### Key Arguments
- `--include-smc-comparison`: Must be set to generate SMC comparison data
- `--n-particles`: Particle count (default: 200)
- `--experiment-name`: Custom name for data storage

## Data Export (UNCHANGED)

The data export system remains intact for reproducibility:
- Complete experimental data saved to `data/` directory
- CSV format with metadata preservation
- Can regenerate the 2 figures without recomputation

## Archive Notes

The original versions before cleanup are preserved in the `archive/` directory:
- `archive/main_original.py` - Original main with 30+ figure generation
- `archive/figs_original.py` - Original figs with all visualization functions
- `archive/CLAUDE_original.md` - Original documentation

The cleaned versions are now the default `main.py` and `figs.py`.

## Performance Characteristics (UNCHANGED)

Same performance as original:
- **Timing (100 particles)**:
  - Basic SMC: ~22ms
  - SMC + HMC: ~53ms  
  - SMC + Locally Optimal: ~30ms

## Summary

This cleaned version maintains all core functionality while removing 30+ unused visualization functions. It generates only the 2 figures actually used in the POPL paper, making the codebase cleaner and more maintainable.