"""
Figure generation utilities for the Game of Life case study - CLEANED VERSION.

This cleaned version only contains functions needed for the two figures used in the paper:
1. create_integrated_showcase_figure() / save_integrated_showcase_figure()
2. create_timing_bar_plot()
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import jax.numpy as jnp
import jax.random as jrand

from .core import (
    run_sampler_and_get_summary,
    GibbsSampler,
    get_gol_sampler_separate_figures,
)
from .data import get_mit_logo, get_popl_logo, get_wizards_logo, get_small_wizards_logo, get_blinker_n

# Import shared GenJAX Research Visualization Standards
from genjax.viz.standard import (
    setup_publication_fonts,
    FIGURE_SIZES,
    get_method_color,
    apply_grid_style,
    apply_standard_ticks,
    save_publication_figure,
    PRIMARY_COLORS,
    LINE_SPECS,
)

# Apply GRVS typography standards
setup_publication_fonts()


def create_timing_bar_plot(timing_data_path="data/gibbs_sweep_timing.json"):
    """
    Create a standalone timing bar plot showing GOL performance across grid sizes.
    
    Args:
        timing_data_path: Path to JSON file containing timing data
        
    Returns:
        matplotlib.figure.Figure: The timing bar plot
    """
    # Try to load timing data
    if os.path.exists(timing_data_path):
        with open(timing_data_path, 'r') as f:
            data = json.load(f)
        
        grid_sizes = data['grid_sizes']
        times_mean = data['times_mean']
        times_std = data['times_std']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar plot
        y_positions = np.arange(len(grid_sizes))
        bars = ax.barh(y_positions, times_mean, xerr=times_std,
                      color=get_method_color("genjax_is"), alpha=0.8,
                      capsize=5, error_kw={'linewidth': 2})
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{size}×{size}" for size in grid_sizes])
        ax.set_xlabel("Time per Gibbs sweep (seconds)", fontweight='bold')
        ax.set_ylabel("Grid size", fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, times_mean, times_std)):
            ax.text(bar.get_width() + std + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{mean:.3f}±{std:.3f}s', va='center', fontsize=12)
        
        # Apply GRVS styling
        apply_grid_style(ax)
        ax.set_xlim(0, max(times_mean) * 1.3)
        
        # Add annotation about JIT compilation
        ax.text(0.95, 0.05, "Times include JIT compilation",
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=12, style='italic', alpha=0.7)
    else:
        # Create placeholder if data not available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Timing data not available\nRun: pixi run gol-timing-optimized',
               ha='center', va='center', fontsize=18, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    return fig


def create_integrated_showcase_figure(
    pattern_type="wizards", size=256, chain_length=500, flip_prob=0.03, seed=42
):
    """
    Create an integrated showcase figure combining schematic and inference results.
    
    This creates a 2-row figure:
    - Top row: Schematic showing the inverse problem (? → GoL → observed)
    - Bottom row: Actual inference results on the pattern
    
    Args:
        pattern_type: Type of pattern to use ("wizards", "mit", "popl", "blinker")
        size: Size of the pattern grid
        chain_length: Number of Gibbs sampling steps
        flip_prob: Probability of rule violations
        seed: Random seed for reproducibility
        
    Returns:
        matplotlib.figure.Figure: The integrated showcase figure
    """
    # Load the appropriate pattern
    if pattern_type == "wizards":
        # For wizards, we need to handle size
        if size == 1024:
            target_pattern = get_wizards_logo()  # Full size
        else:
            target_pattern = get_small_wizards_logo(size)
    elif pattern_type == "mit":
        if size <= 128:
            from .data import get_small_mit_logo
            target_pattern = get_small_mit_logo(size)
        else:
            target_pattern = get_mit_logo()
    elif pattern_type == "popl":
        if size <= 128:
            from .data import get_small_popl_logo
            target_pattern = get_small_popl_logo(size)
        else:
            target_pattern = get_popl_logo()
    elif pattern_type == "blinker":
        target_pattern = get_blinker_n(size)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Run Gibbs sampling
    sampler = GibbsSampler(target_pattern, p_flip=flip_prob)
    key = jrand.key(seed)
    summary = run_sampler_and_get_summary(key, sampler, chain_length, 1)
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(20, 10))
    
    # Create grid: 2 rows with different heights
    gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=0.15,
                               height_ratios=[0.4, 1])  # Schematic smaller
    
    # === TOP ROW: SCHEMATIC ===
    gs_schematic = gs_main[0].subgridspec(1, 3, wspace=0.1,
                                         width_ratios=[1, 0.8, 1])
    
    ax_unknown = fig.add_subplot(gs_schematic[0])
    ax_arrows = fig.add_subplot(gs_schematic[1])
    ax_observed = fig.add_subplot(gs_schematic[2])
    
    # Unknown state (?)
    ax_unknown.set_xlim(0, 1)
    ax_unknown.set_ylim(0, 1)
    ax_unknown.text(0.5, 0.5, '?', fontsize=120, ha='center', va='center',
                   fontweight='bold', color='darkgray')
    ax_unknown.set_xticks([])
    ax_unknown.set_yticks([])
    ax_unknown.set_aspect('equal')
    
    # Add green dashed border
    for spine in ax_unknown.spines.values():
        spine.set_color('green')
        spine.set_linewidth(4)
        spine.set_linestyle('--')
    
    # Arrows
    ax_arrows.axis('off')
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(0, 1)
    
    # Bidirectional arrows
    ax_arrows.annotate('', xy=(0.15, 0.65), xytext=(0.85, 0.65),
                      arrowprops=dict(arrowstyle='->', lw=5, color='green'))
    ax_arrows.annotate('', xy=(0.85, 0.35), xytext=(0.15, 0.35),
                      arrowprops=dict(arrowstyle='->', lw=5, color='black'))
    
    ax_arrows.text(0.5, 0.75, 'inference', ha='center', va='center',
                  fontsize=20, fontweight='bold', color='green')
    ax_arrows.text(0.5, 0.25, '@gen GoL', ha='center', va='center',
                  fontsize=20, fontweight='bold', color='black')
    
    # Observed state (small version of target)
    # Downsample for visibility
    downsample_factor = max(1, size // 64)
    small_target = target_pattern[::downsample_factor, ::downsample_factor]
    
    ax_observed.imshow(small_target, cmap='gray_r', interpolation='nearest')
    ax_observed.set_xticks([])
    ax_observed.set_yticks([])
    ax_observed.set_aspect('equal')
    
    # Add red solid border
    for spine in ax_observed.spines.values():
        spine.set_color('red')
        spine.set_linewidth(4)
    
    # === BOTTOM ROW: INFERENCE RESULTS ===
    gs_inference = gs_main[1].subgridspec(1, 3, wspace=0.1,
                                         width_ratios=[1, 1, 1])
    
    ax_target = fig.add_subplot(gs_inference[0])
    ax_gibbs = fig.add_subplot(gs_inference[1])
    ax_perf = fig.add_subplot(gs_inference[2])
    
    # Target state
    ax_target.imshow(target_pattern, cmap='gray_r', interpolation='nearest')
    ax_target.set_title('Observed Future State', fontsize=24, fontweight='bold', pad=20)
    ax_target.set_xticks([])
    ax_target.set_yticks([])
    
    # Add red border
    for spine in ax_target.spines.values():
        spine.set_color('red')
        spine.set_linewidth(4)
    
    # Gibbs chain visualization (2x4 grid of samples)
    n_samples_to_show = 8
    sample_indices = np.linspace(0, chain_length-1, n_samples_to_show, dtype=int)
    
    gs_samples = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=ax_gibbs.get_subplotspec(),
                                                  wspace=0.05, hspace=0.15)
    ax_gibbs.axis('off')
    
    for i, idx in enumerate(sample_indices):
        row = i // 4
        col = i % 4
        ax_sample = fig.add_subplot(gs_samples[row, col])
        
        # Get the inferred state at this point in the chain
        inferred_state = summary.inferred_image_history[idx]
        ax_sample.imshow(inferred_state, cmap='gray_r', interpolation='nearest')
        ax_sample.set_xticks([])
        ax_sample.set_yticks([])
        
        # Add label
        ax_sample.text(0.05, 0.95, f't={idx}', transform=ax_sample.transAxes,
                      fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.2',
                                                      facecolor='white', alpha=0.7))
        
        # Color first and last differently
        if i == 0:
            for spine in ax_sample.spines.values():
                spine.set_color('green')
                spine.set_linewidth(2)
                spine.set_linestyle('--')
        elif i == n_samples_to_show - 1:
            for spine in ax_sample.spines.values():
                spine.set_color('green')
                spine.set_linewidth(2)
    
    # Add title
    ax_gibbs.text(0.5, 1.08, 'Vectorized Gibbs Chain', transform=ax_gibbs.transAxes,
                 fontsize=24, fontweight='bold', ha='center')
    
    # Performance plot (timing or error)
    # Show reconstruction accuracy over time
    error_history = []
    for t in range(0, chain_length, 10):
        inferred = summary.inferred_image_history[t]
        error_rate = np.mean(inferred != target_pattern)
        error_history.append(error_rate)
    
    timesteps = list(range(0, chain_length, 10))
    ax_perf.plot(timesteps, error_history, linewidth=3, color=get_method_color("genjax_is"))
    ax_perf.set_xlabel('Gibbs Steps', fontweight='bold')
    ax_perf.set_ylabel('Error Rate', fontweight='bold')
    ax_perf.set_title('Reconstruction Quality', fontsize=24, fontweight='bold', pad=20)
    apply_grid_style(ax_perf)
    ax_perf.set_ylim(0, max(error_history) * 1.1)
    
    # Add final accuracy
    final_accuracy = (1 - error_history[-1]) * 100
    ax_perf.text(0.95, 0.95, f'Final: {final_accuracy:.1f}% accurate',
                transform=ax_perf.transAxes, ha='right', va='top',
                fontsize=16, bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig


def save_integrated_showcase_figure(
    pattern_type="wizards", size=256, chain_length=500, flip_prob=0.03, seed=42
):
    """Save the integrated showcase figure."""
    fig = create_integrated_showcase_figure(
        pattern_type, size, chain_length, flip_prob, seed
    )
    filename = f"figs/gol_integrated_showcase_{pattern_type}_{size}.pdf"
    save_publication_figure(fig, filename)
    print(f"Saved integrated showcase figure: {filename}")
    return fig