"""
Visualization functions for the localization case study - CLEANED VERSION.

This cleaned version only contains the visualization functions needed for 
the two figures used in the paper:
1. plot_localization_problem_explanation() 
2. plot_smc_method_comparison()

Plus their dependencies: plot_world(), plot_pose(), plot_lidar_rays()
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List
from .core import Pose, World

# Import shared GenJAX Research Visualization Standards
from genjax.viz.standard import (
    setup_publication_fonts, FIGURE_SIZES, get_method_color,
    apply_grid_style, set_minimal_ticks, apply_standard_ticks, save_publication_figure,
    SMC_METHOD_COLORS, PRIMARY_COLORS, MARKER_SPECS, LINE_SPECS
)

# Apply GRVS typography standards
setup_publication_fonts()


def plot_world(world: World, ax=None, room_label_fontsize=14, show_room_labels=False):
    """Plot the world boundaries and internal walls."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Draw world boundaries
    rect = plt.Rectangle(
        (0, 0), world.width, world.height, fill=False, edgecolor="black", linewidth=3
    )
    ax.add_patch(rect)

    # Draw internal walls
    if world.num_walls > 0:
        for i in range(world.num_walls):
            ax.plot(
                [world.walls_x1[i], world.walls_x2[i]],
                [world.walls_y1[i], world.walls_y2[i]],
                color="black",
                linewidth=2,
                alpha=0.8,
            )

    # Add room labels for clarity (optional)
    if world.num_walls > 0 and show_room_labels:
        # Assuming the 3-room layout we designed
        ax.text(
            2,
            8,
            "Room 1",
            fontsize=room_label_fontsize,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax.text(
            6,
            8,
            "Room 2",
            fontsize=room_label_fontsize,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3),
        )
        ax.text(
            10,
            8,
            "Room 3",
            fontsize=room_label_fontsize,
            ha="center",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3),
        )

    ax.set_xlim(-0.5, world.width + 0.5)
    ax.set_ylim(-0.5, world.height + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    return ax


def plot_pose(
    pose: Pose,
    ax=None,
    color="red",
    label=None,
    arrow_length=0.5,
    marker="o",
    show_arrow=True,
    markersize=100,
):
    """Plot a single robot pose with position and heading."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot position
    ax.scatter(pose.x, pose.y, s=markersize, c=color, marker=marker, label=label, zorder=10)

    # Plot heading arrow
    if show_arrow:
        dx = arrow_length * jnp.cos(pose.theta)
        dy = arrow_length * jnp.sin(pose.theta)
        ax.arrow(
            pose.x,
            pose.y,
            dx,
            dy,
            head_width=0.2,
            head_length=0.1,
            fc=color,
            ec=color,
            zorder=10,
        )

    return ax


def plot_lidar_rays(
    pose: Pose,
    world: World,
    distances=None,
    ax=None,
    n_angles=8,
    ray_color="gray",
    measurement_color="red",
    show_rays=True,
    ray_alpha=0.5,
):
    """Plot LIDAR rays from robot pose."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Generate ray angles relative to robot heading
    angles = jnp.linspace(0, 2 * jnp.pi, n_angles, endpoint=False)
    
    for i, angle in enumerate(angles):
        # Ray direction in world coordinates
        world_angle = pose.theta + angle
        
        # If distances provided, use them; otherwise show max range
        if distances is not None:
            dist = distances[i]
        else:
            dist = jnp.sqrt(world.width**2 + world.height**2)  # Max possible
        
        # End point of ray
        end_x = pose.x + dist * jnp.cos(world_angle)
        end_y = pose.y + dist * jnp.sin(world_angle)
        
        # Plot ray
        if show_rays:
            ax.plot(
                [pose.x, end_x],
                [pose.y, end_y],
                color=ray_color,
                alpha=ray_alpha,
                linewidth=1,
            )
        
        # Plot measurement point
        if distances is not None:
            ax.scatter(
                end_x, end_y, s=30, c=measurement_color, marker='o', zorder=5
            )
    
    return ax


def plot_localization_problem_explanation(
    true_poses: List[Pose],
    observations: jnp.ndarray,
    world: World,
    save_path: str = None,
    timesteps: List[int] = None,
    n_rays: int = 8,
):
    """Create a 1x4 row explaining the localization problem with LIDAR measurements.
    
    Shows the robot at different timesteps with LIDAR rays and measurements,
    illustrating how the robot moves through the environment and senses walls.
    
    Args:
        true_poses: List of true robot poses
        observations: Array of LIDAR observations (shape: [T, n_rays])
        world: World object with walls
        save_path: Optional path to save figure
        timesteps: Which timesteps to show (default: [0, 5, 10, 15])
        n_rays: Number of LIDAR rays
    """
    if timesteps is None:
        # Select 4 evenly spaced timesteps
        n_poses = len(true_poses)
        timesteps = [0, n_poses // 3, 2 * n_poses // 3, n_poses - 1]
        # Ensure timesteps are valid
        timesteps = [min(t, n_poses - 1) for t in timesteps]
    
    # Create 1x4 subplot with GRVS styling (single row)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    for idx, t in enumerate(timesteps):
        ax = axes[idx]
        pose = true_poses[t]
        obs = observations[t]
        
        # Plot world
        plot_world(world, ax)
        
        # Remove axis frames for clean visualization
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Plot trajectory up to this point with fading effect
        if t > 0:
            for i in range(t):
                alpha = 0.2 + 0.6 * (i / t)  # Fade from 0.2 to 0.8
                if i > 0:
                    ax.plot([true_poses[i-1].x, true_poses[i].x],
                           [true_poses[i-1].y, true_poses[i].y],
                           color='gray', linewidth=2, alpha=alpha, zorder=2)
        
        # Plot LIDAR rays and measurements
        from .core import distance_to_wall_lidar
        true_distances = distance_to_wall_lidar(pose, world, n_angles=n_rays)
        
        # Plot true LIDAR rays (what the robot actually sees)
        plot_lidar_rays(
            pose,
            world,
            distances=true_distances,
            ax=ax,
            n_angles=n_rays,
            ray_color="darkgray",
            measurement_color="black",
            show_rays=True,
            ray_alpha=0.3,
        )
        
        # Plot noisy observations as orange dots
        plot_lidar_rays(
            pose,
            world,
            distances=obs,
            ax=ax,
            n_angles=n_rays,
            ray_color="orange",
            measurement_color="orange",
            show_rays=False,  # Don't show rays for noisy observations
        )
        
        # Plot robot pose prominently
        plot_pose(
            pose,
            ax,
            color="red",
            arrow_length=0.6,
            marker="o",
            show_arrow=True,
            markersize=150,
        )
        
        # Add timestep label
        ax.text(0.05, 0.95, f"t = {t}", 
                transform=ax.transAxes,
                fontsize=18, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add simple legend only in first subplot
        if idx == 0:
            # Create proxy artists for legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=10, label='True distances'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=10, label='Noisy observations'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, label='Robot'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=14,
                     frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        save_publication_figure(fig, save_path)
    
    return fig, axes


def plot_smc_method_comparison(
    benchmark_results,
    true_poses,
    world,
    save_path=None,
    n_rays=8,
    n_particles=200,
    K=20,
    n_particles_big_grid=5,
    include_ess_row=False,
    include_legend=False,
):
    """Create comprehensive comparison plot for different SMC methods.
    
    Creates a 4-row visualization showing:
    1. Initial particle distributions
    2. Final particle evolution 
    3. ESS raincloud plots
    4. Timing comparison
    """
    # Define colors first
    colors = {
        "smc_basic": "#1f77b4",
        "smc_hmc": "#2ca02c",
        "smc_locally_optimal_big_grid": "#ff7f0e",
    }

    # Filter out smc_locally_optimal from results
    filtered_results = {k: v for k, v in benchmark_results.items() if k != "smc_locally_optimal"}
    
    # Only count methods we have colors for
    valid_methods = [name for name in filtered_results.keys() if name in colors]
    n_methods = len(valid_methods)

    # Create 4-row layout
    fig = plt.figure(figsize=(6 * n_methods, 20))
    gs_main = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.3], hspace=-0.2)
    gs_first = gs_main[0].subgridspec(1, n_methods, hspace=0.05)  # Initial particles
    gs_second = gs_main[1].subgridspec(1, n_methods, hspace=0.05)  # Final evolution
    gs_third = gs_main[2].subgridspec(1, n_methods, hspace=0.05)   # Raincloud plots
    gs_bottom = gs_main[3]  # Timing

    method_labels = {}
    for method_name, result in filtered_results.items():
        if method_name in colors:
            # Use n_particles_big_grid for the big grid variant
            if method_name == "smc_locally_optimal_big_grid":
                method_n_particles = n_particles_big_grid
            else:
                method_n_particles = result.get("n_particles", n_particles)
            
            if method_name == "smc_basic":
                method_labels[method_name] = f"Bootstrap filter\n(N={method_n_particles})"
            elif method_name == "smc_hmc":
                method_labels[method_name] = f"SMC (N={method_n_particles})\n+ HMC (K=25)"
            elif method_name == "smc_locally_optimal_big_grid":
                method_labels[method_name] = f"SMC (N={method_n_particles})\n+ Locally Optimal (L=25)"

    # Extract timing data early to sort methods by speed
    timing_data = {}
    for method_name, result in filtered_results.items():
        if method_name in colors:
            timing_mean = float(result["timing_stats"][0]) * 1000  # Convert to milliseconds
            timing_data[method_name] = timing_mean
    
    # Keep original order for particle plots (as they appear in colors dict)
    original_order = [name for name in colors.keys() if name in filtered_results]
    
    # Process methods in original order for particle plots
    method_items = [
        (name, filtered_results[name]) for name in original_order
    ]

    # Plot each method
    for i, (method_name, result) in enumerate(method_items):
        particle_history = result["particle_history"]
        weight_history = result["weight_history"]
        diagnostic_weights = result["diagnostic_weights"]

        # Row 1: Initial particles with method title
        ax_initial = fig.add_subplot(gs_first[i])
        plot_world(world, ax_initial)
        
        # Clean visualization
        ax_initial.grid(False)
        for spine in ax_initial.spines.values():
            spine.set_visible(False)
        ax_initial.set_xticks([])
        ax_initial.set_yticks([])
        ax_initial.set_xlabel("")
        ax_initial.set_ylabel("")
        
        # Plot initial particles
        initial_particles = particle_history[0]
        initial_weights = weight_history[0]
        
        xs = [p.x for p in initial_particles]
        ys = [p.y for p in initial_particles]
        
        ax_initial.scatter(
            xs, ys, s=50, alpha=0.6, c=range(len(initial_particles)),
            cmap="viridis", edgecolors="black", linewidth=0.3
        )
        
        # Add Start label
        if i == 0:
            ax_initial.text(-3.5, 5, "Start", fontsize=24, fontweight='bold',
                          ha='center', va='center', rotation=90)
        
        # Add method title
        ax_initial.text(0.5, 1.05, method_labels[method_name],
                       transform=ax_initial.transAxes,
                       fontsize=18, fontweight='bold',
                       ha='center', va='bottom')

        # Row 2: Particle evolution (blended)
        ax_particles = fig.add_subplot(gs_second[i])
        plot_world(world, ax_particles)
        
        ax_particles.grid(False)
        for spine in ax_particles.spines.values():
            spine.set_visible(False)
        ax_particles.set_xticks([])
        ax_particles.set_yticks([])
        ax_particles.set_xlabel("")
        ax_particles.set_ylabel("")
        
        # Plot particles from multiple timesteps with alpha blending
        n_blend_steps = len(particle_history)
        start_idx = 0
        
        for blend_idx, t in enumerate(range(start_idx, len(particle_history))):
            particles = particle_history[t]
            weights = weight_history[t]
            
            alpha = 0.05 + (0.75 * blend_idx / max(1, n_blend_steps - 1))
            
            if weights is not None and jnp.sum(weights) > 0:
                weights_norm = weights / jnp.sum(weights)
                normalized_weights = weights_norm / jnp.max(weights_norm)
                sizes = 50 * normalized_weights
            else:
                sizes = 5
            
            xs = [p.x for p in particles]
            ys = [p.y for p in particles]
            
            ax_particles.scatter(
                xs, ys, s=sizes, alpha=alpha, c=range(len(particles)),
                cmap="viridis", edgecolors="black", linewidth=0.3,
                zorder=3 + blend_idx
            )
        
        # Plot true trajectory
        if len(true_poses) > start_idx:
            true_xs = [true_poses[t].x for t in range(start_idx, min(len(true_poses), len(particle_history)))]
            true_ys = [true_poses[t].y for t in range(start_idx, min(len(true_poses), len(particle_history)))]
            
            for j in range(len(true_xs) - 1):
                blend_factor = j / max(1, len(true_xs) - 2)
                alpha = 0.2 + (0.6 * blend_factor)
                ax_particles.plot(
                    [true_xs[j], true_xs[j + 1]],
                    [true_ys[j], true_ys[j + 1]],
                    color="red", linewidth=2, alpha=alpha, zorder=19
                )
        
        # Plot final true pose
        if len(true_poses) > 0:
            plot_pose(
                true_poses[-1], ax_particles, color="red",
                label="True Pose", arrow_length=0.5, marker="x",
                show_arrow=True, markersize=200
            )
        
        # Add End label
        if i == 0:
            ax_particles.text(-3.5, 5, "End", fontsize=24, fontweight='bold',
                            ha='center', va='center', rotation=90)

        # Row 3: Raincloud plots
        ax_rain = fig.add_subplot(gs_third[i])
        
        # Import raincloud plotting
        from genjax.viz.raincloud import plot_rainclouds
        
        # Prepare data for raincloud
        ess_percentages = []
        timesteps = []
        
        for t, weights in enumerate(weight_history):
            if weights is not None and len(weights) > 0 and jnp.sum(weights) > 0:
                weights_norm = weights / jnp.sum(weights)
                ess = 1.0 / jnp.sum(weights_norm ** 2)
                ess_pct = (ess / len(weights)) * 100
                ess_percentages.append(float(ess_pct))
                timesteps.append(t)
        
        # Create raincloud plot
        plot_rainclouds(
            np.array(timesteps),
            np.array(ess_percentages),
            ax_rain,
            color="#404040",  # Grayscale
            violin_width=0.6,
            point_size=40
        )
        
        ax_rain.set_xlabel("Timestep", fontsize=16, fontweight='bold')
        if i == 0:
            ax_rain.set_ylabel("ESS %", fontsize=16, fontweight='bold')
        ax_rain.set_ylim(0, 105)
        
        # Add ESS percentage annotation
        if ess_percentages:
            mean_ess = np.mean(ess_percentages)
            ax_rain.text(0.5, 0.9, f"ESS: {mean_ess:.0f}%",
                        transform=ax_rain.transAxes,
                        fontsize=16, fontweight='bold',
                        ha='center', va='top')

    # Row 4: Timing comparison (bottom)
    ax_timing = fig.add_subplot(gs_bottom)
    
    sorted_methods = sorted(timing_data.keys(), key=lambda x: timing_data[x])
    
    y_positions = np.arange(len(sorted_methods))
    timing_means = [timing_data[m] for m in sorted_methods]
    timing_stds = [float(filtered_results[m]["timing_stats"][1]) * 1000 for m in sorted_methods]
    
    bars = ax_timing.barh(
        y_positions, timing_means, xerr=timing_stds,
        color=[colors[m] for m in sorted_methods],
        alpha=0.8, capsize=5, error_kw={'linewidth': 2}
    )
    
    ax_timing.set_yticks(y_positions)
    ax_timing.set_yticklabels([method_labels[m] for m in sorted_methods], fontsize=14)
    ax_timing.set_xlabel("Execution Time (ms)", fontsize=16, fontweight='bold')
    ax_timing.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, timing_means, timing_stds):
        ax_timing.text(bar.get_width() + std + 1, bar.get_y() + bar.get_height() / 2,
                      f'{mean:.1f}±{std:.1f}', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_publication_figure(fig, save_path)
    
    return fig