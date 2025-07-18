"""
Main entry point for the localization case study - CLEANED VERSION.

This cleaned version only generates the two figures used in the paper:
1. Localization problem explanation (1x4 row)
2. Comprehensive 4-panel SMC methods analysis
"""

import argparse
import os
import time
from datetime import datetime
import json
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import const, Const

from .core import (
    create_simple_3room_world,
    generate_trajectory,
    run_smc_basic,
    run_smc_with_mh,
    run_smc_with_hmc,
    run_smc_with_locally_optimal_big_grid,
    localization_model,
)
from .data import sample_trajectory_with_world
from .figs import (
    plot_localization_problem_explanation,
    plot_smc_method_comparison,
)
from .export import save_benchmark_results, load_benchmark_results


def generate_data(args):
    """Generate experimental data and save to disk."""
    print("\n=== Running Localization Benchmark ===")
    
    # Configuration
    config = {
        "n_timesteps": args.n_timesteps,
        "n_particles": args.n_particles,
        "k_rejuv": args.k_rejuv,
        "n_rays": args.n_rays,
        "timing_repeats": args.timing_repeats,
        "n_particles_big_grid": 5,  # Fixed for locally optimal big grid
        "world_type": "basic",
        "use_mcts": False,
    }
    
    # Create world and generate trajectory
    print("\n1. Setting up world and trajectory...")
    world = create_simple_3room_world()
    key = jrand.key(42)
    
    # Generate trajectory
    (true_poses, observations, true_lidar_distances) = sample_trajectory_with_world(
        world, key, config["n_timesteps"], config["n_rays"]
    )
    print(f"   Generated {len(true_poses)} timesteps with {config['n_rays']}-ray LIDAR")
    
    # Only run SMC comparison if requested
    benchmark_results = {}
    if args.include_smc_comparison:
        print("\n2. Running SMC method comparison...")
        
        # Run each SMC method
        methods = [
            ("smc_basic", run_smc_basic, "Bootstrap Filter"),
            ("smc_hmc", run_smc_with_hmc, "SMC + HMC"),
            ("smc_locally_optimal_big_grid", run_smc_with_locally_optimal_big_grid, "SMC + Locally Optimal (Big Grid)"),
        ]
        
        for method_key, method_fn, method_name in methods:
            print(f"\n   Running {method_name}...")
            
            # Use appropriate particle count
            if method_key == "smc_locally_optimal_big_grid":
                n_particles = config["n_particles_big_grid"]
            else:
                n_particles = config["n_particles"]
            
            result = method_fn(
                observations=observations,
                true_poses=true_poses,
                world=world,
                n_particles=const(n_particles),
                key=jrand.key(method_key.__hash__() % 1000),
                timing_repeats=config["timing_repeats"],
                K=const(config["k_rejuv"]) if method_key != "smc_basic" else None,
            )
            
            # Store results
            benchmark_results[method_key] = {
                "particle_history": result["particle_history"],
                "weight_history": result["weight_history"],
                "timing_stats": result["timing_stats"],
                "diagnostic_weights": result.get("diagnostic_weights", {}),
                "n_particles": n_particles,
            }
            
            print(f"      Average time: {result['timing_stats'][0]*1000:.2f}ms ± {result['timing_stats'][1]*1000:.2f}ms")
    
    # Create data directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"localization_r{config['n_rays']}_p{config['n_particles']}_{config['world_type']}_{timestamp}"
    data_dir = os.path.join("data", experiment_name)
    
    # Save all results
    print(f"\n3. Saving results to {data_dir}...")
    save_benchmark_results(
        data_dir,
        benchmark_results,
        config,
        true_poses,
        observations,
        true_lidar_distances,
        world,
    )
    
    print(f"\nExperiment saved as: {experiment_name}")
    return experiment_name


def plot_figures(args):
    """Generate figures from saved experimental data."""
    # Determine which experiment to load
    if args.experiment_name:
        data_dir = os.path.join("data", args.experiment_name)
    else:
        # Find most recent experiment
        data_dirs = [d for d in os.listdir("data") if d.startswith("localization_")]
        if not data_dirs:
            raise ValueError("No localization experiments found in data/")
        data_dir = os.path.join("data", sorted(data_dirs)[-1])
    
    print(f"\nLoading experiment from: {data_dir}")
    
    # Load data
    (
        benchmark_results,
        config,
        true_poses,
        observations,
        true_lidar_distances,
        world,
    ) = load_benchmark_results(data_dir)
    
    # Create output directory
    figs_dir = args.output_dir
    os.makedirs(figs_dir, exist_ok=True)
    
    # Generate parameter prefix for filenames
    param_prefix = f"localization_r{config['n_rays']}_p{config['n_particles']}_{config['world_type']}"
    
    print("\nGenerating figures...")
    
    # 1. Localization problem explanation (USED IN PAPER)
    print("  - Localization problem explanation (1x4 row)...")
    fig_explain, axes_explain = plot_localization_problem_explanation(
        true_poses, observations, world, n_rays=config["n_rays"]
    )
    filename_explain = f"{param_prefix}_localization_problem_1x4_explanation.pdf"
    fig_explain.savefig(os.path.join(figs_dir, filename_explain), dpi=300, bbox_inches="tight")
    plt.close(fig_explain)
    print(f"    Saved: {filename_explain}")
    
    # 2. Comprehensive 4-panel SMC methods analysis (USED IN PAPER)
    if benchmark_results:
        print("  - Comprehensive 4-panel SMC methods analysis...")
        comparison_filename = f"{param_prefix}_comprehensive_4panel_smc_methods_analysis.pdf"
        comparison_path = os.path.join(figs_dir, comparison_filename)
        plot_smc_method_comparison(
            benchmark_results,
            true_poses,
            world,
            save_path=comparison_path,
            n_rays=config["n_rays"],
            n_particles=config["n_particles"],
            K=config["k_rejuv"],
            n_particles_big_grid=config.get("n_particles_big_grid", 5),
        )
        print(f"    Saved: {comparison_filename}")
    
    print("\nFigure generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Localization case study - cleaned version")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate data command
    gen_parser = subparsers.add_parser("generate-data", help="Generate experimental data")
    gen_parser.add_argument(
        "--n-timesteps", type=int, default=17, help="Number of timesteps"
    )
    gen_parser.add_argument(
        "--n-particles", type=int, default=200, help="Number of particles"
    )
    gen_parser.add_argument(
        "--k-rejuv", type=int, default=20, help="Number of MCMC rejuvenation steps"
    )
    gen_parser.add_argument(
        "--n-rays", type=int, default=8, help="Number of LIDAR rays"
    )
    gen_parser.add_argument(
        "--timing-repeats", type=int, default=20, help="Number of timing repetitions"
    )
    gen_parser.add_argument(
        "--include-smc-comparison",
        action="store_true",
        help="Include SMC comparison (adds computation time)",
    )
    gen_parser.add_argument(
        "--experiment-name", type=str, help="Custom experiment name"
    )
    
    # Plot figures command
    plot_parser = subparsers.add_parser("plot-figures", help="Plot figures from saved data")
    plot_parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name to load (defaults to most recent)",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default="figs",
        help="Output directory for figures",
    )
    
    # Full pipeline (both generate and plot)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (generate data then plot figures)",
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.full or args.command is None:
        # Run full pipeline with default settings
        class Args:
            n_timesteps = 17
            n_particles = 200
            k_rejuv = 20
            n_rays = 8
            timing_repeats = 20
            include_smc_comparison = True
            experiment_name = None
            output_dir = "figs"
        
        full_args = Args()
        experiment_name = generate_data(full_args)
        full_args.experiment_name = experiment_name
        plot_figures(full_args)
    elif args.command == "generate-data":
        generate_data(args)
    elif args.command == "plot-figures":
        plot_figures(args)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # Use non-interactive backend
    main()