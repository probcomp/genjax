"""
GenJAX Curvefit Case Study - CLEANED Main Entry Point

This cleaned version only supports modes that generate the 5 figures used in the POPL paper.
The 6th figure (curvefit_vectorization_illustration.pdf) is stored in images/.
"""

import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GenJAX Curvefit Case Study - CLEANED VERSION (POPL paper figures only)"
    )

    parser.add_argument(
        "mode",
        choices=["all", "traces", "scaling", "outlier"],
        nargs="?",
        default="all",
        help="Analysis mode: all (generate all 5 figures), traces (trace density figures), scaling (performance analysis), outlier (detection comparison)",
    )

    # Analysis parameters
    parser.add_argument(
        "--n-points", type=int, default=10, help="Number of data points (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials for scaling analysis (default: 100)"
    )

    return parser.parse_args()


def run_traces_mode(args):
    """Generate trace density visualizations."""
    from figs_cleaned import (
        save_multiple_multipoint_traces_with_density,
        save_single_multipoint_trace_with_density,
    )

    print("=== Traces Mode: Generating trace density figures ===")

    print("\n1. Generating prior multipoint traces with density...")
    save_multiple_multipoint_traces_with_density()

    print("\n2. Generating single multipoint trace with density...")
    save_single_multipoint_trace_with_density()

    print("\n✓ Traces mode complete!")
    print("Generated figures:")
    print("  - curvefit_prior_multipoint_traces_density.pdf")
    print("  - curvefit_single_multipoint_trace_density.pdf")


def run_scaling_mode(args):
    """Generate scaling performance analysis."""
    from figs_cleaned import (
        save_inference_scaling_viz,
        save_posterior_scaling_plots,
    )

    print("=== Scaling Mode: Performance analysis ===")

    print("\n1. Generating inference scaling visualization...")
    save_inference_scaling_viz(n_trials=args.n_trials)

    print("\n2. Generating posterior scaling analysis...")
    save_posterior_scaling_plots(seed=args.seed)

    print("\n✓ Scaling mode complete!")
    print("Generated figures:")
    print("  - curvefit_scaling_performance.pdf")
    print("  - curvefit_posterior_scaling_combined.pdf")


def run_outlier_mode(args):
    """Generate outlier detection comparison."""
    from figs_cleaned import save_outlier_detection_comparison

    print("=== Outlier Mode: Detection comparison ===")

    print("\nGenerating outlier detection comparison (IS vs Gibbs+HMC)...")
    save_outlier_detection_comparison()

    print("\n✓ Outlier mode complete!")
    print("Generated figure:")
    print("  - curvefit_outlier_detection_comparison.pdf")
    print("\nThis figure demonstrates:")
    print("  - GenJAX's Cond combinator for natural mixture modeling")
    print("  - Improved robustness with automatic outlier detection")
    print("  - Comparison of inference algorithms on challenging problems")


def run_all_mode(args):
    """Generate all 5 figures used in the paper."""
    print("=== All Mode: Generating all 5 POPL paper figures ===")
    print("Note: curvefit_vectorization_illustration.pdf is stored in images/")
    
    # Run all modes
    run_traces_mode(args)
    print()
    run_scaling_mode(args)
    print()
    run_outlier_mode(args)
    
    print("\n✨ All figures generated successfully!")
    print("\nComplete list of generated figures:")
    print("  1. curvefit_prior_multipoint_traces_density.pdf")
    print("  2. curvefit_single_multipoint_trace_density.pdf")
    print("  3. curvefit_scaling_performance.pdf")
    print("  4. curvefit_posterior_scaling_combined.pdf")
    print("  5. curvefit_outlier_detection_comparison.pdf")
    print("\n(Plus curvefit_vectorization_illustration.pdf from images/)")


def main():
    """Main entry point."""
    args = parse_args()

    print("\n🚀 GenJAX Curvefit Case Study - CLEANED VERSION")
    print("📊 Generating only figures used in POPL paper")
    print(f"Mode: {args.mode}")

    if args.mode == "all":
        run_all_mode(args)
    elif args.mode == "traces":
        run_traces_mode(args)
    elif args.mode == "scaling":
        run_scaling_mode(args)
    elif args.mode == "outlier":
        run_outlier_mode(args)

    print("\n✨ Done!")
    print("Figures saved in examples/curvefit/figs/")


if __name__ == "__main__":
    main()