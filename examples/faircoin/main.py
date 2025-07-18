"""Command-line interface for fair coin - CLEANED VERSION.

This cleaned version only generates the figure used in the POPL paper.
"""

import argparse
from examples.faircoin.figs import combined_comparison_fig


def main():
    """Main CLI entry point for fair coin - CLEANED VERSION."""
    parser = argparse.ArgumentParser(
        description="Fair coin (Beta-Bernoulli) - CLEANED VERSION (POPL paper figure only)"
    )
    parser.add_argument(
        "--num-obs", type=int, default=50, help="Number of observations (default: 50)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,  # Changed default to 2000 for paper
        help="Number of importance samples (default: 2000)",
    )

    args = parser.parse_args()

    print("Fair Coin (Beta-Bernoulli) Analysis - CLEANED VERSION")
    print("=" * 55)
    print("📊 Generating only the figure used in POPL paper")
    print("Configuration:")
    print(f"  - Observations: {args.num_obs}")
    print(f"  - Importance samples: {args.num_samples}")
    print()
    
    # Generate the combined figure used in the paper
    print("Generating combined posterior and timing comparison...")
    combined_comparison_fig(
        num_obs=args.num_obs,
        num_samples=args.num_samples,
    )
    
    print("\n✨ Done!")
    print("Generated figure:")
    print(f"  - faircoin_combined_posterior_and_timing_obs{args.num_obs}_samples{args.num_samples}.pdf")


if __name__ == "__main__":
    main()