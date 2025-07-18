"""
Main entry point for the Game of Life case study - CLEANED VERSION.

This cleaned version only generates the two figures used in the paper:
1. Integrated showcase figure (wizards, 1024x1024)
2. Gibbs timing bar plot
"""

import argparse
from .figs import (
    save_integrated_showcase_figure,
    create_timing_bar_plot,
    save_publication_figure,
)


def main():
    parser = argparse.ArgumentParser(
        description="Game of Life Gibbs sampling case study - cleaned version"
    )
    parser.add_argument(
        "--mode",
        choices=["showcase", "timing", "all"],
        default="all",
        help="Which figures to generate",
    )
    parser.add_argument(
        "--pattern-type",
        choices=["wizards", "mit", "popl", "blinker"],
        default="wizards",
        help="Pattern type for showcase",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Grid size for showcase (default: 1024 for paper figure)",
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=500,
        help="Number of Gibbs steps for showcase",
    )
    parser.add_argument(
        "--flip-prob",
        type=float,
        default=0.03,
        help="Probability of rule violations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=== Game of Life Case Study (Cleaned) ===")
    print(f"Mode: {args.mode}")

    if args.mode in ["showcase", "all"]:
        print(f"\nGenerating integrated showcase figure ({args.pattern_type}, {args.size}x{args.size})...")
        save_integrated_showcase_figure(
            pattern_type=args.pattern_type,
            size=args.size,
            chain_length=args.chain_length,
            flip_prob=args.flip_prob,
            seed=args.seed,
        )
        print(f"✓ Saved: gol_integrated_showcase_{args.pattern_type}_{args.size}.pdf")

    if args.mode in ["timing", "all"]:
        print("\nGenerating Gibbs timing bar plot...")
        fig = create_timing_bar_plot()
        filename = "figs/gol_gibbs_timing_bar_plot.pdf"
        save_publication_figure(fig, filename)
        print(f"✓ Saved: {filename}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()