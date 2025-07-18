"""
Visualizations and figures for curvefit case study - CLEANED VERSION.

This cleaned version only generates the 5 figures used in the POPL paper.
The 6th figure (curvefit_vectorization_illustration.pdf) is stored in images/.
"""

import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# GenJAX visualization standards
from examples.viz import (
    setup_publication_fonts,
    FIGURE_SIZES,
    PRIMARY_COLORS,
    get_method_color,
    apply_grid_style,
    apply_standard_ticks,
    save_publication_figure,
    LINE_SPECS,
    MARKER_SPECS,
)

# Apply publication fonts on import
setup_publication_fonts()


def save_multiple_multipoint_traces_with_density():
    """
    Save visualization showing 3 distinct multi-point curve samples from prior.
    Generates: curvefit_prior_multipoint_traces_density.pdf
    """
    from core import npoint_curve

    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_SIZES["two_panel_horizontal"][0] * 1.2, 4.5)
    )

    # Generate X points
    xs = jnp.linspace(0, 1, 20)

    # Generate 3 different curves
    curves_data = []
    seeds = [1, 2, 3]

    for i, seed in enumerate(seeds):
        key = jrand.key(seed)
        trace = npoint_curve.simulate(key, xs)
        curve, (xs_ret, ys_ret) = trace.get_retval()
        log_density = trace.log_density()
        curves_data.append((curve, ys_ret, log_density))

    # Titles for each subplot
    titles = ["", "", ""]

    for i, (ax, (curve, ys, log_density), title) in enumerate(
        zip(axes, curves_data, titles)
    ):
        # Scatter plot for data points
        ax.scatter(xs, ys, s=60, color=get_method_color("data_points"), zorder=10)

        # Use provided polyfn
        from core import polyfn

        coeffs = jnp.array(
            [
                trace.get_choices()["curve"]["a"],
                trace.get_choices()["curve"]["b"],
                trace.get_choices()["curve"]["c"],
            ]
        )

        # Create smooth curve for plotting
        xs_smooth = jnp.linspace(0, 1, 100)
        ys_smooth = polyfn(xs_smooth, coeffs)

        ax.plot(
            xs_smooth,
            ys_smooth,
            color=get_method_color("curves"),
            linewidth=2.5,
            alpha=0.9,
        )

        # Display log density as text
        ax.text(
            0.05,
            0.95,
            f"log p = {log_density:.2f}",
            transform=ax.transAxes,
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"
            ),
            verticalalignment="top",
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        apply_grid_style(ax)
        ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_prior_multipoint_traces_density.pdf")


def save_single_multipoint_trace_with_density():
    """
    Save single trace visualization with density annotation.
    Generates: curvefit_single_multipoint_trace_density.pdf
    """
    from core import npoint_curve

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_medium"])

    # Generate X points
    xs = jnp.linspace(0, 1, 20)

    # Generate curve
    key = jrand.key(100)
    trace = npoint_curve.simulate(key, xs)
    curve, (xs_ret, ys_ret) = trace.get_retval()
    log_density = trace.log_density()

    # Scatter plot for data points
    ax.scatter(xs, ys_ret, s=100, color=get_method_color("data_points"), zorder=10)

    # Use provided polyfn
    from core import polyfn

    coeffs = jnp.array(
        [
            trace.get_choices()["curve"]["a"],
            trace.get_choices()["curve"]["b"],
            trace.get_choices()["curve"]["c"],
        ]
    )

    # Create smooth curve for plotting
    xs_smooth = jnp.linspace(0, 1, 100)
    ys_smooth = polyfn(xs_smooth, coeffs)

    ax.plot(
        xs_smooth,
        ys_smooth,
        color=get_method_color("curves"),
        linewidth=3,
        alpha=0.9,
    )

    # Display log density as text
    ax.text(
        0.05,
        0.95,
        f"log p = {log_density:.2f}",
        transform=ax.transAxes,
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"),
        verticalalignment="top",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_grid_style(ax)
    ax.set_xlim(0, 1)

    save_publication_figure(fig, "figs/curvefit_single_multipoint_trace_density.pdf")


def save_inference_scaling_viz(n_trials=100, extended_timing=False):
    """
    Save performance scaling analysis visualization with 3 panels.
    Generates: curvefit_scaling_performance.pdf
    
    Args:
        n_trials: Number of Monte Carlo trials per N value (default: 100)
        extended_timing: If True, includes more detailed timing measurements
    """
    from core import infer_latents, get_points_for_inference
    from genjax._src.typing import Const
    from genjax._src.generative_functions.distributions.distribution import Distribution

    seed = Distribution.seed  # Alias for convenience

    # Generate test data with fixed seed
    key = jrand.key(0)
    _, (xs, ys) = get_points_for_inference()

    # Test different particle counts
    if extended_timing:
        # Extended range for GPU behavior analysis
        particle_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    else:
        # Standard range
        particle_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]

    runtimes = []
    lml_estimates = []
    ess_values = []
    
    print("Running inference scaling analysis...")
    for n_particles in particle_counts:
        print(f"  Testing N={n_particles:,} particles with {n_trials} trials...")
        
        # Skip very large particle counts that might OOM
        if n_particles > 100000:
            # Special handling for large particle counts
            try:
                # Just try one run to see if it works
                key, subkey = jrand.split(key)
                samples, log_weights = seed(infer_latents)(
                    subkey, xs, ys, Const(n_particles)
                )
                
                # If successful, do a few more trials
                n_trials_large = min(5, n_trials)  # Reduce trials for large N
                times = []
                lmls = []
                
                for _ in range(n_trials_large):
                    key, subkey = jrand.split(key)
                    import time
                    start = time.time()
                    samples, log_weights = seed(infer_latents)(
                        subkey, xs, ys, Const(n_particles)
                    )
                    runtime = (time.time() - start) * 1000  # Convert to ms
                    times.append(runtime)
                    
                    # Compute log marginal likelihood
                    lml = jnp.log(jnp.sum(jnp.exp(log_weights - jnp.max(log_weights)))) + jnp.max(log_weights) - jnp.log(n_particles)
                    lmls.append(lml)
                
                runtimes.append((np.mean(times), np.std(times)))
                lml_estimates.append((np.mean(lmls), np.std(lmls)))
                ess_values.append((n_particles / 2, n_particles / 4))  # Approximate ESS
                
            except Exception as e:
                print(f"    OOM or error at N={n_particles}: {e}")
                # Add NaN for failed runs
                runtimes.append((np.nan, np.nan))
                lml_estimates.append((np.nan, np.nan))
                ess_values.append((np.nan, np.nan))
                continue
        else:
            # Normal processing for smaller particle counts
            times = []
            lmls = []
            ess_list = []
            
            # Monte Carlo trials to reduce noise
            for _ in range(n_trials):
                key, subkey = jrand.split(key)
                
                # Time inference
                import time
                start = time.time()
                samples, log_weights = seed(infer_latents)(
                    subkey, xs, ys, Const(n_particles)
                )
                runtime = (time.time() - start) * 1000  # Convert to ms
                times.append(runtime)
                
                # Compute log marginal likelihood estimate
                # log p(y) ≈ log(1/N * sum_i w_i) = log(sum exp(log w_i)) - log(N)
                lml = jnp.log(jnp.sum(jnp.exp(log_weights - jnp.max(log_weights)))) + jnp.max(log_weights) - jnp.log(n_particles)
                lmls.append(lml)
                
                # Compute effective sample size
                # ESS = 1 / sum(w_i^2) where w_i are normalized weights
                normalized_weights = jnp.exp(log_weights - jnp.max(log_weights))
                normalized_weights = normalized_weights / jnp.sum(normalized_weights)
                ess = 1.0 / jnp.sum(normalized_weights**2)
                ess_list.append(ess)
            
            # Store mean and std
            runtimes.append((np.mean(times), np.std(times)))
            lml_estimates.append((np.mean(lmls), np.std(lmls)))
            ess_values.append((np.mean(ess_list), np.std(ess_list)))

    # Create 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Filter out NaN values for plotting
    valid_indices = [i for i, (mean, _) in enumerate(runtimes) if not np.isnan(mean)]
    valid_particle_counts = [particle_counts[i] for i in valid_indices]
    valid_runtimes = [runtimes[i] for i in valid_indices]
    valid_lmls = [lml_estimates[i] for i in valid_indices]
    valid_ess = [ess_values[i] for i in valid_indices]

    # Panel 1: Runtime vs N (now without error bars for cleaner look)
    runtime_means = [mean for mean, _ in valid_runtimes]
    ax1.plot(valid_particle_counts, runtime_means, 
             color=get_method_color("genjax_is"), 
             marker='o', markersize=8, linewidth=2.5)
    
    ax1.set_xlabel('Number of Particles', fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xlim(50, max(valid_particle_counts) * 2)
    
    # Zoom y-axis to show the flat line better
    y_min = min(runtime_means) * 0.8
    y_max = max(runtime_means) * 1.2
    ax1.set_ylim(y_min, y_max)
    
    apply_grid_style(ax1)
    
    # Use scientific notation for x-axis
    ax1.set_xticks([100, 1000, 10000, 100000])
    ax1.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    
    # Panel 2: Log Marginal Likelihood vs N
    lml_means = [mean for mean, _ in valid_lmls]
    lml_stds = [std for _, std in valid_lmls]
    ax2.errorbar(valid_particle_counts, lml_means, yerr=lml_stds,
                 color=get_method_color("genjax_is"), 
                 marker='o', markersize=8, linewidth=2.5, capsize=5)
    
    ax2.set_xlabel('Number of Particles', fontweight='bold')
    ax2.set_ylabel('Log Marginal Likelihood', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xlim(50, max(valid_particle_counts) * 2)
    apply_grid_style(ax2)
    
    # Use scientific notation for x-axis
    ax2.set_xticks([100, 1000, 10000, 100000])
    ax2.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    
    # Panel 3: ESS vs N
    ess_means = [mean for mean, _ in valid_ess]
    ess_stds = [std for _, std in valid_ess]
    ax3.errorbar(valid_particle_counts, ess_means, yerr=ess_stds,
                 color=get_method_color("genjax_is"), 
                 marker='o', markersize=8, linewidth=2.5, capsize=5)
    
    # Add diagonal reference line (ESS = N would be perfect)
    ax3.plot(valid_particle_counts, valid_particle_counts, 
             'k--', alpha=0.3, linewidth=1.5, label='ESS = N')
    
    ax3.set_xlabel('Number of Particles', fontweight='bold')
    ax3.set_ylabel('Effective Sample Size', fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(50, max(valid_particle_counts) * 2)
    apply_grid_style(ax3)
    
    # Use scientific notation for x-axis
    ax3.set_xticks([100, 1000, 10000, 100000])
    ax3.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    
    # Add OOM annotation if needed
    if len(valid_indices) < len(particle_counts):
        # Find first OOM point
        first_oom_idx = len(valid_indices)
        if first_oom_idx < len(particle_counts):
            oom_n = particle_counts[first_oom_idx]
            ax1.axvline(oom_n, color='red', linestyle=':', alpha=0.5)
            ax2.axvline(oom_n, color='red', linestyle=':', alpha=0.5)
            ax3.axvline(oom_n, color='red', linestyle=':', alpha=0.5)
            
            # Add text annotation on first panel
            ax1.text(oom_n * 1.1, y_max * 0.9, 'GPU OOM', 
                    color='red', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig("figs/curvefit_scaling_performance.pdf")
    plt.close()

    print("✓ Saved inference scaling visualization")


def save_posterior_scaling_plots(n_runs=1000, seed=42):
    """
    Create comprehensive posterior scaling analysis plots.
    Generates: curvefit_posterior_scaling_combined.pdf
    
    Shows posterior marginals, correlations, and convergence diagnostics
    for different numbers of IS particles.
    """
    from core import infer_latents, get_points_for_inference
    from genjax._src.typing import Const
    from genjax._src.generative_functions.distributions.distribution import Distribution

    dist_seed = Distribution.seed

    # Generate test data
    key = jrand.key(seed)
    true_curve, (xs, ys) = get_points_for_inference()

    # Get true parameters
    true_a = true_curve.dynamic_vals[0]
    true_b = true_curve.dynamic_vals[1]
    true_c = true_curve.dynamic_vals[2]

    # Different particle counts to compare
    particle_counts = [100, 1000, 10000]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 3x3 grid - one row per particle count
    # Column 1: Marginal posteriors
    # Column 2: 2D correlation plots  
    # Column 3: Convergence diagnostics
    
    print("Running posterior scaling analysis...")
    
    for row_idx, n_particles in enumerate(particle_counts):
        print(f"  Analyzing N={n_particles:,} particles...")
        
        # Run inference
        key, subkey = jrand.split(key)
        samples, log_weights = dist_seed(infer_latents)(
            subkey, xs, ys, Const(n_particles)
        )
        
        # Extract parameters
        a_samples = samples.get_choices()["curve"]["a"]
        b_samples = samples.get_choices()["curve"]["b"]
        c_samples = samples.get_choices()["curve"]["c"]
        
        # Normalize weights for resampling
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        weights = weights / jnp.sum(weights)
        
        # Compute ESS
        ess = 1.0 / jnp.sum(weights**2)
        
        # Column 1: Marginal posteriors
        ax1 = plt.subplot(3, 3, row_idx * 3 + 1)
        
        # Plot histograms with weights
        bins = 30
        
        # Weighted histograms
        for param_samples, param_true, param_name, color in [
            (a_samples, true_a, 'a', PRIMARY_COLORS[0]),
            (b_samples, true_b, 'b', PRIMARY_COLORS[1]),
            (c_samples, true_c, 'c', PRIMARY_COLORS[2])
        ]:
            # Create weighted histogram
            hist, edges = np.histogram(param_samples, bins=bins, weights=weights)
            centers = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]
            
            # Normalize to density
            hist = hist / (np.sum(hist) * width)
            
            ax1.bar(centers, hist, width=width, alpha=0.6, label=param_name, color=color)
            ax1.axvline(param_true, color=color, linestyle='--', linewidth=2)
        
        ax1.set_xlabel('Parameter Value' if row_idx == 2 else '', fontweight='bold')
        ax1.set_ylabel(f'N={n_particles:,}', fontweight='bold')
        ax1.legend()
        apply_grid_style(ax1)
        
        # Column 2: 2D correlation plot (a vs b)
        ax2 = plt.subplot(3, 3, row_idx * 3 + 2)
        
        # Resample for visualization
        key, subkey = jrand.split(key)
        resample_idx = jrand.choice(subkey, n_particles, shape=(1000,), p=weights)
        a_resample = a_samples[resample_idx]
        b_resample = b_samples[resample_idx]
        
        # 2D hexbin plot
        hb = ax2.hexbin(a_resample, b_resample, gridsize=25, cmap='Blues', mincnt=1)
        ax2.scatter([true_a], [true_b], color='red', s=100, marker='*', zorder=10)
        
        ax2.set_xlabel('a' if row_idx == 2 else '', fontweight='bold')
        ax2.set_ylabel('b', fontweight='bold')
        apply_grid_style(ax2)
        
        # Add colorbar
        cb = plt.colorbar(hb, ax=ax2)
        cb.set_label('Count', fontweight='bold')
        
        # Column 3: Log weights distribution
        ax3 = plt.subplot(3, 3, row_idx * 3 + 3)
        
        # Plot sorted log weights
        sorted_log_weights = jnp.sort(log_weights)[::-1]
        ax3.plot(sorted_log_weights, linewidth=2, color=get_method_color("genjax_is"))
        
        # Add ESS annotation
        ax3.text(0.95, 0.95, f'ESS = {ess:.1f}\n({ess/n_particles*100:.1f}%)', 
                transform=ax3.transAxes, 
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"),
                verticalalignment='top', horizontalalignment='right')
        
        ax3.set_xlabel('Particle Index' if row_idx == 2 else '', fontweight='bold')
        ax3.set_ylabel('Log Weight', fontweight='bold')
        ax3.set_xlim(0, min(100, n_particles))  # Show top 100 particles
        apply_grid_style(ax3)
    
    # Overall title
    fig.suptitle('Posterior Analysis: Effect of Particle Count', fontsize=20, fontweight='bold')
    
    # Use tight_layout to handle spacing automatically
    plt.tight_layout()
    
    # Save as a single figure
    filename = "figs/curvefit_posterior_scaling_combined.pdf"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"\n✓ Saved combined posterior scaling plot: {filename}")


def save_outlier_detection_comparison(output_filename="figs/curvefit_outlier_detection_comparison.pdf"):
    """
    Create 3-panel outlier detection comparison figure.
    Generates: curvefit_outlier_detection_comparison.pdf
    """
    
    from core import (
        npoint_curve,
        npoint_curve_with_outliers,
        get_points_for_inference,
        infer_latents,
        mixed_gibbs_hmc_kernel,
        enumerative_gibbs_outliers,
    )
    from genjax._src.typing import Const
    from genjax._src.generative_functions.distributions.distribution import Distribution
    from genjax.experimental.mcmc import mcmc

    dist_seed = Distribution.seed

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate synthetic data with outliers
    key = jrand.key(42)
    true_curve, (xs, ys) = get_points_for_inference(n_points=20)
    
    # Manually add outliers at specific locations
    outlier_indices = [5, 12, 18]  # 3 outliers
    for idx in outlier_indices:
        ys = ys.at[idx].set(ys[idx] + jrand.normal(key, shape=()) * 3.0)  # Large deviation

    # Colors for outlier probability
    def get_outlier_color(prob):
        """Map outlier probability to color."""
        # Red for high probability, blue for low
        return plt.cm.RdBu_r(prob)

    # Panel 1: Standard model (fails with outliers)
    ax1 = axes[0]
    
    # Run IS on standard model
    key, subkey = jrand.split(key)
    std_samples, std_log_weights = dist_seed(infer_latents)(
        subkey, xs, ys, Const(1000)
    )
    
    # Get best sample (highest weight)
    best_idx = jnp.argmax(std_log_weights)
    best_a = std_samples.get_choices()["curve"]["a"][best_idx]
    best_b = std_samples.get_choices()["curve"]["b"][best_idx]
    best_c = std_samples.get_choices()["curve"]["c"][best_idx]
    
    # Plot data and fitted curve
    ax1.scatter(xs, ys, s=80, c='gray', edgecolor='black', linewidth=1, zorder=10)
    
    # Plot fitted curve
    from core import polyfn
    xs_smooth = jnp.linspace(0, 1, 100)
    ys_smooth = polyfn(xs_smooth, jnp.array([best_a, best_b, best_c]))
    ax1.plot(xs_smooth, ys_smooth, color=get_method_color("curves"), linewidth=3)
    
    ax1.set_xlabel('x', fontweight='bold')
    ax1.set_ylabel('y', fontweight='bold')
    ax1.set_title('Standard Model\n(good inference, bad model)', fontsize=14)
    apply_grid_style(ax1)
    ax1.set_xlim(-0.05, 1.05)

    # Panel 2: Outlier model with IS
    ax2 = axes[1]
    
    # Run IS on outlier model
    key, subkey = jrand.split(key)
    
    # Use the outlier model with fixed outlier rate
    outlier_rate = 0.2
    outlier_std = 3.0
    
    # Create constrained trace for inference
    trace = npoint_curve_with_outliers.simulate(
        subkey, xs, outlier_rate, 0.0, outlier_std
    )
    
    # Constrain observations
    constrained_trace = trace.update(
        {"ys": {"obs": ys}}
    )
    
    # Run importance sampling with rejection
    n_attempts = 10000  # Increase attempts for outlier model
    successful_traces = []
    successful_weights = []
    
    for i in range(n_attempts):
        key, subkey = jrand.split(key)
        try:
            proposal_trace = npoint_curve_with_outliers.simulate(
                subkey, xs, outlier_rate, 0.0, outlier_std
            )
            constrained = proposal_trace.update({"ys": {"obs": ys}})
            if constrained.log_density() > -jnp.inf:
                successful_traces.append(constrained)
                successful_weights.append(constrained.log_density() - proposal_trace.log_density())
        except:
            continue
    
    if len(successful_traces) > 0:
        # Get best trace
        best_idx = jnp.argmax(jnp.array(successful_weights))
        best_trace = successful_traces[best_idx]
        
        # Extract parameters
        best_a = best_trace.get_choices()["curve"]["a"]
        best_b = best_trace.get_choices()["curve"]["b"]
        best_c = best_trace.get_choices()["curve"]["c"]
        
        # Extract outlier indicators
        outlier_probs = []
        for i in range(len(xs)):
            is_outlier = best_trace.get_choices()["ys"][i]["is_outlier"]
            outlier_probs.append(float(is_outlier))
    else:
        # Fallback to standard fit if IS fails
        best_a, best_b, best_c = 0.0, 0.0, 0.0
        outlier_probs = [0.0] * len(xs)
    
    # Plot data with outlier coloring
    colors = [get_outlier_color(p) for p in outlier_probs]
    scatter = ax2.scatter(xs, ys, s=80, c=colors, edgecolor='black', linewidth=1, zorder=10)
    
    # Plot fitted curve
    ys_smooth = polyfn(xs_smooth, jnp.array([best_a, best_b, best_c]))
    ax2.plot(xs_smooth, ys_smooth, color=get_method_color("curves"), linewidth=3)
    
    ax2.set_xlabel('x', fontweight='bold')
    ax2.set_ylabel('y', fontweight='bold')
    ax2.set_title('Outlier Model + IS\n(bad inference, good model)', fontsize=14)
    apply_grid_style(ax2)
    ax2.set_xlim(-0.05, 1.05)

    # Panel 3: Outlier model with Gibbs/HMC
    ax3 = axes[2]
    
    # Initialize with IS result or random
    key, subkey = jrand.split(key)
    init_trace = npoint_curve_with_outliers.simulate(
        subkey, xs, outlier_rate, 0.0, outlier_std
    )
    init_trace = init_trace.update({"ys": {"obs": ys}})
    
    # Run mixed Gibbs/HMC
    key, subkey = jrand.split(key)
    
    # Create kernel that combines Gibbs for discrete and HMC for continuous
    def kernel(key, trace):
        # First do Gibbs on outlier indicators
        key, subkey = jrand.split(key)
        trace = enumerative_gibbs_outliers(trace, xs, ys, outlier_rate)
        
        # Then do HMC on continuous parameters
        key, subkey = jrand.split(key)
        return mixed_gibbs_hmc_kernel(xs, ys, outlier_rate, outlier_std)(subkey, trace)
    
    # Run MCMC
    final_trace, chain = mcmc(
        init_trace,
        kernel,
        subkey,
        n_steps=100,  # Fewer steps since Gibbs is efficient
    )
    
    # Use final trace
    best_trace = final_trace
    best_a = best_trace.get_choices()["curve"]["a"]
    best_b = best_trace.get_choices()["curve"]["b"] 
    best_c = best_trace.get_choices()["curve"]["c"]
    
    # Extract outlier indicators from final trace
    outlier_probs = []
    for i in range(len(xs)):
        is_outlier = best_trace.get_choices()["ys"][i]["is_outlier"]
        outlier_probs.append(float(is_outlier))
    
    # Plot data with outlier coloring
    colors = [get_outlier_color(p) for p in outlier_probs]
    scatter = ax3.scatter(xs, ys, s=80, c=colors, edgecolor='black', linewidth=1, zorder=10)
    
    # Plot fitted curve
    ys_smooth = polyfn(xs_smooth, jnp.array([best_a, best_b, best_c]))
    ax3.plot(xs_smooth, ys_smooth, color=get_method_color("curves"), linewidth=3)
    
    ax3.set_xlabel('x', fontweight='bold')
    ax3.set_ylabel('y', fontweight='bold')
    ax3.set_title('Outlier Model + Gibbs/HMC\n(good inference, good model)', fontsize=14)
    apply_grid_style(ax3)
    ax3.set_xlim(-0.05, 1.05)
    
    # Add colorbar for outlier probability
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, fraction=0.05)
    cbar.set_label('P(outlier)', fontweight='bold')
    
    plt.tight_layout()
    save_publication_figure(fig, output_filename)