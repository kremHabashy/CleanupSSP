import os
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from cleanup_ssps.sspspace import HexagonalSSPSpace
from utils.evaluation_utils import compute_cleanup_baseline


def run_experiment(average_runs=5):
    sides = 1
    bounds = np.array([[0, sides], [0, sides]])
    snrs = torch.linspace(1, 0, steps=25)
    n_values = list(range(15, 2, -1))
    length_scales = np.logspace(-1, 0.6, num=20)

    results = np.zeros((len(snrs), len(n_values), len(length_scales)))

    for snr_idx, snr in enumerate(snrs):
        for n_idx, n in enumerate(n_values):
            ssp_dim = n * n * 6 + 1
            for scale_idx, length_scale in enumerate(length_scales):
                ssp_space = HexagonalSSPSpace(
                    domain_dim=2, ssp_dim=ssp_dim, domain_bounds=bounds, length_scale=length_scale
                )

                run_similarities = []
                for _ in range(average_runs):
                    mean_cosine_sim = compute_cleanup_baseline(
                        ssp_space, ssp_dim, snr.item(), grid_resolution=64, num_trials=100, device="cuda"
                    )
                    run_similarities.append(mean_cosine_sim)

                # Average over runs
                results[snr_idx, n_idx, scale_idx] = np.mean(run_similarities)

    # Save results for further analysis
    os.makedirs("results", exist_ok=True)
    np.save("results/ssp_cleanup_results.npy", results)

    # Create video visualization
    visualize_results(results, n_values, length_scales, snrs)


def visualize_results(results, n_values, length_scales, snrs, output_dir="results/ls_cleanup_frames", gif_file="results/ls_baseline_performance.gif"):
    """
    Visualize results as a heatmap animation with consistent colorbar scaling,
    overlaid with contour lines, and save frames as PNG files.

    Args:
        results (np.ndarray): 3D array [num_snrs, len(n_values), len(length_scales)].
        n_values (list): List of SSP dimensions.
        length_scales (list): List of length scales.
        snrs (list): List of signal-to-noise ratios.
        output_dir (str): Directory to save PNG frames (default: "results/frames").
        gif_file (str): Path to save the animation (default: "results/baseline_performance.gif").
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [f"{scale:.2f}" for scale in length_scales]
    y_labels = [f"{n}" for n in n_values]

    # Determine global min and max for consistent color scaling
    vmin = 0  # Minimum similarity (absolute scale)
    vmax = 1  # Maximum similarity (absolute scale, dot products max at 1)

    # Initialize heatmap
    heatmap = ax.imshow(
        results[0],
        cmap="viridis",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Cosine Similarity")

    # Set up axes once
    ax.set_xlabel("Length Scale", fontsize=12)
    ax.set_ylabel("Dimension (n)", fontsize=12)
    ax.set_xticks(np.arange(len(length_scales)))
    ax.set_yticks(np.arange(len(n_values)))
    ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=10)  # Labels match n_values

    # Add contour levels (consistent across all frames)
    contour_levels = np.linspace(vmin, vmax, 10)  # Adjust number of levels for more/less detail

    def update(frame):
        # Update heatmap data
        heatmap.set_data(results[frame])

        # Add contour lines on top of the heatmap
        contour = ax.contour(
            results[frame],
            levels=contour_levels,
            colors="white",
            linewidths=0.8,
        )
        # Add contour labels
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

        # Update title
        ax.set_title(f"SSP Signal Strength : {snrs[frame]:.2f}", fontsize=14)

        # Save current frame as PNG
        frame_filename = os.path.join(output_dir, f"frame_{frame:03d}.png")
        plt.savefig(frame_filename)

        # Print progress
        print(f"Saved frame {frame + 1}/{len(snrs)} as {frame_filename}")
        return [heatmap, contour]

    ani = animation.FuncAnimation(fig, update, frames=len(snrs), blit=False, interval=200)

    # Save animation as a GIF
    ani.save(gif_file, writer="pillow")
    plt.close(fig)
    print(f"Animation saved as {gif_file}")


if __name__ == "__main__":
    run_experiment(average_runs=5)
