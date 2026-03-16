import numpy as np
import torch
from cleanup_ssps.sspspace import HexagonalSSPSpace
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def run_experiment(num_dims, length_scales, num_points_per_dim=100, num_runs=5):
    """
    Run the experiment across different SSP dimensions and length scales.

    Args:
        num_dims (list): List of SSP dimensions to test.
        length_scales (list): List of length scales to test.
        num_points_per_dim (int): Number of points per dimension to sample.
        num_runs (int): Number of runs to average results over.

    Returns:
        dims_list: List of SSP dimensions tested.
        results: Dictionary containing results for each length scale.
    """
    dims_range = range(2, num_dims + 2)
    dims_list = [dim**2 * 6 + 1 for dim in dims_range]

    results = {
        length_scale: {
            "avg_dot_products": [],
            "std_dot_products": [],
            "avg_dot_with_avg": [],
            "std_dot_with_avg": []
        }
        for length_scale in length_scales
    }

    for length_scale in tqdm(length_scales, desc="Sweeping length scales"):
        for dim in dims_list:
            dot_products = []
            dot_with_avg = []

            # Initialize SSP space
            ssp_space = HexagonalSSPSpace(
                domain_dim=2,
                ssp_dim=dim,
                domain_bounds=np.array([[0, 1], [0, 1]]),
                length_scale=length_scale,
            )

            for _ in range(num_runs):
                # Sample SSPs using Sobol sampling
                ssps1, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim, method="sobol")
                ssps2, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim, method="sobol")

                # Compute pairwise dot products
                ssps1_tensor = torch.tensor(ssps1, dtype=torch.float32)
                pairwise_dot = torch.einsum("ij,kj->ik", ssps1_tensor, ssps1_tensor).mean().item()
                dot_products.append(pairwise_dot)

                # Compute "average" SSP and its dot product with the first set
                avg_ssp = torch.tensor(ssps2.mean(axis=0), dtype=torch.float32)
                avg_dot = (ssps1_tensor @ avg_ssp).mean().item()
                dot_with_avg.append(avg_dot)

            # Store averaged results and standard deviations
            results[length_scale]["avg_dot_products"].append(np.mean(dot_products))
            results[length_scale]["std_dot_products"].append(np.std(dot_products))
            results[length_scale]["avg_dot_with_avg"].append(np.mean(dot_with_avg))
            results[length_scale]["std_dot_with_avg"].append(np.std(dot_with_avg))

    return dims_list, results


def create_video(dims_list, results, length_scales, output_dir="results/avg_ssp_frames"):
    """
    Save individual frames of the animation as .png images.

    Args:
        dims_list (list): List of SSP dimensions tested.
        results (dict): Results for each length scale.
        length_scales (list): List of length scales.
        output_dir (str): Directory to save the frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for frame, length_scale in enumerate(length_scales):
        avg_dot_products = results[length_scale]["avg_dot_products"]
        std_dot_products = results[length_scale]["std_dot_products"]
        avg_dot_with_avg = results[length_scale]["avg_dot_with_avg"]
        std_dot_with_avg = results[length_scale]["std_dot_with_avg"]

        # Plot Pairwise Dot Products
        ax[0].clear()
        ax[0].errorbar(
            dims_list, avg_dot_products, yerr=std_dot_products, fmt="-o", label="Pairwise Dot Products"
        )
        ax[0].set_title("Average Pairwise Dot Products")
        ax[0].set_xlabel("SSP Dimension")
        ax[0].set_ylabel("Average Dot Product")
        ax[0].legend()
        ax[0].grid()

        # Plot Dot Products with Average SSP
        ax[1].clear()
        ax[1].errorbar(
            dims_list, avg_dot_with_avg, yerr=std_dot_with_avg, fmt="-o", label="Dot Product with Average SSP"
        )
        ax[1].set_title("Dot Product with Average SSP")
        ax[1].set_xlabel("SSP Dimension")
        ax[1].set_ylabel("Average Dot Product")
        ax[1].legend()
        ax[1].grid()

        # Annotate the length scale
        fig.suptitle(f"Length Scale: {length_scale:.2f}")

        # Save the frame
        frame_filename = os.path.join(output_dir, f"frame_{frame:03d}.png")
        plt.savefig(frame_filename)

    plt.close(fig)



if __name__ == "__main__":
    # Define parameters
    num_dims = 10  # Number of SSP dimensions to test
    length_scales = np.linspace(0.000001, 1.0, 25)  # Length scale sweep
    num_points_per_dim = 512
    num_runs = 5

    # Run experiment
    dims_list, results = run_experiment(num_dims, length_scales, num_points_per_dim, num_runs)

    # Create video
    create_video(dims_list, results, length_scales)
