import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.model import MLP
from cleanup_ssps.sspspace import HexagonalSSPSpace
from utils.evaluation_utils import compute_cleanup_baseline

from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import matplotlib.pyplot as plt


def parse_folder_name(folder_name):
    """
    Extract SSP dimension and length scale from folder name.
    """
    parts = folder_name.split('_')
    ssp_dim = int(parts[1])
    length_scale = float(parts[3])
    return ssp_dim, length_scale

def evaluate_dot_product_cleanup(data_dir, ssp_space, ssp_dim, length_scale, signal_strength, grid_resolution=64, num_trials=100):
    """
    Evaluate performance using the dot product cleanup baseline.
    Args:
        data_dir: Path to dataset folder (currently unused; SSPs are generated on the fly).
        ssp_space: Instance of HexagonalSSPSpace defining the SSP space.
        ssp_dim: Dimensionality of the SSPs.
        length_scale: Length scale of the SSP space.
        signal_strength: Signal-to-noise ratio for corrupted SSPs.
        grid_resolution: Resolution of the grid used for cleanup.
        num_trials: Number of random trials to run.
    Returns:
        Average cosine similarity for cleanup performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssp_space.update_lengthscale(length_scale)  # Ensure correct length scale

    # Use compute_cleanup_baseline to calculate performance
    mean_similarity = compute_cleanup_baseline(
        ssp_space=ssp_space,
        ssp_dim=ssp_dim,
        snr=signal_strength,
        grid_resolution=grid_resolution,
        num_trials=num_trials,
        device=device
    )
    return mean_similarity

def run_experiments(base_dir, signal_strengths, grid_resolution=64, num_trials=100):
    """
    Run experiments across all datasets using dot product cleanup.
    """
    results = []
    for folder in os.listdir(base_dir):
        if not folder.startswith("dim"):
            continue
        ssp_dim, length_scale = parse_folder_name(folder)
        data_dir = os.path.join(base_dir, folder)
        ssp_space = HexagonalSSPSpace(
            domain_dim=2,
            ssp_dim=ssp_dim,
            domain_bounds=np.array([[-1, 1], [-1, 1]]),
            length_scale=length_scale
        )
        for signal_strength in signal_strengths:
            avg_similarity = evaluate_dot_product_cleanup(
                data_dir=data_dir,
                ssp_space=ssp_space,
                ssp_dim=ssp_dim,
                length_scale=length_scale,
                signal_strength=signal_strength,
                grid_resolution=grid_resolution,
                num_trials=num_trials
            )
            results.append({
                'ssp_dim': ssp_dim,
                'length_scale': length_scale,
                'signal_strength': signal_strength,
                'performance': avg_similarity
            })
    return results

def bootstrap_linear_regression(df, num_bootstrap=1000, ci=95):
    """
    Perform linear regression with bootstrap confidence intervals.
    """
    X = df[['ssp_dim', 'length_scale', 'signal_strength']].copy()
    X = (X - X.mean()) / X.std()  # Normalize features for interpretability
    y = df['performance']

    model = LinearRegression()
    model.fit(X, y)
    initial_coefficients = model.coef_

    bootstrap_coeffs = []
    for _ in range(num_bootstrap):
        resampled_df = resample(df)
        X_resampled = (resampled_df[['ssp_dim', 'length_scale', 'signal_strength']] - X.mean()) / X.std()
        y_resampled = resampled_df['performance']
        model.fit(X_resampled, y_resampled)
        bootstrap_coeffs.append(model.coef_)

    bootstrap_coeffs = np.array(bootstrap_coeffs)
    lower_bound = np.percentile(bootstrap_coeffs, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(bootstrap_coeffs, 100 - (100 - ci) / 2, axis=0)

    results_df = pd.DataFrame({
        'Parameter': ['ssp_dim', 'length_scale', 'signal_strength'],
        'Coefficient': initial_coefficients,
        f'Lower {ci}% CI': lower_bound,
        f'Upper {ci}% CI': upper_bound
    })

    return results_df

def plot_coefficients(results_df, save_path="/u1/khabashy/CleanupSSP/results/linear_regression_coefficients.png"):
    """
    Plot regression coefficients with confidence intervals and save the plot.
    """
    lower_errors = np.abs(results_df['Coefficient'] - results_df[f'Lower 95% CI'])
    upper_errors = np.abs(results_df[f'Upper 95% CI'] - results_df['Coefficient'])
    yerr = np.array([lower_errors, upper_errors])

    plt.figure(figsize=(8, 6))
    plt.bar(results_df['Parameter'], results_df['Coefficient'], color='skyblue', label='Coefficient')
    plt.errorbar(
        results_df['Parameter'],
        results_df['Coefficient'],
        yerr=yerr,
        fmt='o',
        color='black',
        label='95% CI'
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Parameter')
    plt.ylabel('Impact on Performance')
    plt.title('Relative Impact of Parameters on Model Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    base_dir = "/u1/khabashy/CleanupSSP/data"
    signal_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Run experiments and gather results
    results = run_experiments(base_dir, signal_strengths)
    df = pd.DataFrame(results)

    # Perform linear regression with bootstrap confidence intervals
    results_df = bootstrap_linear_regression(df)

    # Plot coefficients with confidence intervals and save the plot
    plot_coefficients(results_df)
