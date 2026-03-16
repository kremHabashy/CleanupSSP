#!/usr/bin/env python3
import numpy as np
import torch
import wandb
import plotly.graph_objects as go
from cleanup_ssps.sspspace import HexagonalSSPSpace
from tqdm import tqdm

# --- EXPERIMENT CONFIG ---
length_scale     = 0.2
device           = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_STRENGTHS = [round(i * 0.1, 2) for i in range(11)]
GRID_RES         = 128
BASELINE_TRIALS  = 100

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
_baseline_grid_cache = {}

def bootstrap_ci(arr, num_samples=100, ci=95):
    N = len(arr)
    if N == 0:
        return 0.0
    idx = np.random.randint(0, N, size=(num_samples, N))
    boot_means = arr[idx].mean(axis=1)
    lower, upper = np.percentile(boot_means, [(100 - ci) / 2, (100 + ci) / 2])
    return (upper - lower) / 2


def compute_cleanup_baseline(
    ssp_space,
    ssp_dim,
    snr,
    cleanup_method='from-set',
    grid_resolution=64,
    method='grid',
    num_trials=100,
    device="cpu",
    bootstrap_samples=200
):
    """
    Compare cleanup performance for either 'from-set' or 'direct-optim'.
    """
    grid_key = (ssp_space.ssp_dim, grid_resolution, method)
    if grid_key not in _baseline_grid_cache:
        grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(
            num_points_per_dim=grid_resolution,
            method=method
        )
        _baseline_grid_cache[grid_key] = {
            "grid_ssps": torch.tensor(grid_ssps, device=device),
            "grid_pts": grid_pts
        }
    cache = _baseline_grid_cache[grid_key]
    grid_ssps, grid_pts = cache["grid_ssps"], cache["grid_pts"]

    # ground truth
    gt_ssps, gt_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=num_trials,
        method='Rd'
    )
    gt_ssps = torch.tensor(gt_ssps, device=device)

    # corrupt
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)

    corrupted = snr * gt_ssps + (1 - snr) * z
    corrupted = corrupted / corrupted.norm(dim=1, keepdim=True)

    # cleanup
    cleaned_pts = []
    cleaned_ssps = []

    for i in range(num_trials):
        ssp_i = corrupted[i:i+1].cpu().numpy()
        if cleanup_method == 'from-set':
            clean_ssp = ssp_space.clean_up(ssp_i, method='from-set', grid_resolution=grid_resolution)
        elif cleanup_method == 'direct-optim':
            clean_ssp = ssp_space.clean_up(ssp_i, method='direct-optim', grid_resolution=grid_resolution)

        else:
            raise ValueError(f"Unknown cleanup method {cleanup_method}")

        cleaned_ssps.append(np.atleast_2d(clean_ssp))   # (1, D)
        decoded_pt = ssp_space.decode(ssp_i, method=cleanup_method, num_samples=grid_resolution)
        cleaned_pts.append(np.atleast_2d(decoded_pt))   # (1, domain_dim)

    cleaned_ssps = np.vstack(cleaned_ssps)
    cleaned_pts = np.vstack(cleaned_pts)

    # cosine similarity
    cos = np.sum(gt_ssps.cpu().numpy() * cleaned_ssps, axis=1) / (
        np.linalg.norm(gt_ssps.cpu().numpy(), axis=1)
        * np.linalg.norm(cleaned_ssps, axis=1)
    )
    mean_cosine = cos.mean()
    std_cosine = cos.std(ddof=1)
    ci95_cosine = bootstrap_ci(cos, num_samples=bootstrap_samples)

    # rmse
    diffs = cleaned_pts - gt_pts
    rmse = np.linalg.norm(diffs, axis=1)
    mean_rmse = rmse.mean()
    std_rmse = rmse.std(ddof=1)
    ci95_rmse = bootstrap_ci(rmse, num_samples=bootstrap_samples)

    return {
        "mean_cosine": mean_cosine,
        "std_cosine": std_cosine,
        "ci95_cosine": ci95_cosine,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "ci95_rmse": ci95_rmse,
    }

# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(
        project="Clean_Up",
        name="gridres_fromset_vs_directoptim",
        config={
            "length_scale": length_scale,
            "baseline_trials": BASELINE_TRIALS,
        }
    )

    methods = ['from-set', 'direct-optim']
    grid_resolutions = [4, 8, 16, 32]  # adjust as needed
    ssp_dim = 151

    results = {m: {} for m in methods}

    ssp_space = HexagonalSSPSpace(
        domain_dim=2,
        ssp_dim=ssp_dim,
        domain_bounds=np.array([[-1,1],[-1,1]]),
        length_scale=length_scale,
        n_rotates=int(np.sqrt((ssp_dim-1)/6)),  # matches 487 config
        n_scales=int(np.sqrt((ssp_dim-1)/6))
    )

    for grid_res in grid_resolutions:
        for method in methods:
            results[method][grid_res] = {"mean_cosine": [], "ci95_cosine": []}

            for snr in tqdm(SIGNAL_STRENGTHS, desc=f"{method}, grid={grid_res}"):
                stats = compute_cleanup_baseline(
                    ssp_space,
                    ssp_dim=ssp_dim,
                    snr=snr,
                    cleanup_method=method,
                    grid_resolution=grid_res,
                    method='grid',
                    num_trials=BASELINE_TRIALS,
                    device=device,
                    bootstrap_samples=200
                )
                results[method][grid_res]["mean_cosine"].append(stats["mean_cosine"])
                results[method][grid_res]["ci95_cosine"].append(stats["ci95_cosine"])

    # -----------------------------------------------------------------------------
    # Plot A: overlay cosine vs SNR for all grid resolutions
    # -----------------------------------------------------------------------------
    x = np.array(SIGNAL_STRENGTHS)
    fig_overlay = go.Figure()

    for method, color in zip(methods, ['#1f77b4', '#d62728']):  # blue/red
        for grid_res in grid_resolutions:
            fig_overlay.add_trace(go.Scatter(
                x=x,
                y=results[method][grid_res]["mean_cosine"],
                error_y=dict(type="data", array=results[method][grid_res]["ci95_cosine"]),
                mode="lines+markers",
                name=f"{method}, grid={grid_res}",
                line=dict(width=2),
                marker=dict(size=5)
            ))

    fig_overlay.update_layout(
        title="Cleanup Comparison Across Grid Resolutions (SSP dim=487, 95% CI)",
        xaxis_title="Signal Strength (SNR)",
        yaxis_title="Mean Cosine Similarity ±95% CI",
        legend_title="Method / Grid Resolution",
        template="plotly_white"
    )

    # -----------------------------------------------------------------------------
    # Plot B: average across SNRs → single number per grid resolution
    # -----------------------------------------------------------------------------
    fig_avg = go.Figure()
    for method, color in zip(methods, ['#1f77b4', '#d62728']):
        avg_cos = [np.mean(results[method][grid]["mean_cosine"]) for grid in grid_resolutions]
        avg_ci  = [np.mean(results[method][grid]["ci95_cosine"]) for grid in grid_resolutions]
        fig_avg.add_trace(go.Scatter(
            x=grid_resolutions,
            y=avg_cos,
            error_y=dict(type="data", array=avg_ci),
            mode="lines+markers",
            name=method,
            line=dict(width=3),
            marker=dict(size=7),
        ))

    fig_avg.update_layout(
        title="Average Cleanup Performance vs Grid Resolution (SSP dim=487, 95% CI)",
        xaxis_title="Grid Resolution (points per dim)",
        yaxis_title="Average Mean Cosine Similarity",
        legend_title="Method",
        template="plotly_white"
    )

    # -----------------------------------------------------------------------------
    # Log to WandB
    # -----------------------------------------------------------------------------
    wandb.log({
        "Comparison/Overlay": fig_overlay,
        "Comparison/Average": fig_avg
    })
    wandb.finish()
