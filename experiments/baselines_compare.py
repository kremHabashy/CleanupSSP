#!/usr/bin/env python3
import numpy as np
import torch
from pathlib import Path
import wandb
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.dataset import SSPDataset

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

DIM = 487
LENGTH_SCALE = 0.1
SIGNAL_STRENGTHS = np.linspace(0.0, 1.0, 31).round(3).tolist()
BASELINE_TRIALS = 256
BOOTSTRAP_SAMPLES = 200

GRID_SIZES_FROMSET = [64, 128, 256]
GRID_SIZE_DIRECTOPT = 4

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "test" / "coordinate_ssps")
OUT_DIR = PROJECT_ROOT / "trained_models" / "baseline_dim{dim}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# BASELINE FUNCTIONS
# --------------------------------------------------------------------------
def bootstrap_ci(arr, num_samples=200, ci=95):
    """Compute bootstrap confidence interval width."""
    N = len(arr)
    if N == 0:
        return 0.0
    idx = np.random.randint(0, N, size=(num_samples, N))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci)/2, (100 + ci)/2])
    return (hi - lo) / 2

def compute_baseline(ssp_space, snr, cleanup_method, grid_res, num_trials, device="cpu", bootstrap_samples=200):
    """Compute cosine similarity baseline."""
    gt_ssps, gt_pts = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=num_trials, method="Rd")
    gt_ssps = torch.tensor(gt_ssps, device=device)

    # Corrupt
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)
    corrupted = snr * gt_ssps + (1 - snr) * z
    corrupted = corrupted / corrupted.norm(dim=1, keepdim=True)

    if cleanup_method == "from-set":
        grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=grid_res, method="grid")
        grid_ssps = torch.tensor(grid_ssps, device=device)
        grid_pts = grid_pts
        sims = corrupted @ grid_ssps.T
        idx = sims.argmax(dim=1)
        cleaned_ssps = grid_ssps[idx]
        cleaned_pts = grid_pts[idx.cpu().numpy()]
    elif cleanup_method == "direct-optim":
        cleaned_ssp_list, cleaned_pt_list = [], []
        for i in range(num_trials):
            ssp_i = corrupted[i:i+1].detach().cpu().numpy()
            x_hat = ssp_space.decode(
                ssp_i,
                method="direct-optim",
                sampling_method="grid",
                num_samples=grid_res
            )
            cleaned_pt_list.append(np.atleast_2d(x_hat))
            ssp_hat = ssp_space.encode(np.atleast_2d(x_hat))
            cleaned_ssp_list.append(ssp_hat)
        cleaned_pts = np.vstack(cleaned_pt_list)
        cleaned_ssps = torch.tensor(np.vstack(cleaned_ssp_list), device=device)
    else:
        raise ValueError(f"Unknown cleanup_method={cleanup_method!r}")

    cos = (gt_ssps * cleaned_ssps).sum(dim=1) / (gt_ssps.norm(dim=1) * cleaned_ssps.norm(dim=1))
    cos_np = cos.detach().cpu().numpy()
    mean_cos = cos_np.mean()
    ci95_cos = bootstrap_ci(cos_np, num_samples=bootstrap_samples)

    return {"mean_cosine": mean_cos, "ci95_cosine": ci95_cos}

# --------------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project="Clean_Up", name="baselines_dim487")

    print(f"\n🚀 Running baselines for dim={DIM} ...")

    ssp_space = HexagonalSSPSpace(
        domain_dim=2,
        ssp_dim=DIM,
        domain_bounds=np.array([[2,3],[2,3]]),
        length_scale=LENGTH_SCALE,
        n_rotates=3,
        n_scales=3
    )

    results_fromset = {res: {"mean_cosine": [], "ci95_cosine": []} for res in GRID_SIZES_FROMSET}
    results_dopt = {"mean_cosine": [], "ci95_cosine": []}

    # Sweep SNRs
    for snr in SIGNAL_STRENGTHS:
        print(f"→ SNR={snr:.2f}")
        # from-set at 16, 64, 256
        for res in GRID_SIZES_FROMSET:
            r = compute_baseline(ssp_space, snr, "from-set", res, BASELINE_TRIALS, device=device)
            results_fromset[res]["mean_cosine"].append(r["mean_cosine"])
            results_fromset[res]["ci95_cosine"].append(r["ci95_cosine"])
            wandb.log({"SNR": snr, f"fromset_{res}/mean": r["mean_cosine"], f"fromset_{res}/ci95": r["ci95_cosine"]})

        # direct-optim at 16×16
        r_dopt = compute_baseline(ssp_space, snr, "direct-optim", GRID_SIZE_DIRECTOPT, BASELINE_TRIALS, device=device)
        results_dopt["mean_cosine"].append(r_dopt["mean_cosine"])
        results_dopt["ci95_cosine"].append(r_dopt["ci95_cosine"])
        wandb.log({"SNR": snr, "direct_optim_16/mean": r_dopt["mean_cosine"], "direct_optim_16/ci95": r_dopt["ci95_cosine"]})

    # ----------------------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------------------
    x = np.array(SIGNAL_STRENGTHS)
    fig = go.Figure()

    def add_band(fig, x, mean_list, ci_list, color, name, dash="dash"):
        m = np.array(mean_list)
        c = np.array(ci_list)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([m-c, (m+c)[::-1]]),
            fill="toself", fillcolor="rgba(0,0,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{name} 95% CI", legendgroup=name, showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=m, error_y=dict(type="data", array=c),
            mode="lines+markers",
            line=dict(color=color, dash=dash),
            name=name, legendgroup=name
        ))

    # From-set baselines
    color_map = {4: "yellow", 8: "orange", 16: "red", 32: "green", 64: "blue", 128: "purple"}
    for res in GRID_SIZES_FROMSET:
        add_band(fig, x, results_fromset[res]["mean_cosine"], results_fromset[res]["ci95_cosine"],
                 color_map[res], f"from-set {res}×{res}", dash="dot")

    # Direct-optim baseline
    add_band(fig, x, results_dopt["mean_cosine"], results_dopt["ci95_cosine"],
             "red", f"direct-optim {GRID_SIZE_DIRECTOPT}×{GRID_SIZE_DIRECTOPT}", dash="solid")

    fig.update_layout(
        title=f"Baseline Cosine Similarity vs Signal Strength (dim={DIM})",
        xaxis_title="Signal Strength",
        yaxis_title="Mean Cosine Similarity",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(groupclick="togglegroup")
    )

    out_base = OUT_DIR / "Baselines_Comparison"
    fig.write_image(str(out_base) + ".pdf")
    fig.write_image(str(out_base) + ".svg")
    wandb.log({"Baselines_Comparison": fig})

    print(f"\n✅ Saved plots to:\n  {out_base}.pdf\n  {out_base}.svg")
