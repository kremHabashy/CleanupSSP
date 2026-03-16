#!/usr/bin/env python3
"""
Average performance (cosine similarity) over all SNRs as a function of dimensionality.
Includes flows (drift_*.pt), feedforward (feedforward.pt), and baselines
('from-set', 'direct-optim'). Uploads a single Plotly figure (means ± 95% CI) to W&B.

Assumes directory layout:
  trained_models/Hex/dim{DIM}_ls{LS}/drift_{mode}.pt
  trained_models/Hex/dim{DIM}_ls{LS}/feedforward.pt
"""

import os
import re
import glob
import math
import numpy as np
from pathlib import Path

import torch
import plotly.graph_objects as go
import plotly.io as pio
import kaleido  # ensure plotly finds the kaleido engine
import wandb

pio.kaleido.scope.mathjax = None

from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.cleanup_methods import FlowMatching

# =========================
# Config
# =========================
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "trained_models" / "Hex"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 256

# Evaluation hyperparameters
EVAL_STEPS    = 25                          # ODE steps for flows
SNR_LIST      = [round(i * 0.05, 2) for i in range(21)]   # 0.00 .. 1.00
PTS_PER_DIM   = 16                          # clean targets grid per axis (16x16)
PTS_METHOD    = "sobol"                     # 'sobol' or 'grid'
GRID_RES_BASE = 128                         # baseline grid resolution (from-set, direct-optim)

WANDB_PROJ    = "Clean_Up"
RUN_NAME      = "avg_perf_vs_dim_means_and_cis"

# Which series to plot (labels after normalization below)
# Default to exactly what you asked: euc_det, geo_det, and the 3 baselines.
PLOT_ONLY = {
    "euc_det",
    "geo_det",
    "baseline/feedforward",
    "baseline/from-set",
    "baseline/direct-optim",
}

# =========================
# Helpers
# =========================
def renorm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine(a, b):
    a = renorm(a); b = renorm(b)
    return (a * b).sum(dim=1)

def geodesic_log(p, q, eps=1e-8):
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)
    v = q - dot * p
    vn = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * v / vn

def geodesic_exp(p, v, eps=1e-8):
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.cos(nv) * p + torch.sin(nv) * (v / nv)

def geodesic_mix(z0, z1, snr):
    v0 = geodesic_log(z0, z1)
    t  = torch.full((z0.shape[0], 1), float(snr), device=z0.device)
    return geodesic_exp(z0, t * v0)

def euclidean_mix(z0, z1, snr):
    return snr * z1 + (1.0 - snr) * z0

def use_geodesic(sampling_mode: str) -> bool:
    return sampling_mode.startswith("geo_")

def bootstrap_ci_halfwidth(values, num_samples=200, ci=95):
    """Half-width of bootstrap percentile CI for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    idx = np.random.randint(0, arr.size, size=(num_samples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100-ci)/2, (100+ci)/2])
    return float((hi - lo) / 2)

def dim_to_r(d):
    # d = 1 + 6 r^2  → r = sqrt((d-1)/6)
    return int(round(math.sqrt(max(d-1, 0)/6)))

def discover_groups():
    """
    Returns:
      groups: {(dim, ls): {'flows': [(path, mode), ...], 'ff': path|None}}
    """
    pat_dim = re.compile(r"dim(?P<dim>\d+)_ls(?P<ls>[\d.]+)")
    pat_flow = re.compile(r"drift_(?P<mode>[^.]+)\.pt")

    groups = {}
    for f in glob.glob(str(CHECKPOINT_DIR / "dim*_ls*/*.pt")):
        p = Path(f)
        m_dim = pat_dim.search(p.parent.name)
        if not m_dim:
            continue
        dim = int(m_dim.group("dim"))
        ls = float(m_dim.group("ls"))

        entry = groups.setdefault((dim, ls), {"flows": [], "ff": None})
        if p.name == "feedforward.pt":
            entry["ff"] = f
        else:
            m_flow = pat_flow.match(p.name)
            if m_flow:
                entry["flows"].append((f, m_flow.group("mode")))
    # sort modes per group for determinism
    for k in groups:
        groups[k]["flows"].sort(key=lambda t: t[1])
    return groups

def build_eval_targets(ssp_space, device):
    tgt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=PTS_PER_DIM, method=PTS_METHOD)
    tgt = torch.tensor(tgt_ssps, dtype=torch.float32, device=device)
    tgt = renorm(tgt)
    base_noise = renorm(torch.randn_like(tgt))
    return tgt, base_noise

def load_flow_wrapper(ckpt_path: str, ssp_dim: int, device: str, sampling_mode: str) -> FlowMatching:
    model = ResidualMLP(ssp_dim, flow=True).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return FlowMatching(model=model, num_steps=1, sampling=sampling_mode, device=device)

def load_feedforward(ckpt_path: str, ssp_dim: int, device: str) -> torch.nn.Module:
    model = ResidualMLP(ssp_dim, flow=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

# =========================
# Evaluators
# =========================
@torch.no_grad()
def eval_flow_mean_over_snrs(wrapper: FlowMatching,
                             tgt_all: torch.Tensor,
                             noise_all: torch.Tensor,
                             steps: int,
                             snr_list):
    """Mean across SNRs (list) for a fixed flow model and fixed number of steps."""
    geo = use_geodesic(wrapper.sampling)
    per_snr_means = []
    for snr in snr_list:
        if geo:
            z0_all = geodesic_mix(noise_all, tgt_all, snr)
        else:
            z0_all = euclidean_mix(noise_all, tgt_all, snr)

        cos_vals = []
        for i in range(0, tgt_all.shape[0], BATCH_SIZE):
            z1 = tgt_all[i:i+BATCH_SIZE]
            z0 = z0_all[i:i+BATCH_SIZE]
            zT = wrapper.sample_ode(
                z_init=z0,
                N=steps,
                use_sphere=geo,
                t0=torch.tensor([[snr]], device=z0.device)
            )[-1]
            zT = renorm(zT)
            cos_vals.append(cosine(zT, z1).cpu().numpy())
        cos_vals = np.concatenate(cos_vals)
        per_snr_means.append(float(np.mean(cos_vals)))
    return float(np.mean(per_snr_means)), per_snr_means

@torch.no_grad()
def eval_feedforward_mean_over_snrs(model_ff: torch.nn.Module,
                                    tgt_all: torch.Tensor,
                                    noise_all: torch.Tensor,
                                    snr_list):
    """Mean across SNRs for feedforward (Euclidean mixing)."""
    per_snr_means = []
    for snr in snr_list:
        z0_all = euclidean_mix(noise_all, tgt_all, snr)
        cos_vals = []
        for i in range(0, tgt_all.shape[0], BATCH_SIZE):
            z1 = tgt_all[i:i+BATCH_SIZE]
            z0 = z0_all[i:i+BATCH_SIZE]
            out = renorm(model_ff(z0))
            cos_vals.append(cosine(out, z1).cpu().numpy())
        cos_vals = np.concatenate(cos_vals)
        per_snr_means.append(float(np.mean(cos_vals)))
    return float(np.mean(per_snr_means)), per_snr_means

def eval_baseline_mean_over_snrs(ssp_space: HexagonalSSPSpace,
                                 tgt_all_np: np.ndarray,
                                 snr_list,
                                 method: str,
                                 grid_resolution: int):
    """
    Baselines: 'from-set' and 'direct-optim'
    Uses ssp_space.clean_up() and ssp_space.decode() internally.
    """
    gt_ssps = torch.tensor(tgt_all_np, device="cpu", dtype=torch.float32)
    per_snr_means = []

    for snr in snr_list:
        z = torch.randn_like(gt_ssps)
        z = renorm(z)
        corrupted = snr * gt_ssps + (1 - snr) * z
        corrupted = renorm(corrupted)

        cos_vals = []
        for i in range(corrupted.shape[0]):
            ssp_i = corrupted[i:i+1].cpu().numpy()  # (1, D)
            if method == "from-set":
                cleaned_ssp = ssp_space.clean_up(ssp_i, method='from-set', grid_resolution=grid_resolution)
            elif method == "direct-optim":
                cleaned_ssp = ssp_space.clean_up(ssp_i, method='direct-optim', grid_resolution=grid_resolution)
            else:
                raise ValueError("Unknown baseline method.")

            # Ensure both are (1, D)
            cleaned_ssp = np.atleast_2d(cleaned_ssp)
            gt_i = np.atleast_2d(tgt_all_np[i])

            # cosine similarity
            dot = float(np.dot(gt_i, cleaned_ssp.T))  # scalar
            denom = float(np.linalg.norm(gt_i) * np.linalg.norm(cleaned_ssp) + 1e-12)
            c = dot / denom
            cos_vals.append(c)

        per_snr_means.append(float(np.mean(cos_vals)))

    return float(np.mean(per_snr_means)), per_snr_means

# =========================
# Vector export helper (SVG/PDF) + W&B logging
# =========================
def save_vector_and_log(fig: go.Figure, out_base: str):
    svg, pdf, png = f"{out_base}.svg", f"{out_base}.pdf", f"{out_base}.png"
    try:
        # Vector export (requires kaleido==0.2.1 with plotly 5.x)
        fig.write_image(svg)
        fig.write_image(pdf)
        wandb.save(svg); wandb.save(pdf)
        art = wandb.Artifact(name=out_base.replace("/", "_"), type="plot")
        art.add_file(svg); art.add_file(pdf)
        wandb.log_artifact(art)
    except Exception as e:
        # Fallback to PNG so the run never crashes
        fig.write_image(png, scale=2)
        wandb.save(png)
        wandb.alert(
            title="Vector export failed",
            text=f"{out_base}: {type(e).__name__}: {e}\n"
                 f'Ensure kaleido==0.2.1 is installed in the active venv.',
        )

# =========================
# Main
# =========================
def main():
    wandb.init(
        project=WANDB_PROJ,
        name=RUN_NAME,
        config=dict(
            device=DEVICE,
            eval_steps=EVAL_STEPS,
            snr_list=SNR_LIST,
            pts_per_dim=PTS_PER_DIM,
            pts_method=PTS_METHOD,
            plot_only=sorted(PLOT_ONLY),
        ),
    )

    groups = discover_groups()
    if not groups:
        print(f"No checkpoints found under {CHECKPOINT_DIR}")
        return

    total_ckpts = sum(len(v["flows"]) + (1 if v["ff"] else 0) for v in groups.values())
    print(f"Found {total_ckpts} checkpoints across {len(groups)} dimensionalities.")

    # Collect results for a single global figure
    results_by_mode = {}  # normalized_label -> list of (dim, mean, ci)

    # helper for label normalization (so we can filter & style consistently)
    def norm_label(raw_mode: str) -> str:
        if raw_mode == "feedforward":
            return "baseline/feedforward"
        if raw_mode == "from-set":
            return "baseline/from-set"
        if raw_mode == "direct-optim":
            return "baseline/direct-optim"
        return raw_mode  # e.g., euc_det, geo_det, euc_ot, geo_amb_sb, ...

    for (dim, ls), info in sorted(groups.items()):
        print(f"\nEvaluating dim={dim}, length_scale={ls} ...")
        r = dim_to_r(dim)
        ssp_space = HexagonalSSPSpace(
            domain_dim=2,
            ssp_dim=dim,
            domain_bounds=np.array([[2, 3], [2, 3]]),
            length_scale=ls,
            n_rotates=r,
            n_scales=r
        )

        tgt_all, noise_all = build_eval_targets(ssp_space, DEVICE)
        tgt_all_np = tgt_all.cpu().numpy()

        # --- Feedforward (stored under baseline/feedforward)
        if info["ff"] is not None:
            print("→ Evaluating feedforward ...")
            model_ff = load_feedforward(info["ff"], dim, DEVICE)
            mean_val, snr_means = eval_feedforward_mean_over_snrs(model_ff, tgt_all, noise_all, SNR_LIST)
            label = norm_label("feedforward")
            results_by_mode.setdefault(label, []).append(
                (dim, mean_val, bootstrap_ci_halfwidth(snr_means))
            )

        # --- Flows
        for ckpt_path, mode in info["flows"]:
            try:
                wrapper = load_flow_wrapper(ckpt_path, dim, DEVICE, sampling_mode=mode)
            except Exception as e:
                print(f"[SKIP] {ckpt_path}: {e}")
                continue
            print(f"→ Evaluating flow mode {mode} ...")
            mean_val, snr_means = eval_flow_mean_over_snrs(
                wrapper=wrapper,
                tgt_all=tgt_all,
                noise_all=noise_all,
                steps=EVAL_STEPS,
                snr_list=SNR_LIST
            )
            label = norm_label(mode)
            results_by_mode.setdefault(label, []).append(
                (dim, mean_val, bootstrap_ci_halfwidth(snr_means))
            )

        # --- Baselines (stored under baseline/*)
        print("→ Evaluating baselines ('from-set' and 'direct-optim') ...")
        for method in ["from-set", "direct-optim"]:
            mean_val, snr_means = eval_baseline_mean_over_snrs(
                ssp_space=ssp_space,
                tgt_all_np=tgt_all_np,
                snr_list=SNR_LIST,
                method=method,
                grid_resolution=GRID_RES_BASE
            )
            label = norm_label(method)
            results_by_mode.setdefault(label, []).append(
                (dim, mean_val, bootstrap_ci_halfwidth(snr_means))
            )

    # ---- Build one global Plotly figure: x = dimensionality
    fig = go.Figure()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    # Only plot selected labels
    selected_items = [(k, v) for k, v in sorted(results_by_mode.items()) if k in PLOT_ONLY]
    if not selected_items:
        print("Warning: no series matched PLOT_ONLY; plotting all.")
        selected_items = sorted(results_by_mode.items())

    for i, (mode_label, data) in enumerate(selected_items):
        dims  = np.array([d for d, _, _ in data], dtype=float)
        means = np.array([m for _, m, _ in data], dtype=float)
        cis   = np.array([c for _, _, c in data], dtype=float)

        fig.add_trace(go.Scatter(
            x=dims,
            y=means,
            mode="lines+markers",
            name=mode_label,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
            error_y=dict(
                type="data",
                array=cis,        # upper
                arrayminus=cis,   # lower
                thickness=1.2,
                width=3
            )
        ))

    fig.update_layout(
        title="Average Mean Cosine ± 95% CI vs SSP Dimensionality",
        xaxis_title="SSP Dimensionality",
        yaxis_title="Mean Cosine Similarity (avg over SNRs)",
        template="plotly_white",
        legend_title="Model / Method",
        yaxis=dict(range=[0.0, 1.05])
    )

    wandb.log({"AvgOverDims": fig})
    save_vector_and_log(fig, "AvgOverDims_vs_Dim")
    wandb.finish()


if __name__ == "__main__":
    main()
