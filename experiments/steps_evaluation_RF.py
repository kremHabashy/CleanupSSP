#!/usr/bin/env python3
"""
Average performance over all noise levels (SNRs) for a given number of steps,
with separate Plotly figures per dimensionality, showing different model
(modes) means ± 95% CI. Uploads figures to Weights & Biases.

Now includes:
  • Feedforward (as "baseline/feedforward")
  • Baselines: 'from-set' and 'direct-optim'
  • Filtering: choose which series to plot/evaluate via PLOT_ONLY

Assumes checkpoints live at:
  trained_models/Hex/dim{D}_ls{LS}/drift_{mode}.pt
  trained_models/Hex/dim{D}_ls{LS}/feedforward.pt
"""

import re
import glob
import math
import numpy as np
from pathlib import Path

import torch
import plotly.graph_objects as go
import plotly.io as pio
import wandb

pio.kaleido.scope.mathjax = None

# --- project imports ---
from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.cleanup_methods import FlowMatching

# =========================
# Config
# =========================
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "trained_models" / "Hex"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
EVAL_STEPS  = [1, 2, 3, 5, 10, 15, 20, 25]        # steps to sweep (flows only)
SNR_LIST    = [round(i * 0.05, 2) for i in range(21)]  # 0.00..1.00
PTS_PER_DIM = 256                                  # eval targets per dim
PTS_METHOD  = "sobol"                              # or "grid"
WANDB_PROJ  = "Clean_Up"
RUN_NAME    = "avg_perf_steps_by_dim_means_and_cis"

# Which series to evaluate/plot (normalized labels, see norm_label())
# Default to exactly what you asked: Euclidean/Geodesic deterministic + 3 baselines.
PLOT_ONLY = {
    "euc_det",
    # "euc_ot",
    # "euc_sb",
    "geo_det",
    # "geo_amb_const",
    # "geo_amb_sb",
    # "geo_tan_const",
    # "geo_tan_sb",
}

# Baseline grid resolution
GRID_RES_BASE = 64          # from-set
GRID_RES_DOPT = 4           # direct-optim initial grid for L-BFGS-B

# =========================
# Helpers
# =========================
def dim_to_r(d):
    # d = 1 + 6 r^2  → r = sqrt((d-1)/6)
    return int(round(math.sqrt(max(d-1, 0)/6)))

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

def bootstrap_ci_bounds(arr, num_samples=200, ci=95, seed=42):
    """
    Returns (lower, upper) percentile CI for the mean of `arr`.
    Robust to list input, NaNs, and small N.
    """
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    N = x.size
    if N == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(num_samples, N))
    boot_means = x[idx].mean(axis=1)
    alpha = (100 - ci) / 2.0
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(lower), float(upper)

def norm_label(raw_mode: str) -> str:
    """Normalize labels so we can filter/plot consistently."""
    if raw_mode == "feedforward":
        return "baseline/feedforward"
    if raw_mode == "from-set":
        return "baseline/from-set"
    if raw_mode == "direct-optim":
        return "baseline/direct-optim"
    return raw_mode  # e.g., euc_det, geo_det, ...

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

def build_eval_batch(ssp_space, device):
    # targets: clean SSPs
    tgt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=PTS_PER_DIM, method=PTS_METHOD)
    tgt = torch.tensor(tgt_ssps, dtype=torch.float32, device=device)
    tgt = renorm(tgt)
    # base noise (on sphere)
    noise = renorm(torch.randn_like(tgt))
    return tgt, noise

def load_flow(ckpt_path: str, ssp_dim: int, device: str, sampling_mode: str) -> FlowMatching:
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
def mean_over_snrs_for_steps(wrapper: FlowMatching,
                             tgt_all: torch.Tensor,
                             noise_all: torch.Tensor,
                             steps: int,
                             snr_list):
    """Return mean across SNRs (list) for a fixed number of steps (flow)."""
    geo = use_geodesic(wrapper.sampling)
    per_snr_means = []
    for snr in snr_list:
        # build start states per SNR
        if geo:
            z_init_all = geodesic_mix(noise_all, tgt_all, snr)
        else:
            z_init_all = euclidean_mix(noise_all, tgt_all, snr)

        # integrate in batches
        cos_vals = []
        for i in range(0, tgt_all.shape[0], BATCH_SIZE):
            z1 = tgt_all[i:i+BATCH_SIZE]
            z0 = z_init_all[i:i+BATCH_SIZE]
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
    # return the average across SNRs and also the list (for CI)
    return float(np.mean(per_snr_means)), per_snr_means

@torch.no_grad()
def mean_over_snrs_feedforward(model_ff: torch.nn.Module,
                               tgt_all: torch.Tensor,
                               noise_all: torch.Tensor,
                               snr_list):
    """Average over SNRs for feedforward (no steps)."""
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
    return float(np.mean(per_snr_means)), per_snr_means  # mean across SNRs + list for CI

def mean_over_snrs_baseline(ssp_space: HexagonalSSPSpace,
                            tgt_all_np: np.ndarray,
                            snr_list,
                            method: str,
                            grid_resolution: int):
    """
    Baselines: 'from-set' and 'direct-optim'
    Returns (mean over SNRs, list over SNRs) for CI.
    """
    gt_ssps = torch.tensor(tgt_all_np, device="cpu", dtype=torch.float32)
    per_snr_means = []

    for snr in snr_list:
        z = torch.randn_like(gt_ssps)
        z = renorm(z)
        corrupted = renorm(snr * gt_ssps + (1 - snr) * z)

        cos_vals = []
        for i in range(corrupted.shape[0]):
            ssp_i = corrupted[i:i+1].cpu().numpy()  # (1, D)
            if method == "from-set":
                cleaned_ssp = ssp_space.clean_up(
                    ssp_i, method='from-set', grid_resolution=grid_resolution
                )
            elif method == "direct-optim":
                cleaned_ssp = ssp_space.clean_up(
                    ssp_i, method='direct-optim', grid_resolution=grid_resolution
                )
            else:
                raise ValueError("Unknown baseline method.")

            cleaned_ssp = np.atleast_2d(cleaned_ssp)
            gt_i = np.atleast_2d(tgt_all_np[i])

            dot = float(np.dot(gt_i, cleaned_ssp.T))
            denom = float(np.linalg.norm(gt_i) * np.linalg.norm(cleaned_ssp) + 1e-12)
            cos_vals.append(dot / denom)

        per_snr_means.append(float(np.mean(cos_vals)))

    return float(np.mean(per_snr_means)), per_snr_means

def save_vector_and_log(fig, out_base: Path):
    """
    Save a Plotly figure as both SVG and PDF directly via Kaleido,
    with MathJax disabled to avoid the loading badge.
    Also logs outputs to Weights & Biases.
    """
    import plotly.io as pio
    import wandb

    # Disable MathJax for clean exports
    pio.kaleido.scope.mathjax = None

    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")

    # Write vector images
    fig.write_image(str(svg_path))
    fig.write_image(str(pdf_path))

    # Log to W&B
    wandb.save(str(svg_path))
    wandb.save(str(pdf_path))

    art = wandb.Artifact(name=out_base.name.replace("/", "_"), type="plot")
    art.add_file(str(svg_path))
    art.add_file(str(pdf_path))
    wandb.log_artifact(art)

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
    groups = {k: v for k, v in groups.items() if abs(k[1] - 0.1) < 1e-8}
    if not groups:
        print(f"No checkpoints found under {CHECKPOINT_DIR}")
        return

    print(f"Found {sum(len(v['flows']) + (1 if v['ff'] else 0) for v in groups.values())} "
          f"checkpoints across {len(groups)} dimensionalities.")

    # For each dimensionality: build one figure with per-series curves (mean ± CI) vs steps.
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

        # Fixed eval batch for this dimension
        tgt_all, noise_all = build_eval_batch(ssp_space, DEVICE)
        tgt_all_np = tgt_all.cpu().numpy()

        # series curves: norm_label -> dict(steps, mean, ci)
        per_series = {}

        # --- Feedforward (constant line across steps)
        if info["ff"] is not None:
            label = norm_label("feedforward")  # "baseline/feedforward"
            if label in PLOT_ONLY:
                print("→ Evaluating feedforward ...")
                model_ff = load_feedforward(info["ff"], dim, DEVICE)
                ff_mean, ff_snr_means = mean_over_snrs_feedforward(model_ff, tgt_all, noise_all, SNR_LIST)
                lo, hi = bootstrap_ci_bounds(ff_snr_means)
                per_series[label] = dict(
                    steps=list(EVAL_STEPS),
                    mean=[ff_mean] * len(EVAL_STEPS),
                    ci=[(lo, hi)] * len(EVAL_STEPS),
                )

        # --- Baselines (constant lines across steps)
        for method, gres in [("from-set", GRID_RES_BASE), ("direct-optim", GRID_RES_DOPT)]:
            label = norm_label(method)  # baseline/from-set or baseline/direct-optim
            if label in PLOT_ONLY:
                print(f"→ Evaluating baseline '{method}' ...")
                b_mean, b_snr_means = mean_over_snrs_baseline(
                    ssp_space=ssp_space,
                    tgt_all_np=tgt_all_np,
                    snr_list=SNR_LIST,
                    method=method,
                    grid_resolution=gres
                )
                lo, hi = bootstrap_ci_bounds(b_snr_means)
                per_series[label] = dict(
                    steps=list(EVAL_STEPS),
                    mean=[b_mean] * len(EVAL_STEPS),
                    ci=[(lo, hi)] * len(EVAL_STEPS),
                )

        # --- Flows (evaluate only selected modes)
        for ckpt_path, raw_mode in sorted(info["flows"], key=lambda t: t[1]):
            label = norm_label(raw_mode)
            if label not in PLOT_ONLY:
                continue
            try:
                wrapper = load_flow(ckpt_path, dim, DEVICE, sampling_mode=raw_mode)
            except Exception as e:
                print(f"[SKIP] {ckpt_path}: {e}")
                continue

            means = []
            cis   = []
            for N in EVAL_STEPS:
                mean_N, snr_means = mean_over_snrs_for_steps(
                    wrapper=wrapper,
                    tgt_all=tgt_all,
                    noise_all=noise_all,
                    steps=N,
                    snr_list=SNR_LIST
                )
                lo, hi = bootstrap_ci_bounds(snr_means)
                means.append(mean_N)
                cis.append((lo, hi))

            per_series[label] = dict(steps=list(EVAL_STEPS), mean=means, ci=cis)

        if not per_series:
            print(f"[WARN] Nothing matched PLOT_ONLY for dim={dim}.")
            continue

        # ---- Build Plotly figure for this dimension
        steps = np.array(EVAL_STEPS, dtype=float)
        fig = go.Figure()
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]

        for i, (label, curve) in enumerate(sorted(per_series.items())):
            m = np.array(curve["mean"], dtype=float)
            ci_bounds = np.array(curve["ci"], dtype=float)  # shape: [len(steps), 2]

            fig.add_trace(go.Scatter(
                x=steps,
                y=m,
                mode="lines+markers",
                name=label,
                line=dict(color=palette[i % len(palette)], width=3),
                marker=dict(size=6),
                error_y=dict(
                    type="data",
                    array=ci_bounds[:, 1] - m,        # upper deviation
                    arrayminus=m - ci_bounds[:, 0],   # lower deviation
                    visible=True,
                    thickness=1.2,
                    width=3
                )
            ))

        fig.update_layout(
            title=f"Mean Cosine vs Steps — dim={dim}, ls={ls}",
            xaxis_title="ODE Steps",
            yaxis_title="Mean Cosine Similarity (avg over SNRs)",
            template="plotly_white",
            legend_title="Series",
            yaxis=dict(range=[0.0, 1.05]),
        )

        wandb.log({f"AvgOverSteps/Dim_{dim}": fig})
        OUT_DIR = PROJECT_ROOT / "figures" / "AvgOverSteps"
        out_base = OUT_DIR / f"AvgOverSteps_Dim_{dim}_ls_{ls}"
        save_vector_and_log(fig, out_base)

    wandb.finish()


if __name__ == "__main__":
    main()
