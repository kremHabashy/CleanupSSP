#!/usr/bin/env python3
"""
Performance Benefit Analysis (with CIs) + Baseline
--------------------------------------------------
Plots relative performance gains (Δ mean cosine similarity) for key model
comparisons. Produces two kinds of figures:

1) Δ vs Dimensionality (averaged over SNRs)          → one figure per comparison group
2) Δ vs Noise level / SNR (averaged over dimensions) → one figure per comparison group

Also saves vector SVG/PDF assets (no MathJax badge) and logs them + Plotly figures to W&B.

Baseline:
  - Adds "baseline/from-set" computed via NN over a grid of clean SSPs.

Filtering:
  - PLOT_GROUPS   → which comparison groups to render
  - PLOT_METHODS  → only these methods may appear in any figure (others ignored)
"""

import re, glob, math, os
import numpy as np
from pathlib import Path

import torch
import plotly.graph_objects as go
import plotly.io as pio
import wandb

from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.cleanup_methods import FlowMatching

# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "trained_models" / "Hex"
OUT_DIR        = PROJECT_ROOT / "figures" / "PerfBenefits"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 256
EVAL_STEPS     = 10
SNR_LIST       = [round(i * 0.025, 2) for i in range(41)]
PTS_PER_DIM    = 64        # eval targets per dim for models
PTS_METHOD     = "sobol"

# Baseline params
BASELINE_GRID_RES   = 64       # grid resolution for 'from-set'
BASELINE_TRIALS_PER_DIM = 4    # per-axis (e.g., 16→16x16) for baseline eval set

WANDB_PROJ     = "Clean_Up"
RUN_NAME       = "performance_benefits_with_ci_and_baseline"

# Which groups to plot
PLOT_GROUPS = {
    "Euclidean OT",
    "Geodesic OT",
    "Feedforward OT",
    "Geometry",
    "Baseline (from-set)",
}

# Only methods listed here can appear in any figure
PLOT_METHODS = {
    "euc_det", #"euc_ot", "euc_sb",
    "geo_det", #"geo_amb_const", "geo_amb_sb", "geo_tan_const", "geo_tan_sb",
    "feedforward", "feedforward_ot",
    "baseline/from-set",
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def sanitize_artifact_name(name: str) -> str:
    # Allowed: alnum, dashes, underscores, dots
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def autoscale_y_with_errors(fig, pad=0.05):
    """
    Set y-axis range to tightly fit all traces (including error bars).
    Call this right before saving/logging the figure.
    """
    ymins, ymaxs = [], []
    for tr in fig.data:
        if getattr(tr, "y", None) is None:
            continue
        y = np.asarray(tr.y, dtype=float)
        y_low = y.copy()
        y_high = y.copy()
        err = getattr(tr, "error_y", None)
        if err and getattr(err, "type", None) == "data":
            arr = np.asarray(getattr(err, "array", None), dtype=float) if getattr(err, "array", None) is not None else None
            arrminus = np.asarray(getattr(err, "arrayminus", None), dtype=float) if getattr(err, "arrayminus", None) is not None else None
            if arr is not None and arr.size:
                y_high = np.where(np.isfinite(arr), y + arr, y_high)
            if arrminus is not None and arrminus.size:
                y_low = np.where(np.isfinite(arrminus), y - arrminus, y_low)
            elif arr is not None and getattr(err, "symmetric", True):
                y_low = np.where(np.isfinite(arr), y - arr, y_low)
        finite = np.isfinite(y_low) & np.isfinite(y_high)
        if np.any(finite):
            ymins.append(np.min(y_low[finite]))
            ymaxs.append(np.max(y_high[finite]))
    if not ymins:
        fig.update_yaxes(autorange=True)
        return
    ymin = float(np.min(ymins))
    ymax = float(np.max(ymaxs))
    if ymin == ymax:
        eps = 1e-6 if ymin == 0 else abs(ymin) * 1e-6
        ymin -= eps; ymax += eps
    span = ymax - ymin
    ymin -= pad * span
    ymax += pad * span
    fig.update_yaxes(range=[ymin, ymax])

def save_vector_and_log(fig, out_base: Path):
    """
    Save a Plotly figure as both SVG and PDF directly via Kaleido (no MathJax),
    and log the files to W&B so you can download vector assets.
    """
    # Disable MathJax badge during export
    pio.kaleido.scope.mathjax = None
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    fig.write_image(str(svg_path))
    fig.write_image(str(pdf_path))
    wandb.save(str(svg_path))
    wandb.save(str(pdf_path))
    art = wandb.Artifact(name=sanitize_artifact_name(out_base.name), type="plot")
    art.add_file(str(svg_path)); art.add_file(str(pdf_path))
    wandb.log_artifact(art)

def renorm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine(a, b):
    a = renorm(a); b = renorm(b)
    return (a * b).sum(dim=1)

def geodesic_log(p, q, eps=1e-8):
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    v = q - dot * p
    vn = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return theta * v / vn

def geodesic_exp(p, v, eps=1e-8):
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.cos(nv)*p + torch.sin(nv)*(v/nv)

def geodesic_mix(z0, z1, snr):
    v0 = geodesic_log(z0, z1)
    t  = torch.full((z0.shape[0], 1), float(snr), device=z0.device)
    return geodesic_exp(z0, t * v0)

def euclidean_mix(z0, z1, snr):
    return snr * z1 + (1.0 - snr) * z0

def dim_to_r(d):
    return int(round(math.sqrt(max(d-1, 0)/6)))

def bootstrap_ci_halfwidth(values, num_samples=300, ci=95):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    idx = np.random.randint(0, arr.size, size=(num_samples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100-ci)/2, (100+ci)/2])
    return float((hi - lo) / 2)

def discover_groups():
    """
    Returns: {(dim, ls): {mode_name: path, ...}}
      where mode_name ∈ { 'euc_det', 'euc_ot', 'euc_sb', 'geo_det', ...,
                          'feedforward', 'feedforward_ot', 'feedforward_sb' (if present) }
    """
    pat_dim  = re.compile(r"dim(?P<dim>\d+)_ls(?P<ls>[\d.]+)")
    pat_flow = re.compile(r"drift_(?P<mode>[^.]+)\.pt")
    groups = {}
    for f in glob.glob(str(CHECKPOINT_DIR / "dim*_ls*/*.pt")):
        p = Path(f)
        m_dim = pat_dim.search(p.parent.name)
        if not m_dim:
            continue
        dim = int(m_dim.group("dim")); ls = float(m_dim.group("ls"))
        entry = groups.setdefault((dim, ls), {})
        if p.name == "feedforward.pt":
            entry["feedforward"] = f
        elif p.name in ("feedforward_ot.pt", "feedforward_sb.pt"):
            entry[p.name.replace(".pt","")] = f
        else:
            m_flow = pat_flow.match(p.name)
            if m_flow:
                entry[m_flow.group("mode")] = f
    return groups

def build_eval_targets(ssp_space, device, pts_per_dim=PTS_PER_DIM, method=PTS_METHOD):
    tgt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=pts_per_dim, method=method)
    tgt = torch.tensor(tgt_ssps, dtype=torch.float32, device=device)
    tgt = renorm(tgt)
    base_noise = renorm(torch.randn_like(tgt))
    return tgt, base_noise

def load_flow(ckpt_path, ssp_dim, device, mode):
    model = ResidualMLP(ssp_dim, flow=True).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return FlowMatching(model=model, num_steps=EVAL_STEPS, sampling=mode, device=device)

def load_feedforward(ckpt_path, ssp_dim, device):
    model = ResidualMLP(ssp_dim, flow=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model

@torch.no_grad()
def eval_model_mean_and_curve(model_obj, tgt_all, noise_all, snr_list, mode=None):
    """
    Returns:
      mean_over_snrs (float),
      per_snr_means  (List[float]) aligned with snr_list order.
    """
    geo = (mode is not None and mode.startswith("geo_"))
    per_snr_means = []
    for snr in snr_list:
        z0 = geodesic_mix(noise_all, tgt_all, snr) if geo else euclidean_mix(noise_all, tgt_all, snr)
        cos_vals = []
        for i in range(0, tgt_all.shape[0], BATCH_SIZE):
            z1 = tgt_all[i:i+BATCH_SIZE]; z0_b = z0[i:i+BATCH_SIZE]
            if isinstance(model_obj, FlowMatching):
                zT = model_obj.sample_ode(
                    z_init=z0_b, N=EVAL_STEPS, use_sphere=geo,
                    t0=torch.tensor([[snr]], device=z0_b.device)
                )[-1]
            else:
                zT = model_obj(z0_b)
            zT = renorm(zT)
            cos_vals.append(cosine(zT, z1).cpu().numpy())
        per_snr_means.append(float(np.mean(np.concatenate(cos_vals))))
    return float(np.mean(per_snr_means)), per_snr_means

@torch.no_grad()
def eval_baseline_from_set_curve(ssp_space: HexagonalSSPSpace,
                                 snr_list,
                                 grid_resolution: int = BASELINE_GRID_RES,
                                 trials_per_dim: int = BASELINE_TRIALS_PER_DIM,
                                 device: str = "cpu"):
    """
    Vectorized 'from-set' baseline: for each SNR, sample GTs, corrupt, NN over grid.
    Returns per_snr_means aligned with snr_list.
    """
    # Build evaluation GT set for this baseline (smaller than model eval by default)
    gt_ssps_np, gt_pts_np = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=trials_per_dim, method="Rd"
    )
    gt = torch.tensor(gt_ssps_np, dtype=torch.float32, device=device)  # (T, D); cpu ok
    gt = renorm(gt)

    # Build grid once
    grid_ssps_np, _ = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=grid_resolution, method="grid"
    )
    grid = torch.tensor(grid_ssps_np, dtype=torch.float32, device=device)  # (G, D)
    grid = renorm(grid)

    per_snr_means = []
    for snr in snr_list:
        z = torch.randn_like(gt); z = renorm(z)
        corrupted = renorm(snr * gt + (1 - snr) * z)
        sims = corrupted @ grid.T
        idx  = sims.argmax(dim=1)
        cleaned = grid[idx]
        cos = cosine(cleaned, gt).cpu().numpy()
        per_snr_means.append(float(np.mean(cos)))

    return per_snr_means  # len == len(snr_list)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    wandb.init(project=WANDB_PROJ, name=RUN_NAME, config=dict(
        device=DEVICE,
        eval_steps=EVAL_STEPS,
        snr_list=SNR_LIST,
        pts_per_dim=PTS_PER_DIM,
        pts_method=PTS_METHOD,
        baseline_grid=BASELINE_GRID_RES,
        baseline_trials_per_dim=BASELINE_TRIALS_PER_DIM
    ))

    groups = discover_groups()
    groups = {k: v for k, v in groups.items() if abs(k[1] - 0.1) < 1e-8}
    print("Discovered groups:\n", {k: list(v.keys()) for k, v in groups.items()})


    if not groups:
        print(f"No checkpoints found under {CHECKPOINT_DIR}")
        return

    # Store: per dimension
    # perf_means[d][mode]  = scalar mean over SNRs
    # perf_snrs[d][mode]   = list over SNRs
    perf_means = {}
    perf_snrs  = {}

    # 1) Evaluate models + baseline for each (dim, ls)
    for (dim, ls), models in sorted(groups.items()):
        print(f"→ Evaluating dim={dim} (ls={ls})")
        r = dim_to_r(dim)
        ssp_space = HexagonalSSPSpace(
            domain_dim=2, ssp_dim=dim,
            domain_bounds=np.array([[2,3],[2,3]]),
            length_scale=ls, n_rotates=r, n_scales=r
        )

        # Eval batch for models
        tgt_all, noise_all = build_eval_targets(ssp_space, DEVICE, pts_per_dim=PTS_PER_DIM, method=PTS_METHOD)
        perf_means[dim], perf_snrs[dim] = {}, {}

        # Baseline (from-set) — per SNR curve
        baseline_curve = eval_baseline_from_set_curve(
            ssp_space=ssp_space,
            snr_list=SNR_LIST,
            grid_resolution=BASELINE_GRID_RES,
            trials_per_dim=BASELINE_TRIALS_PER_DIM,
            device="cpu"  # baseline runs fine on CPU
        )
        perf_snrs[dim]["baseline/from-set"] = baseline_curve
        perf_means[dim]["baseline/from-set"] = float(np.mean(baseline_curve))

        # Models
        for mode, path in sorted(models.items()):
            try:
                if mode.startswith("feedforward"):
                    if mode not in PLOT_METHODS:
                        continue
                    model = load_feedforward(path, dim, DEVICE)
                    mean_val, snr_means = eval_model_mean_and_curve(model, tgt_all, noise_all, SNR_LIST)
                    perf_means[dim][mode] = mean_val
                    perf_snrs[dim][mode]  = snr_means

                else:
                    if mode not in PLOT_METHODS:
                        continue
                    wrapper = load_flow(path, dim, DEVICE, mode)
                    mean_val, snr_means = eval_model_mean_and_curve(wrapper, tgt_all, noise_all, SNR_LIST, mode=mode)
                    perf_means[dim][mode] = mean_val
                    perf_snrs[dim][mode]  = snr_means
            except Exception as e:
                print(f"[SKIP] {mode} @ dim={dim}: {e}")
                continue

    dims_sorted = sorted(perf_means.keys())

    # Differences helper
    def diff_vs_dim(a, b, d):
        """
        Return (mean_diff, ci_halfwidth) for b − a at dimension d, or (nan, nan) if missing.
        Mean diff computed from mean-over-SNR; CI from per-SNR differences at that dimension.
        """
        if a in perf_means[d] and b in perf_means[d]:
            mean_diff = perf_means[d][b] - perf_means[d][a]
            snr_diffs = np.array(perf_snrs[d][b]) - np.array(perf_snrs[d][a])
            ci = bootstrap_ci_halfwidth(snr_diffs)
            return mean_diff, ci
        return np.nan, np.nan

    def diff_vs_snr(a, b, snr_idx):
        """
        For a fixed SNR index, average (b−a) across dimensions and return (mean, ci_halfwidth).
        CI via bootstrap across available dimensions.
        """
        diffs_across_dims = []
        for d in dims_sorted:
            if a in perf_snrs[d] and b in perf_snrs[d]:
                arr_a = perf_snrs[d][a]
                arr_b = perf_snrs[d][b]
                if snr_idx < len(arr_a) and snr_idx < len(arr_b):
                    diffs_across_dims.append(arr_b[snr_idx] - arr_a[snr_idx])
        if not diffs_across_dims:
            return np.nan, np.nan
        diffs = np.asarray(diffs_across_dims, dtype=float)
        mean = float(np.nanmean(diffs))
        ci   = bootstrap_ci_halfwidth(diffs)
        return mean, ci

    # 2) Define comparisons
    all_comparisons = {
        "Euclidean OT": [
            ("euc_det", "euc_ot"),
            ("euc_det", "euc_sb"),
        ],
        "Geodesic OT": [
            ("geo_det", "geo_amb_const"),
            ("geo_det", "geo_amb_sb"),
            ("geo_det", "geo_tan_const"),
            ("geo_det", "geo_tan_sb"),
        ],
        "Feedforward OT": [
            # Only included if you have those checkpoints and whitelisted
            ("feedforward", "feedforward_ot"),  # no variants by default; keep placeholder if needed
        ],
        "Geometry": [
            ("euc_det", "geo_det"),
        ],
        # Baseline group compares baseline to all other whitelisted methods (except itself)
        "Baseline (from-set)": [
            ("baseline/from-set", m) for m in sorted(PLOT_METHODS)
            if m != "baseline/from-set"
        ],
    }

    # Filter groups by PLOT_GROUPS
    all_comparisons = {g: pairs for g, pairs in all_comparisons.items() if g in PLOT_GROUPS}

    # Palette
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
               "#bcbd22", "#17becf"]

    # 3) Generate, log, and SAVE vector figures
    for group_name, pairs in all_comparisons.items():
        # Keep only pairs where both methods are whitelisted
        pairs = [(a, b) for (a, b) in pairs if a in PLOT_METHODS and b in PLOT_METHODS]
        if not pairs:
            continue

        # ---------- (A) Δ vs Dimensionality (avg over SNRs) ----------
        fig_dim = go.Figure()
        legend_idx = 0
        for (a, b) in pairs:
            ys, cis = [], []
            for d in dims_sorted:
                y, ci = diff_vs_dim(a, b, d)
                ys.append(y); cis.append(ci)
            ys = np.array(ys, dtype=float)
            cis = np.array(cis, dtype=float)

            # Skip if all NaN (models missing)
            if np.all(np.isnan(ys)):
                continue

            fig_dim.add_trace(go.Scatter(
                x=dims_sorted,
                y=ys,
                mode="lines+markers",
                name=f"{b} − {a}",
                line=dict(color=palette[legend_idx % len(palette)], width=2),
                marker=dict(size=6),
                error_y=dict(
                    type="data",
                    array=cis,
                    arrayminus=cis,
                    thickness=1.2,
                    width=3,
                )
            ))
            legend_idx += 1

        fig_dim.update_layout(
            title=f"Performance Benefits — {group_name} (Δ vs Dim, avg over SNRs)",
            xaxis_title="SSP Dimensionality",
            yaxis_title="Δ Mean Cosine (avg over SNRs ± 95% CI)",
            template="plotly_white",
            legend_title="Comparison",
        )
        autoscale_y_with_errors(fig_dim, pad=0.06)

        wandb.log({f"PerfBenefits_Dim/{group_name.replace(' ','_')}": fig_dim})
        out_base = OUT_DIR / f"PerfBenefits_Dim_{group_name.replace(' ','_')}"
        save_vector_and_log(fig_dim, out_base)

        # ---------- (B) Δ vs Noise (avg over dimensions) ----------
        fig_snr = go.Figure()
        legend_idx = 0
        x_snr = np.array(SNR_LIST, dtype=float)
        for (a, b) in pairs:
            ys, cis = [], []
            for si, _snr in enumerate(SNR_LIST):
                y, ci = diff_vs_snr(a, b, si)
                ys.append(y); cis.append(ci)
            ys = np.array(ys, dtype=float)
            cis = np.array(cis, dtype=float)

            if np.all(np.isnan(ys)):
                continue

            fig_snr.add_trace(go.Scatter(
                x=x_snr,
                y=ys,
                mode="lines+markers",
                name=f"{b} − {a}",
                line=dict(color=palette[legend_idx % len(palette)], width=2),
                marker=dict(size=6),
                error_y=dict(
                    type="data",
                    array=cis,
                    arrayminus=cis,
                    thickness=1.2,
                    width=3,
                )
            ))
            legend_idx += 1

        fig_snr.update_layout(
            title=f"Performance Benefits — {group_name} (Δ vs Signal Strength, avg over dims)",
            xaxis_title="Signal Strength",
            yaxis_title="Δ Mean Cosine (avg over dims ± 95% CI)",
            template="plotly_white",
            legend_title="Comparison",
        )
        autoscale_y_with_errors(fig_snr, pad=0.06)

        wandb.log({f"PerfBenefits_SNR/{group_name.replace(' ','_')}": fig_snr})
        out_base_snr = OUT_DIR / f"PerfBenefits_SNR_{group_name.replace(' ','_')}"
        save_vector_and_log(fig_snr, out_base_snr)

    wandb.finish()


if __name__ == "__main__":
    main()
