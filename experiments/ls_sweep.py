#!/usr/bin/env python3
"""
Length-scale sweep with baselines:
x-axis  : length scale
y-axis  : averaged mean cosine over signal strengths and dimensions
series  : feedforward_OT, euc_det, geo_det (+ optional methods),
          baseline/from-set, baseline/direct-optim
"""

from pathlib import Path
import numpy as np
import torch
import wandb
import plotly.graph_objects as go
import plotly.io as pio
import kaleido  # ensure plotly finds the kaleido engine
pio.kaleido.scope.mathjax = None

from torch.utils.data import DataLoader

from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.run import FeedforwardTrainer, FlowTrainer
from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.model import ResidualMLP

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_ROOT   = PROJECT_ROOT / "trained_models" / "Hex"

DATA_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "train" / "coordinate_ssps")
TEST_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "test"  / "coordinate_ssps")

device           = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE       = 256
EPOCHS           = 100
LR               = 1e-4
WD               = 5e-4
VAL_SPLIT        = 0.1

# Evaluate over these signal levels, then average
SIGNAL_STRENGTHS = [round(i * 0.05, 2) for i in range(21)]

# Dimensions & length scales to average over
DIMS          = [55, 151, 295, 487, 727, 1015]
LENGTH_SCALES = [0.1, 0.2, 0.4, 0.8]

# Methods to plot (extend as needed)
# "feedforward" will be labeled "feedforward_OT" in the plot
METHODS = [
    "feedforward",  # OT-trained feedforward checkpoint name handled below
    "euc_det",
    "geo_det",
    # "euc_ot",
    # "euc_sb",
    # "geo_amb_const",
    # "geo_tan_const",
    # "geo_amb_sb",
    # "geo_tan_sb",
]

# Baseline config
GRID_RES_FROMSET = 128  # high-res grid for from-set
GRID_RES_DOPT    = 16   # coarse grid for direct-optim start
BASELINE_TRIALS  = 256  # number of points sampled per baseline eval

# ---------------- Visuals ----------------
COLOR_MAP = {
    "euc_det":              "red",
    "euc_ot":               "orange",
    "euc_sb":               "purple",
    "geo_det":              "green",
    "geo_amb_const":        "teal",
    "geo_tan_const":        "darkgreen",
    "geo_amb_sb":           "brown",
    "geo_tan_sb":           "olive",
    "feedforward":          "blue",
    "baseline/from-set":    "#666666",
    "baseline/direct-optim":"#B22222",
}

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def bootstrap_ci(arr, num_samples=200, ci=95):
    if len(arr) == 0:
        return 0.0
    arr = np.asarray(arr, dtype=float)
    N = len(arr)
    idx = np.random.randint(0, N, size=(num_samples, N))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, (100 + ci) / 2])
    return (hi - lo) / 2

def renorm(x, eps=1e-12): 
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def logmap_sphere(p, q):
    dot = (p*q).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    v = q - dot*p
    return theta * v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))

def expmap_sphere(p, v):
    nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cos(nv)*p + torch.sin(nv)*(v/nv)

def _ot_policy_for_mode(mode: str):
    if mode in ("euc_det", "geo_det"):
        return False, "none", None, None
    if mode == "euc_ot":
        return True, "exact",    None, "euclidean"
    if mode == "euc_sb":
        return True, "sinkhorn", 0.2,  "euclidean"
    if mode in ("geo_amb_const", "geo_tan_const"):
        return True, "exact",    None, "angular"
    if mode in ("geo_amb_sb", "geo_tan_sb"):
        return True, "sinkhorn", 0.2,  "angular"
    return False, "none", None, None

def drift_ckpt_path(mode: str, dim: int, ls: float) -> Path:
    folder = CACHE_ROOT / f"dim{dim}_ls{ls}"
    folder.mkdir(parents=True, exist_ok=True)
    # feedforward OT checkpoint name:
    return folder / ("feedforward_ot.pt" if mode == "feedforward" else f"drift_{mode}.pt")

def _load_test_pairs(ssp_space, snr, data_dir_template, dim, ls, batch_size, device,
                     noise_type="uniform_hypersphere"):
    ds = SSPDataset(
        data_dir=data_dir_template.format(dim=dim, ls=ls),
        ssp_dim=ssp_space.ssp_dim,
        target_type="coordinate",
        noise_type=noise_type,
        signal_strength=snr,
        mode="test",
        device="cpu",
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    inp_list, tgt_list = [], []
    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.squeeze(1).to(device)
            tgt = tgt.squeeze(1).to(device)
            inp_list.append(inp)
            tgt_list.append(tgt)
    if not inp_list:
        raise RuntimeError(f"No test data for dim={dim}, ls={ls}, snr={snr}")
    inp_all = torch.cat(inp_list, dim=0)
    tgt_all = torch.cat(tgt_list, dim=0)
    return inp_all, tgt_all

def eval_cosine_for(dim: int, ls: float, mode: str, model, fm_sampler, snr: float):
    """
    Evaluate mean cosine at a given (dim, ls, mode, snr).
    """
    # Build space with given length scale
    # Use r such that dim = 1 + 6 r^2  -> r = int(sqrt((dim-1)/6))
    r = int(np.sqrt(max((dim - 1) / 6, 0)))
    ssp_space = HexagonalSSPSpace(
        domain_dim=2,
        ssp_dim=dim,
        domain_bounds=np.array([[2,3],[2,3]]),
        length_scale=ls,
        n_rotates=r,
        n_scales=r
    )

    _, tgt_all = _load_test_pairs(
        ssp_space=ssp_space, snr=snr, data_dir_template=TEST_DIR_TMPL,
        dim=dim, ls=ls, batch_size=BATCH_SIZE, device=device
    )

    use_geo = mode.startswith("geo_")
    sims_all = []
    with torch.no_grad():
        chunk = 512
        for i in range(0, tgt_all.shape[0], chunk):
            z1 = tgt_all[i:i+chunk].to(device)
            z_noise = renorm(torch.randn_like(z1))
            if use_geo:
                v0 = logmap_sphere(z_noise, z1)
                z_init = expmap_sphere(z_noise, snr * v0)
            else:
                z_init = renorm(snr * z1 + (1.0 - snr) * z_noise)

            if fm_sampler is None:
                out = model(z_init)
            else:
                # ODE integrate on-sphere for geodesic; Euclidean step otherwise
                out = fm_sampler.sample_ode(
                    z_init=z_init, N=10, use_sphere=use_geo,
                    t0=torch.tensor([[snr]], device=device)
                )[-1]

            out = out / out.norm(dim=1, keepdim=True)
            sims_all.append((out * z1).sum(dim=1).detach().cpu().numpy())

    sims_all = np.concatenate(sims_all)
    return sims_all.mean()

def _load_model_and_sampler(dim: int, ls: float, mode: str):
    """
    Instantiate architecture, load checkpoint, and return (model, sampler_or_None).
    We do not (re)train here: we only load from your trained_models folder.
    """
    arch = ResidualMLP(dim, flow=(mode != "feedforward")).to(device)

    if mode == "feedforward":
        # FF is evaluated as a direct one-step mapping (no sampler)
        ckpt = drift_ckpt_path(mode, dim, ls)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        arch.load_state_dict(torch.load(ckpt, map_location=device))
        return arch, None

    # Flow model: construct FlowTrainer to access its `flow_model` wrapper for ODE sampling
    use_ot, ot_method, ot_reg, ot_cost = _ot_policy_for_mode(mode)
    trainer = FlowTrainer(
        encoded_dim     = dim,
        architecture    = arch,
        data_dir        = DATA_DIR_TMPL.format(dim=dim, ls=ls),
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        lr              = LR,
        weight_decay    = WD,
        val_split       = VAL_SPLIT,
        signal_strength = 0.0,
        noise_type      = "uniform_hypersphere",
        target_type     = "coordinate",
        device          = device,

        sampling_mode   = mode,

        use_ot_train    = use_ot,
        ot_method       = ot_method,
        ot_reg          = ot_reg,
        ot_cost         = ot_cost,

        sigma_min       = 0.1
    )
    fm = trainer.flow_model
    ckpt = drift_ckpt_path(mode, dim, ls)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    fm.model.load_state_dict(torch.load(ckpt, map_location=device))
    return fm.model, fm

# ---------------- Baselines ----------------
_baseline_grid_cache = {}  # (dim, ls, grid_res) -> (grid_ssps_tensor, grid_pts_np)

def _get_space(dim: int, ls: float) -> HexagonalSSPSpace:
    r = int(np.sqrt(max((dim - 1) / 6, 0)))
    return HexagonalSSPSpace(
        domain_dim=2,
        ssp_dim=dim,
        domain_bounds=np.array([[2,3],[2,3]]),
        length_scale=ls,
        n_rotates=r,
        n_scales=r
    )

def _ensure_grid(ssp_space, dim: int, ls: float, grid_res: int, device: str):
    key = (dim, ls, grid_res)
    if key in _baseline_grid_cache:
        return _baseline_grid_cache[key]
    grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=grid_res, method='grid'
    )
    grid_ssps = torch.tensor(grid_ssps, device=device)
    _baseline_grid_cache[key] = (grid_ssps, grid_pts)
    return _baseline_grid_cache[key]

def baseline_from_set_mean_cosine(dim: int, ls: float, snr: float, device: str) -> float:
    """
    Mean cosine for 'from-set' baseline at a given (dim, ls, snr).
    Uses a vectorized nearest-neighbor over a precomputed grid.
    """
    ssp_space = _get_space(dim, ls)
    # sample GT points/ssps
    gt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=BASELINE_TRIALS, method='Rd')
    gt_ssps = torch.tensor(gt_ssps, device=device)
    # corrupt
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)
    corrupted = snr * gt_ssps + (1 - snr) * z
    # grid
    grid_ssps, _grid_pts = _ensure_grid(ssp_space, dim, ls, GRID_RES_FROMSET, device)
    sims = corrupted @ grid_ssps.T
    idx  = sims.argmax(dim=1)
    cleaned = grid_ssps[idx]
    cos = (gt_ssps * cleaned).sum(dim=1) / (gt_ssps.norm(dim=1) * cleaned.norm(dim=1))
    return float(cos.mean().detach().cpu().item())

def baseline_direct_optim_mean_cosine(dim: int, ls: float, snr: float, device: str) -> float:
    """
    Mean cosine for 'direct-optim' baseline at a given (dim, ls, snr).
    Starts from coarse grid and refines by L-BFGS-B through ssp_space.decode.
    """
    ssp_space = _get_space(dim, ls)
    # sample GT points/ssps
    gt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=BASELINE_TRIALS, method='Rd')
    gt_ssps = torch.tensor(gt_ssps, device=device)
    # corrupt
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)
    corrupted = snr * gt_ssps + (1 - snr) * z

    cleaned_list = []
    for i in range(BASELINE_TRIALS):
        ssp_i = corrupted[i:i+1].detach().cpu().numpy()
        x_hat = ssp_space.decode(
            ssp_i,
            method='direct-optim',
            sampling_method='grid',
            num_samples=GRID_RES_DOPT
        )
        ssp_hat = ssp_space.encode(np.atleast_2d(x_hat))  # (1,d) np
        cleaned_list.append(ssp_hat)
    cleaned = torch.tensor(np.vstack(cleaned_list), device=device)

    cos = (gt_ssps * cleaned).sum(dim=1) / (gt_ssps.norm(dim=1) * cleaned.norm(dim=1))
    return float(cos.mean().detach().cpu().item())

# ---------------- Vector export ----------------
def save_vector_and_log(fig: go.Figure, out_base: str):
    svg, pdf, png = f"{out_base}.svg", f"{out_base}.pdf", f"{out_base}.png"
    try:
        fig.write_image(svg)
        fig.write_image(pdf)
        wandb.save(svg); wandb.save(pdf)
        art = wandb.Artifact(name=out_base.replace("/", "_"), type="plot")
        art.add_file(svg); art.add_file(pdf)
        wandb.log_artifact(art)
    except Exception as e:
        fig.write_image(png, scale=2)
        wandb.save(png)
        wandb.alert(
            title="Vector export failed",
            text=f"{out_base}: {type(e).__name__}: {e}\n"
                 f'Ensure kaleido==0.2.1 is installed in the active venv.',
        )

# -----------------------------------------------------------------------------
# Main: evaluate mean over (SNR × DIMS) for each length-scale and method + baselines
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(
        project="Clean_Up",
        name="length_scale_sweep_with_baselines",
        config={
            "dims":            DIMS,
            "length_scales":   LENGTH_SCALES,
            "batch_size":      BATCH_SIZE,
            "snr_grid":        SIGNAL_STRENGTHS,
            "methods":         METHODS,
            "baseline_trials": BASELINE_TRIALS,
            "grid_res_from":   GRID_RES_FROMSET,
            "grid_res_dopt":   GRID_RES_DOPT,
        }
    )

    # Cache models/samplers per (dim, ls, mode) to avoid reloads
    model_cache = {}

    # For each method/baseline, track mean over dims for each ls; then plot vs ls
    per_series_ls_means = {m: [] for m in METHODS}
    per_series_ls_cis   = {m: [] for m in METHODS}
    # Baseline series
    per_series_ls_means["baseline/from-set"]     = []
    per_series_ls_means["baseline/direct-optim"] = []
    per_series_ls_cis["baseline/from-set"]       = []
    per_series_ls_cis["baseline/direct-optim"]   = []

    for ls in LENGTH_SCALES:
        # Accumulate per-dim averages (over SNR) for CI estimation
        per_series_dim_means = {m: [] for m in METHODS}
        per_series_dim_means["baseline/from-set"]     = []
        per_series_dim_means["baseline/direct-optim"] = []

        for dim in DIMS:
            # --- Baselines over SNR ---
            snr_means_from  = []
            snr_means_dopt  = []
            for snr in SIGNAL_STRENGTHS:
                m_from = baseline_from_set_mean_cosine(dim, ls, snr, device=device)
                m_dopt = baseline_direct_optim_mean_cosine(dim, ls, snr, device=device)
                snr_means_from.append(m_from)
                snr_means_dopt.append(m_dopt)

            dim_avg_from = float(np.mean(snr_means_from))
            dim_avg_dopt = float(np.mean(snr_means_dopt))
            per_series_dim_means["baseline/from-set"].append(dim_avg_from)
            per_series_dim_means["baseline/direct-optim"].append(dim_avg_dopt)

            wandb.log({
                "length_scale": ls,
                "dim": dim,
                "baseline/from-set/mean_over_snr": dim_avg_from,
                "baseline/direct-optim/mean_over_snr": dim_avg_dopt,
            })

            # --- Methods over SNR ---
            for mode in METHODS:
                key = (dim, ls, mode)
                if key not in model_cache:
                    model_cache[key] = _load_model_and_sampler(dim, ls, mode)
                model, sampler = model_cache[key]

                snr_means = []
                for snr in SIGNAL_STRENGTHS:
                    mean_cos = eval_cosine_for(dim, ls, mode, model, sampler, snr)
                    snr_means.append(mean_cos)
                dim_avg = float(np.mean(snr_means))
                per_series_dim_means[mode].append(dim_avg)

                wandb.log({
                    "length_scale": ls,
                    "dim": dim,
                    f"{'feedforward_OT' if mode=='feedforward' else mode}/mean_over_snr": dim_avg
                })

        # Average across dims for this ls; CI via bootstrap across dim-averages
        # Baselines
        for base_key in ("baseline/from-set", "baseline/direct-optim"):
            dim_means = per_series_dim_means[base_key]
            ls_mean = float(np.mean(dim_means)) if len(dim_means) else 0.0
            ls_ci   = float(bootstrap_ci(np.array(dim_means), num_samples=400)) if len(dim_means) else 0.0
            per_series_ls_means[base_key].append(ls_mean)
            per_series_ls_cis[base_key].append(ls_ci)
            wandb.log({
                "length_scale": ls,
                f"{base_key}/mean_over_dims_and_snr": ls_mean,
                f"{base_key}/ci95_over_dims": ls_ci,
            })

        # Methods
        for mode in METHODS:
            dim_means = per_series_dim_means[mode]
            ls_mean = float(np.mean(dim_means)) if len(dim_means) else 0.0
            ls_ci   = float(bootstrap_ci(np.array(dim_means), num_samples=400)) if len(dim_means) else 0.0
            per_series_ls_means[mode].append(ls_mean)
            per_series_ls_cis[mode].append(ls_ci)
            label = "feedforward_OT" if mode == "feedforward" else mode
            wandb.log({
                "length_scale": ls,
                f"{label}/mean_over_dims_and_snr": ls_mean,
                f"{label}/ci95_over_dims": ls_ci,
            })

    # ----------------- Plot: x=length_scale, y=avg accuracy (methods + baselines) -----------------
    x = np.array(LENGTH_SCALES, dtype=float)
    fig = go.Figure()

    def add_series(fig, x, means, cis, color, label):
        m = np.array(means, dtype=float)
        c = np.array(cis, dtype=float)
        # CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([m - c, (m + c)[::-1]]),
            fill="toself", fillcolor="rgba(0,0,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} 95% CI", legendgroup=label, showlegend=False
        ))
        # mean line
        fig.add_trace(go.Scatter(
            x=x, y=m, error_y=dict(type="data", array=c),
            mode="lines+markers",
            line=dict(color=color),
            name=label, legendgroup=label
        ))

    # baselines
    add_series(fig, x, per_series_ls_means["baseline/from-set"],     per_series_ls_cis["baseline/from-set"],     COLOR_MAP["baseline/from-set"],     "baseline/from-set")
    add_series(fig, x, per_series_ls_means["baseline/direct-optim"], per_series_ls_cis["baseline/direct-optim"], COLOR_MAP["baseline/direct-optim"], "baseline/direct-optim")

    # methods
    for mode in METHODS:
        label = "feedforward_OT" if mode == "feedforward" else mode
        add_series(fig, x, per_series_ls_means[mode], per_series_ls_cis[mode], COLOR_MAP.get(mode, "black"), label)

    fig.update_layout(
        title="Mean Cosine vs Length Scale",
        xaxis_title="Length scale",
        yaxis_title="Mean Cosine Similarity",
        yaxis=dict(range=[0.0, 1.05]),
        legend=dict(groupclick="togglegroup")
    )

    wandb.log({"LengthScaleSweep/AvgAccuracy_with_Baselines": fig})
    save_vector_and_log(fig, "LengthScaleSweep_AvgAccuracy_with_Baselines")
