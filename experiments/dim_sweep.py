#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import plotly.graph_objects as go
import plotly.io as pio
import kaleido
pio.kaleido.scope.mathjax = None

from torch.utils.data import DataLoader

from cleanup_ssps.sspspace import HexagonalSSPSpace, RandomSSPSpace
from cleanup_ssps.run import FeedforwardTrainer, FlowTrainer
from cleanup_ssps.dataset import SSPDataset
from utils.evaluation_utils import make_unitary
from cleanup_ssps.model import ResidualMLP

# -----------------------------------------------------------------------------
# Absolute paths
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent     # one level up from experiments/
CACHE_ROOT   = PROJECT_ROOT / "trained_models" / "Hex"

DATA_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "train" / "coordinate_ssps")
TEST_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "test"  / "coordinate_ssps")

# --- EXPERIMENT CONFIG ---
rs               = [5, 13]#[3,5,7,11,13]
dims             = [1 + 6 * r * r for r in rs]
length_scale     = 0.1
device           = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE       = 256
EPOCHS           = 150
LR               = 1e-4
WD               = 5e-4
VAL_SPLIT        = 0.1

SIGNAL_STRENGTHS = np.linspace(0.0, 1.0, 31).round(3).tolist()
GRID_RES         = 128
BASELINE_TRIALS  = 256

SIGMA            = 0.01         # noise level for flow training
SB_REG           = 2*SIGMA**2

# -------------------- METHODS --------------------
METHODS = [
    "feedforward",
    # Euclidean
    "euc_det",
    # "euc_ot",
    # "euc_sb",
    # Geodesic (angular OT couplings)
    "geo_det",
    # "geo_amb_const",
    # "geo_tan_const",
    # "geo_amb_sb",
    # "geo_tan_sb",
]

# ---------------------------------------------
# Caches
# ---------------------------------------------
_test_pair_cache = {}        # (id(space), snr, tpl) -> tensors
_baseline_grid_cache = {}    # (id(space), grid_res, method) -> grid

# ---------------------------------------------
# Utils
# ---------------------------------------------
def bootstrap_ci(arr, num_samples=200, ci=95):
    N = len(arr)
    if N == 0:
        return 0.0
    idx = np.random.randint(0, N, size=(num_samples, N))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci)/2, (100 + ci)/2])
    return (hi - lo) / 2

def _load_test_pairs(ssp_space, snr, data_dir_template, dim, length_scale, batch_size, device,
                     noise_type="uniform_hypersphere"):
    key = (id(ssp_space), snr, data_dir_template.format(dim=dim, ls=length_scale))
    if key in _test_pair_cache:
        return _test_pair_cache[key]

    ds = SSPDataset(
        data_dir=data_dir_template.format(dim=dim, ls=length_scale),
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
        raise RuntimeError("No test data loaded.")
    inp_all = torch.cat(inp_list, dim=0)
    tgt_all = torch.cat(tgt_list, dim=0)
    _test_pair_cache[key] = (inp_all, tgt_all)
    return inp_all, tgt_all

def compute_cleanup_baseline(
    ssp_space,
    ssp_dim,
    snr,
    cleanup_method='from-set',          # 'from-set' or 'direct-optim'
    grid_resolution=64,
    num_trials=100,
    device="cpu",
    bootstrap_samples=200
):
    """
    Baselines:
      • 'from-set'    : NN over a grid of clean SSPs (fast, vectorized)
      • 'direct-optim': start from coarse sampling and L-BFGS-B refine (uses .decode)
    Returns mean + 95% CI of cosine; also computes RMSE internally (not returned here).
    """
    # --- Sample ground truth (GT) ---
    gt_ssps, gt_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=num_trials, method='Rd'
    )
    gt_ssps = torch.tensor(gt_ssps, device=device)

    # --- Corrupt GT at specified SNR ---
    z = torch.randn_like(gt_ssps)
    z = z / z.norm(dim=1, keepdim=True)
    corrupted = snr * gt_ssps + (1 - snr) * z

    if cleanup_method == 'from-set':
        grid_key = (id(ssp_space), grid_resolution, 'grid')
        if grid_key not in _baseline_grid_cache:
            grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(
                num_points_per_dim=grid_resolution, method='grid'
            )
            _baseline_grid_cache[grid_key] = {
                "grid_ssps": torch.tensor(grid_ssps, device=device),
                "grid_pts":  grid_pts
            }
        grid_ssps = _baseline_grid_cache[grid_key]["grid_ssps"]  # (G,d)
        grid_pts  = _baseline_grid_cache[grid_key]["grid_pts"]   # (G,2)

        sims = corrupted @ grid_ssps.T
        idx  = sims.argmax(dim=1)
        cleaned_ssps = grid_ssps[idx]
        cleaned_pts  = grid_pts[idx.cpu().numpy()]

    elif cleanup_method == 'direct-optim':
        cleaned_ssp_list, cleaned_pt_list = [], []
        for i in range(num_trials):
            ssp_i = corrupted[i:i+1].detach().cpu().numpy()
            x_hat = ssp_space.decode(
                ssp_i,
                method='direct-optim',
                sampling_method='grid',
                num_samples=grid_resolution
            )
            cleaned_pt_list.append(np.atleast_2d(x_hat))
            ssp_hat = ssp_space.encode(np.atleast_2d(x_hat))
            cleaned_ssp_list.append(ssp_hat)

        cleaned_pts  = np.vstack(cleaned_pt_list)
        cleaned_ssps = torch.tensor(np.vstack(cleaned_ssp_list), device=device)

    else:
        raise ValueError(f"Unknown cleanup_method={cleanup_method!r}")

    # --- Cosine similarity vs GT clean SSPs ---
    cos = (gt_ssps * cleaned_ssps).sum(dim=1) / (
        gt_ssps.norm(dim=1) * cleaned_ssps.norm(dim=1)
    )
    cos_np = cos.detach().cpu().numpy()
    mean_cosine = cos_np.mean()
    ci95_cosine = bootstrap_ci(cos_np, num_samples=bootstrap_samples)

    # (Optional) RMSE (not returned)
    diffs = cleaned_pts - gt_pts
    _rmse = np.linalg.norm(diffs, axis=1)  # noqa: F841

    return {"mean_cosine": mean_cosine, "ci95_cosine": ci95_cosine}

def renorm(x, eps=1e-12): return x / (x.norm(dim=-1, keepdim=True) + eps)

def logmap_sphere(p, q):
    dot = (p*q).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    v = q - dot*p
    return theta * v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))

def expmap_sphere(p, v):
    nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cos(nv)*p + torch.sin(nv)*(v/nv)

def eval_cosine(mode, model, fm_sampler, ssp_space, snr, bootstrap_samples=200):
    """
    Fair eval:
      • Build z_init per SNR from target + fresh noise.
      • geo_* uses geodesic mixing; euc_* uses linear.
      • Flows integrate from t0=snr→1.0; FF maps once.
    """
    sims_all = []
    _, tgt_all = _load_test_pairs(
        ssp_space=ssp_space, snr=snr, data_dir_template=TEST_DIR_TMPL,
        dim=ssp_space.ssp_dim, length_scale=length_scale,
        batch_size=BATCH_SIZE, device=device
    )
    use_geo = mode.startswith("geo_")
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
                if use_geo:
                    out = fm_sampler.sample_ode(
                        z_init=z_init, N=10, use_sphere=use_geo,
                        t0=torch.tensor([[snr]], device=device)
                    )[-1]
                else:
                    out = fm_sampler.sample_ode(
                        z_init=z_init, N=1, use_sphere=use_geo,
                        t0=torch.tensor([[snr]], device=device)
                    )[-1]

            # out = make_unitary(out)
            out = out / out.norm(dim=1, keepdim=True)
            sims_all.append((out * z1).sum(dim=1).cpu().numpy())

    sims_all = np.concatenate(sims_all)
    mean = sims_all.mean()
    ci = bootstrap_ci(sims_all, num_samples=bootstrap_samples)
    return {"mean_cosine": mean, "ci95_cosine": ci}

def drift_ckpt_path(mode: str, dim: int) -> Path:
    folder = CACHE_ROOT / f"dim{dim}_ls{length_scale}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / ("feedforward_ot.pt" if mode == "feedforward" else f"drift_{mode}.pt")

def _ot_policy_for_mode(mode: str):
    """
    Return (use_ot_train, ot_method, ot_reg, ot_cost) for a given mode.
      - Euclidean modes use Euclidean cost.
      - Geodesic modes use angular cost.
    """
    if mode in ("euc_det", "geo_det"):
        return False, "none", None, None

    if mode == "euc_ot":
        return True, "exact",    None,   "euclidean"
    if mode == "euc_sb":
        return True, "sinkhorn", 0.2,   "euclidean"

    if mode in ("geo_amb_const", "geo_tan_const"):
        return True, "exact",    None,   "angular"
    if mode in ("geo_amb_sb", "geo_tan_sb"):
        return True, "sinkhorn", 0.2,   "angular"

    return False, "none", None, None

# ---------------------------------------------
# Vector export helper (SVG/PDF) + W&B logging
# ---------------------------------------------
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

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    wandb.init(
        project="Clean_Up",
        name="dim_sweep_ls0.8",
        config={
            "length_scale":    length_scale,
            "batch_size":      BATCH_SIZE,
            "epochs":          EPOCHS,
            "lr":              LR,
            "wd":              WD,
            "val_split":       VAL_SPLIT,
            "grid_res":        GRID_RES,
            "baseline_trials": BASELINE_TRIALS,
            "model_repeats":   1
        }
    )

    color_map = {
        "euc_det":        "red",
        "euc_ot":         "orange",
        "euc_sb":         "purple",
        "geo_det":        "green",
        "geo_amb_const":  "teal",
        "geo_tan_const":  "darkgreen",
        "geo_amb_sb":     "brown",
        "geo_tan_sb":     "olive",
        "baseline":       "gray",
        "feedforward":    "blue",
    }

    for dim, r in zip(dims, rs):
        print(f"\n→ Dimension {dim} (r={r})")
        ssp_space = HexagonalSSPSpace(
            domain_dim=2,
            ssp_dim=dim,
            domain_bounds=np.array([[2,3],[2,3]]),
            length_scale=length_scale,
            n_rotates=r,
            n_scales=r
        )

        models, samplers = {}, {}
        for mode in METHODS:
            arch = ResidualMLP(dim, flow=(mode != "feedforward")).to(device)

            if mode == "feedforward":
                use_ot, ot_method, ot_reg, ot_cost = _ot_policy_for_mode("euc_det")
                ff_trainer = FeedforwardTrainer(
                    encoded_dim     = dim,
                    architecture    = arch,
                    data_dir        = DATA_DIR_TMPL.format(dim=dim, ls=length_scale),
                    batch_size      = BATCH_SIZE,
                    epochs          = EPOCHS,
                    lr              = LR,
                    weight_decay    = WD,
                    val_split       = VAL_SPLIT,
                    signal_strength = 0.0,
                    noise_type      = "uniform_hypersphere",
                    target_type     = "coordinate",
                    device          = device,

                    use_ot_train = use_ot,
                    ot_method    = ot_method,
                    ot_reg       = ot_reg,
                )

                ckpt = drift_ckpt_path(mode, dim)
                if ckpt.exists():
                    arch.load_state_dict(torch.load(ckpt, map_location=device))
                    model_ff, loss_ff, val_ff = arch, [], []
                else:
                    model_ff, loss_ff, val_ff = ff_trainer.train()
                    if isinstance(model_ff, (tuple, list)):
                        model_ff = model_ff[0]
                    torch.save(model_ff.state_dict(), ckpt)
                    for ep, (tr, vl) in enumerate(zip(loss_ff, val_ff)):
                        wandb.log({"mode":"feedforward","epoch":ep,"dim":dim,"train_loss":tr,"val_loss":vl})

                models[mode], samplers[mode] = model_ff, None

            else:
                use_ot, ot_method, ot_reg, ot_cost = _ot_policy_for_mode(mode)

                trainer = FlowTrainer(
                    encoded_dim     = dim,
                    architecture    = arch,
                    data_dir        = DATA_DIR_TMPL.format(dim=dim, ls=length_scale),
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

                    sigma_min       = SIGMA
                )
                fm = trainer.flow_model

                ckpt = drift_ckpt_path(mode, dim)
                if ckpt.exists():
                    fm.model.load_state_dict(torch.load(ckpt, map_location=device))
                    loss_rf, val_rf = [], []
                else:
                    out_model, loss_rf, val_rf = trainer.train()
                    if isinstance(out_model, (tuple, list)): out_model = out_model[0]
                    torch.save(out_model.state_dict(), ckpt)
                    for ep, (tr, vl) in enumerate(zip(loss_rf, val_rf)):
                        wandb.log({"mode":mode,"epoch":ep,"dim":dim,"train_loss":tr,"val_loss":vl})
                    fm.model.load_state_dict(torch.load(ckpt, map_location=device))
                models[mode], samplers[mode] = fm.model, fm

        # ---- sweep over SNR (cosine mean ± 95% CI only)
        baseline_from = {"mean_cosine": [], "ci95_cosine": []}   # high-res from-set (GRID_RES)
        baseline_dopt = {"mean_cosine": [], "ci95_cosine": []}   # coarse direct-optim (16×16)
        results       = {m: {"mean_cosine": [], "ci95_cosine": []} for m in METHODS}

        for snr in SIGNAL_STRENGTHS:
            # 1) from-set baseline @ GRID_RES
            bs_from = compute_cleanup_baseline(
                ssp_space,
                ssp_dim=dim,
                snr=snr,
                cleanup_method='from-set',
                grid_resolution=GRID_RES,
                num_trials=BASELINE_TRIALS,
                device=device,
                bootstrap_samples=200
            )
            baseline_from["mean_cosine"].append(bs_from["mean_cosine"])
            baseline_from["ci95_cosine"].append(bs_from["ci95_cosine"])

            # 2) direct-optim baseline @ coarse 16×16
            bs_dopt = compute_cleanup_baseline(
                ssp_space,
                ssp_dim=dim,
                snr=snr,
                cleanup_method='direct-optim',
                grid_resolution=16,
                num_trials=BASELINE_TRIALS,
                device=device,
                bootstrap_samples=200
            )
            baseline_dopt["mean_cosine"].append(bs_dopt["mean_cosine"])
            baseline_dopt["ci95_cosine"].append(bs_dopt["ci95_cosine"])

            # 3) method curves (feedforward included here; we just relabel it later)
            for mode in METHODS:
                cos_stats = eval_cosine(
                    mode, models[mode], samplers[mode], ssp_space, snr, bootstrap_samples=200
                )
                results[mode]["mean_cosine"].append(cos_stats["mean_cosine"])
                results[mode]["ci95_cosine"].append(cos_stats["ci95_cosine"])

            # log to wandb (map feedforward -> baseline/feedforward)
            logd = {
                "dim": dim, "SNR": snr,
                "baseline_from/mean_cosine":  baseline_from["mean_cosine"][-1],
                "baseline_from/ci95_cosine":  baseline_from["ci95_cosine"][-1],
                "baseline_dopt/mean_cosine":  baseline_dopt["mean_cosine"][-1],
                "baseline_dopt/ci95_cosine":  baseline_dopt["ci95_cosine"][-1],
            }
            for mode in METHODS:
                log_key = "feedforward_OT" if mode == "feedforward" else mode
                logd[f"{log_key}/mean_cosine"] = results[mode]["mean_cosine"][-1]
                logd[f"{log_key}/ci95_cosine"] = results[mode]["ci95_cosine"][-1]
            wandb.log(logd)

        # ---- plot
        x = np.array(SIGNAL_STRENGTHS)
        fig = go.Figure()

        def add_band(fig, x, mean_list, ci_list, color, name, dash="dash"):
            m = np.array(mean_list); c = np.array(ci_list)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([m-c, (m+c)[::-1]]),
                fill="toself", fillcolor="rgba(0,0,0,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{name} 95% CI", legendgroup=name, showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x, y=m, error_y=dict(type="data", array=c),
                mode="lines+markers", line=dict(color=color, dash=dash),
                name=name, legendgroup=name
            ))

        # baselines
        add_band(fig, x, baseline_from["mean_cosine"], baseline_from["ci95_cosine"], "#666", "baseline/from-set", dash="dash")
        add_band(fig, x, baseline_dopt["mean_cosine"], baseline_dopt["ci95_cosine"],  "#B22222", "baseline/direct-optim", dash="dot")

        # method curves (map feedforward -> baseline/feedforward in label)
        for mode in METHODS:
            label = "feedforward_OT" if mode == "feedforward" else mode
            m = np.array(results[mode]["mean_cosine"])
            c = np.array(results[mode]["ci95_cosine"])

            # CI band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([m-c, (m+c)[::-1]]),
                fill="toself", fillcolor="rgba(0,0,0,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{label} 95% CI", legendgroup=label, showlegend=False
            ))
            # mean line + error bars
            fig.add_trace(go.Scatter(
                x=x, y=m, error_y=dict(type="data", array=c),
                mode="lines+markers",
                line=dict(color=color_map.get(mode, "black")),
                name=label, legendgroup=label
            ))

        fig.update_layout(
            title=f"Cosine Similarity (mean ± 95% CI), dim={dim}",
            xaxis_title="Signal Strength",
            yaxis_title="Mean Cosine Similarity",
            yaxis=dict(range=[0.0, 1.05]),
            legend=dict(groupclick="togglegroup")
        )

        # interactive in W&B + vector exports
        wandb.log({f"Cosine/Dim_{dim}": fig})
        save_vector_and_log(fig, f"Cosine_Dim_{dim}")
