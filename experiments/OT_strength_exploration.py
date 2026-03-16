#!/usr/bin/env python3
"""
Sweep over different sigma (σ) values for *all uncommented modes* including:
feedforward, feedforward_ot, feedforward_sb, euc_ot, euc_sb,
geo_amb_const, geo_tan_const, geo_amb_sb, geo_tan_sb.

Each sigma controls the minimum noise level during flow training.
Creates subfolders under:
    OT_strength_results/dim{D}_ls{LS}_sigma{σ}/
and logs cosine similarity results to W&B.
"""

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

# project imports
from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.run import FeedforwardTrainer, FlowTrainer
from cleanup_ssps.dataset import SSPDataset
from utils.evaluation_utils import make_unitary


# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_ROOT   = PROJECT_ROOT / "OT_strength_results"

DATA_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "train" / "coordinate_ssps")
TEST_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "test"  / "coordinate_ssps")

rs           = [5,7,11]
dims         = [1 + 6 * r * r for r in rs]
length_scale = 0.2
device       = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE   = 256
EPOCHS       = 80
LR           = 1e-4
WD           = 5e-4
VAL_SPLIT    = 0.1

GRID_RES         = 128
BASELINE_TRIALS  = 256
SIGNAL_STRENGTHS = [round(i * 0.05, 2) for i in range(21)]
SIGMAS           = np.logspace(-2, 1, 5)  # 0.01 → 10

# active modes
METHODS = [
    "feedforward_ot",
    "feedforward_sb",
    "euc_ot",
    "euc_sb",
    "geo_amb_const",
    "geo_tan_const",
    "geo_amb_sb",
    "geo_tan_sb",
]


# ============================================================
# UTILITIES
# ============================================================
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

def bootstrap_ci(arr, num_samples=200, ci=95):
    if len(arr) == 0:
        return 0.0
    idx = np.random.randint(0, len(arr), size=(num_samples, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci)/2, (100 + ci)/2])
    return (hi - lo) / 2

def _load_test_pairs(ssp_space, snr, data_dir_template, dim, length_scale, batch_size, device,
                     noise_type="uniform_hypersphere"):
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
    return torch.cat(inp_list, dim=0), torch.cat(tgt_list, dim=0)

def eval_cosine(mode, model, fm_sampler, ssp_space, snr, bootstrap_samples=200):
    sims_all = []
    _, tgt_all = _load_test_pairs(
        ssp_space=ssp_space, snr=snr,
        data_dir_template=TEST_DIR_TMPL,
        dim=ssp_space.ssp_dim,
        length_scale=length_scale,
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
                out = fm_sampler.sample_ode(
                    z_init=z_init, N=10, use_sphere=use_geo,
                    t0=torch.tensor([[snr]], device=device)
                )[-1]

            out = make_unitary(out)
            sims_all.append((out * z1).sum(dim=1).cpu().numpy())

    sims_all = np.concatenate(sims_all)
    mean = sims_all.mean()
    ci = bootstrap_ci(sims_all, num_samples=bootstrap_samples)
    return {"mean_cosine": mean, "ci95_cosine": ci}

def _ot_policy_for_mode(mode, sigma: float):
    """Return (use_ot_train, ot_method, ot_reg, ot_cost) given the mode and sigma."""
    if mode.endswith("_ot"):
        # Euclidean OT (exact)
        return True, "exact", None, "euclidean"
    if mode.endswith("_sb"):
        # Euclidean Sinkhorn (entropic regularization)
        return True, "sinkhorn", 2 * sigma**2, "euclidean"
    if "geo" in mode:
        if "sb" in mode:
            # Geodesic Sinkhorn
            return True, "sinkhorn", 2 * sigma**2, "angular"
        else:
            # Geodesic exact OT
            return True, "exact", None, "angular"
    # Default (no OT)
    return False, "none", None, None


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
        wandb.alert(title="Vector export failed", text=f"{out_base}: {e}")


# ============================================================
# MAIN EXPERIMENT
# ============================================================
if __name__ == "__main__":
    wandb.init(
        project="Clean_Up",
        name="sigma_sweep_OT_strength_all_modes",
        config={
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WD,
            "batch_size": BATCH_SIZE,
            "val_split": VAL_SPLIT,
            "sigmas": SIGMAS.tolist(),
            "methods": METHODS
        }
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

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

        for mode in METHODS:
            print(f"\n=== Mode: {mode} ===")
            fig = go.Figure()

            for idx, sigma in enumerate(SIGMAS):
                print(f"→ σ={sigma:.3f}")
                folder = CACHE_ROOT / f"dim{dim}_ls{length_scale}_sigma{sigma:.3f}"
                folder.mkdir(parents=True, exist_ok=True)
                ckpt = folder / f"{mode}.pt"

                use_ot, ot_method, ot_reg, ot_cost = _ot_policy_for_mode(mode, sigma)
                is_feedforward = mode.startswith("feedforward")

                arch = ResidualMLP(dim, flow=not is_feedforward).to(device)

                if is_feedforward:
                    trainer = FeedforwardTrainer(
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

                    if ckpt.exists():
                        arch.load_state_dict(torch.load(ckpt, map_location=device))
                        print(f"Loaded feedforward model from {ckpt}")
                        model_ff, fm = arch, None
                    else:
                        model_ff, loss_ff, val_ff = trainer.train()
                        if isinstance(model_ff, (tuple, list)): model_ff = model_ff[0]
                        torch.save(model_ff.state_dict(), ckpt)
                        print(f"Saved feedforward model to {ckpt}")
                        for ep, (tr, vl) in enumerate(zip(loss_ff, val_ff)):
                            wandb.log({"mode": mode, "sigma": sigma, "epoch": ep, "train_loss": tr, "val_loss": vl})
                        fm = None

                else:
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
                        sigma_min       = sigma,
                    )

                    fm = trainer.flow_model
                    if ckpt.exists():
                        fm.model.load_state_dict(torch.load(ckpt, map_location=device))
                        print(f"Loaded flow model from {ckpt}")
                    else:
                        out_model, loss_rf, val_rf = trainer.train()
                        if isinstance(out_model, (tuple, list)): out_model = out_model[0]
                        torch.save(out_model.state_dict(), ckpt)
                        print(f"Saved flow model to {ckpt}")
                        for ep, (tr, vl) in enumerate(zip(loss_rf, val_rf)):
                            wandb.log({"mode": mode, "sigma": sigma, "epoch": ep, "train_loss": tr, "val_loss": vl})

                # ---- Evaluation ----
                means, cis = [], []
                for snr in SIGNAL_STRENGTHS:
                    stats = eval_cosine(mode, arch, fm, ssp_space, snr)
                    means.append(stats["mean_cosine"])
                    cis.append(stats["ci95_cosine"])
                    wandb.log({
                        "dim": dim, "mode": mode, "sigma": sigma, "SNR": snr,
                        "mean_cosine": stats["mean_cosine"], "ci95": stats["ci95_cosine"]
                    })

                x = np.array(SIGNAL_STRENGTHS)
                fig.add_trace(go.Scatter(
                    x=x, y=np.array(means),
                    error_y=dict(type="data", array=np.array(cis)),
                    mode="lines+markers",
                    line=dict(color=colors[idx % len(colors)]),
                    name=f"σ={sigma:.2g}"
                ))

            fig.update_layout(
                title=f"OT Strength Sweep (mode={mode}, dim={dim})",
                xaxis_title="Signal Strength (SNR)",
                yaxis_title="Mean Cosine Similarity",
                legend_title="σ values",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(groupclick="toggleitem"),
            )

            out_base = str(CACHE_ROOT / f"Cosine_dim{dim}_{mode}_sigma_sweep")
            save_vector_and_log(fig, out_base)
            wandb.log({f"SigmaSweep/Dim_{dim}/{mode}": fig})

    print("\n✅ Sigma sweep across all active modes complete.")
