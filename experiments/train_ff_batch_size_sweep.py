#!/usr/bin/env python3
import time
from pathlib import Path
import numpy as np
import torch
import wandb
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.run import FeedforwardTrainer
from utils.evaluation_utils import make_unitary
from cleanup_ssps.model import ResidualMLP
from torch.utils.data import DataLoader
from cleanup_ssps.dataset import SSPDataset

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

DIM = 487
LENGTH_SCALE = 0.2
EPOCHS = 100
LR = 1e-4
WD = 5e-4
VAL_SPLIT = 0.1
GRID_RES = 128
BASELINE_TRIALS = 256

BATCH_SIZES = [32, 64, 128, 256, 512, 2048]
SIGNAL_STRENGTHS = np.linspace(0.0, 1.0, 21).round(3).tolist()

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "train" / "coordinate_ssps")
TEST_DIR_TMPL = str(PROJECT_ROOT / "data" / "dim_{dim}_scale_{ls}" / "test" / "coordinate_ssps")
OUT_DIR = PROJECT_ROOT / "trained_models" / "ff_batch_size_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_test_pair_cache = {}        # (id(space), snr, tpl) -> tensors
_baseline_grid_cache = {}    # (id(space), grid_res, method) -> grid

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def renorm(x, eps=1e-12): 
    return x / (x.norm(dim=-1, keepdim=True) + eps)

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
        ssp_dim=DIM,
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


def eval_cosine(model, ssp_space, snr, batch_size, device, bootstrap_samples=200):
    sims_all = []
    _, tgt_all = _load_test_pairs(
        ssp_space=ssp_space, snr=snr, data_dir_template=TEST_DIR_TMPL,
        dim=DIM, length_scale=LENGTH_SCALE,
        batch_size=batch_size, device=device
    )
    with torch.no_grad():
        chunk = 512
        for i in range(0, tgt_all.shape[0], chunk):
            z1 = tgt_all[i:i+chunk].to(device)
            z_noise = renorm(torch.randn_like(z1))
            z_init = renorm(snr * z1 + (1.0 - snr) * z_noise)
            out = model(z_init)

            out = make_unitary(out)
            out = out / out.norm(dim=1, keepdim=True)
            sims_all.append((out * z1).sum(dim=1).cpu().numpy())

    sims_all = np.concatenate(sims_all)
    mean = sims_all.mean()
    ci = bootstrap_ci(sims_all, num_samples=bootstrap_samples)
    return {"mean_cosine": mean, "ci95_cosine": ci}

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project="Clean_Up", name="feedforward_OT_batch_sweep")

    ssp_space = HexagonalSSPSpace(
        domain_dim=2,
        ssp_dim=DIM,
        domain_bounds=np.array([[2,3],[2,3]]),
        length_scale=LENGTH_SCALE,
        n_rotates=9,
        n_scales=9
    )

    results = {}

    for bs in BATCH_SIZES:
        print(f"\n🧠 Training feedforward_OT, batch_size={bs}")
        arch = ResidualMLP(DIM, flow=False).to(device)

        trainer = FeedforwardTrainer(
            encoded_dim     = DIM,
            architecture    = arch,
            data_dir        = DATA_DIR_TMPL.format(dim=DIM, ls=LENGTH_SCALE),
            batch_size      = bs,
            epochs          = EPOCHS,
            lr              = LR,
            weight_decay    = WD,
            val_split       = VAL_SPLIT,
            signal_strength = 0.0,
            noise_type      = "uniform_hypersphere",
            target_type     = "coordinate",
            device          = device,

            use_ot_train = True,
            ot_method    = "exact",
            ot_reg       = 0,
        )

        ckpt_path = OUT_DIR / f"feedforward_OT_dim{DIM}_bs{bs}.pt"
        if ckpt_path.exists():
            arch.load_state_dict(torch.load(ckpt_path, map_location=device))
            model_ff = arch
            print(f"✅ Loaded existing model for batch size {bs}")
        else:
            model_ff, loss_ff, val_ff = trainer.train()
            torch.save(model_ff.state_dict(), ckpt_path)
            print(f"💾 Saved model to {ckpt_path}")

            for ep, (tr, vl) in enumerate(zip(loss_ff, val_ff)):
                wandb.log({"batch_size": bs, "epoch": ep, "train_loss": tr, "val_loss": vl})

        # Evaluate
        mean_list, ci_list = [], []
        for snr in SIGNAL_STRENGTHS:
            cos = eval_cosine(model_ff, ssp_space, snr, bs, device)
            mean_list.append(cos["mean_cosine"])
            ci_list.append(cos["ci95_cosine"])
            wandb.log({"batch_size": bs, "SNR": snr, "mean_cosine": cos["mean_cosine"], "ci95": cos["ci95_cosine"]})

        results[bs] = {"mean_cosine": mean_list, "ci95_cosine": ci_list}

    # ----------------------------------------------------------------------
    # Plotting comparison across batch sizes
    # ----------------------------------------------------------------------
    x = np.array(SIGNAL_STRENGTHS)
    fig = go.Figure()

    for bs, vals in results.items():
        m = np.array(vals["mean_cosine"])
        c = np.array(vals["ci95_cosine"])
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([m-c, (m+c)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,0,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"bs={bs} 95% CI",
            legendgroup=str(bs),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=m, error_y=dict(type="data", array=c),
            mode="lines+markers",
            name=f"Batch {bs}",
            legendgroup=str(bs)
        ))

    fig.update_layout(
        title=f"Feedforward_OT Performance vs Batch Size (dim={DIM})",
        xaxis_title="Signal Strength (SNR)",
        yaxis_title="Mean Cosine Similarity",
        yaxis=dict(range=[0.0, 1.05]),
        legend=dict(title="Batch Size")
    )

    wandb.log({"Feedforward_OT_Batch_Comparison": fig})
    fig.write_image(str(OUT_DIR / "Feedforward_OT_Batch_Comparison.pdf"))
    fig.write_image(str(OUT_DIR / "Feedforward_OT_Batch_Comparison.svg"))
    print(f"\n📊 Saved comparison plots to {OUT_DIR}")
