#!/usr/bin/env python3
"""
CPU-only Time & Memory Benchmark — clean labels
Shows *one* value per x-group (the tallest bar only), placed above the bar.
Removes bar-outline artifacts that looked like "slits".
"""

import re, glob, math, time, random
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
import wandb

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------
# Config
# -----------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR  = PROJECT_ROOT / "trained_models" / "Hex"
OUT_DIR         = PROJECT_ROOT / "figures" / "CPU_Benchmark"

DEVICE          = "cpu"            # << CPU ONLY
BATCH_SIZE      = 1                # inference batch for NN models
PTS_PER_DIM     = 64               # evaluation batch size per dimension: 64x64 targets
PTS_METHOD      = "sobol"          # 'sobol' or 'grid'
SNR_EVAL        = 0.50             # representative SNR
FLOW_STEPS      = 5
GRID_RES        = 64
DOPT_COARSE     = 4
WANDB_PROJ      = "Clean_Up"
RUN_NAME        = "cpu_time_mem_benchmark"

# -----------------------
# Project imports
# -----------------------
from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.cleanup_methods import FlowMatching

# -----------------------
# Helpers
# -----------------------
def save_vector_and_log(fig, out_base: Path):
    pio.kaleido.scope.mathjax = None
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg")
    pdf_path = out_base.with_suffix(".pdf")
    fig.write_image(str(svg_path))
    fig.write_image(str(pdf_path))
    wandb.save(str(svg_path)); wandb.save(str(pdf_path))
    art = wandb.Artifact(name=out_base.name.replace("/", "_"), type="plot")
    art.add_file(str(svg_path)); art.add_file(str(pdf_path))
    wandb.log_artifact(art)

def renorm(x, eps=1e-12): return x / (x.norm(dim=-1, keepdim=True) + eps)
def dim_to_r(d): return int(round(math.sqrt(max(d-1, 0)/6)))
def bytes_to_mb(x): return float(x) / (1024.0 ** 2)

def euclidean_mix(z0, z1, snr): return renorm(snr * z1 + (1.0 - snr) * z0)

def discover_groups():
    """
    Discover model checkpoints grouped by (dim, length_scale).
    Keeps exactly one feedforward and one flow checkpoint per group.
    Flow preference order:
        1. geo_det
        2. euc_det
        3. first available flow checkpoint
    """
    pat_dim  = re.compile(r"dim(?P<dim>\d+)_ls(?P<ls>[\d.]+)")
    pat_flow = re.compile(r"drift_(?P<mode>[^.]+)\.pt")

    groups = {}
    for f in glob.glob(str(CHECKPOINT_DIR / "dim*_ls*/*.pt")):
        p = Path(f)
        m_dim = pat_dim.search(p.parent.name)
        if not m_dim:
            continue
        dim = int(m_dim.group("dim"))
        ls  = float(m_dim.group("ls"))
        entry = groups.setdefault((dim, ls), {"flow": None, "ff": None})

        if p.name == "feedforward_ot.pt":
            # Keep only the first feedforward (ignore duplicates)
            if entry["ff"] is None:
                entry["ff"] = f
            continue

        m_flow = pat_flow.match(p.name)
        if m_flow:
            mode = m_flow.group("mode")
            # Choose flow checkpoint by preference
            if entry["flow"] is None:
                entry["flow"] = (f, mode)
            else:
                # Replace if higher priority
                _, current_mode = entry["flow"]
                priority = ["geo_det", "euc_det"]
                def rank(m): return priority.index(m) if m in priority else len(priority)
                if rank(mode) < rank(current_mode):
                    entry["flow"] = (f, mode)

    return groups


def build_eval_batch(ssp_space: HexagonalSSPSpace, device: str):
    tgt_ssps, _ = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=PTS_PER_DIM, method=PTS_METHOD)
    tgt = torch.tensor(tgt_ssps, dtype=torch.float32, device=device)
    tgt = renorm(tgt)
    noise = renorm(torch.randn_like(tgt))
    return tgt, noise  # (N, D)

def load_flow_wrapper(ckpt_path: str, ssp_dim: int, device: str, sampling_mode: str) -> FlowMatching:
    model = ResidualMLP(ssp_dim, flow=True).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return FlowMatching(model=model, num_steps=FLOW_STEPS, sampling=sampling_mode, device=device)

def load_feedforward(ckpt_path: str, ssp_dim: int, device: str) -> torch.nn.Module:
    model = ResidualMLP(ssp_dim, flow=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ---- timing ----
@torch.no_grad()
def time_feedforward_cpu(model: torch.nn.Module, z0_all: torch.Tensor):
    _ = model(z0_all[:min(len(z0_all), 2)])
    t0 = time.perf_counter()
    for i in range(0, z0_all.shape[0], BATCH_SIZE):
        _  = model(z0_all[i:i+BATCH_SIZE])
    return time.perf_counter() - t0

@torch.no_grad()
def time_flow_cpu(wrapper: FlowMatching, z0_all: torch.Tensor, use_sphere: bool):
    _ = wrapper.sample_ode(z_init=z0_all[:min(len(z0_all), 2)], N=FLOW_STEPS, use_sphere=use_sphere)[-1]
    t0 = time.perf_counter()
    for i in range(0, z0_all.shape[0], BATCH_SIZE):
        _  = wrapper.sample_ode(z_init=z0_all[i:i+BATCH_SIZE], N=FLOW_STEPS, use_sphere=use_sphere)[-1]
    return time.perf_counter() - t0

def build_grid_bank(ssp_space: HexagonalSSPSpace, grid_res: int):
    grid_ssps, grid_pts = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=grid_res, method="grid")
    grid_t = torch.tensor(grid_ssps, dtype=torch.float32)  # CPU (G, D)
    mem_mb = bytes_to_mb(grid_t.numel() * grid_t.element_size())  # float32
    return grid_t, grid_pts, mem_mb

def time_grid_from_set_cpu(grid_t: torch.Tensor, z0_all_cpu: np.ndarray):
    _ = torch.from_numpy(z0_all_cpu[:min(len(z0_all_cpu), 2)]).float() @ grid_t.T
    t0 = time.perf_counter()
    sims = torch.from_numpy(z0_all_cpu).float() @ grid_t.T  # (N, G)
    _ = sims.argmax(dim=1)
    return time.perf_counter() - t0

def time_direct_optim_cpu(ssp_space: HexagonalSSPSpace,
                          z0_all_cpu: np.ndarray,
                          coarse_grid_t: torch.Tensor,
                          coarse_pts,
                          num_trials: int = 512):
    N = min(num_trials, z0_all_cpu.shape[0])
    t0 = time.perf_counter()
    sims = torch.from_numpy(z0_all_cpu[:N]).float() @ coarse_grid_t.T  # (N, Gc)
    _ = sims.argmax(dim=1).cpu().numpy()
    t_nn = time.perf_counter() - t0
    t1 = time.perf_counter()
    for i in range(N):
        ssp_i = np.atleast_2d(z0_all_cpu[i])
        _ = ssp_space.decode(
            ssp_i, method='direct-optim', sampling_method='grid', num_samples=DOPT_COARSE
        )
    t_refine = time.perf_counter() - t1
    return t_nn + t_refine, N

# -----------------------
# Plot helpers
# -----------------------
def tallest_labels_per_group(series_dict, x_labels):
    """
    series_dict: {name: list_of_y}
    Returns: {name: list_of_texts} where only the tallest bar in each x position gets a label.
    """
    names = list(series_dict.keys())
    Y = np.vstack([np.array(series_dict[n], dtype=float) for n in names])  # (K, X)
    # treat NaN as -inf so it never wins
    Y_compare = np.where(np.isfinite(Y), Y, -np.inf)
    winners = np.argmax(Y_compare, axis=0)  # (X,)
    texts = {n: [""] * len(x_labels) for n in names}
    for j, name_idx in enumerate(winners):
        val = Y[name_idx, j]
        if np.isfinite(val):
            texts[names[name_idx]][j] = f"{val:.2f}"
    return texts

def none_for_nans(values):
    v = np.array(values, dtype=float)
    return [None if not np.isfinite(t) else float(t) for t in v]

# -----------------------
# Main
# -----------------------
def main():
    wandb.init(project=WANDB_PROJ, name=RUN_NAME, config=dict(
        device=DEVICE, pts_per_dim=PTS_PER_DIM, snr=SNR_EVAL,
        flow_steps=FLOW_STEPS, grid_res=GRID_RES, dopt_coarse=DOPT_COARSE
    ))

    # -------------- discover_groups with ls as STRING --------------
    pat_dim  = re.compile(r"dim(?P<dim>\d+)_ls(?P<ls>[\d.]+)")
    pat_flow = re.compile(r"drift_(?P<mode>[^.]+)\.pt")
    groups = {}
    for f in glob.glob(str(CHECKPOINT_DIR / "dim*_ls*/*.pt")):
        p = Path(f)
        m_dim = pat_dim.search(p.parent.name)
        if not m_dim:
            continue
        dim = int(m_dim.group("dim"))
        ls_str = m_dim.group("ls")  # keep as string
        entry = groups.setdefault((dim, ls_str), {"flows": [], "ff": None})
        if p.name == "feedforward_ot.pt":
            entry["ff"] = f
        else:
            m_flow = pat_flow.match(p.name)
            if m_flow:
                entry["flows"].append((f, m_flow.group("mode")))
    for k in groups:
        groups[k]["flows"].sort(key=lambda t: t[1])

    # -------------- Filter only one LS (string match) --------------
    TARGET_LS = "0.1"
    groups = {k: v for k, v in groups.items() if k[1] == TARGET_LS}
    print(f"✅ Using only length_scale = {TARGET_LS}")
    print(f"Loaded {len(groups)} dimensions:", [k[0] for k in groups.keys()])

    colors = {
        "grid_from_set": "#7f7f7f",
        "direct_optim":  "#B22222",
        "feedforward":   "#1f77b4",
        "flow_5steps":   "#2ca02c",
    }

    dims_list, time_bars, mem_bars = [], {k: [] for k in colors}, {k: [] for k in colors}

    # ---------------------------------------------------------
    # Main loop per dimension
    # ---------------------------------------------------------
    for (dim, ls_str), info in sorted(groups.items()):
        ls = float(ls_str)
        print(f"\n== Dimension {dim}, ls={ls} ==")
        r = dim_to_r(dim)
        ssp_space = HexagonalSSPSpace(
            domain_dim=2, ssp_dim=dim,
            domain_bounds=np.array([[2, 3], [2, 3]]),
            length_scale=ls, n_rotates=r, n_scales=r
        )

        tgt_all, noise_all = build_eval_batch(ssp_space, DEVICE)
        z0_all = euclidean_mix(noise_all, tgt_all, SNR_EVAL)
        z0_cpu = z0_all.detach().cpu().numpy()

        # --- Grid baseline ---
        grid_t, grid_pts, grid_mem_mb = build_grid_bank(ssp_space, GRID_RES)
        dt_grid = time_grid_from_set_cpu(grid_t, z0_cpu)
        ms_grid = 1000.0 * dt_grid / float(z0_all.shape[0])
        time_bars["grid_from_set"].append(ms_grid)
        mem_bars["grid_from_set"].append(grid_mem_mb)

        # --- Direct-optim baseline ---
        coarse_t, coarse_pts, coarse_mem_mb = build_grid_bank(ssp_space, DOPT_COARSE)
        dt_dopt, n_trials = time_direct_optim_cpu(
            ssp_space, z0_cpu, coarse_t, coarse_pts, num_trials=min(512, z0_cpu.shape[0])
        )
        ms_dopt = 1000.0 * dt_dopt / float(n_trials)
        time_bars["direct_optim"].append(ms_dopt)
        mem_bars["direct_optim"].append(coarse_mem_mb)

        # --- Feedforward ---
        if info.get("ff") is not None:
            ff = load_feedforward(info["ff"], dim, DEVICE)
            ff_param_mb = bytes_to_mb(count_params(ff) * 4)
            dt_ff = time_feedforward_cpu(ff, z0_all)
            ms_ff = 1000.0 * dt_ff / float(z0_all.shape[0])
            time_bars["feedforward"].append(ms_ff)
            mem_bars["feedforward"].append(ff_param_mb)
        else:
            print(f"⚠️ No feedforward checkpoint found for dim={dim}, ls={ls}")
            time_bars["feedforward"].append(float("nan"))
            mem_bars["feedforward"].append(float("nan"))

        # --- Flow ---
        flow_choice = None
        flow_list = info.get("flows", [])
        if flow_list:
            for path, mode in flow_list:
                if mode == "geo_det":
                    flow_choice = (path, mode)
                    break
            if flow_choice is None:
                flow_choice = flow_list[0]
        else:
            print(f"⚠️ No flow checkpoints found for dim={dim}, ls={ls}")

        if flow_choice is not None:
            fpath, mode = flow_choice
            wrapper = load_flow_wrapper(fpath, dim, DEVICE, sampling_mode=mode)
            flow_param_mb = bytes_to_mb(count_params(wrapper.model) * 4)
            dt_flow = time_flow_cpu(wrapper, z0_all, use_sphere=mode.startswith("geo_"))
            ms_flow = 1000.0 * dt_flow / float(z0_all.shape[0])
            time_bars["flow_5steps"].append(ms_flow)
            mem_bars["flow_5steps"].append(flow_param_mb)
        else:
            time_bars["flow_5steps"].append(float("nan"))
            mem_bars["flow_5steps"].append(float("nan"))

        wandb.log({
            "dim": dim, "ls": ls,
            "time_ms_per_sample/grid_from_set": ms_grid,
            "time_ms_per_sample/direct_optim":  ms_dopt,
            "time_ms_per_sample/feedforward":   time_bars["feedforward"][-1],
            "time_ms_per_sample/flow_5steps":   time_bars["flow_5steps"][-1],
            "mem_MB/grid_from_set":             grid_mem_mb,
            "mem_MB/direct_optim_init_grid":    coarse_mem_mb,
            "mem_MB/feedforward_params":        mem_bars["feedforward"][-1],
            "mem_MB/flow_params":               mem_bars["flow_5steps"][-1],
        })

        dims_list.append(dim)

    # ---------------------------------------------------------
    # Plot A: CPU Time vs Dimensionality
    # ---------------------------------------------------------
    x = [str(d) for d in dims_list]
    fig_time = go.Figure()

    for key, color in colors.items():
        y_vals = np.array(time_bars[key], dtype=float)
        fig_time.add_trace(go.Bar(
            x=x,
            y=y_vals,
            name=key,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:.2f} ms/sample<extra>%{fullData.name}</extra>",
        ))

    # ➕ Numeric labels on top of bars
    for trace in fig_time.data:
        if isinstance(trace, go.Bar):
            trace.text = [f"{y:.2f}" if np.isfinite(y) else "" for y in trace.y]
            trace.textposition = "outside"
            trace.texttemplate = "%{text}"

    fig_time.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.25,
        uniformtext_minsize=10,
        uniformtext_mode="show",
        title=f"CPU Compute Time (ms/sample) vs SSP Dimensionality (ls={TARGET_LS})",
        xaxis_title="SSP Dimensionality",
        yaxis_title="Time (ms/sample)",
        template="plotly_white",
        legend_title="Method"
    )

    wandb.log({"CPU_Time_vs_Dim": fig_time})
    save_vector_and_log(fig_time, OUT_DIR / "CPU_Time_vs_Dim")

    # ---------------------------------------------------------
    # Plot B: Memory (log scale)
    # ---------------------------------------------------------
    fig_mem_log = go.Figure()
    for key, color in colors.items():
        y_vals = np.array(mem_bars[key], dtype=float)
        fig_mem_log.add_trace(go.Bar(
            x=x,
            y=y_vals,
            name=key,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:.2f} MB<extra>%{fullData.name}</extra>",
        ))

    # ➕ Numeric labels on top of bars
    for trace in fig_mem_log.data:
        if isinstance(trace, go.Bar):
            trace.text = [f"{y:.2f}" if np.isfinite(y) else "" for y in trace.y]
            trace.textposition = "outside"
            trace.texttemplate = "%{text}"

    fig_mem_log.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.25,
        title=f"Memory Footprint vs SSP Dimensionality (ls={TARGET_LS})",
        xaxis_title="SSP Dimensionality",
        yaxis_title="Memory (MB)",
        template="plotly_white",
        legend_title="Method"
    )
    fig_mem_log.update_yaxes(type="log")

    wandb.log({"Memory_vs_Dim_log": fig_mem_log})
    save_vector_and_log(fig_mem_log, OUT_DIR / "Memory_vs_Dim_log")


if __name__ == "__main__":
    main()
