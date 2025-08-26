import torch
import numpy as np
from torch.utils.data import DataLoader
import plotly.graph_objs as go

from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.cleanup_methods import FlowMatching
from utils.ot_utils import OTPlanSampler
from utils.evaluation_utils import compute_cleanup_baseline, compute_ssp_mean, make_unitary

from utils.wandb_utils import log_metrics

class EvaluationManager:
    def __init__(
        self,
        training_results,
        test_dir,
        device="cpu",
        signal_strengths=None,
        eval_steps=None,
        repeats=5,
        # new OT flags:
        use_ot_eval: bool = False,
        ot_method:    str  = "sinkhorn",
        ot_reg:       float= 0.05,
    ):
        self.results           = training_results
        self.test_dir          = test_dir
        self.device            = device
        self.signal_strengths = signal_strengths or [0.0,0.25,0.5,0.75,1.0]
        self.eval_steps       = eval_steps   or [1,2,5,10,50]
        self.repeats          = repeats

        self.use_ot_eval = use_ot_eval
        if self.use_ot_eval:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)

    def _label(self, name, mode):
        if name.endswith("_FF"):       return "FeedForward"
        if mode=="hyperspherical_fm":  return "HypersphericalFM"
        if mode=="deterministic":      return "detFlowMatching"
        if mode=="improved_fm":        return "improvedFlowMatching"
        if mode=="schrodinger":        return "SchrodingerBridge"
        if mode=="vp_diffusion":       return "DiffusionVP"
        return f"{name} ({mode})"

    def evaluate_model(self, name, mode, model_obj, dataset, batch_size=128, N=10):
        """
        Returns per-sample mean and std of cosine similarities over the test set.
        """
        # Unpack models
        if name.endswith("_FF"):
            ff_model = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
            ff_model = ff_model.to(self.device).eval()
            flow_model, denoiser_model = None, None
        else:
            if isinstance(model_obj, (tuple, list)) and len(model_obj) == 2:
                flow_model, denoiser_model = model_obj
            else:
                flow_model, denoiser_model = model_obj, None
            flow_model = flow_model.to(self.device).eval()
            if denoiser_model is not None:
                denoiser_model = denoiser_model.to(self.device).eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        sims_list = []

        with torch.no_grad():
            for inputs, targets in loader:
                inp_all = inputs.squeeze(1).to(self.device)
                tgt_all = targets.squeeze(1).to(self.device)

                if name.endswith("_FF"):
                    preds = ff_model(inp_all)
                else:
                    # instantiate a fresh FlowMatching for sampling
                    fm = FlowMatching(
                        model=flow_model,
                        sampling=mode,
                        num_steps=N,
                        device=self.device,
                        sigma_min=getattr(flow_model, "sigma_min", 0.1),
                        beta_min=getattr(flow_model, "beta_min", 0.1),
                        beta_max=getattr(flow_model, "beta_max", 20.0),
                    )
                    if denoiser_model is not None:
                        fm.denoiser_model = denoiser_model
                        preds = fm.sample_sde(z_init=inp_all, N=N)[-1]
                    else:
                        preds = fm.sample_ode(z_init=inp_all, N=N)[-1]

                # project back to hypersphere
                preds = make_unitary(preds)
                preds = preds / preds.norm(dim=1, keepdim=True)

                # compute per-sample cosine
                sims = torch.sum(preds * tgt_all, dim=1)
                sims_list.append(sims.cpu())

        # concatenate all samples and compute mean/std
        sims_all = torch.cat(sims_list)  # shape [num_samples]
        return sims_all.mean().item(), sims_all.std().item()

    def evaluate_noise_levels(self, ssp_space, N, batch_size=128):
        """
        Plots avg cosine vs signal_strength for N steps, and
        overlays Euclid‐mean & Geodesic‐mean baselines.
        Uses per-sample std instead of run-to-run variability.
        """
        # compute baseline once (per-sample)
        baseline_mean, baseline_std = compute_cleanup_baseline(
            ssp_space, 
            ssp_dim=ssp_space.ssp_dim,
            snr=None,  # assume function updated to return per-sample stats
            grid_resolution=64, 
            method='sobol', 
            num_trials=2000, 
            device=self.device
        )

        fig = go.Figure()

        # horizontal baseline with CI band
        fig.add_hline(
            y=baseline_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline ±1σ ({baseline_std:.3f})",
            annotation_position="bottom left"
        )

        # each model
        for (name, mode), (model_obj, *_) in self.results.items():
            means, stds = [], []
            for sf in self.signal_strengths:
                ds = SSPDataset(
                    data_dir        = self.test_dir,
                    ssp_dim         = ssp_space.ssp_dim,
                    target_type     = 'coordinate',
                    noise_type      = 'uniform_hypersphere',
                    signal_strength = sf,
                    mode            = 'test'
                )
                steps = 1 if name.endswith("_FF") else N
                mean, std = self.evaluate_model(
                    name, mode, model_obj, ds, batch_size, N=steps
                )
                means.append(mean)
                stds.append(std)

            fig.add_trace(go.Scatter(
                x=self.signal_strengths,
                y=means,
                error_y=dict(type='data', array=stds),
                mode='lines+markers',
                name=self._label(name, mode)
            ))

        fig.update_layout(
            title=f"Avg Cosine vs Signal Strength @ N={N}",
            xaxis_title="Signal Strength",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )
        log_metrics({f"NoiseEval_N{N}": fig})

    def evaluate_steps(self, ssp_space, signal_strength, batch_size=128):
        """
        Plots avg cosine vs number of steps using per-sample std.
        """
        fig = go.Figure()
        for (name, mode), (model_obj, *_) in self.results.items():
            means, stds = [], []
            for N in self.eval_steps:
                if name.endswith("_FF") and N > 1:
                    means.append(None)
                    stds.append(None)
                    continue

                ds = SSPDataset(
                    data_dir        = self.test_dir,
                    ssp_dim         = ssp_space.ssp_dim,
                    target_type     = 'coordinate',
                    noise_type      = 'uniform_hypersphere',
                    signal_strength = signal_strength,
                    mode            = 'test'
                )
                steps = 1 if name.endswith("_FF") else N
                mean, std = self.evaluate_model(
                    name, mode, model_obj, ds, batch_size, N=steps
                )
                means.append(mean)
                stds.append(std)

            xs, ys, errs = [], [], []
            for s, m, e in zip(self.eval_steps, means, stds):
                if m is not None:
                    xs.append(s); ys.append(m); errs.append(e)

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                error_y=dict(type='data', array=errs),
                mode='lines+markers',
                name=self._label(name, mode)
            ))

        fig.update_layout(
            title=f"Avg Cosine vs Steps @ Signal Strength={signal_strength}",
            xaxis_title="Number of Steps",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )
        log_metrics({f"StepsEval_Signal{signal_strength}": fig})

    def run_all(self, ssp_space, batch_size=128):
        # 1) Noise‐level sweeps
        for N in self.eval_steps:
            self.evaluate_noise_levels(ssp_space, N=N, batch_size=batch_size)

        # 2) Steps‐vs‐signal sweeps
        for sf in self.signal_strengths:
            self.evaluate_steps(ssp_space, signal_strength=sf, batch_size=batch_size)

        # 3) Cleanup baseline by grid resolution
        grid_resolutions = [16, 32, 64, 128, 256]
        fig_base = go.Figure()

        for gr in grid_resolutions:
            means, stds = [], []
            for sf in self.signal_strengths:
                mean_sim, std_sim = compute_cleanup_baseline(
                    ssp_space,
                    ssp_dim         = ssp_space.ssp_dim,
                    snr             = sf,
                    grid_resolution = gr,
                    method          = 'sobol',
                    num_trials      = 100,
                    device          = self.device
                )
                means.append(mean_sim)
                stds.append(std_sim)

            fig_base.add_trace(go.Scatter(
                x=self.signal_strengths,
                y=means,
                error_y=dict(type='data', array=stds),
                mode='lines+markers',
                name=f'Grid {gr}×{gr}'
            ))

        fig_base.update_layout(
            title="Cleanup Baseline: Avg Cosine vs Signal Strength",
            xaxis_title="Signal Strength",
            yaxis_title="Avg Cosine Similarity",
            legend_title="Method"
        )

        # add the same two baselines here too
        sims_euc = compute_ssp_mean(
            ssp_space,
            num_samples=2000,
            device=self.device
        )
        fig_base.add_hline(
            y=sims_euc,
            line_dash="dash",
            line_color="gray",
            annotation_text="Euclid‐mean",
            annotation_position="bottom left"
        )

        log_metrics({"Baseline_Cosine_by_Grid": fig_base})
