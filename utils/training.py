from utils.wandb_utils import log_metrics
import numpy as np
import plotly.graph_objects as go
from cleanup_ssps.model import MLP_Small, ResidualMLP
from cleanup_ssps.run import FlowTrainer, FeedforwardTrainer

class TrainingManager:
    def __init__(self, ssp_space, trainer_configs, ssp_config,
                 sampling_modes=None):
        self.ssp_space       = ssp_space
        self.trainer_configs = trainer_configs
        self.ssp_config      = ssp_config
        self.device          = trainer_configs.get("device", "cpu")

        self.sampling_modes = sampling_modes or [
            "hyperspherical_fm",
            "deterministic",
            "improved_fm",
            "schrodinger",
            "vp_diffusion"
        ]

    def train(self):
        results = {}

        # 1) Pure feed-forward baseline
        ff_arch = ResidualMLP(self.ssp_space.ssp_dim, flow=False).to(self.device)
        ff_trainer = FeedforwardTrainer(
            encoded_dim = self.ssp_space.ssp_dim,
            data_dir    = self.trainer_configs["data_dir"],
            batch_size  = self.trainer_configs["batch_size"],
            epochs      = self.trainer_configs["epochs"],
            lr          = self.trainer_configs["lr"],
            weight_decay= self.trainer_configs["weight_decay"],
            val_split   = self.trainer_configs["val_split"],
            noise_type  = self.trainer_configs["noise_type"],
            target_type = self.trainer_configs["target_type"],
            architecture= ff_arch,
            device      = self.device,

            # ← enable OT pairing here if you want:
            use_ot_train = True,
            ot_method    = self.trainer_configs.get("ot_method", "sinkhorn"),
            ot_reg       = self.trainer_configs.get("ot_reg",    0.005),
        )
        print("Training FeedForward (with OT pairing)" if ff_trainer.use_ot_train else "… random pairing")
        model_ff, loss_ff, val_ff = ff_trainer.train()
        results[("ResidualMLP_FF", "deterministic")] = ((model_ff,), loss_ff, val_ff)

        for epoch, (tr, vl) in enumerate(zip(loss_ff, val_ff)):
            log_metrics({
                "trainer":   "ResidualMLP_FF",
                "sampling":  "deterministic",
                "epoch":     epoch,
                "train_loss": tr,
                "val_loss":   vl
            })

        # 2) Flow-matching (and diffusion) variants
        E = self.trainer_configs["epochs"]
        for sampling in self.sampling_modes:
            print(f"--- Sampling mode: {sampling} ---")
            rf_arch = ResidualMLP(self.ssp_space.ssp_dim, flow=True).to(self.device)

            rf_trainer = FlowTrainer(
                encoded_dim   = self.ssp_space.ssp_dim,
                architecture  = rf_arch,
                data_dir      = self.trainer_configs["data_dir"],
                batch_size    = self.trainer_configs["batch_size"],
                epochs        = E,
                lr            = self.trainer_configs["lr"],
                weight_decay  = self.trainer_configs["weight_decay"],
                val_split     = self.trainer_configs["val_split"],
                noise_type    = self.trainer_configs["noise_type"],
                target_type   = self.trainer_configs["target_type"],
                device        = self.device,
                sampling_mode = sampling,
                sigma_min     = self.trainer_configs.get("sigma_min", 0.1),
                beta_min      = self.trainer_configs.get("beta_min", 0.1),
                beta_max      = self.trainer_configs.get("beta_max", 20.0),
            )

            models_rf, loss_rf, val_rf = rf_trainer.train()
            results[("ResidualMLP_RF", sampling)] = (models_rf, loss_rf, val_rf)


            for epoch, (tr, vl) in enumerate(zip(loss_rf, val_rf)):
                log_metrics({
                    "trainer":    "ResidualMLP_RF",
                    "sampling":   sampling,
                    "epoch":      epoch,
                    "train_loss": tr,
                    "val_loss":   vl
                })

        # 3) Combined loss curves
        self.plot_training_results(results)
        return results

    def plot_training_results(self, training_results):
        def get_label(name, sampling):
            if name.endswith("_FF"):
                return "FeedForward"
            if sampling == "hyperspherical_fm":
                return "hypersphere_fm"
            if sampling == "deterministic":
                return "detFlowMatching"
            if sampling == "improved_fm":
                return "improvedFlowMatching"
            if sampling == "schrodinger":
                return "SchrodingerBridge"
            if sampling == "vp_diffusion":
                return "DiffusionVP"
            return f"{name} ({sampling})"

        # one plot for all training curves
        fig_train = go.Figure()
        fig_val   = go.Figure()

        for (name, sampling), (_, train_l, val_l) in training_results.items():
            label = get_label(name, sampling)
            fig_train.add_trace(go.Scatter(
                x=list(range(len(train_l))),
                y=train_l,
                mode='lines+markers',
                name=label
            ))
            fig_val.add_trace(go.Scatter(
                x=list(range(len(val_l))),
                y=val_l,
                mode='lines+markers',
                name=label
            ))

        fig_train.update_layout(
            title="Training Losses: All Methods",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Method"
        )
        fig_val.update_layout(
            title="Validation Losses: All Methods",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Method"
        )

        log_metrics({"All_Train_Losses": fig_train})
        log_metrics({"All_Val_Losses":   fig_val})
