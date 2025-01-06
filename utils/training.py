from utils.wandb_utils import log_metrics
import numpy as np
import plotly.graph_objects as go
from cleanup_ssps.sspspace import HexagonalSSPSpace
from cleanup_ssps.model import MLP
from cleanup_ssps.run import RectifiedFlowTrainer, FeedforwardTrainer
    
class TrainingManager:
    def __init__(self, ssp_space, trainer_configs):
        self.ssp_space = ssp_space
        self.trainer_configs = trainer_configs

    def train_with_different_initial_distributions(self, signal_strengths):
        training_results = {}

        for signal_strength in signal_strengths:
            print(f"Training with Signal Strength: {signal_strength}")
            for trainer_name, architecture in {
                "RF_MLP": MLP(self.ssp_space.ssp_dim, flow=True),  # Rectified Flow
                "FF_MLP": MLP(self.ssp_space.ssp_dim, flow=False),  # Feedforward
            }.items():
                if "FF" in trainer_name:
                    trainer = FeedforwardTrainer(
                        encoded_dim=self.ssp_space.ssp_dim,
                        data_dir=self.trainer_configs["data_dir"],
                        batch_size=self.trainer_configs["batch_size"],
                        epochs=self.trainer_configs["epochs"],
                        lr=self.trainer_configs["lr"],
                        weight_decay=self.trainer_configs["weight_decay"],
                        val_split=self.trainer_configs["val_split"],
                        signal_strength=signal_strength,
                        noise_type=self.trainer_configs["noise_type"],
                        target_type=self.trainer_configs["target_type"],
                        architecture=architecture
                    )
                elif "RF" in trainer_name:
                    trainer = RectifiedFlowTrainer(
                        encoded_dim=self.ssp_space.ssp_dim,
                        data_dir=self.trainer_configs["data_dir"],
                        batch_size=self.trainer_configs["batch_size"],
                        epochs=self.trainer_configs["epochs"],
                        lr=self.trainer_configs["lr"],
                        weight_decay=self.trainer_configs["weight_decay"],
                        val_split=self.trainer_configs["val_split"],
                        signal_strength=signal_strength,
                        noise_type=self.trainer_configs["noise_type"],
                        target_type=self.trainer_configs["target_type"],
                        logit_m=self.trainer_configs.get("logit_m", -1.0),
                        logit_s=self.trainer_configs.get("logit_s", 2.0),
                        architecture=architecture
                    )

                model, loss_curve, val_loss_curve = trainer.train()
                training_results[(trainer_name, signal_strength)] = (model, loss_curve, val_loss_curve)

                # Log training and validation loss to W&B using log_metrics
                for epoch, (train_loss, val_loss) in enumerate(zip(loss_curve, val_loss_curve)):
                    log_metrics({
                        "trainer_name": trainer_name,
                        "signal_strength": signal_strength,
                        "target_type": self.trainer_configs["target_type"],
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })

        self.plot_training_results(training_results)
        return training_results


    def plot_training_results(self, training_results):
        # Group results by Signal Strength and trainer name
        grouped_results = {}
        for (trainer_name, signal_strength), (model, loss_curve, val_loss_curve) in training_results.items():
            if signal_strength not in grouped_results:
                grouped_results[signal_strength] = {}
            grouped_results[signal_strength][trainer_name] = (loss_curve, val_loss_curve)

        # Plot for each Signal Strength
        for signal_strength, results in grouped_results.items():
            # Training loss plots
            fig_train = go.Figure()
            # Validation loss plots
            fig_val = go.Figure()

            for trainer_name, (loss_curve, val_loss_curve) in results.items():
                fig_train.add_trace(go.Scatter(
                    x=list(range(len(loss_curve))),
                    y=loss_curve,
                    mode='lines+markers',
                    name=f'{trainer_name}_train',
                    line=dict(shape='linear')
                ))
                fig_val.add_trace(go.Scatter(
                    x=list(range(len(val_loss_curve))),
                    y=val_loss_curve,
                    mode='lines+markers',
                    name=f'{trainer_name}_val',
                    line=dict(shape='linear', dash='dash')
                ))

            fig_train.update_layout(
                title=f"Training Losses for Signal Strength {signal_strength}",
                xaxis_title="Epochs",
                yaxis_title="Loss",
                legend_title="Model"
            )

            fig_val.update_layout(
                title=f"Validation Losses for Signal Strength {signal_strength}",
                xaxis_title="Epochs",
                yaxis_title="Loss",
                legend_title="Model"
            )

            # Use log_metrics instead of wandb.log
            log_metrics({f"Training_Losses_Signal_Strength_{signal_strength}": fig_train})
            log_metrics({f"Validation_Losses_Signal_Strength_{signal_strength}": fig_val})
