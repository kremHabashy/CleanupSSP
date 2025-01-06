import torch
import numpy as np
from torch.utils.data import DataLoader
from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.cleanup_methods import RectifiedFlow
import plotly.graph_objs as go

from utils.evaluation_utils import compute_cleanup_baseline
from utils.wandb_utils import log_metrics

class EvaluationManager:
    def __init__(self, training_results, test_dir):
        self.results = training_results
        self.test_dir = test_dir
        self.models = {}
        self.training_losses = {}
        self.validation_losses = {}
        self.evaluation_results = {'Signal Strengths': {}, 'steps': {}}

    def prepare_data(self):
        for key, value in self.results.items():
            self.models[key] = value[0]
            self.training_losses[key] = value[1]
            self.validation_losses[key] = value[2]

    def get_models(self):
        return self.models

    def get_training_losses(self):
        return self.training_losses

    def get_validation_losses(self):
        return self.validation_losses

    def compare_parameters(self):
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name}: {param_count:,} parameters")

    def evaluate_model(self, model, dataset, batch_size=128, N=10):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        total_cosine_similarity = 0
        total_samples = 0
        criterion = torch.nn.CosineEmbeddingLoss(reduction='none')  # Use the same criterion as training
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch[0], batch[1]
                inputs, targets = inputs.squeeze(1), targets.squeeze(1)
                
                # Get model outputs
                if hasattr(model, 'flow') and model.flow:
                    rectified_flow = RectifiedFlow(model=model, num_steps=N)
                    outputs = rectified_flow.sample_ode(z_init=inputs, N=N)[-1]
                else:
                    outputs = model(inputs)
                
                # Normalize both outputs and targets (unit vectors)
                outputs = outputs / torch.norm(outputs, p=2, dim=1, keepdim=True)
                targets = targets / torch.norm(targets, p=2, dim=1, keepdim=True)
                
                # Compute cosine similarity as 1 - criterion
                cosine_similarities = 1 - criterion(outputs, targets, torch.ones(outputs.shape[0], device=outputs.device))
                total_cosine_similarity += cosine_similarities.sum().item()
                total_samples += len(cosine_similarities)

        # Return mean cosine similarity
        mean_cosine_similarity = total_cosine_similarity / total_samples
        return mean_cosine_similarity


    def make_unitary(self, ssp):
        fssp = torch.fft.fft(ssp)
        fssp = fssp / torch.maximum(torch.sqrt(fssp.real**2 + fssp.imag**2), torch.tensor(1e-8, device=fssp.device))
        return torch.fft.ifft(fssp).real

    def evaluate_models_over_sig_strengths(self, ssp_space, ssps_per_domain_dim=16, method='sobol', batch_size=250, N=10):
        evaluation_results_sig_strengths = {}

        self.cleanup_baseline = self.evaluate_cleanup_baseline(
            ssp_space, ssp_space.ssp_dim, ssps_per_domain_dim=ssps_per_domain_dim, method=method, batch_size=batch_size
        )

        # Evaluate each model
        for (trainer_name, signal_strength), model in self.models.items():
            print(f"Evaluating {trainer_name} over Signal Strengths...")
            signal_strengths, avg_cosine_similarities = self.evaluate_over_sig_strengths(
                model, ssp_space, ssps_per_domain_dim, method, batch_size, N
            )
            if signal_strength not in evaluation_results_sig_strengths:
                evaluation_results_sig_strengths[signal_strength] = {}

            evaluation_results_sig_strengths[signal_strength][trainer_name] = (signal_strengths, avg_cosine_similarities)

        self.evaluation_results['Signal Strengths'] = evaluation_results_sig_strengths

    def evaluate_over_sig_strengths(self, model, ssp_space, ssps_per_domain_dim=16, method='sobol', batch_size=250, N=10):
        signal_strengths = (1 - torch.linspace(0, 1, steps=11)).tolist()
        avg_cosine_similarities = []

        for signal_strength in signal_strengths:
            # Create a dataset with the specified Signal Strength
            dataset = SSPDataset(
                data_dir= self.test_dir,
                ssp_dim=ssp_space.ssp_dim,
                target_type='coordinate',
                noise_type='uniform_hypersphere',
                signal_strength=signal_strength,
                mode='test'
            )
            avg_cosine_similarity = self.evaluate_model(model, dataset, batch_size, N=N)
            avg_cosine_similarities.append(avg_cosine_similarity)

        return signal_strengths, avg_cosine_similarities


    def evaluate_models_over_steps(self, ssp_space, ssps_per_domain_dim=16, method='sobol', max_steps=50, batch_size=128):
        evaluation_results_steps = {}
        # Iterate over RectifiedFlow models and evaluate over steps
        for (trainer_name, signal_strength), model in self.models.items():  # Adjust unpacking
            if 'RF' in trainer_name:
                print(f"Evaluating {trainer_name} over steps...")
                steps, avg_cosine_similarities = self.evaluate_over_steps(model, ssp_space, ssps_per_domain_dim, method, max_steps, batch_size)
                if signal_strength not in evaluation_results_steps:
                    evaluation_results_steps[signal_strength] = {}
                evaluation_results_steps[signal_strength][trainer_name] = (steps, avg_cosine_similarities)
        self.evaluation_results['steps'] = evaluation_results_steps

    def evaluate_over_steps(self, model, ssp_space, ssps_per_domain_dim=16, method='sobol', max_steps=50, batch_size=128):
        steps = list(range(1, max_steps + 1))
        avg_cosine_similarities = []

        for step in steps:
            # Create a dataset with a default Signal Strength
            dataset = SSPDataset(
                data_dir=self.test_dir,
                ssp_dim=ssp_space.ssp_dim,
                target_type='coordinate',
                noise_type='uniform_hypersphere',
                signal_strength=0.5,  # Default Signal Strength
                mode='test'
            )
            avg_cosine_similarity = self.evaluate_model(model, dataset, batch_size, N=step)
            avg_cosine_similarities.append(avg_cosine_similarity)

        return steps, avg_cosine_similarities


    def evaluate_models(self, ssp_space, ssps_per_domain_dim=16, method='sobol', max_steps=50, batch_size=128):
        # First, evaluate over Signal Strengths
        self.evaluate_models_over_sig_strengths(ssp_space, ssps_per_domain_dim, method, batch_size)
        # Then, evaluate over steps
        self.evaluate_models_over_steps(ssp_space, ssps_per_domain_dim, method, max_steps, batch_size)
        # Finally, plot the results
        self.plot_evaluation_results()


    def evaluate_cleanup_baseline(self, ssp_space, ssp_dim, ssps_per_domain_dim=16, method='sobol', batch_size=250, num_trials=100):
        signal_strengths = (1 - torch.linspace(0, 1, steps=11)).tolist()
        average_cosine_similarities = []

        for signal_strength in signal_strengths:
            mean_cosine_sim = compute_cleanup_baseline(ssp_space, ssp_dim, signal_strength, grid_resolution=ssps_per_domain_dim, num_trials=num_trials)
            average_cosine_similarities.append(mean_cosine_sim)

        return signal_strengths, average_cosine_similarities


    def plot_evaluation_results(self):
        # Plotting for Signal Strengths
        if self.evaluation_results['Signal Strengths']:
            for signal_strength, model_results in self.evaluation_results['Signal Strengths'].items():
                fig_signal_strengths = go.Figure()

                # Plot model results
                for model_name, (sig_strengths, avg_cosine_similarities) in model_results.items():
                    fig_signal_strengths.add_trace(go.Scatter(
                        x=[int(100 * (1 - signal_strength)) for signal_strength in sig_strengths],
                        y=avg_cosine_similarities,
                        mode='lines+markers',
                        name=f"{model_name}",
                        line=dict(shape='linear')
                    ))

                # Plot cleanup baseline
                if self.cleanup_baseline:
                    baseline_sig_strengths, baseline_cosine_similarities = self.cleanup_baseline
                    fig_signal_strengths.add_trace(go.Scatter(
                        x=[int(100 * (1 - signal_strength)) for signal_strength in baseline_sig_strengths],
                        y=baseline_cosine_similarities,
                        mode='lines+markers',
                        name="Cleanup Baseline",
                        line=dict(dash='dash', shape='linear')
                    ))

                fig_signal_strengths.update_layout(
                    title=f"Cleanup Evaluation for Signal Strength={signal_strength}",
                    xaxis_title="Noise Percentage in Test Data",
                    yaxis_title="Average Cosine Similarities",
                    legend_title="Model Name"
                )

                log_metrics({f"Signal_strengths_Evaluation_Signal_strength_{signal_strength}": fig_signal_strengths})


        # Plotting for steps
        if self.evaluation_results['steps']:
            for signal_strength, model_results in self.evaluation_results['steps'].items():
                fig_steps = go.Figure()
                for model_name, (steps, avg_dot_products) in model_results.items():
                    fig_steps.add_trace(go.Scatter(
                        x=steps,
                        y=avg_dot_products,
                        mode='lines+markers',
                        name=f"{model_name}",
                        line=dict(shape='linear')  # Use a valid shape
                    ))

                fig_steps.update_layout(
                    title=f"Steps Evaluation for Signal Strength={signal_strength}",
                    xaxis_title="Number of Steps",
                    yaxis_title="Average Cosine Similarities",
                    legend_title="Model Name"
                )
                log_metrics({f"Steps_Evaluation_Signal_Strength_{signal_strength}": fig_steps})
