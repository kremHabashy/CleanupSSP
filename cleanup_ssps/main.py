import numpy as np
from cleanup_ssps.sspspace import HexagonalSSPSpace
from utils.training import TrainingManager
from utils.evaluation import EvaluationManager
from utils.wandb_utils import initialize_wandb
from utils.config_loader import load_experiments

def main():
    experiments = load_experiments("configs/experiments.yaml")
    
    for experiment in experiments:
        ssp_config = experiment["ssp_config"]
        trainer_config = experiment["trainer_config"]

        n_rotates = experiment["ssp_config"]["n_rotates"]
        n_scales = experiment["ssp_config"]["n_scales"]
        encoded_dim = n_rotates * n_scales * 6
        ssp_config["encoded_dim"] = encoded_dim

        # Initialize W&B
        initialize_wandb(
            project_name="Clean_Up",
            experiment_name=experiment["name"],
            tags=experiment["tags"],
            config={**ssp_config, **trainer_config}
        )

        # Create SSP space using SSP config
        ssp_space = HexagonalSSPSpace(
            domain_dim=2,
            ssp_dim=ssp_config["encoded_dim"],
            domain_bounds= np.tile([0, 1], (2, 1)),
            length_scale=ssp_config["length_scale"],
            n_rotates=ssp_config["n_rotates"],
            n_scales=ssp_config["n_scales"]
        )

        # Create training manager
        training_manager = TrainingManager(ssp_space, trainer_config)
        training_signal_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
        training_results = training_manager.train_with_different_initial_distributions(training_signal_strengths)

        # Evaluation experiments
        evaluation_manager = EvaluationManager(training_results, trainer_config["test_dir"])
        evaluation_manager.prepare_data()
        evaluation_manager.compare_parameters()
        evaluation_manager.evaluate_models(ssp_space, max_steps=10)

if __name__ == "__main__":
    main()
