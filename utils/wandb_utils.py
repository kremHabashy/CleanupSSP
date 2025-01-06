import wandb

def initialize_wandb(project_name, experiment_name, tags=None, config=None):
    wandb.init(project=project_name, name=experiment_name, tags=tags)
    if config:
        wandb.config.update(config)

def log_metrics(metrics):
    wandb.log(metrics)
