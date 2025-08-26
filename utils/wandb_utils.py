import wandb

api_key = "f7d5d329df98b1d6cd80d721946dd72bf96460de"
def initialize_wandb(project_name, experiment_name, tags=None, config=None, api_key=api_key):
    
    if api_key:
        wandb.login(key=api_key)

    wandb.init(project=project_name, name=experiment_name, tags=tags)
    if config:
        wandb.config.update(config)

def log_metrics(metrics):
    wandb.log(metrics)
