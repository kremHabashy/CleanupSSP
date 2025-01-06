import yaml

def load_experiments(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)["experiments"]
