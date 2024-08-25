import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        config = {}
    return config

config = load_config()
