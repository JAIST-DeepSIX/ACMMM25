"""
There are two ways to run this script:
1. Using the command line:
a. Fix the `config.yaml` file with your parameters.
#    python main.py --config config.yaml

2. Importing the `train` function in another script:
#    from main import train
#    train(model_id="Qwen/Qwen2.5-72B-Instruct", path_load_model=<Your path_load_model>, summary_data_path=<Your summary_data_path>)
"""
import argparse
import yaml
import os
import sys
import argparse
from train_for_classification import train

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        sys.exit(1)
        
def validate_args(args):
    config['learning_rate'] = float(config['learning_rate'])
    if not isinstance(args['model_id'], str) or args['model_id'].strip() == "":
        raise ValueError("model_id must be a non-empty string.")

    if args['path_load_model'] is not None:
        if not os.path.exists(args['path_load_model']):
            raise FileNotFoundError(f"The path_load_model '{args['path_load_model']}' does not exist.")

    if not os.path.exists(args['summary_data_path']):
        raise FileNotFoundError(f"The summary_data_path '{args['summary_data_path']}' does not exist.")

    if not os.path.exists(args['train_save_dir']):
        print(f"Warning: The directory '{args['train_save_dir']}' does not exist. It will be created.")
        os.makedirs(args['train_save_dir'], exist_ok=True)

    if args['learning_rate'] <= 0:
        raise ValueError("learning_rate must be greater than 0.")

    if args['per_device_train_batch_size'] <= 0:
        raise ValueError("per_device_train_batch_size must be a positive integer.")

    if args['per_device_eval_batch_size'] <= 0:
        raise ValueError("per_device_eval_batch_size must be a positive integer.")

    if args['eval_steps'] <= 0:
        raise ValueError("eval_steps must be a positive integer.")

    if args['num_train_epochs'] <= 0:
        raise ValueError("num_train_epochs must be a positive integer.")

    if args['logging_steps'] <= 0:
        raise ValueError("logging_steps must be a positive integer.")

    if args['weight_decay'] < 0:
        raise ValueError("weight_decay cannot be negative.")

    if args['save_total_limit'] <= 0:
        raise ValueError("save_total_limit must be a positive integer.")

    if args['early_stopping_patience'] <= 0:
        raise ValueError("early_stopping_patience must be a positive integer.")

    valid_strategies = ["epoch", "steps", "no"]
    if args['eval_strategy'] not in valid_strategies:
        raise ValueError(f"eval_strategy must be one of {valid_strategies}.")

    if args['save_strategy'] not in valid_strategies:
        raise ValueError(f"save_strategy must be one of {valid_strategies}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")

    args = parser.parse_args()

    config = load_config(args.config)

    validate_args(config)

    try:
        train(
            model_id=config['model_id'],
            path_load_model=config['path_load_model'],
            summary_data_path=config['summary_data_path'],
            train_save_dir=config['train_save_dir'],
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['per_device_train_batch_size'],
            per_device_eval_batch_size=config['per_device_eval_batch_size'],
            eval_steps=config['eval_steps'],
            num_train_epochs=config['num_train_epochs'],
            logging_steps=config['logging_steps'],
            weight_decay=config['weight_decay'],
            eval_strategy=config['eval_strategy'],
            save_strategy=config['save_strategy'],
            load_best_model_at_end=config['load_best_model_at_end'],
            save_total_limit=config['save_total_limit'],
            early_stopping_patience=config['early_stopping_patience']
        )
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)