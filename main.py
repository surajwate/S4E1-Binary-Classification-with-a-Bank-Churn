import os
import platform
import subprocess
import argparse
from src import model_dispatcher
from src import config
import logging

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    return parser.parse_args()

def validate_model(selected_model, available_models):
    """Validate the selected model."""
    if selected_model not in available_models:
        raise ValueError(f"Invalid model. Choose from {list(available_models.keys())}")

def construct_commands(selected_model, os_type):
    """Construct the training commands based on the OS type."""
    commands = []
    if os_type == "Windows":
        commands = [f"python ./src/train.py --fold {fold} --model {selected_model}" for fold in range(5)]
    elif os_type in ("Linux", "Darwin"):  # Darwin is for macOS
        commands = [f"python3 ./src/train.py --fold {fold} --model {selected_model}" for fold in range(5)]
    else:
        raise Exception(f"Unsupported OS: {os_type}")
    return commands

def run_commands(commands):
    """Run the list of shell commands."""
    for command in commands:
        subprocess.run(command, shell=True)

def main():
    # Parse arguments
    args = parse_arguments()
    selected_model = args.model

    # Dynamically get the available models from model_dispatcher
    available_models = model_dispatcher.models

    # Validate the model
    validate_model(selected_model, available_models)

    # Get the OS type
    os_type = platform.system()

    # Construct and run commands
    commands = construct_commands(selected_model, os_type)
    logging.info(f"===== Starting training for model: {selected_model} ======")
    run_commands(commands)
    logging.info(f"Training completed for {selected_model} model.")
    logging.info("-----------------------------------")

if __name__ == "__main__":
    main()
