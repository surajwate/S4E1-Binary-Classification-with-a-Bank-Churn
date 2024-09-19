import subprocess
from src import model_dispatcher
import os

# Inherit the current environment
env = os.environ.copy()

# Get the full path to the Python executable in the environment
python_executable = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')

# Get the models from model_dispatcher
models = model_dispatcher.models.keys()

# Run each model from model_dispatcher
for model in models:
    print(f"Running model: {model}")
    subprocess.run(["python", "main.py", "--model", model], env=env)
