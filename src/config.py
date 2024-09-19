import logging

# File paths
TRAINING_FILE = "./input/train_folds.csv"
MODEL_OUTPUT = "./models/"
LOG_FILE = "./logs/logs.txt"

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)