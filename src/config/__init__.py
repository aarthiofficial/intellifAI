"""
Configuration module for model, data, and training settings.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent
WEIGHTS_DIR = PROJECT_ROOT / "pretrained_weights"

# Example config values
MODEL_NAME = "resnet50"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / f"{MODEL_NAME}.pth"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "WEIGHTS_DIR",
    "MODEL_NAME",
    "DEFAULT_WEIGHTS_PATH",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS"
]
