"""
utils/helpers.py
=================
Small, reusable helper functions shared across train.py, predict.py,
evaluate.py, change_detection.py, and visualization.py. Keeping them
here avoids code duplication and hardcoded logic scattered across files.
"""

import os
import json
import numpy as np
import tensorflow as tf

import config


def set_global_seed(seed: int = config.RANDOM_SEED):
    """Set random seeds for numpy and TensorFlow for reproducible runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def label_mask_to_rgb(label_mask: np.ndarray) -> np.ndarray:
    """
    Convert an integer label mask (H, W) into a color RGB image (H, W, 3)
    using the fixed palette defined in config.CLASS_COLORS. Used by every
    visualization function so colors stay consistent across all figures.

    Args:
        label_mask: Integer class-label array, values 0..NUM_CLASSES-1.

    Returns:
        uint8 RGB image, shape (H, W, 3).
    """
    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in config.CLASS_COLORS.items():
        rgb[label_mask == class_id] = color
    return rgb


def save_json(data: dict, path: str):
    """Save a dictionary as a pretty-printed JSON file, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON report to: {path}")


def load_model_safely(model_path: str):
    """
    Load a saved Keras model, wiring up the project's custom loss and
    metric functions so the model can be deserialized correctly.

    Args:
        model_path: Path to a .h5 or SavedModel file.

    Returns:
        A compiled tf.keras.Model ready for inference.
    """
    from losses import combined_loss, dice_loss, dice_coefficient
    from metrics import iou_metric, dice_metric

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Train a model first with: python train.py"
        )

    custom_objects = {
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
        "dice_metric": dice_metric,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def ensure_dir(path: str):
    """Create a directory (including parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
    return path
