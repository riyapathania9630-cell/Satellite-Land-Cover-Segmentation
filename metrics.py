"""
metrics.py
==========
Evaluation metrics used both as Keras training metrics (IoU, Dice) and
as standalone numpy functions for the final evaluation report
(evaluate.py): IoU, Dice, Precision, Recall, F1, Accuracy, and the
Confusion Matrix.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import config
from losses import dice_coefficient


# --------------------------------------------------------------------------
# KERAS TRAINING METRICS (differentiable, used during model.fit)
# --------------------------------------------------------------------------
def iou_metric(y_true, y_pred, smooth: float = config.DICE_SMOOTH):
    """
    Intersection-over-Union (Jaccard Index) metric, computed per batch
    during training/validation.

    IoU = |A ∩ B| / |A ∪ B|
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(K.round(y_pred))  # round softmax/sigmoid to hard labels
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_metric(y_true, y_pred):
    """Wrapper exposing dice_coefficient as a Keras training metric."""
    return dice_coefficient(y_true, y_pred)


def get_training_metrics():
    """Metrics list passed to model.compile()."""
    return [iou_metric, dice_metric, "accuracy"]


# --------------------------------------------------------------------------
# NUMPY EVALUATION METRICS (used post-hoc on full test set predictions)
# --------------------------------------------------------------------------
def masks_to_labels(mask_batch: np.ndarray) -> np.ndarray:
    """
    Convert a batch of one-hot or softmax masks (N, H, W, C) into integer
    label maps (N, H, W). If the mask is already single-channel binary,
    it is thresholded at 0.5 instead.
    """
    if mask_batch.shape[-1] > 1:
        return np.argmax(mask_batch, axis=-1)
    return (mask_batch[..., 0] > 0.5).astype(np.uint8)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = config.NUM_CLASSES) -> dict:
    """
    Compute the full evaluation metric suite required by the project:
    IoU, Dice, Precision, Recall, F1, Accuracy, and the Confusion Matrix.

    Args:
        y_true: Ground truth masks, shape (N, H, W, C) one-hot/binary.
        y_pred: Predicted masks, shape (N, H, W, C) softmax/sigmoid probs.
        num_classes: Number of segmentation classes.

    Returns:
        Dictionary with scalar metrics (macro-averaged across classes)
        and the raw confusion matrix.
    """
    true_labels = masks_to_labels(y_true).flatten()
    pred_labels = masks_to_labels(y_pred).flatten()

    labels_range = list(range(num_classes)) if num_classes > 1 else [0, 1]

    cm = confusion_matrix(true_labels, pred_labels, labels=labels_range)

    precision = precision_score(true_labels, pred_labels, labels=labels_range,
                                 average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, labels=labels_range,
                           average="macro", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, labels=labels_range,
                   average="macro", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    # Per-class + mean IoU computed directly from the confusion matrix.
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou_per_class = intersection / np.maximum(union, 1)
    mean_iou = float(np.mean(iou_per_class))

    # Dice per class, then averaged.
    dice_per_class = (2 * intersection) / np.maximum(cm.sum(axis=0) + cm.sum(axis=1), 1)
    mean_dice = float(np.mean(dice_per_class))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }
