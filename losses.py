"""
losses.py
=========
Loss functions for training the U-Net segmentation model: Dice Loss,
Categorical/Binary Cross-Entropy, and a combined Dice + Cross-Entropy
loss, which typically yields the most stable convergence for imbalanced
land-cover classes (e.g. water/flood pixels are usually a small minority).
"""

import tensorflow as tf
from tensorflow.keras import backend as K

import config


def dice_coefficient(y_true, y_pred, smooth: float = config.DICE_SMOOTH):
    """
    Compute the Dice coefficient (a.k.a. F1 over pixels) between the
    ground truth and predicted masks.

    Dice = (2 * |A ∩ B| + smooth) / (|A| + |B| + smooth)

    Args:
        y_true: Ground truth mask, one-hot or binary.
        y_pred: Predicted probability mask (softmax/sigmoid output).
        smooth: Small constant to avoid division by zero.

    Returns:
        Scalar tensor with the mean Dice coefficient across the batch.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient. Directly optimizes overlap."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Combined loss = Dice Loss + Cross-Entropy Loss.

    Cross-entropy provides stable, well-behaved gradients early in
    training, while Dice loss directly targets the region-overlap metric
    the project is evaluated on and helps with class imbalance. Using
    both together is standard practice in medical/remote-sensing
    segmentation literature.
    """
    if config.USE_CATEGORICAL:
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    else:
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ce = tf.reduce_mean(ce)
    return ce + dice_loss(y_true, y_pred)


def get_loss():
    """
    Factory function that returns the loss to use for model.compile(),
    selected automatically based on config.USE_CATEGORICAL.
    """
    return combined_loss
