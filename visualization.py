"""
visualization.py
=================
Generates every figure required by the project specification:
segmentation maps, overlays, difference maps, urbanization/deforestation/
flood maps, training curves, area statistics (pie/bar charts), and the
confusion matrix heatmap.

All functions save a PNG to disk and also return the matplotlib Figure
object so they can be reused interactively (e.g. inside a Colab
notebook).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts & Colab
import matplotlib.pyplot as plt

import config
from utils.helpers import label_mask_to_rgb, ensure_dir


# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})


def _legend_patches():
    """Build matplotlib legend patches for the fixed class color palette."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=np.array(config.CLASS_COLORS[i]) / 255.0, label=name)
        for i, name in enumerate(config.CLASS_NAMES)
    ]


# --------------------------------------------------------------------------
# PHASE 1: SEGMENTATION VISUALIZATIONS
# --------------------------------------------------------------------------
def plot_segmentation_result(original_rgb, ground_truth_labels, predicted_labels, save_path: str):
    """
    Figure showing: Original Image | Ground Truth | Predicted Mask | Overlay.

    Args:
        original_rgb: (H, W, 3) uint8 original satellite image.
        ground_truth_labels: (H, W) integer ground-truth mask, or None.
        predicted_labels: (H, W) integer predicted mask.
        save_path: Output PNG path.
    """
    gt_rgb = label_mask_to_rgb(ground_truth_labels) if ground_truth_labels is not None else None
    pred_rgb = label_mask_to_rgb(predicted_labels)
    overlay = (0.6 * original_rgb + 0.4 * pred_rgb).astype(np.uint8)

    panels = [("Original Image", original_rgb)]
    if gt_rgb is not None:
        panels.append(("Ground Truth", gt_rgb))
    panels.append(("Predicted Mask", pred_rgb))
    panels.append(("Overlay", overlay))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.5))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.legend(handles=_legend_patches(), loc="lower center", ncol=len(config.CLASS_NAMES), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_curves(history_csv: str, save_path: str):
    """
    Plot loss curve and accuracy/IoU curves side-by-side from the
    training history CSV produced by train.py.
    """
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["loss"], label="Train Loss")
    if "val_loss" in df:
        axes[0].plot(df["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metric_cols = [c for c in ["accuracy", "iou_metric", "dice_metric"] if c in df.columns]
    for m in metric_cols:
        axes[1].plot(df[m], label=f"Train {m}")
        val_col = f"val_{m}"
        if val_col in df.columns:
            axes[1].plot(df[val_col], label=f"Val {m}", linestyle="--")
    axes[1].set_title("Accuracy / IoU / Dice Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """Plot the confusion matrix as an annotated heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# PHASE 2 & 3: CHANGE DETECTION VISUALIZATIONS
# --------------------------------------------------------------------------
def plot_before_after_diff(before_rgb, after_rgb, change_map, save_path: str):
    """Figure showing: Before Image | After Image | Difference Map."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(before_rgb); axes[0].set_title("Before Image"); axes[0].axis("off")
    axes[1].imshow(after_rgb); axes[1].set_title("After Image"); axes[1].axis("off")
    axes[2].imshow(change_map, cmap="hot"); axes[2].set_title("Difference Map"); axes[2].axis("off")
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_category_map(binary_mask: np.ndarray, title: str, save_path: str, cmap: str = "Reds"):
    """Plot a single named change-category map (urban/deforestation/flood/etc.)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(binary_mask, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_area_pie_chart(stats: dict, save_path: str):
    """Pie chart of land cover class distribution in the 'after' image."""
    labels = list(stats["per_class"].keys())
    values = [stats["per_class"][name]["after"]["pixels"] for name in labels]
    colors = [np.array(config.CLASS_COLORS[i]) / 255.0 for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Land Cover Distribution (After)")
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_area_bar_chart(stats: dict, save_path: str):
    """Bar chart comparing before/after area (hectares) for every class."""
    labels = list(stats["per_class"].keys())
    before_vals = [stats["per_class"][name]["before"]["area_hectares"] for name in labels]
    after_vals = [stats["per_class"][name]["after"]["area_hectares"] for name in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, before_vals, width, label="Before")
    ax.bar(x + width / 2, after_vals, width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Land Cover Area: Before vs After")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_change_detection_summary(before_labels: np.ndarray, after_labels: np.ndarray,
                                   stats: dict, output_dir: str):
    """
    Orchestrates the full set of Phase 2/3 change-detection figures and
    saves all of them into output_dir. Called automatically at the end
    of change_detection.py.
    """
    from change_detection import compute_change_map, build_category_maps

    ensure_dir(output_dir)
    before_rgb = label_mask_to_rgb(before_labels)
    after_rgb = label_mask_to_rgb(after_labels)
    change_map = compute_change_map(before_labels, after_labels)

    plot_before_after_diff(before_rgb, after_rgb, change_map,
                            os.path.join(output_dir, "before_after_diff.png"))

    category_maps = build_category_maps(before_labels, after_labels)
    plot_category_map(category_maps["urban_expansion"], "Urban Expansion Map",
                       os.path.join(output_dir, "urban_expansion_map.png"), cmap="Reds")
    plot_category_map(category_maps["deforestation"], "Deforestation Map",
                       os.path.join(output_dir, "deforestation_map.png"), cmap="Greens")
    plot_category_map(category_maps["flood"], "Flood Detection Map",
                       os.path.join(output_dir, "flood_map.png"), cmap="Blues")
    plot_category_map(category_maps["bare_land_increase"], "Bare Land Increase Map",
                       os.path.join(output_dir, "bare_land_map.png"), cmap="Oranges")

    plot_area_pie_chart(stats, os.path.join(output_dir, "area_pie_chart.png"))
    plot_area_bar_chart(stats, os.path.join(output_dir, "area_bar_chart.png"))
