"""
evaluate.py
===========
Evaluates a trained U-Net model on the held-out test set and produces
the complete metric suite required by the project: IoU, Dice,
Precision, Recall, F1, Accuracy, and the Confusion Matrix. Also saves a
JSON report and a confusion matrix figure.

Usage:
    python evaluate.py
    python evaluate.py --model models/final_unet_model.h5
"""

import argparse
import os
import numpy as np

import config
from dataset import build_dataset
from metrics import compute_all_metrics
from utils.helpers import load_model_safely, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the U-Net model on the test set.")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH,
                         help="Path to the trained model file.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def gather_predictions(model, test_ds):
    """
    Run the model over the entire test dataset and collect ground-truth
    and predicted masks into two large numpy arrays for metric
    computation.
    """
    y_true_list, y_pred_list = [], []
    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.append(masks.numpy())
        y_pred_list.append(preds)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    return y_true, y_pred


def main():
    args = parse_args()

    print(f"Loading model from: {args.model}")
    model = load_model_safely(args.model)

    print("Building test dataset...")
    test_ds = build_dataset(
        config.TEST_IMAGE_DIR, config.TEST_MASK_DIR,
        batch_size=args.batch_size, augment=False, shuffle=False,
    )

    print("Running inference on the full test set...")
    y_true, y_pred = gather_predictions(model, test_ds)

    print("Computing metrics: IoU, Dice, Precision, Recall, F1, Accuracy...")
    results = compute_all_metrics(y_true, y_pred)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Accuracy      : {results['accuracy']:.4f}")
    print(f"Precision     : {results['precision']:.4f}")
    print(f"Recall        : {results['recall']:.4f}")
    print(f"F1 Score      : {results['f1_score']:.4f}")
    print(f"Mean IoU      : {results['mean_iou']:.4f}")
    print(f"Mean Dice     : {results['mean_dice']:.4f}")
    print("\nPer-class IoU:")
    for name, val in zip(config.CLASS_NAMES, results["iou_per_class"]):
        print(f"  {name:12s}: {val:.4f}")

    report_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.json")
    save_json(results, report_path)

    print("\nGenerating confusion matrix figure...")
    from visualization import plot_confusion_matrix
    cm_path = os.path.join(config.RESULTS_IMAGES_DIR, "confusion_matrix.png")
    plot_confusion_matrix(np.array(results["confusion_matrix"]), config.CLASS_NAMES, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
