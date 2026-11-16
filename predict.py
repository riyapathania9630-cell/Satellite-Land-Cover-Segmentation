"""
predict.py
==========
Runs inference with a trained U-Net model on new satellite images and
saves the predicted segmentation masks and overlay visualizations.

Usage:
    python predict.py --input data/test/images --output outputs/predictions
    python predict.py --input path/to/single_image.png
"""

import argparse
import os
import glob
import numpy as np
import cv2

import config
from dataset import _read_image, resize_image, normalize_image
from utils.helpers import load_model_safely, label_mask_to_rgb, ensure_dir
from metrics import masks_to_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference on satellite images.")
    parser.add_argument("--input", type=str, default=config.TEST_IMAGE_DIR,
                         help="Path to a single image file or a directory of images.")
    parser.add_argument("--output", type=str, default=os.path.join(config.OUTPUT_DIR, "predictions"),
                         help="Directory to save predicted masks and overlays.")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH,
                         help="Path to the trained model file.")
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Overlay transparency (0=only original, 1=only mask).")
    return parser.parse_args()


def predict_single_image(model, image_path: str):
    """
    Run the model on one image and return the original image, the
    predicted color mask, and an overlay blend.

    Args:
        model: Loaded Keras U-Net model.
        image_path: Path to the input image.

    Returns:
        Tuple (original_rgb_uint8, predicted_label_mask, predicted_rgb_mask).
    """
    original = _read_image(image_path)
    resized = resize_image(original)
    normalized = normalize_image(resized)

    pred = model.predict(np.expand_dims(normalized, axis=0), verbose=0)
    pred_labels = masks_to_labels(pred)[0]           # (H, W) int labels
    pred_rgb = label_mask_to_rgb(pred_labels)          # (H, W, 3) color mask

    return resized, pred_labels, pred_rgb


def make_overlay(original_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the original image and the colored predicted mask."""
    return cv2.addWeighted(original_rgb, 1 - alpha, mask_rgb, alpha, 0)


def main():
    args = parse_args()
    ensure_dir(args.output)

    print(f"Loading model from: {args.model}")
    model = load_model_safely(args.model)

    if os.path.isdir(args.input):
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "*.png")) +
            glob.glob(os.path.join(args.input, "*.jpg")) +
            glob.glob(os.path.join(args.input, "*.tif"))
        )
    else:
        image_paths = [args.input]

    if not image_paths:
        raise ValueError(f"No images found at '{args.input}'.")

    print(f"Running inference on {len(image_paths)} image(s)...")
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        original, pred_labels, pred_rgb = predict_single_image(model, path)
        overlay = make_overlay(original, pred_rgb, args.alpha)

        cv2.imwrite(os.path.join(args.output, f"{stem}_mask.png"),
                    cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(args.output, f"{stem}_labels.npy"), pred_labels)

        print(f"  -> {stem}: mask, overlay, and label array saved.")

    print(f"\nAll predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
