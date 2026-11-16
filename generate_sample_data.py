"""
generate_sample_data.py
=======================
Creates a small SYNTHETIC dataset so you can smoke-test the whole pipeline
(train -> evaluate -> predict -> change detection) without downloading
real satellite imagery.

The images are artificial (colored regions + noise) and are NOT meaningful
for real land-cover research. Use your own dataset for real results.

Usage:
    python generate_sample_data.py              # 60 samples
    python generate_sample_data.py --samples 120
"""

import argparse
import os

import cv2
import numpy as np

import config

# Base RGB color per class id (0..4), loosely resembling satellite imagery.
_BASE_COLORS = {
    0: (30, 30, 30),       # background
    1: (150, 150, 155),    # urban
    2: (40, 120, 40),      # vegetation
    3: (30, 70, 160),      # water
    4: (190, 160, 110),    # bare land
}


def _make_sample(rng: np.random.Generator, size: int = 256):
    """Return one (image, mask) pair built from random filled blobs."""
    mask = np.full((size, size), 2, dtype=np.uint8)  # vegetation background
    for _ in range(rng.integers(4, 8)):
        cls = int(rng.integers(1, config.NUM_CLASSES))
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        axes = (int(rng.integers(20, 70)), int(rng.integers(20, 70)))
        angle = float(rng.integers(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, cls, -1)

    image = np.zeros((size, size, 3), dtype=np.float32)
    for cid, color in _BASE_COLORS.items():
        image[mask == cid] = color
    image += rng.normal(0, 8, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, mask


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset.")
    parser.add_argument("--samples", type=int, default=60, help="Number of image/mask pairs.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(config.RAW_DATA_DIR, "images")
    msk_dir = os.path.join(config.RAW_DATA_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(args.samples):
        image, mask = _make_sample(rng, config.IMAGE_HEIGHT)
        name = f"sample_{i:04d}.png"
        cv2.imwrite(os.path.join(img_dir, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(msk_dir, name), mask)

    # Two extra images for the change-detection demo.
    before, _ = _make_sample(rng, config.IMAGE_HEIGHT)
    after, _ = _make_sample(rng, config.IMAGE_HEIGHT)
    cv2.imwrite(os.path.join(config.CHANGE_DETECTION_BEFORE_DIR, "example_before.png"),
                cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(config.CHANGE_DETECTION_AFTER_DIR, "example_after.png"),
                cv2.cvtColor(after, cv2.COLOR_RGB2BGR))

    print(f"Created {args.samples} synthetic samples in {config.RAW_DATA_DIR}")
    print("Created example_before.png / example_after.png for change detection.")


if __name__ == "__main__":
    main()
