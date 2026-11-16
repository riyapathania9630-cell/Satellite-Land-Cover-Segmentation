"""
change_detection.py
====================
Multi-temporal change detection module. Compares U-Net segmentation
predictions from two satellite images of the same location taken in
different years, and computes:

    - A pixel-wise class transition (difference) map
    - Newly urbanized areas (-> urban)
    - Lost vegetation (vegetation -> anything else)
    - Flooded regions (-> water, previously non-water)
    - Water body changes (increase / decrease)
    - Bare land increase
    - Area and percentage statistics for every transition, converted to
      real-world units using config.PIXEL_RESOLUTION_M

Usage:
    python change_detection.py --before path/to/2015.png --after path/to/2023.png
"""

import argparse
import os
import numpy as np
import cv2

import config
from dataset import _read_image, resize_image, normalize_image
from utils.helpers import load_model_safely, label_mask_to_rgb, ensure_dir, save_json
from metrics import masks_to_labels

# Convenience lookup from class name to integer id (built from config).
CLASS_ID = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-temporal land cover change detection.")
    parser.add_argument("--before", type=str, required=False,
                         default=os.path.join(config.CHANGE_DETECTION_BEFORE_DIR, "example_before.png"),
                         help="Path to the earlier-year satellite image.")
    parser.add_argument("--after", type=str, required=False,
                         default=os.path.join(config.CHANGE_DETECTION_AFTER_DIR, "example_after.png"),
                         help="Path to the later-year satellite image.")
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH,
                         help="Path to the trained U-Net model.")
    parser.add_argument("--output", type=str, default=os.path.join(config.OUTPUT_DIR, "change_detection"),
                         help="Directory to save change maps and statistics.")
    return parser.parse_args()


def segment_image(model, image_path: str) -> np.ndarray:
    """Run the U-Net model on one image and return its integer label mask."""
    image = _read_image(image_path)
    image = resize_image(image)
    image = normalize_image(image)
    pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
    return masks_to_labels(pred)[0]


def compute_change_map(before_labels: np.ndarray, after_labels: np.ndarray) -> np.ndarray:
    """
    Build a binary change map: 1 where the predicted class changed
    between the two dates, 0 where it stayed the same.
    """
    return (before_labels != after_labels).astype(np.uint8)


def compute_transition_matrix(before_labels: np.ndarray, after_labels: np.ndarray,
                               num_classes: int = config.NUM_CLASSES) -> np.ndarray:
    """
    Build a (num_classes, num_classes) transition matrix where entry
    [i, j] counts the number of pixels that were class i in the 'before'
    image and became class j in the 'after' image.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    flat_before = before_labels.flatten()
    flat_after = after_labels.flatten()
    for b, a in zip(flat_before, flat_after):
        matrix[b, a] += 1
    return matrix


def pixels_to_area(num_pixels: int) -> dict:
    """Convert a pixel count into real-world area (m^2 and hectares)."""
    area_m2 = num_pixels * (config.PIXEL_RESOLUTION_M ** 2)
    return {"pixels": int(num_pixels), "area_m2": float(area_m2), "area_hectares": float(area_m2 / 10000.0)}


def compute_change_statistics(before_labels: np.ndarray, after_labels: np.ndarray) -> dict:
    """
    Compute the full set of change statistics required by the project:
    total area changed, percentage changed, urban growth, forest
    (vegetation) loss, water increase/decrease, and flood-affected area.

    Returns:
        Dictionary with all statistics, ready to be saved as JSON and
        used by visualization.py to build charts.
    """
    total_pixels = before_labels.size
    change_map = compute_change_map(before_labels, after_labels)
    changed_pixels = int(change_map.sum())

    stats = {
        "total_pixels": int(total_pixels),
        "changed_pixels": changed_pixels,
        "percent_changed": float(100.0 * changed_pixels / total_pixels),
        "significant_change_flag": bool(
            (changed_pixels / total_pixels) >= config.CHANGE_SIGNIFICANCE_THRESHOLD
        ),
    }

    # --- Per-class area before / after / delta ---
    per_class = {}
    for name, cid in CLASS_ID.items():
        before_count = int(np.sum(before_labels == cid))
        after_count = int(np.sum(after_labels == cid))
        per_class[name] = {
            "before": pixels_to_area(before_count),
            "after": pixels_to_area(after_count),
            "delta_pixels": after_count - before_count,
            "delta_percent_of_total": float(100.0 * (after_count - before_count) / total_pixels),
        }
    stats["per_class"] = per_class

    # --- Named change categories required by the project spec ---
    urban_id = CLASS_ID.get("urban")
    veg_id = CLASS_ID.get("vegetation")
    water_id = CLASS_ID.get("water")
    bare_id = CLASS_ID.get("bare_land")

    newly_urbanized = int(np.sum((before_labels != urban_id) & (after_labels == urban_id)))
    lost_vegetation = int(np.sum((before_labels == veg_id) & (after_labels != veg_id)))
    flooded_regions = int(np.sum((before_labels != water_id) & (after_labels == water_id)))
    water_decrease = int(np.sum((before_labels == water_id) & (after_labels != water_id)))
    bare_land_increase = int(np.sum((before_labels != bare_id) & (after_labels == bare_id)))

    stats["urban_growth"] = pixels_to_area(newly_urbanized)
    stats["urban_growth"]["percent_of_total"] = float(100.0 * newly_urbanized / total_pixels)

    stats["forest_loss"] = pixels_to_area(lost_vegetation)
    stats["forest_loss"]["percent_of_total"] = float(100.0 * lost_vegetation / total_pixels)

    stats["flood_affected_area"] = pixels_to_area(flooded_regions)
    stats["flood_affected_area"]["percent_of_total"] = float(100.0 * flooded_regions / total_pixels)

    stats["water_increase"] = pixels_to_area(flooded_regions)  # new water pixels
    stats["water_decrease"] = pixels_to_area(water_decrease)
    stats["net_water_change_pixels"] = flooded_regions - water_decrease

    stats["bare_land_increase"] = pixels_to_area(bare_land_increase)
    stats["bare_land_increase"]["percent_of_total"] = float(100.0 * bare_land_increase / total_pixels)

    return stats


def build_category_maps(before_labels: np.ndarray, after_labels: np.ndarray) -> dict:
    """
    Build binary maps (as uint8 0/255 images) highlighting each named
    change category, for use as overlays in visualization.py.

    Returns:
        Dict of category_name -> binary mask (H, W) uint8, 255 = changed.
    """
    urban_id = CLASS_ID.get("urban")
    veg_id = CLASS_ID.get("vegetation")
    water_id = CLASS_ID.get("water")
    bare_id = CLASS_ID.get("bare_land")

    maps = {
        "urban_expansion": ((before_labels != urban_id) & (after_labels == urban_id)).astype(np.uint8) * 255,
        "deforestation": ((before_labels == veg_id) & (after_labels != veg_id)).astype(np.uint8) * 255,
        "flood": ((before_labels != water_id) & (after_labels == water_id)).astype(np.uint8) * 255,
        "bare_land_increase": ((before_labels != bare_id) & (after_labels == bare_id)).astype(np.uint8) * 255,
        "general_change": compute_change_map(before_labels, after_labels) * 255,
    }
    return maps


def main():
    args = parse_args()
    ensure_dir(args.output)

    print(f"Loading model from: {args.model}")
    model = load_model_safely(args.model)

    print(f"Segmenting BEFORE image: {args.before}")
    before_labels = segment_image(model, args.before)
    print(f"Segmenting AFTER image: {args.after}")
    after_labels = segment_image(model, args.after)

    print("Computing change statistics...")
    stats = compute_change_statistics(before_labels, after_labels)
    stats_path = os.path.join(args.output, "change_statistics.json")
    save_json(stats, stats_path)

    print("Building category change maps...")
    category_maps = build_category_maps(before_labels, after_labels)
    for name, mask in category_maps.items():
        cv2.imwrite(os.path.join(args.output, f"{name}.png"), mask)

    print("Saving colored before/after segmentation maps...")
    before_rgb = label_mask_to_rgb(before_labels)
    after_rgb = label_mask_to_rgb(after_labels)
    cv2.imwrite(os.path.join(args.output, "before_segmentation.png"), cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output, "after_segmentation.png"), cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR))

    print("\n===== CHANGE DETECTION SUMMARY =====")
    print(f"Total area changed   : {stats['percent_changed']:.2f}%")
    print(f"Urban growth         : {stats['urban_growth']['percent_of_total']:.2f}% "
          f"({stats['urban_growth']['area_hectares']:.2f} ha)")
    print(f"Forest loss          : {stats['forest_loss']['percent_of_total']:.2f}% "
          f"({stats['forest_loss']['area_hectares']:.2f} ha)")
    print(f"Flood affected area  : {stats['flood_affected_area']['percent_of_total']:.2f}% "
          f"({stats['flood_affected_area']['area_hectares']:.2f} ha)")
    print(f"Bare land increase   : {stats['bare_land_increase']['percent_of_total']:.2f}% "
          f"({stats['bare_land_increase']['area_hectares']:.2f} ha)")

    print("\nGenerating change detection visualizations...")
    from visualization import plot_change_detection_summary
    plot_change_detection_summary(before_labels, after_labels, stats, args.output)

    print(f"\nAll change detection outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
