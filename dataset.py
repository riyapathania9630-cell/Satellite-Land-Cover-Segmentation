"""
dataset.py
==========
Data loading, preprocessing, augmentation, and tf.data pipeline
construction for the land cover segmentation project.

This module is deliberately independent of any specific dataset provider.
It expects the following folder layout (already created by config.py):

    data/train/images/*.png|jpg|tif
    data/train/masks/*.png|jpg|tif
    data/val/images/...
    data/val/masks/...
    data/test/images/...
    data/test/masks/...

Each mask must be a single-channel image where the pixel value equals the
integer class label (0..NUM_CLASSES-1), matching config.CLASS_NAMES.

If you are using a dataset that ships with georeferenced GeoTIFFs (e.g.
Sentinel-2 tiles), rasterio is used automatically to read them; otherwise
OpenCV is used for standard image formats.
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# --------------------------------------------------------------------------
# LOW LEVEL IO
# --------------------------------------------------------------------------
def _read_image(path: str) -> np.ndarray:
    """
    Read an image from disk as an RGB uint8 numpy array.

    Supports standard formats (png/jpg) via OpenCV and GeoTIFF (.tif/.tiff)
    via rasterio when available. Falls back to OpenCV for .tif if rasterio
    is not installed.

    Args:
        path: Full path to the image file.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff") and RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            # Read first 3 bands (assumes RGB or RGB-like ordering).
            bands = min(3, src.count)
            arr = src.read(list(range(1, bands + 1)))          # (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))                 # (H, W, C)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            # Normalize arbitrary bit depth to uint8 for consistent handling.
            arr = arr.astype(np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            return arr.astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: str) -> np.ndarray:
    """
    Read a segmentation mask as a single-channel integer label array.

    Args:
        path: Full path to the mask file.

    Returns:
        np.ndarray of shape (H, W), dtype uint8, values in
        [0, config.NUM_CLASSES - 1].
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff") and RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            mask = src.read(1)
        return mask.astype(np.uint8)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask at: {path}")
    return mask.astype(np.uint8)


# --------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------
def resize_image(image: np.ndarray, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) -> np.ndarray:
    """Resize an image with bilinear interpolation (smooth, good for RGB)."""
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)) -> np.ndarray:
    """Resize a mask with nearest-neighbour interpolation (preserves labels)."""
    return cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale uint8 image [0,255] to float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def one_hot_encode_mask(mask: np.ndarray, num_classes: int = config.NUM_CLASSES) -> np.ndarray:
    """
    Convert an integer label mask (H, W) into a one-hot mask (H, W, C).

    Args:
        mask: Integer label mask.
        num_classes: Number of segmentation classes.

    Returns:
        One-hot float32 array of shape (H, W, num_classes).
    """
    return tf.keras.utils.to_categorical(mask, num_classes=num_classes).astype(np.float32)


def preprocess_pair(image_path: str, mask_path: str, augment: bool = False):
    """
    Full preprocessing pipeline for one (image, mask) pair: read, resize,
    optionally augment, normalize, and one-hot encode.

    Args:
        image_path: Path to the RGB image.
        mask_path: Path to the label mask.
        augment: Whether to apply light random augmentation.

    Returns:
        (image, mask) tuple ready to feed into the model. image has shape
        (H, W, 3) float32 in [0,1]; mask has shape (H, W, NUM_CLASSES) if
        config.USE_CATEGORICAL else (H, W, 1) float32 in {0,1}.
    """
    image = _read_image(image_path)
    mask = _read_mask(mask_path)

    image = resize_image(image)
    mask = resize_mask(mask)

    if augment:
        image, mask = _augment(image, mask)

    image = normalize_image(image)

    if config.USE_CATEGORICAL:
        mask = one_hot_encode_mask(mask)
    else:
        mask = (mask > 0).astype(np.float32)[..., np.newaxis]

    return image, mask


def _augment(image: np.ndarray, mask: np.ndarray):
    """
    Apply simple, label-safe augmentations: horizontal/vertical flip and
    90-degree rotation. Both image and mask receive identical geometric
    transforms so pixel-label alignment is preserved.
    """
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    if np.random.rand() > 0.5:
        k = np.random.choice([1, 2, 3])
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


# --------------------------------------------------------------------------
# FILE DISCOVERY
# --------------------------------------------------------------------------
def _list_paired_files(image_dir: str, mask_dir: str):
    """
    Find matching (image, mask) file pairs by filename stem. Raises a
    clear error if a mask is missing for an image, instead of silently
    skipping data.
    """
    valid_ext = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    for pattern in valid_ext:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_paths = sorted(image_paths)

    pairs = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_candidates = glob.glob(os.path.join(mask_dir, stem + ".*"))
        if not mask_candidates:
            raise FileNotFoundError(
                f"No mask found for image '{img_path}'. "
                f"Expected a file named '{stem}.*' inside '{mask_dir}'."
            )
        pairs.append((img_path, mask_candidates[0]))
    return pairs


# --------------------------------------------------------------------------
# TF.DATA PIPELINE
# --------------------------------------------------------------------------
def _tf_wrapper(image_path, mask_path, augment):
    """Wrap the numpy preprocessing function for use inside tf.data."""
    def _py_fn(img_p, msk_p):
        img_p = img_p.numpy().decode("utf-8")
        msk_p = msk_p.numpy().decode("utf-8")
        image, mask = preprocess_pair(img_p, msk_p, augment=augment)
        return image, mask

    image, mask = tf.py_function(
        func=_py_fn, inp=[image_path, mask_path], Tout=[tf.float32, tf.float32]
    )
    mask_channels = config.NUM_CLASSES if config.USE_CATEGORICAL else 1
    image.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS])
    mask.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, mask_channels])
    return image, mask


def build_dataset(image_dir: str, mask_dir: str, batch_size: int = config.BATCH_SIZE,
                   augment: bool = False, shuffle: bool = False) -> tf.data.Dataset:
    """
    Build a batched, prefetching tf.data.Dataset from a directory of
    images and masks.

    Args:
        image_dir: Directory containing input images.
        mask_dir: Directory containing matching label masks.
        batch_size: Batch size.
        augment: Apply random augmentation (use only for training data).
        shuffle: Shuffle file order every epoch (use only for training data).

    Returns:
        A tf.data.Dataset yielding (image_batch, mask_batch).
    """
    pairs = _list_paired_files(image_dir, mask_dir)
    if not pairs:
        raise ValueError(
            f"No image/mask pairs found in '{image_dir}' / '{mask_dir}'. "
            f"Populate these folders before building the dataset "
            f"(see README.md -> Dataset section)."
        )

    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda i, m: _tf_wrapper(i, m, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def split_dataset_if_needed():
    """
    Convenience utility: if the user only populated data/raw/images and
    data/raw/masks (a single flat folder), this function performs a
    train/val/test split and copies files into the expected
    data/train|val|test structure. Safe to call multiple times; it skips
    work if the split already exists.
    """
    import shutil

    if os.listdir(config.TRAIN_IMAGE_DIR):
        print("Train/val/test split already exists. Skipping split.")
        return

    raw_image_dir = os.path.join(config.RAW_DATA_DIR, "images")
    raw_mask_dir = os.path.join(config.RAW_DATA_DIR, "masks")
    if not os.path.isdir(raw_image_dir):
        print(f"No raw data found at '{raw_image_dir}'. Nothing to split.")
        return

    pairs = _list_paired_files(raw_image_dir, raw_mask_dir)
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
        random_state=config.RANDOM_SEED,
    )
    relative_test_size = config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=relative_test_size, random_state=config.RANDOM_SEED
    )

    split_map = {
        "train": (train_pairs, config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR),
        "val": (val_pairs, config.VAL_IMAGE_DIR, config.VAL_MASK_DIR),
        "test": (test_pairs, config.TEST_IMAGE_DIR, config.TEST_MASK_DIR),
    }

    for split_name, (split_pairs, img_dst, mask_dst) in split_map.items():
        for img_path, mask_path in split_pairs:
            shutil.copy(img_path, os.path.join(img_dst, os.path.basename(img_path)))
            shutil.copy(mask_path, os.path.join(mask_dst, os.path.basename(mask_path)))
        print(f"{split_name}: {len(split_pairs)} samples")
