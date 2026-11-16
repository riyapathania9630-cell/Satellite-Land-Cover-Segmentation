"""
config.py
=========
Central configuration file for the Land Cover Segmentation and
Multi-Temporal Change Detection project.

All hyperparameters, paths, and constants are defined here so that
no file in the project ever hardcodes a path or a magic number.
Edit this file to change dataset locations, image size, training
hyperparameters, or class definitions.
"""

import os

# --------------------------------------------------------------------------
# BASE PATHS
# --------------------------------------------------------------------------
# BASE_DIR is resolved relative to this file so the project runs identically
# on Google Colab, a local machine, or a server, without any manual edits.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

# Two time-stamped folders used only by the change-detection module.
CHANGE_DETECTION_BEFORE_DIR = os.path.join(DATA_DIR, "change_detection", "before")
CHANGE_DETECTION_AFTER_DIR = os.path.join(DATA_DIR, "change_detection", "after")

MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_IMAGES_DIR = os.path.join(RESULTS_DIR, "images")

# Automatically create every directory that the pipeline needs so the user
# never hits a "file not found" error caused by a missing folder.
for _dir in [
    DATA_DIR, RAW_DATA_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR,
    VAL_MASK_DIR, TEST_IMAGE_DIR, TEST_MASK_DIR, CHANGE_DETECTION_BEFORE_DIR,
    CHANGE_DETECTION_AFTER_DIR, MODEL_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    REPORT_DIR, RESULTS_DIR, RESULTS_IMAGES_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# --------------------------------------------------------------------------
# IMAGE / MASK PROPERTIES
# --------------------------------------------------------------------------
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3          # RGB. Set to 4 if you use an extra NIR band.
MASK_CHANNELS = 1           # Set automatically from NUM_CLASSES below.

# --------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------
# Land cover classes used across the whole project. Index position is the
# integer label that appears inside every mask (0 .. NUM_CLASSES-1).
CLASS_NAMES = [
    "background",   # 0 - unlabeled / no-data
    "urban",        # 1 - buildings, roads, impervious surfaces
    "vegetation",   # 2 - forest, crops, dense greenery
    "water",        # 3 - rivers, lakes, sea, flooded areas
    "bare_land",    # 4 - soil, rock, sand, barren land
]
NUM_CLASSES = len(CLASS_NAMES)

# A fixed, visually distinct RGB color for every class. Used by every
# visualization function so that colors are consistent across the project.
CLASS_COLORS = {
    0: (0, 0, 0),        # background - black
    1: (255, 0, 0),      # urban - red
    2: (0, 170, 0),      # vegetation - green
    3: (0, 100, 255),    # water - blue
    4: (200, 170, 100),  # bare land - tan / brown
}

# --------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# --------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Loss / metric behaviour
USE_CATEGORICAL = True     # True -> multi-class softmax U-Net (NUM_CLASSES)
                            # False -> binary sigmoid U-Net (1 channel)
DICE_SMOOTH = 1e-6

# EarlyStopping / ReduceLROnPlateau / ModelCheckpoint
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# --------------------------------------------------------------------------
# FILE NAMES
# --------------------------------------------------------------------------
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_unet_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_unet_model.h5")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_checkpoint_epoch_{epoch:02d}.h5")
TRAINING_HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")
TRAINING_CURVES_PNG = os.path.join(RESULTS_IMAGES_DIR, "training_curves.png")

# --------------------------------------------------------------------------
# CHANGE DETECTION THRESHOLDS
# --------------------------------------------------------------------------
# Minimum fraction of a tile's area (0-1) that must change class before the
# tile is reported as "significantly changed" in the summary statistics.
CHANGE_SIGNIFICANCE_THRESHOLD = 0.05

# Pixel resolution in meters/pixel — used to convert pixel counts into real
# world area (m^2 / hectares). Update this to match your dataset's GSD.
PIXEL_RESOLUTION_M = 10.0  # e.g. 10m for Sentinel-2
