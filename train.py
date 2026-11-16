"""
train.py
========
Trains the U-Net land cover segmentation model.

Usage (Google Colab or local terminal):
    python train.py
    python train.py --epochs 40 --batch-size 16 --resume

This script:
    1. Builds train/val tf.data pipelines from dataset.py
    2. Builds the U-Net model from model.py
    3. Compiles it with the combined Dice+CE loss and IoU/Dice metrics
    4. Trains with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
    5. Saves the final model and a CSV of the training history
    6. Plots and saves loss/accuracy/IoU curves
"""

import argparse
import os
import pandas as pd
import tensorflow as tf

import config
from dataset import build_dataset, split_dataset_if_needed
from model import build_unet
from losses import get_loss
from metrics import get_training_metrics
from utils.helpers import set_global_seed, load_model_safely


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net land cover segmentation model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                         help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                         help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                         help="Initial learning rate.")
    parser.add_argument("--resume", action="store_true",
                         help="Resume training from the best saved checkpoint if it exists.")
    return parser.parse_args()


def get_callbacks():
    """
    Build the standard Keras callback set required by the project spec:
    EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.CHECKPOINT_PATH,
            save_freq="epoch",
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(config.TRAINING_HISTORY_CSV, append=False),
    ]
    return callbacks


def main():
    args = parse_args()
    set_global_seed()

    # If the user only populated data/raw, automatically create the
    # train/val/test split expected by the rest of the pipeline.
    split_dataset_if_needed()

    print("Building datasets...")
    train_ds = build_dataset(
        config.TRAIN_IMAGE_DIR, config.TRAIN_MASK_DIR,
        batch_size=args.batch_size, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        config.VAL_IMAGE_DIR, config.VAL_MASK_DIR,
        batch_size=args.batch_size, augment=False, shuffle=False,
    )

    if args.resume and os.path.exists(config.BEST_MODEL_PATH):
        print(f"Resuming training from checkpoint: {config.BEST_MODEL_PATH}")
        model = load_model_safely(config.BEST_MODEL_PATH)
    else:
        print("Building a new U-Net model...")
        model = build_unet()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=get_loss(),
            metrics=get_training_metrics(),
        )

    model.summary()

    print(f"\nStarting training for up to {args.epochs} epochs "
          f"(batch size {args.batch_size}, lr {args.lr})...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=get_callbacks(),
    )

    model.save(config.FINAL_MODEL_PATH)
    print(f"\nFinal model saved to: {config.FINAL_MODEL_PATH}")
    print(f"Best model (lowest val_loss) saved to: {config.BEST_MODEL_PATH}")

    # history.history is also written incrementally by CSVLogger, but we
    # save a clean final copy here as well for downstream plotting.
    pd.DataFrame(history.history).to_csv(config.TRAINING_HISTORY_CSV, index=False)
    print(f"Training history saved to: {config.TRAINING_HISTORY_CSV}")

    print("\nGenerating training curve plots...")
    from visualization import plot_training_curves
    plot_training_curves(config.TRAINING_HISTORY_CSV, config.TRAINING_CURVES_PNG)
    print(f"Training curves saved to: {config.TRAINING_CURVES_PNG}")


if __name__ == "__main__":
    main()
