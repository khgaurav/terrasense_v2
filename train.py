#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for UNET semantic segmentation on the RUGD dataset.

Usage:
    python train.py --epochs 50 --batch_size 8 --lr 0.01
    python train.py --epochs 1 --batch_size 4  # quick smoke test
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)

from dataset import RUGDDataset, NUM_CLASSES, RUGD_CLASSES
from model import build_unet

# ─── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Train UNET on RUGD dataset")
    ap.add_argument("--data_root", default="data/RUGD",
                    help="Path to RUGD dataset directory")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="Initial learning rate")
    ap.add_argument("--img_height", type=int, default=224,
                    help="Input image height")
    ap.add_argument("--img_width", type=int, default=224,
                    help="Input image width")
    ap.add_argument("--output_dir", default="output",
                    help="Directory for output files (models, plots)")
    return ap.parse_args()


# ─── IoU computation ────────────────────────────────────────────────────────

def compute_iou(y_true_idx, y_pred_idx, n_classes, class_names):
    """Compute per-class IoU and mean IoU."""
    ious = []
    for c in range(n_classes):
        tp = np.sum((y_true_idx == c) & (y_pred_idx == c))
        fp = np.sum((y_true_idx != c) & (y_pred_idx == c))
        fn = np.sum((y_true_idx == c) & (y_pred_idx != c))
        if tp + fp + fn == 0:
            iou = float('nan')
        else:
            iou = tp / float(tp + fp + fn)
        print(f"  class ({c:2d}) {class_names[c]:>24s}: "
              f"#TP={tp:7d}, #FP={fp:7d}, #FN={fn:7d}, IoU={iou:.3f}")
        ious.append(iou)
    valid_ious = [x for x in ious if not np.isnan(x)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    print(f"  {'─' * 60}")
    print(f"  Mean IoU: {mean_iou:.3f}")
    return mean_iou, ious


# ─── Main training loop ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    keras_model_dir = os.path.join(args.output_dir, "keras_model")
    os.makedirs(keras_model_dir, exist_ok=True)
    saved_model_dir = os.path.join(args.output_dir, "saved_model")
    rpt_dir = os.path.join(args.output_dir, "rpt")
    os.makedirs(rpt_dir, exist_ok=True)

    img_size = (args.img_height, args.img_width)

    print("=" * 70)
    print("UNET v2 Training on RUGD Dataset")
    print("=" * 70)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size:  {img_size}")
    print(f"  Classes:     {NUM_CLASSES}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_gen = RUGDDataset(args.data_root, split="train",
                           batch_size=args.batch_size,
                           img_size=img_size, augment=True)
    val_gen = RUGDDataset(args.data_root, split="val",
                         batch_size=args.batch_size,
                         img_size=img_size, augment=False)
    test_gen = RUGDDataset(args.data_root, split="test",
                          batch_size=args.batch_size,
                          img_size=img_size, augment=False)
    print()

    # ── Model ────────────────────────────────────────────────────────
    print("Building UNET v2 model...")
    model = build_unet(NUM_CLASSES, args.img_height, args.img_width)
    model.summary(print_fn=lambda x: None)  # suppress verbose summary
    print(f"  Total parameters: {model.count_params():,}")
    print()

    # ── Compile ──────────────────────────────────────────────────────
    sgd = SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )

    # ── Callbacks ────────────────────────────────────────────────────
    keras_path = os.path.join(
        keras_model_dir,
        f"ep{args.epochs}_trained_unet_v2_{args.img_width}x{args.img_height}.keras"
    )
    callbacks = [
        ModelCheckpoint(keras_path, monitor='val_loss',
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=10, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=25,
                     restore_best_weights=True, verbose=1),
    ]

    # ── Training ─────────────────────────────────────────────────────
    print("Starting training...")
    t_start = datetime.now()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
    )

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    print(f"\nTraining completed in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    # ── Save final model ─────────────────────────────────────────────
    final_keras = os.path.join(
        keras_model_dir,
        f"ep{args.epochs}_final_unet_v2_{args.img_width}x{args.img_height}.keras"
    )
    model.save(final_keras)
    print(f"Final model saved (.keras): {final_keras}")

    # Also save as legacy HDF5 for Vitis AI Docker (TF1.15 compatibility)
    final_h5 = os.path.join(
        keras_model_dir,
        f"ep{args.epochs}_final_unet_v2_{args.img_width}x{args.img_height}.h5"
    )
    model.save(final_h5)
    print(f"Final model saved (.h5):    {final_h5}")

    # Also save as SavedModel format (for TF2-based Vitis AI flows)
    model.export(saved_model_dir)
    print(f"SavedModel exported: {saved_model_dir}")

    # ── Training curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='train_loss')
    axes[0].plot(history.history['val_loss'], label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['accuracy'], label='train_acc')
    axes[1].plot(history.history['val_accuracy'], label='val_acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curves_path = os.path.join(
        rpt_dir,
        f"unet_v2_training_curves_{args.img_width}x{args.img_height}.png"
    )
    plt.savefig(curves_path, dpi=150)
    print(f"Training curves saved: {curves_path}")

    # ── Evaluate on test set ─────────────────────────────────────────
    print("\nEvaluating on test set...")
    X_test, Y_test = test_gen.get_all_data()
    y_pred = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred, axis=3)
    y_true_idx = np.argmax(Y_test, axis=3)

    print("\nTest set IoU:")
    mean_iou, _ = compute_iou(y_true_idx, y_pred_idx,
                               NUM_CLASSES, RUGD_CLASSES)

    print(f"\n{'=' * 70}")
    print(f"Training complete. Mean IoU on test set: {mean_iou:.3f}")
    print(f"Model saved at: {final_keras}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
