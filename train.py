#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for UNET semantic segmentation on the RELLIS-3D dataset.

Usage:
    python train.py --epochs 50 --batch_size 8 --lr 0.01
    python train.py --epochs 1 --batch_size 4  # quick smoke test
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
import tensorflow.keras.backend as K

from dataset import RELLIS3DDataset, NUM_CLASSES, TOTAL_CLASSES, KINEMATIC_CLASSES
from model import build_unet

# ─── Heuristic Inverse Frequency Class Weights ──────────────────────────────
# 0: Rigid, 1: Granular, 2: Veg, 3: Hazard, 4: Nature, 5: Rigid_Obstacle, 6: Void
CLASS_WEIGHTS = [1.5, 1.0, 0.2, 10.0, 0.5, 3.0, 0.0]

def get_weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # Vitis-AI model outputs ReLU, we must apply Softmax for CE loss
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1.0 - K.epsilon())
        weights_tensor = K.constant(weights)
        cce = -y_true * K.log(y_pred_softmax) * weights_tensor
        return K.mean(K.sum(cce, axis=-1))
    return loss

# ─── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Train UNET on RELLIS-3D dataset")
    ap.add_argument("--data_root", default="data/RELLIS-3D",
                    help="Path to RELLIS-3D dataset directory")
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
    # Ignore the void class (index 6, out of active classes 0-5)
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
    print(f"  Mean IoU (Active {n_classes} Classes): {mean_iou:.3f}")
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
    print("UNET v2 Training on RELLIS-3D Dataset (RGBD Early Fusion)")
    print("=" * 70)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size:  {img_size}")
    print(f"  Classes:     {NUM_CLASSES} active (+ 1 Void)")
    print(f"  Output dir:  {args.output_dir}")
    print()

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_gen = RELLIS3DDataset(args.data_root, split="train",
                           batch_size=args.batch_size,
                           img_size=img_size, augment=True)
    val_gen = RELLIS3DDataset(args.data_root, split="val",
                         batch_size=args.batch_size,
                         img_size=img_size, augment=False)
    test_gen = RELLIS3DDataset(args.data_root, split="test",
                          batch_size=args.batch_size,
                          img_size=img_size, augment=False)
    print()

    # ── Model ────────────────────────────────────────────────────────
    print("Building UNET v2 model...")
    # 4 channels for early fusion (RGB + D), outputting 7 classes (including void)
    model = build_unet(TOTAL_CLASSES, args.img_height, args.img_width)
    # model.summary(print_fn=lambda x: None)  # suppress verbose summary
    print(f"  Total parameters: {model.count_params():,}")
    print()

    # ── Compile ──────────────────────────────────────────────────────
    sgd = SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    
    custom_loss = get_weighted_categorical_crossentropy(CLASS_WEIGHTS)
    
    model.compile(
        loss=custom_loss,
        optimizer=sgd,
        metrics=['accuracy']
    )

    # ── Callbacks ────────────────────────────────────────────────────
    keras_path = os.path.join(
        keras_model_dir,
        f"ep{args.epochs}_trained_unet_v2_rgbd_{args.img_width}x{args.img_height}.keras"
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
        f"ep{args.epochs}_final_unet_v2_rgbd_{args.img_width}x{args.img_height}.keras"
    )
    model.save(final_keras)
    print(f"Final model saved (.keras): {final_keras}")

    # Also save as legacy HDF5 for Vitis AI Docker (TF1.15 compatibility)
    final_h5 = os.path.join(
        keras_model_dir,
        f"ep{args.epochs}_final_unet_v2_rgbd_{args.img_width}x{args.img_height}.h5"
    )
    model.save(final_h5)
    print(f"Final model saved (.h5):    {final_h5}")

    # Also save as SavedModel format (for TF2-based Vitis AI flows)
    model.export(saved_model_dir)
    print(f"SavedModel exported: {saved_model_dir}")

    # ── Evaluate on test set ─────────────────────────────────────────
    print("\nEvaluating on test set...")
    y_pred_list = []
    y_true_list = []
    for i in range(len(test_gen)):
        X_batch, Y_batch = test_gen[i]
        pred_batch = model.predict(X_batch, verbose=0)
        y_pred_list.append(np.argmax(pred_batch, axis=3))
        y_true_list.append(np.argmax(Y_batch, axis=3))
    
    y_pred_idx = np.concatenate(y_pred_list, axis=0)
    y_true_idx = np.concatenate(y_true_list, axis=0)

    print("\nTest set IoU:")
    mean_iou, _ = compute_iou(y_true_idx, y_pred_idx,
                               NUM_CLASSES, KINEMATIC_CLASSES)

    print(f"\n{'=' * 70}")
    print(f"Training complete. Mean IoU on test set: {mean_iou:.3f}")
    print(f"Model saved at: {final_keras}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
