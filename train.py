#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for UNET semantic segmentation on the RELLIS-3D dataset.

Usage:
    python train.py --epochs 50 --batch_size 8 --lr 0.01
    python train.py --epochs 1 --batch_size 4  # quick smoke test
"""

import os
# Configure environment for local tensorflow/nvidia CUDA libraries if not already set
if 'XLA_FLAGS' not in os.environ:
    try:
        import nvidia
        nvidia_path = list(nvidia.__path__)[0]
        cuda_nvcc_dir = os.path.join(nvidia_path, 'cuda_nvcc')
        if os.path.isdir(cuda_nvcc_dir):
            os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_nvcc_dir}'
    except Exception:
        pass

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
        
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
import tensorflow.keras.backend as K

from dataset import RELLIS3DDataset, NUM_CLASSES, TOTAL_CLASSES, KINEMATIC_CLASSES
from model import build_unet

# ─── Heuristic Weighted Focal Loss ──────────────────────────────────────────
# 0: Smooth Drivable, 1: Grass, 2: Dirt, 3: Sand, 4: Gravel, 5: Mulch, 6: Puddle, 7: Mud, 8: Soft Obstacle, 9: Obstacle, 10: Void
# Heavily penalise errors on underrepresented classes (Granular/Rough = 15.0, Puddle/Mud = 5.0)
CLASS_WEIGHTS = [3.0, 0.1, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 0.3, 0.4, 0.0]

def get_weighted_focal_loss(weights, gamma=2.0):
    def loss(y_true, y_pred):
        # Cast inputs to float32 to avoid numerical underflow in float16
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Vitis-AI model outputs ReLU, we must apply Softmax for loss computation
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1.0 - K.epsilon())
        weights_tensor = K.constant(weights)
        
        # Focal factor: (1 - p)**gamma
        focal_term = K.pow(1.0 - y_pred_softmax, gamma)
        
        # Weighted Categorical Cross Entropy with Focal factor
        weighted_loss = -y_true * focal_term * K.log(y_pred_softmax) * weights_tensor
        return K.mean(K.sum(weighted_loss, axis=-1))
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
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"],
                    help="Optimizer to use (adam or sgd)")
    ap.add_argument("--lr", type=float, default=None,
                    help="Initial learning rate. Defaults to 0.001 for adam, 0.01 for sgd")
    ap.add_argument("--img_height", type=int, default=224,
                    help="Input image height")
    ap.add_argument("--img_width", type=int, default=224,
                    help="Input image width")
    ap.add_argument("--output_dir", default="output",
                    help="Directory for output files (models, plots)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit the number of samples per dataset split (for testing)")
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
    print(f"  {'-' * 60}")
    print(f"  Mean IoU (Active {n_classes} Classes): {mean_iou:.3f}")
    return mean_iou, ious


# ─── Main training loop ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve learning rate based on optimizer
    if args.lr is None:
        lr_val = 0.001 if args.optimizer == "adam" else 0.01
    else:
        lr_val = args.lr

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
    print(f"  Optimizer:   {args.optimizer.upper()}")
    print(f"  Learning rate: {lr_val}")
    print(f"  Image size:  {img_size}")
    print(f"  Classes:     {NUM_CLASSES} active (+ 1 Void)")
    print(f"  Output dir:  {args.output_dir}")
    print()

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_gen = RELLIS3DDataset(args.data_root, split="train",
                           batch_size=args.batch_size,
                           img_size=img_size, augment=True,
                           max_samples=args.max_samples)
    val_gen = RELLIS3DDataset(args.data_root, split="val",
                         batch_size=args.batch_size,
                         img_size=img_size, augment=False,
                         max_samples=args.max_samples)
    test_gen = RELLIS3DDataset(args.data_root, split="test",
                          batch_size=args.batch_size,
                          img_size=img_size, augment=False,
                          max_samples=args.max_samples)
    print()

    # ── Model ────────────────────────────────────────────────────────
    print("Building UNET v2 model...")
    # 4 channels for early fusion (RGB + D), outputting 7 classes (including void)
    model = build_unet(TOTAL_CLASSES, args.img_height, args.img_width)
    # model.summary(print_fn=lambda x: None)  # suppress verbose summary
    print(f"  Total parameters: {model.count_params():,}")
    print()

    # ── Compile ──────────────────────────────────────────────────────
    if args.optimizer == "adam":
        opt = Adam(learning_rate=lr_val)
    else:
        opt = SGD(learning_rate=lr_val, momentum=0.9, nesterov=True)
    
    custom_loss = get_weighted_focal_loss(CLASS_WEIGHTS, gamma=2.0)
    
    model.compile(
        loss=custom_loss,
        optimizer=opt,
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
