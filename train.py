#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the early-fusion RGB-D UNet on RELLIS-3D. Saves .h5, which is what
vitis_ai/run_kr260.sh freezes. Use eval_metrics.py for test-set metrics.

    python train.py --epochs 50 --batch_size 8 --data_root data/RELLIS-3D_full
"""

import os

# Point XLA at the pip-installed CUDA toolkit if the caller has not already.
if 'XLA_FLAGS' not in os.environ:
    try:
        import nvidia
        cuda_nvcc_dir = os.path.join(list(nvidia.__path__)[0], 'cuda_nvcc')
        if os.path.isdir(cuda_nvcc_dir):
            os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_nvcc_dir}'
    except ImportError:
        pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam

from dataset import IMG_SIZE, NUM_CLASSES, RELLIS3DDataset, TOTAL_CLASSES
from model import build_unet

# Per-class loss weights, indexed by kinematic class. The granular surfaces
# (Dirt/Sand/Gravel/Mulch) are rare but matter most for traversability, so they are
# weighted heavily; Void is excluded outright.
# 0 Smooth 1 Grass 2 Dirt 3 Sand 4 Gravel 5 Mulch 6 Puddle 7 Mud 8 Soft 9 Obstacle 10 Void
CLASS_WEIGHTS = [3.0, 0.1, 15.0, 15.0, 15.0, 15.0, 5.0, 3.0, 0.3, 0.4, 0.0]


def weighted_focal_dice_loss(weights, gamma=2.0, dice_weight=1.0):
    """Weighted focal cross-entropy plus a weighted soft-Dice term.

    The model ends in ReLU rather than softmax (the DPU has no softmax), so the
    softmax is applied here.
    """
    weight_tensor = K.constant(weights)
    active_weight_tensor = K.constant(weights[:NUM_CLASSES])

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        probabilities = K.clip(tf.nn.softmax(tf.cast(y_pred, tf.float32), axis=-1),
                               K.epsilon(), 1.0 - K.epsilon())

        focal = -y_true * K.pow(1.0 - probabilities, gamma) * K.log(probabilities)
        focal_loss = K.mean(K.sum(focal * weight_tensor, axis=-1))

        intersection = K.sum(y_true * probabilities, axis=[0, 1, 2])
        totals = K.sum(K.square(y_true) + K.square(probabilities), axis=[0, 1, 2])
        dice = (2.0 * intersection + K.epsilon()) / (totals + K.epsilon())
        dice_loss = K.mean((1.0 - dice)[:NUM_CLASSES] * active_weight_tensor)

        return focal_loss + dice_weight * dice_loss

    return loss


def parse_args():
    ap = argparse.ArgumentParser(description="Train the RGB-D UNet on RELLIS-3D")
    ap.add_argument("--data_root", default="data/RELLIS-3D_full", help="Dataset root")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate (default: 0.001 for adam, 0.01 for sgd)")
    ap.add_argument("--output_dir", default="output", help="Where to save models and plots")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit samples for a quick check")
    return ap.parse_args()


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, title) in zip(axes, (('loss', 'Loss'), ('accuracy', 'Accuracy'))):
        ax.plot(history.history[metric], label=f'train_{metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Training curves saved: {path}")


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else (0.001 if args.optimizer == "adam" else 0.01)

    keras_model_dir = os.path.join(args.output_dir, "keras_model")
    rpt_dir = os.path.join(args.output_dir, "rpt")
    os.makedirs(keras_model_dir, exist_ok=True)
    os.makedirs(rpt_dir, exist_ok=True)

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    height, width = IMG_SIZE
    print(f"Training RGB-D UNet on RELLIS-3D: {args.epochs} epochs, "
          f"batch {args.batch_size}, {args.optimizer} lr={lr}")
    print(f"{width}x{height} early-fusion input, {NUM_CLASSES} active classes (+ Void)")

    train_gen = RELLIS3DDataset(args.data_root, split="train", batch_size=args.batch_size,
                                augment=True, max_samples=args.max_samples)
    val_gen = RELLIS3DDataset(args.data_root, split="val", batch_size=args.batch_size,
                              max_samples=args.max_samples)

    model = build_unet(TOTAL_CLASSES, height, width)
    print(f"  Total parameters: {model.count_params():,}\n")

    optimizer = (Adam(learning_rate=lr) if args.optimizer == "adam"
                 else SGD(learning_rate=lr, momentum=0.9, nesterov=True))
    model.compile(loss=weighted_focal_dice_loss(CLASS_WEIGHTS),
                  optimizer=optimizer, metrics=['accuracy'])

    stem = f"unet_v2_rgbd_{width}x{height}"
    best_path = os.path.join(keras_model_dir, f"ep{args.epochs}_best_{stem}.h5")
    final_path = os.path.join(keras_model_dir, f"ep{args.epochs}_final_{stem}.h5")

    print("Starting training...")
    started = datetime.now()
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=args.epochs, verbose=2,
        callbacks=[
            ModelCheckpoint(best_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                              min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=25,
                          restore_best_weights=True, verbose=1),
        ])
    elapsed = (datetime.now() - started).total_seconds()
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    model.save(final_path)
    print(f"Best model:  {best_path}")
    print(f"Final model: {final_path}")

    plot_curves(history, os.path.join(rpt_dir, f"{stem}_training_curves.png"))
    print("\nEvaluate with:")
    print(f"  python eval_metrics.py --model {best_path} --data_root {args.data_root}")


if __name__ == "__main__":
    main()
