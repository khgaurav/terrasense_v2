#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference / evaluation script for trained UNET on RELLIS-3D dataset (RGBD).

Usage:
    python infer.py --image data/RELLIS-3D/rgb/sample.jpg --depth data/RELLIS-3D/depth/sample.png
    python infer.py --eval_test          # evaluate Mean IoU on the full test set
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

import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset import (
    RELLIS3DDataset, NUM_CLASSES, TOTAL_CLASSES, KINEMATIC_CLASSES, KINEMATIC_COLORMAP,
    class_index_to_rgb, NORM_FACTOR, normalize_depth
)

# Kinematic Void index, used to skip the unlabelled class when annotating output.
VOID_CLASS = TOTAL_CLASSES - 1

def parse_args():
    ap = argparse.ArgumentParser(description="UNET RELLIS-3D inference (RGBD)")
    ap.add_argument("--model", default="output/keras_model/ep50_trained_unet_v2_224x224.keras",
                    help="Path to trained .keras model")
    ap.add_argument("--image", default=None,
                    help="Path to a single RGB image for inference")
    ap.add_argument("--depth", default=None,
                    help="Path to the corresponding Depth image")
    ap.add_argument("--eval_test", action="store_true",
                    help="Evaluate IoU on the full test set")
    ap.add_argument("--data_root", default="data/RELLIS-3D",
                    help="Path to RELLIS-3D dataset directory")
    ap.add_argument("--output_dir", default="output/predictions",
                    help="Directory for saving prediction outputs")
    ap.add_argument("--img_height", type=int, default=224)
    ap.add_argument("--img_width", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size used by --eval_test")
    return ap.parse_args()


def predict_single_pair(model, image_path, depth_path, img_size=(224, 224)):
    """Run inference on an RGB + Depth pair and return prediction overlay."""
    # Load original image
    img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Process RGB
    img_resized = cv2.resize(img_orig, (img_size[1], img_size[0]))
    img_norm = img_resized.astype(np.float32) / NORM_FACTOR - 1.0
    
    # Process Depth. Must use the training-time scaling from dataset.py: this used
    # to divide by 65535 while training divided by 10000, so every depth value the
    # network saw at inference was ~6.5x smaller than during training.
    if depth_path and os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        depth_norm = normalize_depth(depth)
    else:
        print("WARNING: Depth image missing or invalid, using zero depth.")
        depth_norm = np.full((img_size[0], img_size[1], 1), -1.0, dtype=np.float32)

    fused_input = np.concatenate([img_norm, depth_norm], axis=-1)
    img_batch = np.expand_dims(fused_input, axis=0)

    # Predict
    pred = model.predict(img_batch, verbose=0)
    # Output shape is (1, H, W, TOTAL_CLASSES). Model outputs logits/ReLU, apply argmax
    pred_class = np.argmax(pred[0], axis=-1)  # (H, W)

    # Convert class map to RGB
    pred_rgb = class_index_to_rgb(pred_class)
    pred_rgb_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    # Create overlay
    overlay = cv2.addWeighted(img_resized, 0.4, pred_rgb_bgr, 0.6, 0)

    return img_resized, pred_class, pred_rgb, overlay


def compute_iou(y_true_idx, y_pred_idx, n_classes, class_names):
    """Compute per-class IoU and mean IoU."""
    ious = []
    # Evaluate across the active K=10 kinematic classes
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
    print(f"  Mean IoU: {mean_iou:.3f}")
    return mean_iou


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    # Compile=False is standard for inference to avoid needing custom loss in scope
    model = load_model(args.model, compile=False)
    print("Model loaded successfully.")

    img_size = (args.img_height, args.img_width)

    if args.image:
        print(f"\nRunning inference on: {args.image} and {args.depth}")
        img_resized, pred_class, pred_rgb, overlay = predict_single_pair(
            model, args.image, args.depth, img_size
        )

        # Save outputs
        basename = os.path.splitext(os.path.basename(args.image))[0]

        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input RGB Image")
        axes[0].axis('off')

        # --- Add labels to the middle image (pred_rgb) ---
        labeled_pred_rgb = pred_rgb.copy()
        for class_idx in np.unique(pred_class):
            if class_idx == VOID_CLASS:  # Skip 'Void'
                continue
                
            class_mask = (pred_class == class_idx).astype(np.uint8)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        class_name = KINEMATIC_CLASSES[class_idx]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        
                        cv2.putText(labeled_pred_rgb, class_name, (cX - 15, cY), font, 
                                    font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                        cv2.putText(labeled_pred_rgb, class_name, (cX - 15, cY), font, 
                                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        axes[1].imshow(labeled_pred_rgb)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"{basename}_prediction.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    if args.eval_test:
        print("\nEvaluating on test set...")
        test_gen = RELLIS3DDataset(args.data_root, split="test",
                              batch_size=args.batch_size,
                              img_size=img_size)
        X_test, Y_test = test_gen.get_all_data()

        y_pred = model.predict(X_test, verbose=1)
        y_pred_idx = np.argmax(y_pred, axis=3)
        y_true_idx = np.argmax(Y_test, axis=3)

        print("\nTest set IoU:")
        compute_iou(y_true_idx, y_pred_idx, NUM_CLASSES, KINEMATIC_CLASSES)

    if not args.image and not args.eval_test:
        print("No action specified. Use --image (and --depth) or --eval_test.")
        print("Example:")
        print("  python infer.py --image data/RELLIS-3D/rgb/sample.jpg --depth data/RELLIS-3D/depth/sample.png")
        print("  python infer.py --eval_test")


if __name__ == "__main__":
    main()
