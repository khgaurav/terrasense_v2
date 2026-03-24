#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference / evaluation script for trained UNET on RUGD dataset.

Usage:
    python infer.py --image data/RUGD/RUGD_frames-with-annotations/creek/creek_00001.png
    python infer.py --eval_test          # evaluate Mean IoU on the full test set
"""

import os
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
    RUGDDataset, NUM_CLASSES, RUGD_CLASSES, RUGD_COLORMAP_RGB,
    rgb_mask_to_class_index, class_index_to_rgb, NORM_FACTOR
)


def parse_args():
    ap = argparse.ArgumentParser(description="UNET RUGD inference")
    ap.add_argument("--model", default="output/keras_model/ep50_trained_unet_v2_224x224.keras",
                    help="Path to trained .hdf5 model")
    ap.add_argument("--image", default=None,
                    help="Path to a single image for inference")
    ap.add_argument("--eval_test", action="store_true",
                    help="Evaluate IoU on the full test set")
    ap.add_argument("--data_root", default="data/RUGD",
                    help="Path to RUGD dataset directory")
    ap.add_argument("--output_dir", default="output/predictions",
                    help="Directory for saving prediction outputs")
    ap.add_argument("--img_height", type=int, default=224)
    ap.add_argument("--img_width", type=int, default=224)
    return ap.parse_args()


def predict_single_image(model, image_path, img_size=(224, 224)):
    """Run inference on a single image and return prediction overlay."""
    # Load original image
    img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Pre-process
    img_resized = cv2.resize(img_orig, (img_size[1], img_size[0]))
    img_norm = img_resized.astype(np.float32) / NORM_FACTOR - 1.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Predict
    pred = model.predict(img_batch, verbose=0)
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
    return mean_iou


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    print("Model loaded successfully.")

    img_size = (args.img_height, args.img_width)

    if args.image:
        print(f"\nRunning inference on: {args.image}")
        img_resized, pred_class, pred_rgb, overlay = predict_single_image(
            model, args.image, img_size
        )

        # Save outputs
        basename = os.path.splitext(os.path.basename(args.image))[0]

        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        # --- Add labels to the middle image (pred_rgb) ---
        labeled_pred_rgb = pred_rgb.copy()
        for class_idx in np.unique(pred_class):
            if class_idx == 0:  # Skip 'void' / background
                continue
                
            # Create a binary mask for this class
            class_mask = (pred_class == class_idx).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Only label the largest contiguous region for each class to avoid clutter
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 100:  # Only label reasonably sized regions
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        class_name = RUGD_CLASSES[class_idx]
                        
                        # Draw text with a thin black outline for visibility
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        
                        # Black outline
                        cv2.putText(labeled_pred_rgb, class_name, (cX - 15, cY), font, 
                                    font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                        # White text
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
        test_gen = RUGDDataset(args.data_root, split="test",
                              batch_size=args.batch_size if hasattr(args, 'batch_size') else 8,
                              img_size=img_size)
        X_test, Y_test = test_gen.get_all_data()

        y_pred = model.predict(X_test, verbose=1)
        y_pred_idx = np.argmax(y_pred, axis=3)
        y_true_idx = np.argmax(Y_test, axis=3)

        print("\nTest set IoU:")
        compute_iou(y_true_idx, y_pred_idx, NUM_CLASSES, RUGD_CLASSES)

    if not args.image and not args.eval_test:
        print("No action specified. Use --image or --eval_test.")
        print("Example:")
        print("  python infer.py --image data/RUGD/RUGD_frames-with-annotations/creek/creek_00001.png")
        print("  python infer.py --eval_test")


if __name__ == "__main__":
    main()
