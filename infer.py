#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the trained RGB-D UNet on one RGB + depth pair and save a visualisation.
Use eval_metrics.py for test-set metrics.

    python infer.py --model output/keras_model/ep50_final_unet_v2_rgbd_224x224.h5 \
        --image data/RELLIS-3D_full/rgb/frame000000-1581624652_750.jpg \
        --depth data/RELLIS-3D_full/depth/frame000000-1581624652_750.png
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import (IMG_SIZE, KINEMATIC_CLASSES, NORM_FACTOR, TOTAL_CLASSES,
                     class_index_to_rgb, normalize_depth)

VOID_CLASS = TOTAL_CLASSES - 1


def parse_args():
    ap = argparse.ArgumentParser(description="RGB-D UNet single-pair inference")
    ap.add_argument("--model", required=True, help="Trained model (.keras or .h5)")
    ap.add_argument("--image", required=True, help="RGB image")
    ap.add_argument("--depth", required=True, help="Matching 16-bit depth map (mm)")
    ap.add_argument("--output_dir", default="output/predictions",
                    help="Where to save the visualisation")
    return ap.parse_args()


def predict(model, image_path, depth_path):
    """Returns the resized BGR frame, the class map, its colouring and an overlay."""
    rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    h, w = IMG_SIZE
    rgb = cv2.resize(rgb, (w, h))                                   # BGR, as trained
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    fused = np.concatenate([rgb.astype(np.float32) / NORM_FACTOR - 1.0,
                            normalize_depth(depth)], axis=-1)

    class_map = np.argmax(model.predict(np.expand_dims(fused, 0), verbose=0)[0], axis=-1)
    mask_rgb = class_index_to_rgb(class_map)
    overlay = cv2.addWeighted(rgb, 0.4, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR), 0.6, 0)
    return rgb, class_map, mask_rgb, overlay


def label_regions(mask_rgb, class_map):
    """Write each predicted class name onto its largest region."""
    labelled = mask_rgb.copy()
    for class_idx in np.unique(class_map):
        if class_idx == VOID_CLASS:
            continue
        contours, _ = cv2.findContours((class_map == class_idx).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) <= 100:
            continue
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"]) - 15
        cy = int(moments["m01"] / moments["m00"])
        for colour, thickness in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.putText(labelled, KINEMATIC_CLASSES[class_idx], (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness, cv2.LINE_AA)
    return labelled


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print(f"Running inference on: {args.image} + {args.depth}")
    rgb, class_map, mask_rgb, overlay = predict(model, args.image, args.depth)

    panels = [
        ("Input RGB", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)),
        ("Predicted Segmentation", label_regions(mask_rgb, class_map)),
        ("Overlay", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 6))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.image))[0]}_prediction.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
