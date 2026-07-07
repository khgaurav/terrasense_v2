#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibration input function for vai_q_tensorflow.

Called as:  --input_fn graph_input_fn.calib_input
            --calib_iter 10           (10 batches × BATCH_SIZE images)

Loads RGB frames, optionally loads matching depth maps, resizes to 224×224, normalises to [-1, 1].
Compatible with Python 3.6 (Vitis AI Docker environment).
"""

import os
import glob
import numpy as np
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────
CALIB_DIR = os.environ.get(
    'CALIB_DIR',
    'data/RELLIS-3D_full'
)

IMG_HEIGHT   = 224
IMG_WIDTH    = 224
NORM_FACTOR  = 127.5
BATCH_SIZE   = 10

# Collect all image paths once at import time
_all_paths = sorted(
    glob.glob(os.path.join(CALIB_DIR, '**/*.png'), recursive=True) +
    glob.glob(os.path.join(CALIB_DIR, '**/*.jpg'), recursive=True)
)

if not _all_paths:
    raise RuntimeError(
        "graph_input_fn: no images found under CALIB_DIR={}. "
        "Set the CALIB_DIR environment variable or run from the project root.".format(CALIB_DIR)
    )

print("graph_input_fn: found {} calibration images in {}".format(
    len(_all_paths), CALIB_DIR))


def get_depth_path(rgb_path):
    """Gets matching depth map path for a given RGB image path, if it exists."""
    parts = rgb_path.split(os.sep)
    if 'rgb' in parts:
        idx = len(parts) - 1 - parts[::-1].index('rgb')
        parts[idx] = 'depth'
        base, _ = os.path.splitext(os.sep.join(parts))
        # Check for png first, then jpg
        for ext in ['.png', '.jpg']:
            depth_path = base + ext
            if os.path.exists(depth_path):
                return depth_path
    return None


def load_depth(path):
    """Loads a 16-bit depth map and normalizes it to [-1, 1]."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        
    depth = cv2.resize(depth, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    
    MAX_DEPTH_VAL = 65535.0
    if depth.max() > 0:            
         depth = depth / MAX_DEPTH_VAL * 2.0 - 1.0
    else:
         depth = depth * 0.0 - 1.0 # default to background distance
         
    if len(depth.shape) == 2:
        depth = np.expand_dims(depth, axis=-1)
    return depth


def calib_input(iter):
    """
    Return one batch of calibration images.

    Args:
        iter: batch index (0-based), used by vai_q_tensorflow per --calib_iter
    Returns:
        dict  {'input_1': list of (H, W, C) float32 arrays in [-1, 1]}
    """
    images = []
    
    # Parse target number of channels from INPUT_SHAPES env var
    input_shapes_env = os.environ.get('INPUT_SHAPES', '?,224,224,3')
    try:
        num_channels = int(input_shapes_env.split(',')[-1])
    except Exception:
        num_channels = 3

    for idx in range(BATCH_SIZE):
        path_idx = (iter * BATCH_SIZE + idx) % len(_all_paths)
        path = _all_paths[path_idx]

        # Load RGB
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        else:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / NORM_FACTOR - 1.0

        # Load and concatenate depth channel if target input shape has 4 channels
        if num_channels == 4:
            depth_path = get_depth_path(path)
            if depth_path is not None:
                depth = load_depth(depth_path)
            else:
                depth = np.full((IMG_HEIGHT, IMG_WIDTH, 1), -1.0, dtype=np.float32)
            img = np.concatenate([img, depth], axis=-1)

        images.append(img)

    return {'input_1': images}
