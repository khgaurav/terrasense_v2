#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RELLIS-3D Dataset Loader for Semantic Segmentation with Early Fusion.

Loads RGB images from rgb/, Depth images from depth/, and color/1D annotations
from annotations/, converting them to the 6 kinematic class ontology + Void
using an Early Fusion (4-channel) architecture.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# ─── Kinematic Ontology (K=6) ───────────────────────────────────────────────
KINEMATIC_CLASSES = [
    "Rigid Drivable",      # 0 (Asphalt, Concrete)
    "Granular Drivable",   # 1 (Dirt, Gravel)
    "Vegetation Drivable", # 2 (Grass)
    "Deformable Hazard",   # 3 (Mud, Puddle)
    "Non-Drivable Nature", # 4 (Tree, Bush, Water, Log)
    "Non-Drivable Rigid",  # 5 (Barrier, Rubble, Building, Fence, Vehicle, Person, Object, Pole)
    "Void"                 # 6 (Sky, Unlabeled, etc.)
]

NUM_CLASSES = 6 # The active K=6 kinematic classes models predict on (0-5)
TOTAL_CLASSES = 7 # Including Void

KINEMATIC_COLORMAP = np.array([
    [128, 128, 128],  # 0: Rigid Drivable (Gray)
    [139,  69,  19],  # 1: Granular Drivable (Brown)
    [  0, 255,   0],  # 2: Vegetation Drivable (Green)
    [255, 140,   0],  # 3: Deformable Hazard (Orange/Dark Mud)
    [  0, 100,   0],  # 4: Non-Drivable Nature (Dark Green)
    [255,   0,   0],  # 5: Non-Drivable Rigid (Red)
    [  0,   0,   0]   # 6: Void (Black)
], dtype=np.uint8)

# Map original RELLIS-3D IDs (0-34) to Kinematic classes (0-6)
RELLIS_MAPPING = {
    0: 6,    # void
    1: 1,    # dirt -> Granular
    3: 2,    # grass -> Vegetation
    4: 4,    # tree -> Nature
    5: 5,    # pole -> Rigid Obstacle
    6: 4,    # water -> Nature
    7: 6,    # sky -> Void
    8: 5,    # vehicle -> Rigid Obstacle
    9: 5,    # object -> Rigid Obstacle
    10: 0,   # asphalt -> Rigid
    12: 5,   # building -> Rigid Obstacle
    15: 4,   # log -> Nature
    17: 5,   # person -> Rigid Obstacle
    18: 5,   # fence -> Rigid Obstacle
    19: 4,   # bush -> Nature
    23: 0,   # concrete -> Rigid
    27: 5,   # barrier -> Rigid Obstacle
    31: 3,   # puddle -> Deformable Hazard
    33: 3,   # mud -> Deformable Hazard
    34: 5    # rubble -> Rigid Obstacle
}

# Precompute mapping array for fast vectorized lookup
MAPPING_LUT = np.full(256, 6, dtype=np.uint8)
for k, v in RELLIS_MAPPING.items():
    MAPPING_LUT[k] = v

NORM_FACTOR = 127.5

def rellis_mask_to_kinematic(mask_1d):
    """Map a 1D label mask containing RELLIS IDs to the 6 kinematic classes + Void."""
    return MAPPING_LUT[mask_1d]

def class_index_to_rgb(class_map):
    """Convert a class index map (H, W) back to an RGB image (H, W, 3)."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(TOTAL_CLASSES):
        mask = class_map == c
        rgb[mask] = KINEMATIC_COLORMAP[c]
    return rgb

# ─── Keras Sequence Data Generator ──────────────────────────────────────────

class RELLIS3DDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for RELLIS-3D with early fusion (RGB + Depth).

    Args:
        data_root:  Standard structure containing rgb/, depth/, and annotations/
        split:      'train', 'val', or 'test'
        batch_size: Batch size
        img_size:   (height, width) tuple
        augment:    Whether to apply data augmentation (horizontal flip)
    """

    def __init__(self, data_root, split="train", batch_size=8,
                 img_size=(224, 224), augment=False):
        self.data_root = data_root
        self.batch_size = batch_size
        self.img_size = img_size  # (H, W)
        self.augment = augment
        self.n_classes = NUM_CLASSES # 6 active classes
        self.total_classes = TOTAL_CLASSES # 7 including void

        self.rgb_dir = os.path.join(data_root, "rgb")
        self.depth_dir = os.path.join(data_root, "depth")
        self.annot_dir = os.path.join(data_root, "annotations")

        self.rgb_paths = []
        self.depth_paths = []
        self.annot_paths = []

        if not os.path.exists(self.rgb_dir):
            print(f"WARNING: RELLIS-3D rgb dir not found: {self.rgb_dir}")
            return
            
        all_rgb_files = sorted(os.listdir(self.rgb_dir))
        
        # Simple split logic if explicit lists aren't used:
        # 80% train, 10% val, 10% test
        np.random.seed(42) # Consistent splits
        shuffled_files = all_rgb_files.copy()
        np.random.shuffle(shuffled_files)
        
        n_total = len(shuffled_files)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if split == "train":
            split_files = shuffled_files[:n_train]
        elif split == "val":
            split_files = shuffled_files[n_train:n_train+n_val]
        else: # test
            split_files = shuffled_files[n_train+n_val:]

        for fname in split_files:
            base = os.path.splitext(fname)[0]
            rgb_path = os.path.join(self.rgb_dir, fname)
            # Find matching depth and annot. Extension could be png or jpg.
            depth_path = os.path.join(self.depth_dir, base + ".png")
            if not os.path.exists(depth_path):
                 depth_path = os.path.join(self.depth_dir, base + ".jpg")
                 
            annot_path = os.path.join(self.annot_dir, base + ".png")
            if not os.path.exists(annot_path):
                 annot_path = os.path.join(self.annot_dir, base + ".jpg")

            if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(annot_path):
                self.rgb_paths.append(rgb_path)
                self.depth_paths.append(depth_path)
                self.annot_paths.append(annot_path)

        print(f"[RELLIS3DDataset] split={split}, explicitly matching items={len(self.rgb_paths)}")

    def __len__(self):
        return int(np.ceil(len(self.rgb_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.rgb_paths))

        X_batch = []
        Y_batch = []
        for i in range(start, end):
            # Load and process inputs
            rgb_norm = self._load_rgb(self.rgb_paths[i])
            depth_norm = self._load_depth(self.depth_paths[i])
            # Early Fusion: Concatenate RGB and Depth -> (H, W, 4)
            fused_input = np.concatenate([rgb_norm, depth_norm], axis=-1)

            # Load and process masks
            mask_categorical = self._load_mask(self.annot_paths[i])

            if self.augment and np.random.rand() > 0.5:
                fused_input = np.fliplr(fused_input)
                mask_categorical = np.fliplr(mask_categorical)

            X_batch.append(fused_input)
            Y_batch.append(mask_categorical)

        return np.array(X_batch), np.array(Y_batch)

    def _load_rgb(self, path):
        """Load and normalize RGB image to [-1, 1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / NORM_FACTOR - 1.0
        return img  # (H, W, 3) range [-1, 1]

    def _load_depth(self, path):
        """Load depth and normalize."""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32)
            
        depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        
        # Simple normalization map 16-bit to [-1.0, 1.0]
        # In practice depending on the sensor range (e.g. 50m limit) this might be
        # clamped or adaptive. Since the Vitis DPU handles max activations well with INT8 QAT,
        # we'll map utilizing a standard UINT16 divisor assumption:
        MAX_DEPTH_VAL = 65535.0
        if depth.max() > 0:            
             depth = depth / MAX_DEPTH_VAL * 2.0 - 1.0
        else:
             depth = depth * 0.0 - 1.0 # default to background distance
             
        return np.expand_dims(depth, axis=-1) # (H, W, 1)

    def _load_mask(self, path):
        """Load annotation mask and convert to one-hot for the active K=6 classes + Void."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # RELLIS-3D provides ID-based masks
        if mask is None:
             # Return empty one-hot indicating Void everywhere
             one_hot = np.zeros((self.img_size[0], self.img_size[1], self.total_classes), dtype=np.float32)
             one_hot[:, :, 6] = 1.0
             return one_hot
             
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        class_map = rellis_mask_to_kinematic(mask)
        
        # One-hot encode for all 7 classes (including void)
        one_hot = np.zeros((self.img_size[0], self.img_size[1], self.total_classes), dtype=np.float32)
        for c in range(self.total_classes):
            one_hot[:, :, c] = (class_map == c).astype(np.float32)
            
        return one_hot

    def get_all_data(self, max_images=None):
        """Load all images into numpy arrays."""
        n = len(self.rgb_paths) if max_images is None else min(max_images, len(self.rgb_paths))
        X = np.zeros((n, self.img_size[0], self.img_size[1], 4), dtype=np.float32)
        Y = np.zeros((n, self.img_size[0], self.img_size[1], self.total_classes), dtype=np.float32)
        for i in range(n):
            rgb = self._load_rgb(self.rgb_paths[i])
            depth = self._load_depth(self.depth_paths[i])
            X[i] = np.concatenate([rgb, depth], axis=-1)
            Y[i] = self._load_mask(self.annot_paths[i])
        return X, Y

    def on_epoch_end(self):
        """Shuffle data at end of each epoch."""
        indices = np.arange(len(self.rgb_paths))
        np.random.shuffle(indices)
        self.rgb_paths = [self.rgb_paths[i] for i in indices]
        self.depth_paths = [self.depth_paths[i] for i in indices]
        self.annot_paths = [self.annot_paths[i] for i in indices]
