#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RELLIS-3D Dataset Loader for Semantic Segmentation with Early Fusion.

Loads RGB images from rgb/, Depth images from depth/, and color/1D annotations
from annotations/, converting them to the 7-class kinematic ontology (6 active + Void)
using an Early Fusion (4-channel) architecture with simulated RealSense depth.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# ─── Kinematic Ontology (K=11: 10 active + Void) ───────────────────────────
KINEMATIC_CLASSES = [
    "Smooth Drivable",     # 0 (Asphalt, Concrete)
    "Grass",               # 1 (Grass)
    "Dirt",                # 2 (Dirt)
    "Sand",                # 3 (Sand)
    "Gravel",              # 4 (Gravel)
    "Mulch",               # 5 (Mulch)
    "Puddle",              # 6 (Puddle)
    "Mud",                 # 7 (Mud)
    "Soft Obstacle",       # 8 (Bush, Log, Rubble — potentially traversable for larger robots)
    "Obstacle",            # 9 (Tree, Water, Vehicle, Barrier, Building, Person, Fence, Pole, etc.)
    "Void"                 # 10 (Sky, Unlabeled)
]

NUM_CLASSES = 10 # The active K=10 kinematic classes models predict on (0-9)
TOTAL_CLASSES = 11 # Including Void

KINEMATIC_COLORMAP = np.array([
    [128, 128, 128],  # 0: Smooth Drivable (Gray)
    [  0, 255,   0],  # 1: Grass (Green)
    [139,  69,  19],  # 2: Dirt (Brown)
    [244, 164,  96],  # 3: Sand (Sandy Brown)
    [211, 211, 211],  # 4: Gravel (Light Gray)
    [101,  67,  33],  # 5: Mulch (Dark Brown)
    [  0,   0, 255],  # 6: Puddle (Blue)
    [ 85, 107,  47],  # 7: Mud (Dark Olive Green)
    [144, 238, 144],  # 8: Soft Obstacle (Light Green)
    [255,   0,   0],  # 9: Obstacle (Red)
    [  0,   0,   0]   # 10: Void (Black)
], dtype=np.uint8)

# Traversability costs for a 4-wheeled robot (scale: 0-255)
# 0: Smooth Drivable (Asphalt, Concrete) -> Lowest resistance (Cost: 1)
# 1: Grass                                -> Low-medium friction (Cost: 10)
# 2: Dirt                                 -> Good unpaved traction (Cost: 5)
# 3: Sand                                 -> Very high slip, sinkage risk (Cost: 30)
# 4: Gravel                               -> Hard loose surface, some slip (Cost: 15)
# 5: Mulch                                -> Spongy organic cover, low traction (Cost: 25)
# 6: Puddle                               -> Water pocket, unknown bottom (Cost: 40)
# 7: Mud                                  -> High slip, sinkage, trapping risk (Cost: 90)
# 8: Soft Obstacle (Bush, Log, Rubble)    -> Passable with risk (Cost: 80)
# 9: Obstacle (Tree, Water, Vehicle, etc.)-> Lethal / collision (Cost: 255)
# 10: Void (Sky, Unknown)                 -> Ignore / lethal (Cost: 255)
CLASS_COSTS = {
    0: 1,
    1: 10,
    2: 5,
    3: 30,
    4: 15,
    5: 25,
    6: 40,
    7: 90,
    8: 80,
    9: 255,
    10: 255
}

# Map original RELLIS-3D IDs (0-34) to Kinematic classes (0-10)
RELLIS_MAPPING = {
    0: 10,   # void -> Void
    1: 2,    # dirt -> Dirt
    2: 3,    # sand -> Sand
    3: 1,    # grass -> Grass
    4: 9,    # tree -> Obstacle
    5: 9,    # pole -> Obstacle
    6: 9,    # water -> Obstacle
    7: 10,   # sky -> Void
    8: 9,    # vehicle -> Obstacle
    9: 9,    # container/generic-object -> Obstacle
    10: 0,   # asphalt -> Smooth Drivable
    11: 4,   # gravel -> Gravel
    12: 9,   # building -> Obstacle
    13: 5,   # mulch -> Mulch
    14: 9,   # rock-bed -> Obstacle
    15: 8,   # log -> Soft Obstacle
    16: 9,   # bicycle -> Obstacle
    17: 9,   # person -> Obstacle
    18: 9,   # fence -> Obstacle
    19: 8,   # bush -> Soft Obstacle
    20: 9,   # sign -> Obstacle
    21: 9,   # rock -> Obstacle
    22: 9,   # bridge -> Obstacle
    23: 0,   # concrete -> Smooth Drivable
    24: 9,   # picnic-table -> Obstacle
    27: 9,   # barrier -> Obstacle
    31: 6,   # puddle -> Puddle
    33: 7,   # mud -> Mud
    34: 8    # rubble -> Soft Obstacle
}

# Precompute mapping array for fast vectorized lookup
MAPPING_LUT = np.full(256, 10, dtype=np.uint8)
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

def simulate_realsense_depth(depth_raw, rng):
    """
    Simulate Intel RealSense depth map from raw LiDAR projected depth map (H, W, 1).
    
    Steps:
    1. Densification: Morphological dilation to fill sparse laser rings.
    2. Range Clamping: Clamp depth values to a reliable 10-meter range.
    3. Quadratic Noise: Add distance-dependent noise: sigma = 0.002 * depth_meters^2.
    """
    # 1. Morphological dilation to fill sparse laser rings (H, W)
    depth_2d = depth_raw[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(depth_2d, kernel)
    
    # 2. Convert to meters for clamping and noise
    z = dilated.astype(np.float32) / 1000.0
    
    # 3. Clamp reliable range to 10 meters
    z_clamped = np.minimum(z, 10.0)
    
    # 4. Add quadratic noise (RealSense stereo noise: sigma = 0.002 * z^2)
    noise_std = 0.002 * (z_clamped ** 2)
    try:
        noise = rng.standard_normal(z_clamped.shape, dtype=np.float32) * noise_std
    except TypeError:
        noise = (rng.standard_normal(z_clamped.shape) * noise_std).astype(np.float32)
    z_noisy = z_clamped + noise
    
    # 5. Convert back to uint16 mm and restore channel dimension
    simulated_depth = np.clip(z_noisy * 1000.0, 0, 65535).astype(np.uint16)
    return np.expand_dims(simulated_depth, axis=-1)

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
        max_samples: Maximum number of samples to load (useful for testing)
    """

    def __init__(self, data_root, split="train", batch_size=8,
                 img_size=(224, 224), augment=False, max_samples=None):
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
            
        # Locate the split list file
        split_dir = os.path.join(data_root, "Rellis_3D_image_split")
        lst_file = os.path.join(split_dir, f"{split}.lst")
        
        if not os.path.exists(lst_file):
            print(f"WARNING: RELLIS-3D split list file not found: {lst_file}")
            return

        # Parse and match files listed in the split list
        with open(lst_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                rgb_base = os.path.basename(parts[0])
                label_base = os.path.basename(parts[1])
                
                rgb_path = os.path.join(self.rgb_dir, rgb_base)
                annot_path = os.path.join(self.annot_dir, label_base)
                
                base_name = os.path.splitext(rgb_base)[0]
                depth_path = os.path.join(self.depth_dir, base_name + ".png")
                if not os.path.exists(depth_path):
                    depth_path = os.path.join(self.depth_dir, base_name + ".jpg")
                
                # Check that all three modalities exist (RGB, projected depth, and label)
                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(annot_path):
                    self.rgb_paths.append(rgb_path)
                    self.depth_paths.append(depth_path)
                    self.annot_paths.append(annot_path)

        if max_samples is not None:
            self.rgb_paths = self.rgb_paths[:max_samples]
            self.depth_paths = self.depth_paths[:max_samples]
            self.annot_paths = self.annot_paths[:max_samples]

        print(f"[RELLIS3DDataset] split={split}, explicitly matching items={len(self.rgb_paths)}")

        # RAM Caching to bypass slow on-the-fly OpenCV file loading/decoding
        # Uses lazy initialization to prevent massive startup delays
        self.cache = True
        self.cached_rgb = [None] * len(self.rgb_paths)
        self.cached_depth = [None] * len(self.rgb_paths)
        self.cached_annot = [None] * len(self.rgb_paths)
        
        # Instantiate RNG for RealSense simulation noise
        try:
            self.rng = np.random.default_rng(42)
        except AttributeError:
            self.rng = np.random.RandomState(42)

    def __len__(self):
        return int(np.ceil(len(self.rgb_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.rgb_paths))

        X_batch = np.zeros((end - start, self.img_size[0], self.img_size[1], 4), dtype=np.float32)
        Y_batch = np.zeros((end - start, self.img_size[0], self.img_size[1], self.total_classes), dtype=np.float32)

        for i_batch, i in enumerate(range(start, end)):
            if self.cache:
                if self.cached_rgb[i] is None:
                    # Load raw RGB (H, W, 3) as uint8
                    rgb = cv2.imread(self.rgb_paths[i], cv2.IMREAD_COLOR)
                    if rgb is None:
                        rgb = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                    else:
                        rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]))
                    self.cached_rgb[i] = rgb

                    # Load depth (H, W, 1) as uint16
                    depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        depth = np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.uint16)
                    else:
                        depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                        if len(depth.shape) == 2:
                            depth = np.expand_dims(depth, axis=-1)
                    self.cached_depth[i] = depth

                    # Load mask (H, W) as uint8
                    mask = cv2.imread(self.annot_paths[i], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.full((self.img_size[0], self.img_size[1]), 6, dtype=np.uint8)
                    else:
                        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                    self.cached_annot[i] = mask

                # Normalize RGB to [-1, 1]
                rgb = self.cached_rgb[i].astype(np.float32) / NORM_FACTOR - 1.0
                
                # Normalize Depth to [-1, 1] (with simulated RealSense profile)
                depth_raw = self.cached_depth[i]
                depth_sim = simulate_realsense_depth(depth_raw, self.rng)
                depth = depth_sim.astype(np.float32)
                MAX_DEPTH_VAL = 10000.0
                if depth.max() > 0:            
                     depth = depth / MAX_DEPTH_VAL * 2.0 - 1.0
                else:
                     depth = depth * 0.0 - 1.0
                
                fused_input = np.concatenate([rgb, depth], axis=-1)
                mask = self.cached_annot[i]
                class_map = rellis_mask_to_kinematic(mask)
            else:
                rgb = self._load_rgb(self.rgb_paths[i])
                depth = self._load_depth(self.depth_paths[i])
                mask_raw = cv2.imread(self.annot_paths[i], cv2.IMREAD_GRAYSCALE)
                if mask_raw is None:
                    mask = np.full((self.img_size[0], self.img_size[1]), 6, dtype=np.uint8)
                else:
                    mask = cv2.resize(mask_raw, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                fused_input = np.concatenate([rgb, depth], axis=-1)
                class_map = rellis_mask_to_kinematic(mask)

            # Fast vectorized one-hot encoding using broadcasting
            mask_categorical = (class_map[..., None] == np.arange(self.total_classes)).astype(np.float32)

            if self.augment and np.random.rand() > 0.5:
                fused_input = np.fliplr(fused_input)
                mask_categorical = np.fliplr(mask_categorical)

            X_batch[i_batch] = fused_input
            Y_batch[i_batch] = mask_categorical

        return X_batch, Y_batch

    def _load_rgb(self, path):
        """Load and normalize RGB image to [-1, 1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        return img.astype(np.float32) / NORM_FACTOR - 1.0

    def _load_depth(self, path):
        """Load, simulate RealSense depth, and normalize to [-1, 1]."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Apply RealSense simulation on raw image
        img_sim = simulate_realsense_depth(img, self.rng)
        
        img_norm = img_sim.astype(np.float32)
        MAX_DEPTH_VAL = 10000.0
        if img_norm.max() > 0:
            return img_norm / MAX_DEPTH_VAL * 2.0 - 1.0
        else:
            return img_norm * 0.0 - 1.0

    def _load_mask(self, path):
        """Load annotation and map to kinematic classes one-hot."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.full((self.img_size[0], self.img_size[1]), 6, dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        class_map = rellis_mask_to_kinematic(mask)
        one_hot = (class_map[..., None] == np.arange(self.total_classes)).astype(np.float32)
        return one_hot

    def get_all_data(self, max_images=None):
        """Load all images into numpy arrays."""
        n = len(self.rgb_paths) if max_images is None else min(max_images, len(self.rgb_paths))
        X = np.zeros((n, self.img_size[0], self.img_size[1], 4), dtype=np.float32)
        Y = np.zeros((n, self.img_size[0], self.img_size[1], self.total_classes), dtype=np.float32)
        for i in range(n):
            if self.cache:
                if self.cached_rgb[i] is None:
                    # Populate cache element
                    rgb = cv2.imread(self.rgb_paths[i], cv2.IMREAD_COLOR)
                    if rgb is None:
                        rgb = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                    else:
                        rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]))
                    self.cached_rgb[i] = rgb

                    depth = cv2.imread(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        depth = np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.uint16)
                    else:
                        depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                        if len(depth.shape) == 2:
                            depth = np.expand_dims(depth, axis=-1)
                    self.cached_depth[i] = depth

                    mask = cv2.imread(self.annot_paths[i], cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.full((self.img_size[0], self.img_size[1]), 6, dtype=np.uint8)
                    else:
                        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                    self.cached_annot[i] = mask

                # Get from cache
                rgb = self.cached_rgb[i].astype(np.float32) / NORM_FACTOR - 1.0
                # Normalize Depth to [-1, 1] (with simulated RealSense profile)
                depth_raw = self.cached_depth[i]
                depth_sim = simulate_realsense_depth(depth_raw, self.rng)
                depth = depth_sim.astype(np.float32)
                MAX_DEPTH_VAL = 10000.0
                if depth.max() > 0:            
                     depth = depth / MAX_DEPTH_VAL * 2.0 - 1.0
                else:
                     depth = depth * 0.0 - 1.0
                
                X[i] = np.concatenate([rgb, depth], axis=-1)
                class_map = rellis_mask_to_kinematic(self.cached_annot[i])
                Y[i] = (class_map[..., None] == np.arange(self.total_classes)).astype(np.float32)
            else:
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
        if self.cache:
            self.cached_rgb = [self.cached_rgb[i] for i in indices]
            self.cached_depth = [self.cached_depth[i] for i in indices]
            self.cached_annot = [self.cached_annot[i] for i in indices]
