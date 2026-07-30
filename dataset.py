#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RELLIS-3D dataset for the early-fusion RGB-D UNet.

Reads RGB from rgb/, projected LiDAR depth from depth/ (see generate_depth.py) and
label ids from annotations/, remaps the RELLIS-3D ontology to 11 kinematic classes
(10 active + Void) and yields 4-channel RGB-D inputs.

CHANNEL ORDER: RGB frames stay in OpenCV's native BGR order and are scaled to
[-1, 1]; depth is scaled by normalize_depth(). Inference and DPU calibration must
reuse both. Decoded frames are cached in RAM on first access.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

# Input resolution of the whole pipeline: training, calibration and the compiled
# xmodel all use this shape.
IMG_SIZE = (224, 224)  # (H, W)

# Scale factor mapping uint8 pixels to [-1, 1].
NORM_FACTOR = 127.5

# Channel order the network is trained on, declared rather than left implicit.
CHANNEL_ORDER = "bgr"

# ─── Kinematic ontology (11 classes: 10 active + Void) ─────────────────────
KINEMATIC_CLASSES = [
    "Smooth Drivable",     # 0 (Asphalt, Concrete)
    "Grass",               # 1
    "Dirt",                # 2
    "Sand",                # 3
    "Gravel",              # 4
    "Mulch",               # 5
    "Puddle",              # 6
    "Mud",                 # 7
    "Soft Obstacle",       # 8 (Bush, Log, Rubble — passable for larger robots)
    "Obstacle",            # 9 (Tree, Water, Vehicle, Barrier, Building, Person, ...)
    "Void"                 # 10 (Sky, Unlabeled)
]

NUM_CLASSES = 10    # active classes the model is scored on (0-9)
TOTAL_CLASSES = 11  # including Void, which the model still predicts

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

# Original RELLIS-3D label ids (0-34) -> kinematic classes (0-10)
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

# Unmapped ids fall through to Void.
MAPPING_LUT = np.full(256, 10, dtype=np.uint8)
for _raw_id, _kinematic in RELLIS_MAPPING.items():
    MAPPING_LUT[_raw_id] = _kinematic

# ─── Depth scaling (single source of truth) ─────────────────────────────────
# Depth maps are 16-bit millimetres. Only the first DEPTH_CLAMP_M metres are
# treated as reliable (matching the RealSense simulation below), and that range is
# mapped linearly onto [-1, 1].
DEPTH_CLAMP_M = 10.0
DEPTH_MAX_MM = DEPTH_CLAMP_M * 1000.0


def normalize_depth(depth_mm):
    """Scale a depth map in millimetres to [-1, 1] the way training does.

    Accepts (H, W) or (H, W, 1) and always returns (H, W, 1) float32.
    Absent depth (all zeros) comes out at -1.0, i.e. "no return".
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim == 2:
        depth = np.expand_dims(depth, axis=-1)
    return np.minimum(depth, DEPTH_MAX_MM) / DEPTH_MAX_MM * 2.0 - 1.0


def rellis_mask_to_kinematic(mask_ids):
    """Map raw RELLIS-3D label ids to the 10 kinematic classes + Void."""
    return MAPPING_LUT[mask_ids]


def class_index_to_rgb(class_map):
    """Colour a class index map (H, W) for visualisation."""
    return KINEMATIC_COLORMAP[class_map]


def simulate_realsense_depth(depth_mm, rng):
    """Make projected LiDAR depth look like an Intel RealSense stereo sensor.

    Dilate to fill the gaps between laser rings, clamp to the reliable range, then
    add the distance-dependent noise of a stereo sensor (sigma = 0.002 * z^2).
    Training on degraded depth is what lets the model transfer to the cheaper
    sensor the robot actually carries.
    """
    dilated = cv2.dilate(depth_mm[:, :, 0],
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    z = np.minimum(dilated.astype(np.float32) / 1000.0, DEPTH_CLAMP_M)
    noise_std = 0.002 * (z ** 2)
    try:
        noise = rng.standard_normal(z.shape, dtype=np.float32) * noise_std
    except TypeError:                       # older numpy has no dtype argument
        noise = (rng.standard_normal(z.shape) * noise_std).astype(np.float32)

    noisy_mm = np.clip((z + noise) * 1000.0, 0, 65535).astype(np.uint16)
    return np.expand_dims(noisy_mm, axis=-1)


class RELLIS3DDataset(tf.keras.utils.Sequence):
    """Keras Sequence over one RELLIS-3D split.

    Yields (4-channel RGB-D in [-1, 1], one-hot masks over TOTAL_CLASSES).
    """

    def __init__(self, data_root, split="train", batch_size=8, augment=False,
                 max_samples=None):
        self.batch_size = batch_size
        self.augment = augment
        self.rng = np.random.default_rng(42)

        rgb_dir = os.path.join(data_root, "rgb")
        depth_dir = os.path.join(data_root, "depth")
        annot_dir = os.path.join(data_root, "annotations")
        lst_file = os.path.join(data_root, "Rellis_3D_image_split", f"{split}.lst")

        self.samples = []  # (rgb_path, depth_path, annotation_path)
        if not os.path.exists(lst_file):
            print(f"WARNING: split list not found: {lst_file}")
        else:
            with open(lst_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    rgb = os.path.join(rgb_dir, os.path.basename(parts[0]))
                    annot = os.path.join(annot_dir, os.path.basename(parts[1]))
                    stem = os.path.splitext(os.path.basename(parts[0]))[0]
                    depth = os.path.join(depth_dir, stem + ".png")
                    if all(os.path.exists(p) for p in (rgb, depth, annot)):
                        self.samples.append((rgb, depth, annot))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[RELLIS3DDataset] split={split}, samples={len(self.samples)}")
        self._cache = [None] * len(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def _load(self, i):
        """Decode and resize one sample; the raw arrays are cached."""
        rgb_path, depth_path, annot_path = self.samples[i]
        h, w = IMG_SIZE

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)            # BGR
        if rgb is None:
            raise FileNotFoundError(f"Unreadable image: {rgb_path}")
        rgb = cv2.resize(rgb, (w, h))

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)    # uint16 mm
        if depth is None:
            raise FileNotFoundError(f"Unreadable depth map: {depth_path}")
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=-1)

        mask = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)     # raw RELLIS ids
        if mask is None:
            raise FileNotFoundError(f"Unreadable annotation: {annot_path}")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return rgb, depth, rellis_mask_to_kinematic(mask)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.samples))
        h, w = IMG_SIZE

        X = np.zeros((end - start, h, w, 4), dtype=np.float32)
        Y = np.zeros((end - start, h, w, TOTAL_CLASSES), dtype=np.float32)
        eye = np.eye(TOTAL_CLASSES, dtype=np.float32)

        for n, i in enumerate(range(start, end)):
            if self._cache[i] is None:
                self._cache[i] = self._load(i)
            rgb, depth, class_map = self._cache[i]

            fused = np.concatenate([
                rgb.astype(np.float32) / NORM_FACTOR - 1.0,
                normalize_depth(simulate_realsense_depth(depth, self.rng)),
            ], axis=-1)
            onehot = eye[class_map]

            if self.augment and np.random.rand() > 0.5:
                fused, onehot = np.fliplr(fused), np.fliplr(onehot)

            X[n] = fused
            Y[n] = onehot
        return X, Y

    def on_epoch_end(self):
        """Shuffle sample order (and the cache alongside it) between epochs."""
        order = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in order]
        self._cache = [self._cache[i] for i in order]
