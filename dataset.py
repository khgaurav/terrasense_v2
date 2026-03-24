#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RUGD Dataset Loader for Semantic Segmentation.

Loads images from RUGD_frames-with-annotations/ and color-coded annotation
masks from RUGD_annotations/, converting RGB colors to class indices using
the 25-class RUGD colormap.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# ─── RUGD 25‑class colormap (RGB) from RUGD_annotation-colormap.txt ─────────
RUGD_CLASSES = [
    "void", "dirt", "sand", "grass", "tree", "pole", "water", "sky",
    "vehicle", "container/generic-object", "asphalt", "gravel", "building",
    "mulch", "rock-bed", "log", "bicycle", "person", "fence", "bush",
    "sign", "rock", "bridge", "concrete", "picnic-table"
]

# (R, G, B) tuples – order matches class index 0..24
RUGD_COLORMAP_RGB = np.array([
    [0,     0,   0],  # 0  void
    [108,  64,  20],  # 1  dirt
    [255, 229, 204],  # 2  sand
    [0,   102,   0],  # 3  grass
    [0,   255,   0],  # 4  tree
    [0,   153, 153],  # 5  pole
    [0,   128, 255],  # 6  water
    [0,     0, 255],  # 7  sky
    [255, 255,   0],  # 8  vehicle
    [255,   0, 127],  # 9  container/generic-object
    [64,   64,  64],  # 10 asphalt
    [255, 128,   0],  # 11 gravel
    [255,   0,   0],  # 12 building
    [153,  76,   0],  # 13 mulch
    [102, 102,   0],  # 14 rock-bed
    [102,   0,   0],  # 15 log
    [0,   255, 128],  # 16 bicycle
    [204, 153, 255],  # 17 person
    [102,   0, 204],  # 18 fence
    [255, 153, 204],  # 19 bush
    [0,   102, 102],  # 20 sign
    [153, 204, 255],  # 21 rock
    [102, 255, 255],  # 22 bridge
    [101, 101,  11],  # 23 concrete
    [114,  85,  47],  # 24 picnic-table
], dtype=np.uint8)

NUM_CLASSES = len(RUGD_CLASSES)  # 25

# Default train / val / test split by sequence name
DEFAULT_TRAIN_SEQS = [
    "park-1", "park-2", "park-8",
    "trail", "trail-3", "trail-4", "trail-5", "trail-6", "trail-7",
    "trail-11", "trail-12", "trail-13", "trail-14", "trail-15",
    "village",
]
DEFAULT_VAL_SEQS = ["trail-9", "trail-10"]
DEFAULT_TEST_SEQS = ["creek"]

# Normalization (same as Vitis AI tutorial)
NORM_FACTOR = 127.5

# ─── Helper: RGB mask → class index map ─────────────────────────────────────

def _build_color_to_class_lut():
    """Build a lookup table from (R,G,B) tuple to class index."""
    lut = {}
    for idx, rgb in enumerate(RUGD_COLORMAP_RGB):
        lut[tuple(rgb)] = idx
    return lut

_COLOR_TO_CLASS = _build_color_to_class_lut()


def rgb_mask_to_class_index(mask_rgb):
    """
    Convert an RGB annotation mask (H, W, 3) to a class index map (H, W).

    Uses a vectorized approach: builds a single uint32 key from R,G,B and
    looks up via a pre-built array.
    """
    # Build a flat lookup array indexed by packed RGB
    # Pack: key = R * 256*256 + G * 256 + B
    flat_lut = np.zeros(256 * 256 * 256, dtype=np.uint8)
    for rgb_tuple, cls_idx in _COLOR_TO_CLASS.items():
        key = rgb_tuple[0] * 65536 + rgb_tuple[1] * 256 + rgb_tuple[2]
        flat_lut[key] = cls_idx

    # Convert mask to uint32 keys
    r = mask_rgb[:, :, 0].astype(np.uint32)
    g = mask_rgb[:, :, 1].astype(np.uint32)
    b = mask_rgb[:, :, 2].astype(np.uint32)
    keys = r * 65536 + g * 256 + b

    return flat_lut[keys]


def class_index_to_rgb(class_map):
    """Convert a class index map (H, W) back to an RGB image (H, W, 3)."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        mask = class_map == c
        rgb[mask] = RUGD_COLORMAP_RGB[c]
    return rgb


# ─── Keras Sequence Data Generator ──────────────────────────────────────────

class RUGDDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for the RUGD dataset.

    Args:
        data_root:  Path to the RUGD directory (containing RUGD_frames-with-annotations/ and RUGD_annotations/)
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
        self.n_classes = NUM_CLASSES

        frames_dir = os.path.join(data_root, "RUGD_frames-with-annotations")
        annot_dir = os.path.join(data_root, "RUGD_annotations")

        if split == "train":
            seqs = DEFAULT_TRAIN_SEQS
        elif split == "val":
            seqs = DEFAULT_VAL_SEQS
        elif split == "test":
            seqs = DEFAULT_TEST_SEQS
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_paths = []
        self.mask_paths = []
        for seq in seqs:
            seq_frames = os.path.join(frames_dir, seq)
            seq_annots = os.path.join(annot_dir, seq)
            if not os.path.isdir(seq_frames):
                print(f"WARNING: sequence directory not found: {seq_frames}")
                continue
            if not os.path.isdir(seq_annots):
                print(f"WARNING: annotation directory not found: {seq_annots}")
                continue
            fnames = sorted(os.listdir(seq_frames))
            annot_fnames = set(os.listdir(seq_annots))
            for fn in fnames:
                if fn in annot_fnames:
                    self.image_paths.append(os.path.join(seq_frames, fn))
                    self.mask_paths.append(os.path.join(seq_annots, fn))

        print(f"[RUGDDataset] split={split}, images={len(self.image_paths)}")

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))

        X_batch = []
        Y_batch = []
        for i in range(start, end):
            img = self._load_image(self.image_paths[i])
            mask = self._load_mask(self.mask_paths[i])

            if self.augment and np.random.rand() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            X_batch.append(img)
            Y_batch.append(mask)

        return np.array(X_batch), np.array(Y_batch)

    def _load_image(self, path):
        """Load and normalize image to [-1, 1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype(np.float32) / NORM_FACTOR - 1.0
        return img  # (H, W, 3) in BGR, range [-1, 1]

    def _load_mask(self, path):
        """Load annotation mask and convert to one-hot (H, W, C)."""
        mask_bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, (self.img_size[1], self.img_size[0]),
                              interpolation=cv2.INTER_NEAREST)
        class_map = rgb_mask_to_class_index(mask_rgb)  # (H, W)
        # One-hot encode
        one_hot = np.zeros((self.img_size[0], self.img_size[1], self.n_classes),
                           dtype=np.float32)
        for c in range(self.n_classes):
            one_hot[:, :, c] = (class_map == c).astype(np.float32)
        return one_hot

    def get_all_data(self, max_images=None):
        """Load all images and masks into numpy arrays (for small datasets)."""
        n = len(self.image_paths) if max_images is None else min(max_images, len(self.image_paths))
        X = np.zeros((n, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        Y = np.zeros((n, self.img_size[0], self.img_size[1], self.n_classes), dtype=np.float32)
        for i in range(n):
            X[i] = self._load_image(self.image_paths[i])
            Y[i] = self._load_mask(self.mask_paths[i])
        return X, Y

    def on_epoch_end(self):
        """Shuffle data at end of each epoch."""
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]
