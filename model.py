#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNET v2 Model for Semantic Segmentation.

Adapted from the Vitis AI tutorial (Keras_FCN8_UNET_segmentation/files/code/config/unet.py).
Uses UpSampling2D + Conv2D (DPU-compatible) instead of Conv2DTranspose.

Architecture: UNET v2 with batch normalization and dropout.
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D,
    Concatenate, BatchNormalization, Activation
)

IMAGE_ORDERING = "channels_last"


def conv2d_block(input_tensor, n_filters, kernel_size=3,
                 batchnorm=True, activation=True):
    """Two Conv2D layers with optional BatchNorm and ReLU."""
    # First layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer='he_normal',
        padding='same',
        data_format=IMAGE_ORDERING
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    # Second layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer='he_normal',
        padding='same',
        data_format=IMAGE_ORDERING
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)

    return x


def build_unet(n_classes, input_height=224, input_width=224,
               n_filters=64, dropout=0.1, batchnorm=True):
    """
    Build UNET v2 model.

    Uses UpSampling2D + Conv2D in the decoder (DPU-compatible).
    Output has 'relu' activation (no softmax) — softmax is applied in
    the loss function or post-processing, following the Vitis AI pattern.

    Args:
        n_classes:    Number of segmentation classes
        input_height: Input image height (must be divisible by 16)
        input_width:  Input image width (must be divisible by 16)
        n_filters:    Base number of filters (doubled at each level)
        dropout:      Dropout rate
        batchnorm:    Whether to use batch normalization
    Returns:
        tf.keras.Model
    """
    assert input_height % 16 == 0, f"input_height must be divisible by 16, got {input_height}"
    assert input_width % 16 == 0, f"input_width must be divisible by 16, got {input_width}"

    img_input = Input(shape=(input_height, input_width, 4))

    # Encoder (Downsampling)
    # Block 1: n_filters
    c1 = conv2d_block(img_input, n_filters * 1, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    # Block 2: n_filters * 2
    c2 = conv2d_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    # Block 3: n_filters * 4
    c3 = conv2d_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    # Block 4: n_filters * 8
    c4 = conv2d_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck: n_filters * 16
    c5 = conv2d_block(p4, n_filters * 16, batchnorm=batchnorm)
    p5 = Dropout(dropout)(c5)

    # Decoder (Upsampling)
    # Block 6
    up6 = UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING,
                       interpolation="bilinear")(p5)
    u6 = Conv2D(n_filters * 8, kernel_size=2, activation="relu",
                padding="same", kernel_initializer="he_normal",
                data_format=IMAGE_ORDERING)(up6)
    m6 = Concatenate(axis=3)([u6, c4])
    m6 = Dropout(dropout)(m6)
    c6 = conv2d_block(m6, n_filters * 8, batchnorm=batchnorm)

    # Block 7
    up7 = UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING,
                       interpolation="bilinear")(c6)
    u7 = Conv2D(n_filters * 4, kernel_size=2, activation="relu",
                padding="same", kernel_initializer="he_normal",
                data_format=IMAGE_ORDERING)(up7)
    m7 = Concatenate(axis=3)([u7, c3])
    m7 = Dropout(dropout)(m7)
    c7 = conv2d_block(m7, n_filters * 4, batchnorm=batchnorm)

    # Block 8
    up8 = UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING,
                       interpolation="bilinear")(c7)
    u8 = Conv2D(n_filters * 2, kernel_size=2, activation="relu",
                padding="same", kernel_initializer="he_normal",
                data_format=IMAGE_ORDERING)(up8)
    m8 = Concatenate(axis=3)([u8, c2])
    m8 = Dropout(dropout)(m8)
    c8 = conv2d_block(m8, n_filters * 2, batchnorm=batchnorm)

    # Block 9
    up9 = UpSampling2D(size=(2, 2), data_format=IMAGE_ORDERING,
                       interpolation="bilinear")(c8)
    u9 = Conv2D(n_filters * 1, kernel_size=2, activation="relu",
                padding="same", kernel_initializer="he_normal",
                data_format=IMAGE_ORDERING)(up9)
    m9 = Concatenate(axis=3)([u9, c1])
    m9 = Dropout(dropout)(m9)
    c9 = conv2d_block(m9, n_filters * 1, batchnorm=batchnorm)

    # Output
    # Use relu activation (no softmax) for Vitis AI DPU compatibility.
    # Softmax is computed in software by the ARM CPU.
    c10 = Conv2D(
        filters=n_classes, kernel_size=3,
        activation="relu", padding="same",
        kernel_initializer="he_normal",
        data_format=IMAGE_ORDERING
    )(c9)

    model = Model(inputs=img_input, outputs=c10, name="unet_v2_rugd")
    return model


if __name__ == "__main__":
    model = build_unet(7, 224, 224)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Input:  {model.input_shape}")
    print(f"Output: {model.output_shape}")
