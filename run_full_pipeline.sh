#!/bin/bash
set -e
cd /media/gauravkh/B6F6CF21F6CEE12B/Users/gauravkh/git_clones/terrasense-v2

mkdir -p /tmp/rellis
echo "Downloading Annotations (94MB)..."
gdown "16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5" -O /tmp/rellis/annotations.zip || { echo "Failed to download annotations"; exit 1; }

echo "Downloading Images (11GB)..."
gdown "1F3Leu0H_m6aPVpZITragfreO_SGtL2yV" -O /tmp/rellis/images.zip || { echo "Failed to download images"; exit 1; }

echo "Extracting Datasets..."
rm -rf data/RELLIS-3D_full/{rgb,annotations,depth}
mkdir -p data/RELLIS-3D_full/{rgb,annotations,depth}

unzip -q /tmp/rellis/annotations.zip -d /tmp/rellis/annot_ex
unzip -q /tmp/rellis/images.zip -d /tmp/rellis/img_ex

echo "Flattening Dataset Structure..."
find /tmp/rellis/img_ex -type f -name "*.jpg" -exec mv {} data/RELLIS-3D_full/rgb/ \;
find /tmp/rellis/annot_ex -type f -name "*.png" -exec mv {} data/RELLIS-3D_full/annotations/ \;
find /tmp/rellis/annot_ex -type f -name "*.jpg" -exec mv {} data/RELLIS-3D_full/annotations/ \;

echo "Projecting LiDAR Point Clouds to Generate 2D Depth Maps..."
python3 generate_depth.py

echo "Cleaning up temporary files..."
rm -rf /tmp/rellis

echo "Starting 50-Epoch Full Training..."
python3 train.py --epochs 50 --batch_size 8 --lr 0.01 --data_root data/RELLIS-3D_full
