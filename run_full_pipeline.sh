#!/bin/bash
# RELLIS-3D end-to-end pipeline: download → flatten → project depth → train.
#
# Re-extracting wipes ${DATASET_DIR}/{rgb,annotations,depth} — which includes the
# depth maps generate_depth.py spends a long time producing — so the delete is
# confirmed interactively first.
#
#   DATASET_DIR=...  where the flattened dataset lives (default below)
#   EPOCHS=...       training epochs (default: 50)
#   FORCE=1          delete without asking (unattended runs)
set -euo pipefail

# Resolve the repo root from this script's own location instead of hardcoding it.
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-data/RELLIS-3D_full}"
EPOCHS="${EPOCHS:-50}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rellis.XXXXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Repo root : ${REPO_ROOT}"
echo "Dataset   : ${DATASET_DIR}"
echo "Staging   : ${TMP_DIR}"

# Confirm before destroying previously extracted data and generated depth maps.
EXISTING=""
for sub in rgb annotations depth; do
    if [ -d "${DATASET_DIR}/${sub}" ] && [ -n "$(ls -A "${DATASET_DIR}/${sub}" 2>/dev/null)" ]; then
        EXISTING="${EXISTING} ${DATASET_DIR}/${sub}"
    fi
done

if [ -n "${EXISTING}" ]; then
    echo ""
    echo "The following non-empty directories will be DELETED and rebuilt:"
    for d in ${EXISTING}; do
        echo "  ${d} ($(find "${d}" -type f | wc -l) files)"
    done
    if [ "${FORCE:-0}" = "1" ]; then
        echo "FORCE=1 set, deleting without confirmation."
    else
        printf "Continue? [y/N] "
        read -r reply
        case "${reply}" in
            [yY]|[yY][eE][sS]) ;;
            *) echo "Aborted; nothing was deleted."; exit 1 ;;
        esac
    fi
fi

echo "Downloading Annotations (94MB)..."
gdown "16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5" -O "${TMP_DIR}/annotations.zip" \
    || { echo "Failed to download annotations"; exit 1; }

echo "Downloading Images (11GB)..."
gdown "1F3Leu0H_m6aPVpZITragfreO_SGtL2yV" -O "${TMP_DIR}/images.zip" \
    || { echo "Failed to download images"; exit 1; }

echo "Extracting Datasets..."
rm -rf "${DATASET_DIR}/rgb" "${DATASET_DIR}/annotations" "${DATASET_DIR}/depth"
mkdir -p "${DATASET_DIR}/rgb" "${DATASET_DIR}/annotations" "${DATASET_DIR}/depth"

unzip -q "${TMP_DIR}/annotations.zip" -d "${TMP_DIR}/annot_ex"
unzip -q "${TMP_DIR}/images.zip" -d "${TMP_DIR}/img_ex"

echo "Flattening Dataset Structure..."
find "${TMP_DIR}/img_ex" -type f -name "*.jpg" -exec mv {} "${DATASET_DIR}/rgb/" \;
find "${TMP_DIR}/annot_ex" -type f -name "*.png" -exec mv {} "${DATASET_DIR}/annotations/" \;
find "${TMP_DIR}/annot_ex" -type f -name "*.jpg" -exec mv {} "${DATASET_DIR}/annotations/" \;

echo "Projecting LiDAR Point Clouds to Generate 2D Depth Maps..."
python3 generate_depth.py

# Staging directory is removed by the EXIT trap.

echo "Starting ${EPOCHS}-Epoch Full Training..."
python3 train.py --epochs "${EPOCHS}" --batch_size 8 --lr 0.01 --data_root "${DATASET_DIR}"
