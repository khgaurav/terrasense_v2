#!/bin/bash
# =============================================================================
# TerraSense RGB UNet → ZCU102 Deployment Pipeline (Vitis AI TF1 flow)
#
# TWO-STAGE PROCESS:
#
#   Stage 1 — On the HOST (TF2, run once):
#     python3 vitis_ai/freeze.py \
#       --model  output/keras_model/ep50_final_unet_v2_224x224.h5 \
#       --output_dir build_vai/freeze
#
#   Stage 2 — Inside the Vitis AI Docker (this script):
#     docker run --gpus all -it --rm \
#       -v $(pwd):/workspace \
#       xilinx/vitis-ai-tensorflow-gpu:3.5.0.001- bash -c \
#       "cd /workspace && bash vitis_ai/run_zcu102.sh"
#
# ARGUMENTS (all optional)
#   $1  Path to frozen graph .pb     default: build_vai/freeze/frozen_graph.pb
#   $2  Calibration image root dir   default: data/RELLIS-3D_full
#   $3  Number of calibration batches (calib_iter)  default: 10
# =============================================================================

set -euo pipefail

export PATH=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin:$PATH

CONDA_PYTHON=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/python
VAI_Q=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_q_tensorflow
VAI_C=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow

FROZEN_PB="${1:-build_vai/freeze/frozen_graph.pb}"
CALIB_DIR="${2:-data/RELLIS-3D_full}"
CALIB_ITER="${3:-10}"

NET_NAME="terrasense_unet"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"

INPUT_NODE="input_1"
OUTPUT_NODE="Identity"
INPUT_SHAPES="?,224,224,4"

# Internal container paths to avoid NTFS permission/lock issues
TMP_FREEZE_DIR="/tmp/freeze"
TMP_QUANT_DIR="/tmp/quantized"
TMP_COMPILE_DIR="/tmp/compiled"

# Workspace output paths
BUILD_DIR="build_vai"
QUANT_DIR="${BUILD_DIR}/quantized"
COMPILE_DIR="${BUILD_DIR}/compiled"
TARGET_DIR="${BUILD_DIR}/target_zcu102/model"

echo "======================================================================"
echo "  TerraSense RGB UNet  →  ZCU102 (Vitis AI TF1 flow)"
echo "======================================================================"
echo "  Frozen graph  : ${FROZEN_PB}"
echo "  Calibration   : ${CALIB_DIR}  (${CALIB_ITER} batches × 10 images)"
echo "  Output root   : ${BUILD_DIR}/"
echo "======================================================================"

# Sanity checks
[ -f "${FROZEN_PB}" ] || { echo "ERROR: frozen graph not found: ${FROZEN_PB}"; exit 1; }
[ -d "${CALIB_DIR}" ] || { echo "ERROR: calib dir not found: ${CALIB_DIR}"; exit 1; }
[ -f "${ARCH}" ]      || { echo "ERROR: arch.json not found: ${ARCH}"; exit 1; }

mkdir -p "${TMP_FREEZE_DIR}" "${TMP_QUANT_DIR}" "${TMP_COMPILE_DIR}"
mkdir -p "${QUANT_DIR}" "${COMPILE_DIR}" "${TARGET_DIR}"

# Copy frozen graph to tmp
cp "${FROZEN_PB}" "${TMP_FREEZE_DIR}/frozen_graph.pb"
TMP_FROZEN_PB="${TMP_FREEZE_DIR}/frozen_graph.pb"

# ── Step 1: Inspect frozen graph ─────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 1/3: Inspect frozen graph"
echo "======================================================================"
${VAI_Q} inspect --input_frozen_graph "${TMP_FROZEN_PB}"

# ── Step 2: Quantise (FP32 → INT8) ───────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 2/3: Quantise with vai_q_tensorflow"
echo "======================================================================"

# vai_q_tensorflow --input_fn expects the module to be on PYTHONPATH
# We export CALIB_DIR and INPUT_SHAPES so graph_input_fn.py can find images and channels in the container
export CALIB_DIR="${CALIB_DIR}"
export INPUT_SHAPES="${INPUT_SHAPES}"

${VAI_Q} quantize \
    --input_frozen_graph  "${TMP_FROZEN_PB}"      \
    --input_nodes         "${INPUT_NODE}"          \
    --input_shapes        "${INPUT_SHAPES}"        \
    --output_nodes        "${OUTPUT_NODE}"         \
    --output_dir          "${TMP_QUANT_DIR}"       \
    --method              1                        \
    --input_fn            vitis_ai.graph_input_fn.calib_input \
    --calib_iter          "${CALIB_ITER}"          \
    --gpu                 0

TMP_QUANT_PB="${TMP_QUANT_DIR}/quantize_eval_model.pb"
# Copy quantized model and logs back to workspace
cp "${TMP_QUANT_PB}" "${QUANT_DIR}/"
cp -r "${TMP_QUANT_DIR}/"* "${QUANT_DIR}/" 2>/dev/null || true
QUANT_PB="${QUANT_DIR}/quantize_eval_model.pb"

echo ""
echo "Quantized model: ${QUANT_PB}"

# ── Step 3: Compile for ZCU102 ────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 3/3: Compile for ZCU102 DPUCZDX8G with vai_c_tensorflow"
echo "======================================================================"

${VAI_C} \
    --frozen_pb   "${TMP_QUANT_PB}" \
    --arch        "${ARCH}"       \
    --output_dir  "${TMP_COMPILE_DIR}" \
    --options     "{'mode':'normal'}" \
    --net_name    "${NET_NAME}"

# Copy compiled results back to workspace
cp -r "${TMP_COMPILE_DIR}/"* "${COMPILE_DIR}/"

# Copy artefacts to target directory
cp "${COMPILE_DIR}/${NET_NAME}.xmodel"    "${TARGET_DIR}/"
cp "${COMPILE_DIR}/${NET_NAME}.json"      "${TARGET_DIR}/" 2>/dev/null || true

# Package for SCP to board
TARBALL="${BUILD_DIR}/target_zcu102.tar.gz"
tar -czf "${TARBALL}" "${BUILD_DIR}/target_zcu102/"

echo ""
echo "======================================================================"
echo "DONE"
echo ""
echo "  Quantized model  : ${QUANT_PB}"
echo "  Compiled xmodel  : ${TARGET_DIR}/${NET_NAME}.xmodel"
echo "  Board tarball    : ${TARBALL}"
echo ""
echo "Deploy to ZCU102:"
echo "  scp ${TARBALL} root@<board_ip>:~/"
echo "  ssh root@<board_ip> 'tar -xzf target_zcu102.tar.gz'"
echo "======================================================================"
