#!/bin/bash
# =============================================================================
# Freeze the trained RGB-D UNet, quantize it to INT8 and compile it for the KR260
# DPU. Run from the repository root: bash vitis_ai/run_kr260.sh
#
# The freeze step runs on the host because it needs TF2 to load the Keras model;
# the container only has TF1.15. Point PYTHON at your TF2 interpreter if `python3`
# is not it.
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DOCKER_IMAGE="xilinx/vitis-ai-tensorflow-gpu:3.5.0.001-"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

MODEL_PATH="output/keras_model/ep50_final_unet_v2_rgbd_224x224.h5"
DATA_ROOT="data/RELLIS-3D_full"
FREEZE_DIR="build_vai/freeze"
QUANT_DIR="build_vai/quantized"
COMPILE_DIR="build_vai/compiled"
TARGET_DIR="build_vai/target_kr260/model"
NET_NAME="terrasense_unet_rgbd"
INPUT_SHAPES="?,224,224,4"          # 4 channels: BGR + depth
INPUT_NODE="input_1"
OUTPUT_NODE="Identity"

[ -f "${MODEL_PATH}" ] || { echo "ERROR: no trained model at ${MODEL_PATH}; run train.py first"; exit 1; }
[ -d "${DATA_ROOT}" ]  || { echo "ERROR: no dataset at ${DATA_ROOT}"; exit 1; }

mkdir -p "${FREEZE_DIR}" "${QUANT_DIR}" "${COMPILE_DIR}" "${TARGET_DIR}"

# Run a command inside the Vitis AI TensorFlow conda environment.
vai_run() {
    docker run --rm -v "$(pwd)":/workspace -w /workspace "${DOCKER_IMAGE}" \
        bash -c "source /opt/vitis_ai/conda/bin/activate vitis-ai-tensorflow && $1"
}

echo "== 1/3 Freeze the Keras model (host, ${PYTHON}) =="
# Freezing is a graph transformation, so keep it off the GPU entirely.
CUDA_VISIBLE_DEVICES="" "${PYTHON}" vitis_ai/freeze.py \
    --model "${MODEL_PATH}" --output_dir "${FREEZE_DIR}"

echo "== 2/3 Quantize to INT8 =="
vai_run "export CALIB_DIR=${DATA_ROOT} && export INPUT_SHAPES=${INPUT_SHAPES} && \
    export PYTHONPATH=\$PYTHONPATH:/workspace && \
    vai_q_tensorflow quantize \
        --input_frozen_graph ${FREEZE_DIR}/frozen_graph.pb \
        --input_nodes ${INPUT_NODE} \
        --input_shapes ${INPUT_SHAPES} \
        --output_nodes ${OUTPUT_NODE} \
        --output_dir ${QUANT_DIR} \
        --method 1 \
        --input_fn vitis_ai.graph_input_fn.calib_input \
        --calib_iter 10"

echo "== 3/3 Compile for KR260 =="
vai_run "vai_c_tensorflow \
    --frozen_pb ${QUANT_DIR}/quantize_eval_model.pb \
    --arch ${ARCH} \
    --output_dir ${COMPILE_DIR} \
    --net_name ${NET_NAME}"

cp "${COMPILE_DIR}/${NET_NAME}.xmodel" "${TARGET_DIR}/"
TARBALL="build_vai/target_kr260.tar.gz"
tar -czf "${TARBALL}" -C "build_vai" "target_kr260"

echo
echo "Compiled xmodel : ${TARGET_DIR}/${NET_NAME}.xmodel"
echo "Deploy tarball  : ${TARBALL}"
