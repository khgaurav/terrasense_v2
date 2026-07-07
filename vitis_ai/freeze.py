#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freeze the trained TerraSense RGB UNet to a TF1-compatible frozen graph (.pb).

Run on the HOST (requires TF2, NOT inside the Vitis AI Docker):
    python3 vitis_ai/freeze.py \
        --model  output/keras_model/ep50_final_unet_v2_224x224.h5 \
        --output_dir build_vai/freeze

Produces:
    build_vai/freeze/frozen_graph.pb   ← input to vai_q_tensorflow
    build_vai/freeze/node_names.txt    ← input/output node names for quantise step
"""

import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',      required=True,
                    help='Trained FP32 model (.h5 or .keras)')
    ap.add_argument('--output_dir', default='build_vai/freeze',
                    help='Directory for frozen_graph.pb (default: build_vai/freeze)')
    ap.add_argument('--input_height', type=int, default=224)
    ap.add_argument('--input_width',  type=int, default=224)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    n_channels = model.input_shape[-1]   # 3 for RGB
    n_classes  = model.output_shape[-1]  # 25 for RUGD

    # ── Trace to concrete function ─────────────────────────────────────────
    # Use training=False so BN uses inference statistics (frozen mean/var)
    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, args.input_height, args.input_width, n_channels],
            dtype=tf.float32,
            name='input_1',           # name the input node explicitly
        )
    ])
    def serving_fn(input_1):
        return model(input_1, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    print(f"\nConcrete function inputs:  {[t.name for t in concrete_fn.inputs]}")
    print(f"Concrete function outputs: {[t.name for t in concrete_fn.outputs]}")

    # ── Freeze (fold variables into constants) ─────────────────────────────
    print("\nFreezing graph…")
    frozen_fn     = convert_variables_to_constants_v2(concrete_fn)
    frozen_graph  = frozen_fn.graph.as_graph_def()

    # ── Strip TF2-only artefacts unsupported by TF1.15 / vai_c_tensorflow ──
    # 1. TF2 MaxPool carries 'explicit_paddings' which TF1.15 doesn't know.
    # 2. TF2 adds NoOp nodes that vai_c_tensorflow can't handle.
    from tensorflow.core.framework.graph_pb2 import GraphDef
    clean_graph = GraphDef()
    
    removed_mp = 0
    noop_names = {node.name for node in frozen_graph.node if node.op == 'NoOp'}
    
    for node in frozen_graph.node:
        if node.name in noop_names:
            continue
            
        new_node = clean_graph.node.add()
        new_node.CopyFrom(node)
        
        if new_node.op == 'MaxPool' and 'explicit_paddings' in new_node.attr:
            del new_node.attr['explicit_paddings']
            removed_mp += 1
            
        # Clean control dependencies pointing to removed NoOp nodes
        new_inputs = []
        for inp in new_node.input:
            if inp.startswith('^'):
                dep_name = inp[1:]
                if dep_name in noop_names:
                    continue
            new_inputs.append(inp)
            
        del new_node.input[:]
        new_node.input.extend(new_inputs)
        
    frozen_graph = clean_graph
    removed_noop = len(noop_names)

    if removed_mp:
        print(f"  Stripped 'explicit_paddings' from {removed_mp} MaxPool node(s)")
    if removed_noop:
        print(f"  Removed {removed_noop} NoOp node(s)")

    # ── Report node names ──────────────────────────────────────────────────
    # Input node: the tensor named 'input_1' (strip ':0' suffix)
    input_nodes  = [t.name.split(':')[0] for t in frozen_fn.inputs
                    if not t.name.startswith('unknown')]
    output_nodes = [t.name.split(':')[0] for t in frozen_fn.outputs]

    # Fallback: find ReLU output nodes by inspecting graph ops
    if not output_nodes:
        output_nodes = [
            op.name for op in frozen_graph.node
            if op.op == 'Relu' or op.op == 'Identity'
        ][-1:]

    print(f"\nInput  node(s) : {input_nodes}")
    print(f"Output node(s) : {output_nodes}")

    # ── Save frozen graph ──────────────────────────────────────────────────
    pb_path = os.path.join(args.output_dir, 'frozen_graph.pb')
    tf.io.write_graph(frozen_graph, args.output_dir,
                      'frozen_graph.pb', as_text=False)
    print(f"\nFrozen graph saved → {pb_path}")

    # ── Save node names for the quantise step ──────────────────────────────
    names_path = os.path.join(args.output_dir, 'node_names.txt')
    with open(names_path, 'w') as f:
        f.write(f"INPUT_NODE={input_nodes[0] if input_nodes else 'input_1'}\n")
        f.write(f"OUTPUT_NODE={output_nodes[0] if output_nodes else ''}\n")
        f.write(f"N_CHANNELS={n_channels}\n")
        f.write(f"N_CLASSES={n_classes}\n")
        f.write(f"IMG_HEIGHT={args.input_height}\n")
        f.write(f"IMG_WIDTH={args.input_width}\n")
    print(f"Node names saved → {names_path}")

    # ── Inspect with vai_q_tensorflow if available ────────────────────────
    print("\nTo inspect with Vitis AI (inside Docker):")
    print(f"  vai_q_tensorflow inspect --input_frozen_graph {pb_path}")
    print("\nAll done.")


if __name__ == '__main__':
    main()
