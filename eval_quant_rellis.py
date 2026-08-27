#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Vitis AI INT8 Quantized Frozen Graph (.pb) on RELLIS-3D dataset.
Runs inside the Vitis AI Docker container with vai_q_tensorflow.
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import vai_q_tensorflow  # Registers FixNeuron custom ops

from dataset import RELLIS3DDataset, KINEMATIC_CLASSES, NUM_CLASSES, TOTAL_CLASSES

VOID_CLASS = TOTAL_CLASSES - 1


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate INT8 Quantized Model on RELLIS-3D")
    ap.add_argument("--pb_file", default="build_vai/quantized_rgbd_4ch/quantize_eval_model.pb",
                    help="Path to quantized .pb graph")
    ap.add_argument("--data_root", default="data/RELLIS-3D_full", help="RELLIS-3D root")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_json", default="build_vai/quant_rellis_results.json")
    return ap.parse_args()


def compute_metrics(y_true, y_pred):
    labelled = y_true != VOID_CLASS
    y_true = y_true[labelled]
    y_pred = y_pred[labelled]

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        def ratio(num, den):
            return float(num / den) if den else float('nan')

        per_class[KINEMATIC_CLASSES[c]] = {
            "TP": tp, "FP": fp, "FN": fn,
            "IoU": ratio(tp, tp + fp + fn),
            "Precision": ratio(tp, tp + fp),
            "Recall": ratio(tp, tp + fn),
            "F1-Score": ratio(2 * tp, 2 * tp + fp + fn),
        }

    def mean_of(key):
        vals = [m[key] for m in per_class.values() if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "overall_pixel_accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "mean_iou": mean_of("IoU"),
        "mean_precision": mean_of("Precision"),
        "mean_recall": mean_of("Recall"),
        "mean_f1_score": mean_of("F1-Score"),
        "total_pixels": int(y_true.size),
        "class_details": per_class,
    }


def main():
    args = parse_args()

    print(f"Loading quantized graph: {args.pb_file}")
    with tf.io.gfile.GFile(args.pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    input_tensor = graph.get_tensor_by_name("input_1:0")
    output_tensor = graph.get_tensor_by_name("Identity:0")
    print(f"Graph loaded: {input_tensor.name} -> {output_tensor.name}")

    dataset = RELLIS3DDataset(args.data_root, split="test", batch_size=args.batch_size, max_samples=args.max_samples)
    print(f"Test split samples: {len(dataset) * args.batch_size}")

    targets_list, preds_list = [], []
    with tf.compat.v1.Session(graph=graph) as sess:
        for idx in range(len(dataset)):
            X, Y = dataset[idx]
            preds = np.argmax(sess.run(output_tensor, feed_dict={input_tensor: X}), axis=-1)
            targets = np.argmax(Y, axis=-1)
            targets_list.append(targets)
            preds_list.append(preds)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataset):
                print(f"  processed batch {idx + 1}/{len(dataset)}", flush=True)

    y_true = np.concatenate(targets_list)
    y_pred = np.concatenate(preds_list)

    summary = compute_metrics(y_true, y_pred)
    print("\n========== INT8 CPU BENCHMARK RESULTS (RELLIS-3D) ==========")
    print(f"{'Class Name':<20} {'TP':>12} {'FP':>12} {'FN':>12} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
    for name, m in summary["class_details"].items():
        print(f"{name:<20} {m['TP']:12,d} {m['FP']:12,d} {m['FN']:12,d} {m['IoU']:8.4f} {m['Precision']:10.4f} {m['Recall']:8.4f} {m['F1-Score']:9.4f}")

    print("-" * 85)
    print(f"{'Mean':<20} {'-':>12} {'-':>12} {'-':>12} {summary['mean_iou']:8.4f} {summary['mean_precision']:10.4f} {summary['mean_recall']:8.4f} {summary['mean_f1_score']:9.4f}")
    print(f"Overall Pixel Accuracy: {summary['overall_pixel_accuracy']:.4f}")

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
