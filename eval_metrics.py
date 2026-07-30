#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the trained RGB-D UNet on the RELLIS-3D test split.

Scores the 10 active kinematic classes and ignores Void, reporting IoU, precision,
recall and F1 per class plus the means, as JSON and Markdown.

    python eval_metrics.py --model output/keras_model/ep50_final_unet_v2_224x224.h5
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from dataset import KINEMATIC_CLASSES, NUM_CLASSES, RELLIS3DDataset, TOTAL_CLASSES

VOID_CLASS = TOTAL_CLASSES - 1


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate the RGB-D UNet on RELLIS-3D")
    ap.add_argument("--model", required=True, help="Trained model (.keras or .h5)")
    ap.add_argument("--data_root", default="data/RELLIS-3D_full", help="Dataset root")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit samples for a quick check")
    ap.add_argument("--output_json", default="output/rpt/test_metrics.json")
    ap.add_argument("--output_md", default="output/rpt/test_metrics.md")
    return ap.parse_args()


def predict_split(model, dataset):
    """Return (ground truth, predictions) as flat class-index arrays."""
    targets, preds = [], []
    for idx in range(len(dataset)):
        X, Y = dataset[idx]
        preds.append(np.argmax(model.predict(X, verbose=0), axis=-1))
        targets.append(np.argmax(Y, axis=-1))
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataset):
            print(f"  processed batch {idx + 1}/{len(dataset)}")
    return np.concatenate(targets), np.concatenate(preds)


def compute_metrics(y_true, y_pred):
    """Per-class IoU / precision / recall / F1 over the active classes, ignoring Void."""
    labelled = y_true != VOID_CLASS
    y_true = y_true[labelled]
    y_pred = y_pred[labelled]

    per_class = {}
    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        def ratio(numerator, denominator):
            return float(numerator / denominator) if denominator else float('nan')

        per_class[KINEMATIC_CLASSES[c]] = {
            "TP": tp, "FP": fp, "FN": fn,
            "IoU": ratio(tp, tp + fp + fn),
            "Precision": ratio(tp, tp + fp),
            "Recall": ratio(tp, tp + fn),
            "F1-Score": ratio(2 * tp, 2 * tp + fp + fn),
        }

    def mean_of(key):
        values = [m[key] for m in per_class.values() if not np.isnan(m[key])]
        return float(np.mean(values)) if values else 0.0

    return {
        "overall_pixel_accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "mean_iou": mean_of("IoU"),
        "mean_precision": mean_of("Precision"),
        "mean_recall": mean_of("Recall"),
        "mean_f1_score": mean_of("F1-Score"),
        "total_pixels": int(y_true.size),
        "class_details": per_class,
    }


def print_report(summary):
    print("\nper-class metrics (Void ignored):")
    print(f"  {'class':<20} {'IoU':>7} {'prec':>7} {'recall':>7} {'F1':>7}")
    for name, m in summary["class_details"].items():
        print(f"  {name:<20} {m['IoU']:7.4f} {m['Precision']:7.4f} "
              f"{m['Recall']:7.4f} {m['F1-Score']:7.4f}")
    print(f"  {'mean':<20} {summary['mean_iou']:7.4f} {summary['mean_precision']:7.4f} "
          f"{summary['mean_recall']:7.4f} {summary['mean_f1_score']:7.4f}")
    print(f"\noverall pixel accuracy {summary['overall_pixel_accuracy']:.4f}")
    print(f"evaluated pixels       {summary['total_pixels']:,}")


def save_reports(summary, model_path, json_path, md_path):
    for path in (json_path, md_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"JSON report: {json_path}")

    with open(md_path, "w") as f:
        f.write("# TerraSense RGB-D Segmentation Report\n\n")
        f.write(f"- **Model**: `{model_path}`\n")
        f.write("- **Dataset**: RELLIS-3D test split\n")
        f.write(f"- **Date**: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## Summary\n\n| Metric | Value |\n| :--- | :--- |\n")
        for label, key in (("Overall Pixel Accuracy", "overall_pixel_accuracy"),
                           ("Mean IoU", "mean_iou"),
                           ("Mean Precision", "mean_precision"),
                           ("Mean Recall", "mean_recall"),
                           ("Mean F1-Score", "mean_f1_score")):
            f.write(f"| **{label}** | `{summary[key]:.4f}` |\n")
        f.write(f"| **Evaluated Pixels** | `{summary['total_pixels']:,}` |\n\n")

        f.write("## Per-Class\n\n")
        f.write("| Class | TP | FP | FN | IoU | Precision | Recall | F1 |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for name, m in summary["class_details"].items():
            f.write(f"| **{name}** | {m['TP']:,} | {m['FP']:,} | {m['FN']:,} | "
                    f"`{m['IoU']:.4f}` | `{m['Precision']:.4f}` | "
                    f"`{m['Recall']:.4f}` | `{m['F1-Score']:.4f}` |\n")
    print(f"Markdown report: {md_path}")


def main():
    args = parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = RELLIS3DDataset(args.data_root, split="test", batch_size=args.batch_size,
                              max_samples=args.max_samples)

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    summary = compute_metrics(*predict_split(model, dataset))
    print_report(summary)
    save_reports(summary, args.model, args.output_json, args.output_md)


if __name__ == "__main__":
    main()
