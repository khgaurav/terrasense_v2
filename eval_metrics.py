#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for TerraSense UNet on the RELLIS-3D dataset.
Computes comprehensive segmentation metrics (Accuracy, IoU, Precision, Recall, F1-score)
for each active class and mean metrics.

Supports Keras models (.keras, .h5) and TF1-compatible frozen graphs (.pb).
"""

import sys
import argparse
import os

# Parse args before loading TensorFlow so we can disable v2 behavior if needed
def get_args():
    ap = argparse.ArgumentParser(description="Evaluate segmentation models on RELLIS-3D test set")
    ap.add_argument("--model", default="output/keras_model/ep50_final_unet_v2_224x224.keras",
                    help="Path to the model file (.keras, .h5, or .pb)")
    ap.add_argument("--model_type", default="keras", choices=["keras", "pb"],
                    help="Model type: 'keras' for TF2 Keras, 'pb' for TF1 frozen graph")
    ap.add_argument("--data_root", default="data/RELLIS-3D_full",
                    help="Path to the RELLIS-3D dataset root")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for evaluation")
    ap.add_argument("--output_json", default="output/rpt/test_metrics.json",
                    help="Path to save JSON metrics report")
    ap.add_argument("--output_md", default="output/rpt/test_metrics.md",
                    help="Path to save Markdown report")
    return ap.parse_args()

args = get_args()

# Force CPU execution to prevent CUDA driver mismatch / OOM / initialization issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2

if args.model_type == "pb":
    print("Initializing TF1 Compatibility Mode for Frozen Graph...")
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    try:
        import tensorflow.contrib.decent_q
    except ImportError:
        pass
else:
    print("Initializing TF2 Mode for Keras model...")
    import tensorflow as tf

from dataset import RELLIS3DDataset, NUM_CLASSES, TOTAL_CLASSES, KINEMATIC_CLASSES

def evaluate_keras(model_path, test_gen):
    print(f"Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Running inference on test set...")
    
    y_true_list = []
    y_pred_list = []
    
    n_batches = len(test_gen)
    for i in range(n_batches):
        X_batch, Y_batch = test_gen[i]
        pred_batch = model.predict(X_batch, verbose=0)
        
        y_true_list.append(np.argmax(Y_batch, axis=-1))
        y_pred_list.append(np.argmax(pred_batch, axis=-1))
        
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  Processed batch {i + 1}/{n_batches}")
            
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)

def evaluate_pb(pb_path, test_gen):
    print(f"Loading Frozen Graph: {pb_path}")
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
    # Standard Vitis-AI node names for early fusion UNet
    INPUT_NODE = "input_1:0"
    OUTPUT_NODE = "Identity:0"
    
    try:
        input_tensor = graph.get_tensor_by_name(INPUT_NODE)
        output_tensor = graph.get_tensor_by_name(OUTPUT_NODE)
    except KeyError as e:
        print(f"Error: Tensor name not found in graph. Inspecting nodes...")
        # Fallback to listing operations
        ops = graph.get_operations()
        print("First 10 operations:", [op.name for op in ops[:10]])
        print("Last 10 operations:", [op.name for op in ops[-10:]])
        raise e
        
    y_true_list = []
    y_pred_list = []
    
    n_batches = len(test_gen)
    with tf.Session(graph=graph) as sess:
        print("Running inference on test set...")
        for i in range(n_batches):
            X_batch, Y_batch = test_gen[i]
            pred_batch = sess.run(output_tensor, feed_dict={input_tensor: X_batch})
            
            y_true_list.append(np.argmax(Y_batch, axis=-1))
            y_pred_list.append(np.argmax(pred_batch, axis=-1))
            
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  Processed batch {i + 1}/{n_batches}")
                
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)

def compute_metrics(y_true, y_pred, num_classes=6, void_class=6):
    """
    Compute Accuracy, IoU, Precision, Recall, F1-score.
    Ignores the void class.
    """
    # Create mask for pixels that are NOT void in the ground truth
    valid_mask = (y_true != void_class)
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Overall Pixel Accuracy
    pixel_acc = np.mean(y_true_valid == y_pred_valid) if len(y_true_valid) > 0 else 0.0
    
    class_metrics = {}
    total_tp = 0
    total_pixels = len(y_true_valid)
    
    print("\nPer-Class Metrics (ignoring Void class):")
    print(f"{'Class Name':<25} | {'IoU':<6} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8}")
    print("-" * 65)
    
    ious = []
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        class_name = KINEMATIC_CLASSES[c]
        
        tp = np.sum((y_true_valid == c) & (y_pred_valid == c))
        fp = np.sum((y_true_valid != c) & (y_pred_valid == c))
        fn = np.sum((y_true_valid == c) & (y_pred_valid != c))
        
        total_tp += tp
        
        # Intersection over Union
        union = tp + fp + fn
        iou = tp / float(union) if union > 0 else float('nan')
        
        # Precision
        prec = tp / float(tp + fp) if (tp + fp) > 0 else float('nan')
        
        # Recall / Sensitivity
        rec = tp / float(tp + fn) if (tp + fn) > 0 else float('nan')
        
        # F1-Score / Dice Coefficient
        f1 = 2 * tp / float(2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
        
        class_metrics[class_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "IoU": float(iou),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1)
        }
        
        if not np.isnan(iou): ious.append(iou)
        if not np.isnan(prec): precisions.append(prec)
        if not np.isnan(rec): recalls.append(rec)
        if not np.isnan(f1): f1s.append(f1)
        
        print(f"{class_name:<25} | {iou:.4f} | {prec:.4f}    | {rec:.4f} | {f1:.4f}")
        
    mIoU = np.mean(ious) if ious else 0.0
    mPrecision = np.mean(precisions) if precisions else 0.0
    mRecall = np.mean(recalls) if recalls else 0.0
    mF1 = np.mean(f1s) if f1s else 0.0
    
    print("-" * 65)
    print(f"{'Mean / Overall':<25} | {mIoU:.4f} | {mPrecision:.4f}    | {mRecall:.4f} | {mF1:.4f}")
    print(f"Overall Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Total Evaluated Pixels: {total_pixels:,}")
    
    summary = {
        "overall_pixel_accuracy": float(pixel_acc),
        "mean_iou": float(mIoU),
        "mean_precision": float(mPrecision),
        "mean_recall": float(mRecall),
        "mean_f1_score": float(mF1),
        "total_pixels": int(total_pixels),
        "class_details": class_metrics
    }
    
    return summary

def save_reports(summary, model_path, model_type):
    # Save JSON report
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    import json
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"JSON metrics report saved to: {args.output_json}")
    
    # Save Markdown report
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, 'w') as f:
        f.write(f"# TerraSense Semantic Segmentation Report\n\n")
        f.write(f"- **Model Path**: `{model_path}`\n")
        f.write(f"- **Model Type**: `{model_type.upper()}`\n")
        f.write(f"- **Dataset**: RELLIS-3D Test Set\n")
        f.write(f"- **Date**: {os.popen('date').read().strip()}\n\n")
        
        f.write(f"## Summary Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Overall Pixel Accuracy** | `{summary['overall_pixel_accuracy']:.4f}` ({summary['overall_pixel_accuracy']*100:.2f}%) |\n")
        f.write(f"| **Mean IoU (mIoU)** | `{summary['mean_iou']:.4f}` ({summary['mean_iou']*100:.2f}%) |\n")
        f.write(f"| **Mean Precision** | `{summary['mean_precision']:.4f}` |\n")
        f.write(f"| **Mean Recall** | `{summary['mean_recall']:.4f}` |\n")
        f.write(f"| **Mean F1-Score (Dice)** | `{summary['mean_f1_score']:.4f}` |\n")
        f.write(f"| **Total Valid Pixels** | `{summary['total_pixels']:,}` |\n\n")
        
        f.write(f"## Per-Class Metrics\n\n")
        f.write(f"| Class Name | TP (Pixels) | FP (Pixels) | FN (Pixels) | IoU | Precision | Recall | F1-Score |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for class_name, metrics in summary['class_details'].items():
            f.write(f"| **{class_name}** | {metrics['TP']:,} | {metrics['FP']:,} | {metrics['FN']:,} | `{metrics['IoU']:.4f}` | `{metrics['Precision']:.4f}` | `{metrics['Recall']:.4f}` | `{metrics['F1-Score']:.4f}` |\n")
            
        f.write(f"| **Mean / Overall** | - | - | - | **`{summary['mean_iou']:.4f}`** | **`{summary['mean_precision']:.4f}`** | **`{summary['mean_recall']:.4f}`** | **`{summary['mean_f1_score']:.4f}`** |\n")
        
    print(f"Markdown report saved to: {args.output_md}")

def main():
    print(f"Evaluating {args.model_type.upper()} model from {args.model}...")
    
    # Load dataset generator
    test_gen = RELLIS3DDataset(args.data_root, split="test", batch_size=args.batch_size, augment=False)
    
    if args.model_type == "pb":
        y_true, y_pred = evaluate_pb(args.model, test_gen)
    else:
        y_true, y_pred = evaluate_keras(args.model, test_gen)
        
    summary = compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES, void_class=TOTAL_CLASSES-1)
    save_reports(summary, args.model, args.model_type)

if __name__ == "__main__":
    main()
