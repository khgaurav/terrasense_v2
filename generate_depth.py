#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to project RELLIS-3D Ouster LiDAR point clouds (.bin files) 
into Basler camera coordinates using extrinsic and intrinsic calibration 
parameters to generate sparse depth maps (.png files).
"""

import os
import sys
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def load_calibration(data_root, seq):
    """Load calibration matrices for a specific sequence."""
    trans_yaml = os.path.join(data_root, "calibration", "extracted", "Rellis_3D", seq, "transforms.yaml")
    cam_info_txt = os.path.join(data_root, "calibration", "extracted", "Rellis-3D", seq, "camera_info.txt")
    
    if not os.path.exists(trans_yaml) or not os.path.exists(cam_info_txt):
        return None, None, None, None
        
    # Load extrinsics
    with open(trans_yaml, "r") as f:
        calib = yaml.safe_load(f)
    trans = calib["os1_cloud_node-pylon_camera_node"]
    q = trans["q"]
    t = trans["t"]
    
    rot = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    translation = np.array([t["x"], t["y"], t["z"]])
    
    # Invert to transform from LiDAR to Camera coordinate system
    RT = np.eye(4)
    RT[:3, :3] = rot
    RT[:3, 3] = translation
    RT_inv = np.linalg.inv(RT)
    
    R_vc = RT_inv[:3, :3]
    T_vc = RT_inv[:3, 3].reshape(3, 1)
    
    # Load intrinsics
    with open(cam_info_txt, "r") as f:
        line = f.readline().strip()
    fx, fy, cx, cy = map(float, line.split())
    
    P = np.zeros((3, 3))
    P[0, 0] = fx
    P[1, 1] = fy
    P[2, 2] = 1.0
    P[0, 2] = cx
    P[1, 2] = cy
    
    # Distortion coefficients (default Basler coefficients used in RELLIS-3D)
    dist_coeff = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0]).reshape((5, 1))
    
    return R_vc, T_vc, P, dist_coeff

def main():
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(workspace_root, "data")
    dataset_root = os.path.join(data_root, "RELLIS-3D_full")
    
    depth_dir = os.path.join(dataset_root, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    split_dir = os.path.join(dataset_root, "Rellis_3D_image_split")
    
    # Pre-cache calibration parameters for all 5 sequences
    calibrations = {}
    for seq_id in ["00000", "00001", "00002", "00003", "00004"]:
        calibrations[seq_id] = load_calibration(data_root, seq_id)
        
    print("Starting depth map generation from LiDAR point clouds...")
    total_processed = 0
    total_missing_bin = 0
    
    for split in splits:
        lst_file = os.path.join(split_dir, f"{split}.lst")
        if not os.path.exists(lst_file):
            print(f"Split file not found: {lst_file}, skipping.")
            continue
            
        print(f"Processing split: {split}")
        with open(lst_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                # Example path: 00000/pylon_camera_node/frame000000-1581624652_750.jpg
                rgb_relative = parts[0]
                seq = rgb_relative.split("/")[0]
                rgb_base = os.path.basename(rgb_relative)
                base_name = os.path.splitext(rgb_base)[0]
                
                # Parse frame index (e.g. frame000000 -> 000000)
                frame_idx_str = base_name.split("-")[0].replace("frame", "")
                
                # Locate raw LiDAR file (.bin)
                bin_path = os.path.join(dataset_root, "os1_cloud_node_kitti_bin", seq, f"{frame_idx_str}.bin")
                
                if not os.path.exists(bin_path):
                    total_missing_bin += 1
                    continue
                    
                R_vc, T_vc, P, dist_coeff = calibrations[seq]
                if R_vc is None:
                    print(f"Missing calibration files for sequence {seq}, skipping.")
                    continue
                    
                # 1. Load point cloud points
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                xyz = points[:, :3]
                
                # 2. Transform points to camera frame to calculate depth (Z coordinate)
                pts_cam = (R_vc @ xyz.T).T + T_vc.T[0]
                z_c = pts_cam[:, 2]
                
                # Keep only points in front of the camera
                mask = z_c > 0.0
                xyz_filtered = xyz[mask]
                z_c_filtered = z_c[mask]
                
                # 3. Project to pixel coordinates applying distortion coefficients
                rvec, _ = cv2.Rodrigues(R_vc)
                tvec = T_vc
                imgpoints, _ = cv2.projectPoints(xyz_filtered, rvec, tvec, P, dist_coeff)
                imgpoints = np.squeeze(imgpoints, 1) # Shape: (N, 2)
                
                u = imgpoints[:, 0]
                v = imgpoints[:, 1]
                
                # 4. Filter pixels within the camera frame bounds (1200x1920)
                h, w = 1200, 1920
                valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u_idx = u[valid_mask].astype(np.int32)
                v_idx = v[valid_mask].astype(np.int32)
                depth_vals = z_c_filtered[valid_mask]
                
                # 5. Create the depth map using z-buffering (minimum depth value)
                proj_depth = np.zeros((h, w), dtype=np.float32)
                min_depth = np.full((h, w), np.inf, dtype=np.float32)
                
                for uu, vv, d in zip(u_idx, v_idx, depth_vals):
                    if d < min_depth[vv, uu]:
                        min_depth[vv, uu] = d
                        proj_depth[vv, uu] = d
                        
                # Convert depth in meters to millimeters and cast to uint16
                proj_depth_mm = (proj_depth * 1000.0).astype(np.uint16)
                
                # 6. Save depth map as a 16-bit PNG image
                out_path = os.path.join(depth_dir, f"{base_name}.png")
                cv2.imwrite(out_path, proj_depth_mm)
                
                total_processed += 1
                if total_processed % 500 == 0:
                    print(f"Generated {total_processed} depth maps...")
                    
    print(f"\nSuccessfully generated {total_processed} depth maps.")
    if total_missing_bin > 0:
        print(f"Warning: {total_missing_bin} LiDAR .bin files were missing.")

if __name__ == "__main__":
    main()
