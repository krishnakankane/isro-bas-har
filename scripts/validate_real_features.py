"""Validate and report statistics for real extracted HAR feature archive."""

import json
from pathlib import Path
import numpy as np

def validate_features():
    npz_path = Path("data/temporal_har/features/SES_001_NOMINAL_windows.npz")
    print(f"File exists: {npz_path.exists()} ({npz_path})")
    assert npz_path.exists(), "NPZ archive does not exist!"

    data = np.load(npz_path)
    print(f"Keys in NPZ archive: {list(data.keys())}")

    windows = data["windows"]
    labels = data["labels"]
    ranges = data["ranges"]
    actor_id = str(data["actor_id"])
    session_id = str(data["session_id"])
    video_id = str(data["video_id"])
    feature_dim = int(data["feature_dim"])
    window_size = int(data["window_size"])
    rejected_windows = int(data["rejected_windows"])
    total_frames = int(data["total_source_frames"])

    print("\n--- 1. TENSOR SHAPE & DTYPE VALIDATION ---")
    print(f"Windows Shape : {windows.shape}")
    print(f"Windows dtype : {windows.dtype}")
    print(f"Labels Shape  : {labels.shape}")
    print(f"Labels dtype  : {labels.dtype}")
    print(f"Ranges Shape  : {ranges.shape}")
    print(f"Ranges dtype  : {ranges.dtype}")
    print(f"Actor ID      : {actor_id}")
    print(f"Session ID    : {session_id}")
    print(f"Video ID      : {video_id}")
    print(f"Window Length : {window_size}")
    print(f"Feature Dim   : {feature_dim}")
    print(f"Total Frames  : {total_frames}")
    print(f"Rejected Wins : {rejected_windows}")

    assert windows.shape == (212, 60, 154), f"Unexpected shape {windows.shape}"
    assert windows.dtype == np.float32, f"Unexpected dtype {windows.dtype}"
    assert labels.shape == (212,), f"Unexpected label shape {labels.shape}"
    assert labels.dtype == np.int64, f"Unexpected label dtype {labels.dtype}"
    assert len(windows) == len(labels) == len(ranges), "Sample count mismatch!"

    print("\n--- 2. NUMERICAL INTEGRITY & FINITENESS CHECKS ---")
    nan_count = int(np.isnan(windows).sum())
    inf_count = int(np.isinf(windows).sum())
    is_finite = bool(np.all(np.isfinite(windows)))
    print(f"NaN count     : {nan_count}")
    print(f"Inf count     : {inf_count}")
    print(f"All values finite: {is_finite}")
    assert is_finite, "Found non-finite values!"

    print("\n--- 3. PER-CLASS SAMPLE DISTRIBUTION ---")
    action_names = {
        0: "ACTION_IDLE",
        1: "ACTION_SANITIZE",
        2: "ACTION_HOLD_BOTTLE",
        3: "ACTION_PLACE_BOTTLE",
        4: "ACTION_HOLD_BOX",
        5: "ACTION_OPEN_BOX",
        6: "ACTION_PICK_OBJECT",
        7: "ACTION_TRANSFER_OBJECT",
        8: "ACTION_RETURN_OBJECT",
        9: "ACTION_CLOSE_BOX",
    }
    unique_classes, counts = np.unique(labels, return_counts=True)
    for c_id in range(10):
        c_count = int(np.sum(labels == c_id))
        pct = (c_count / len(labels)) * 100
        print(f"  Class {c_id:02d} [{action_names[c_id]}]: {c_count:3d} windows ({pct:5.1f}%)")

    # Check all 10 classes are represented
    assert len(unique_classes) == 10, f"Expected 10 classes, got {len(unique_classes)}"

    print("\n--- 4. FEATURE SIGNAL VARIATION QUALITY CHECKS ---")
    print(f"Global min: {windows.min():.4f}, max: {windows.max():.4f}, mean: {windows.mean():.4f}, std: {windows.std():.4f}")
    
    pose_detected_mean = float(windows[:, :, 138].mean())
    left_hand_mean = float(windows[:, :, 139].mean())
    right_hand_mean = float(windows[:, :, 140].mean())
    left_grasp_mean = float(windows[:, :, 136].mean())
    right_grasp_mean = float(windows[:, :, 137].mean())
    torso_angle_min = float(windows[:, :, 141].min())
    torso_angle_max = float(windows[:, :, 141].max())
    obj_conf_mean = float(windows[:, :, 142].mean())
    obj_conf_max = float(windows[:, :, 142].max())
    left_wrist_dist_min = float(windows[:, :, 146].min())
    right_wrist_dist_min = float(windows[:, :, 147].min())

    print(f"Pose detected flag (slot 138) mean : {pose_detected_mean:.3f} (present in {pose_detected_mean*100:.1f}% of frame instances)")
    print(f"Left hand detected (slot 139) mean : {left_hand_mean:.3f} ({left_hand_mean*100:.1f}%)")
    print(f"Right hand detected (slot 140) mean: {right_hand_mean:.3f} ({right_hand_mean*100:.1f}%)")
    print(f"Left hand grasping (slot 136) mean : {left_grasp_mean:.3f}")
    print(f"Right hand grasping (slot 137) mean: {right_grasp_mean:.3f}")
    print(f"Torso angle range (slot 141)       : [{torso_angle_min:.3f}, {torso_angle_max:.3f}]")
    print(f"Object detection conf (slot 142)   : mean={obj_conf_mean:.3f}, max={obj_conf_max:.3f}")
    print(f"Wrist-object distance (146, 147)   : L_min={left_wrist_dist_min:.3f}, R_min={right_wrist_dist_min:.3f}")
    
    hoi_slot_sums = windows[:, :, 148:154].sum(axis=(0, 1))
    hoi_names = ["NONE", "UNKNOWN", "APPROACHING", "GRASPING", "MANIPULATING", "DWELL/RELEASED"]
    print("HOI State Flag Occurrences:")
    for h_name, h_sum in zip(hoi_names, hoi_slot_sums):
        print(f"  {h_name:15s}: {h_sum:8.1f}")

    per_dim_std = windows.std(axis=(0, 1))
    active_dims = int(np.count_nonzero(per_dim_std > 0.0))
    print(f"\nActive Varying Feature Dimensions   : {active_dims} / {feature_dim} ({active_dims/feature_dim*100:.1f}%)")
    
    print("\n--- 5. FRAME RANGE CONTINUITY & MONOTONICITY ---")
    for i in range(min(5, len(ranges))):
        print(f"  Window {i:03d}: frames {ranges[i, 0]:04d} -> {ranges[i, 1]:04d} | label = {labels[i]} ({action_names[labels[i]]})")
    print("  ...")
    for i in range(max(0, len(ranges)-5), len(ranges)):
        print(f"  Window {i:03d}: frames {ranges[i, 0]:04d} -> {ranges[i, 1]:04d} | label = {labels[i]} ({action_names[labels[i]]})")

    assert np.all(ranges[:, 1] - ranges[:, 0] == 60), "Window length mismatch in ranges!"
    assert np.all(np.diff(ranges[:, 0]) >= 10), "Window stride inconsistency!"
    print("\n[ALL VALIDATION CHECKS PASSED SUCCESSFULLY]")

if __name__ == "__main__":
    validate_features()
