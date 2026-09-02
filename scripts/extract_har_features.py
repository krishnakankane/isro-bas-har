"""ISRO BAS HAR — Batch Video to Temporal Feature Tensor Extractor.

Converts annotated experiment videos into structured (N × 60 × 154) temporal feature
sliding windows using the existing Perception + HOI + Feature Extraction pipeline.

Preserves full metadata lineage: actor ID, session ID, video ID, action label, and frame range.

DOES NOT TRAIN ANY MODELS — PREPARES PRE-EXTRACTED TENSORS FOR HAR TRAINING.

Usage
-----
# Extract features from all annotated manifests:
    python -m scripts.extract_har_features --all

# Extract features from a single manifest:
    python -m scripts.extract_har_features --manifest data/temporal_har/manifests/session01_actions.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import FramePerceptionResult
from src.perception.object_detector import ObjectDetector
from src.perception.pose_estimator import PoseEstimator
from src.perception.hand_tracker import HandTracker
from src.perception.hoi_detector import HOIDetector, HOIConfig
from src.activity_engine.feature_extractor import PerceptionFeatureExtractor, FEATURE_DIM
from src.activity_engine.temporal_buffer import TemporalFeatureBuffer


def process_video_to_features(
    video_path: Path,
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    window_size: int = 60,
    stride: int = 10,
    detector: Optional[ObjectDetector] = None,
    pose_estimator: Optional[PoseEstimator] = None,
    hand_tracker: Optional[HandTracker] = None,
    hoi_detector: Optional[HOIDetector] = None,
    extractor: Optional[PerceptionFeatureExtractor] = None,
) -> Dict[str, object]:
    """Process an annotated video into sliding window feature tensors.

    Args:
        video_path: Path to video file.
        manifest_path: Path to action interval JSON manifest.
        output_dir: Folder to write .npz output (default: data/temporal_har/features/).
        window_size: Frames per temporal window (default: 60).
        stride: Temporal window step stride (default: 10 frames).
        ... (optional pre-instantiated model instances for batch efficiency).

    Returns:
        Summary dict containing tensor shapes, window counts, and output file path.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = video_path.stem
    out_dir = output_dir or (_PROJECT_ROOT / "data" / "temporal_har" / "features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz_path = out_dir / f"{video_stem}_windows.npz"

    # Read manifest metadata if available
    manifest_data = {}
    segments: List[Dict] = []
    actor_id = "ACTOR_01"
    session_id = video_stem

    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            segments = manifest_data.get("segments", [])
            actor_id = manifest_data.get("actor_id", actor_id)
            session_id = manifest_data.get("session_id", session_id)

    # Initialise perception modules if not supplied
    yolo_dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = detector or ObjectDetector(model_name_or_path="yolov8n.pt", device=yolo_dev)
    pose_est = pose_estimator or PoseEstimator(model_path="models/mediapipe/pose_landmarker_lite.task")
    hand_trk = hand_tracker or HandTracker(model_path="models/mediapipe/hand_landmarker.task")
    hoi_det = hoi_detector or HOIDetector(config=HOIConfig())
    feat_ext = extractor or PerceptionFeatureExtractor()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[FeatureExtractor] Processing: {video_path.name} ({total_frames} frames @ {fps:.1f} FPS)")

    start_wall_time = time.time()
    per_frame_vectors: List[np.ndarray] = []
    per_frame_labels: List[int] = []
    frame_idx = 0

    hoi_det.reset()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 1. Perception
        objects = det.detect(frame)
        pose_res = pose_est.estimate(frame)
        lh, rh = hand_trk.track(frame)
        interactions = hoi_det.analyze_interaction(objects, lh, rh)

        # 2. Extract 154-d vector
        fr_result = FramePerceptionResult(
            frame_id=frame_idx,
            timestamp_sec=frame_idx / fps,
            objects=objects,
            body_pose=pose_res,
            left_hand=lh,
            right_hand=rh,
            interactions=interactions,
        )
        fv = feat_ext.extract(fr_result)
        per_frame_vectors.append(fv.data.copy())

        # 3. Label matching from manifest segments
        # If segments are present, frames outside annotated intervals are marked -1 (UNLABELLED_GAP)
        matched_action_id = -1 if segments else 0
        for seg in segments:
            if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                matched_action_id = seg["action_id"]
                break
        per_frame_labels.append(matched_action_id)

        frame_idx += 1
        if frame_idx % 200 == 0 or frame_idx == total_frames:
            print(f"[FeatureExtractor] Frame {frame_idx:05d}/{total_frames:05d} ({frame_idx/total_frames*100:.1f}%)")

    cap.release()

    if not per_frame_vectors:
        raise RuntimeError(f"No frames processed from {video_path.name}")

    all_features = np.stack(per_frame_vectors, axis=0)  # Shape: (T_total, 154)
    all_labels = np.array(per_frame_labels, dtype=np.int64)

    # 4. Generate Sliding Windows (N × 60 × 154)
    window_tensors: List[np.ndarray] = []
    window_labels: List[int] = []
    window_ranges: List[Tuple[int, int]] = []
    rejected_count = 0

    min_valid_ratio = 0.5  # At least 50% (30 frames) must belong to a dominant labeled action
    min_dominant_frames = int(window_size * min_valid_ratio)

    T = len(all_features)
    for start_f in range(0, max(1, T - window_size + 1), stride):
        end_f = start_f + window_size
        win = all_features[start_f:end_f]
        if len(win) < window_size:
            pad = np.zeros((window_size - len(win), FEATURE_DIM), dtype=np.float32)
            win = np.vstack([pad, win])

        sub_labels = all_labels[start_f:end_f]

        # Filter out windows dominating unannotated transition gaps
        if segments:
            valid_sub = sub_labels[sub_labels >= 0]
            if len(valid_sub) < min_dominant_frames:
                # Majority of window is unlabelled transition gap
                rejected_count += 1
                continue

            # Check if there is a dominant labeled action with >= min_dominant_frames (50%)
            counts = np.bincount(valid_sub)
            maj_label = int(counts.argmax())
            if counts[maj_label] < min_dominant_frames:
                # Ambiguous boundary window between two actions without a clear 50% majority
                rejected_count += 1
                continue
        else:
            valid_sub = sub_labels[sub_labels >= 0]
            maj_label = int(np.bincount(valid_sub).argmax()) if len(valid_sub) > 0 else 0

        window_tensors.append(win)
        window_labels.append(maj_label)
        window_ranges.append((start_f, end_f))

    windows_arr = np.stack(window_tensors, axis=0).astype(np.float32) if window_tensors else np.zeros((0, window_size, FEATURE_DIM), dtype=np.float32)
    labels_arr = np.array(window_labels, dtype=np.int64)
    ranges_arr = np.array(window_ranges, dtype=np.int32) if window_ranges else np.zeros((0, 2), dtype=np.int32)

    elapsed_time = time.time() - start_wall_time

    # Save to compressed NPZ archive
    np.savez_compressed(
        out_npz_path,
        windows=windows_arr,
        labels=labels_arr,
        ranges=ranges_arr,
        actor_id=actor_id,
        session_id=session_id,
        video_id=video_stem,
        feature_dim=FEATURE_DIM,
        window_size=window_size,
        stride=stride,
        rejected_windows=rejected_count,
        total_source_frames=T,
    )

    class_counts = {int(c): int(np.sum(labels_arr == c)) for c in range(10)} if len(labels_arr) > 0 else {}

    summary = {
        "video_id": video_stem,
        "actor_id": actor_id,
        "session_id": session_id,
        "total_source_frames": T,
        "windows_generated": len(windows_arr),
        "windows_rejected": rejected_count,
        "tensor_shape": list(windows_arr.shape),
        "class_distribution": class_counts,
        "output_npz_path": str(out_npz_path),
        "file_size_bytes": out_npz_path.stat().st_size if out_npz_path.exists() else 0,
        "processing_time_sec": round(elapsed_time, 2),
        "mean_fps": round(T / elapsed_time, 2) if elapsed_time > 0 else 0.0,
        "mean_ms_per_frame": round((elapsed_time / T) * 1000.0, 2) if T > 0 else 0.0,
    }

    print("\n" + "=" * 62)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 62)
    print(f"  Video ID           : {video_stem}")
    print(f"  Source Frames      : {T} frames")
    print(f"  Windows Created    : {summary['windows_generated']} tensors")
    print(f"  Windows Rejected   : {summary['windows_rejected']} transition/gap windows")
    print(f"  Tensor Shape       : {summary['tensor_shape']}")
    print(f"  Processing Time    : {summary['processing_time_sec']}s ({summary['mean_fps']} FPS / {summary['mean_ms_per_frame']} ms/frame)")
    print(f"  Output Archive     : {out_npz_path}")
    print("=" * 62)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Video to Temporal Feature Tensor Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", default="", help="Path to action manifest JSON")
    parser.add_argument("--video", default="", help="Path to raw video file")
    parser.add_argument("--all", action="store_true", help="Process all manifests in data/temporal_har/manifests/")
    parser.add_argument("--stride", type=int, default=10, help="Window stride in frames (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all:
        man_dir = _PROJECT_ROOT / "data" / "temporal_har" / "manifests"
        manifests = list(man_dir.glob("*.json"))
        print(f"[FeatureExtractor] Batch processing {len(manifests)} manifest(s)...")
        for m in manifests:
            with open(m, "r", encoding="utf-8") as f:
                d = json.load(f)
            vpath = Path(d.get("video_path", ""))
            if vpath.exists():
                process_video_to_features(video_path=vpath, manifest_path=m, stride=args.stride)
    elif args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            d = json.load(f)
        vpath = Path(d.get("video_path", ""))
        process_video_to_features(video_path=vpath, manifest_path=Path(args.manifest), stride=args.stride)
    elif args.video:
        process_video_to_features(video_path=Path(args.video), stride=args.stride)
    else:
        print("[FeatureExtractor] Please specify --manifest <path>, --video <path>, or --all. See --help.")
