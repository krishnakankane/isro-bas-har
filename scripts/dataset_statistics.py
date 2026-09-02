"""ISRO BAS HAR — Master Dataset Statistics & Inventory Reporter.

Scans the local data/ hierarchy and generates a comprehensive inventory report:
- Raw video inventory, total recording hours, actor count, session distribution.
- Object detection dataset metrics: image counts per split, bounding box counts per class, imbalance metrics.
- Temporal HAR dataset metrics: segment counts, duration per action, 60-frame window counts.
- Sequence validation test trial counts.

Usage
-----
    python -m scripts.dataset_statistics
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OBJECT_CLASSES = {
    0: "micropipette",
    1: "pipette_tip_box",
    2: "sample_vial",
    3: "well_plate",
    4: "centrifuge",
    5: "waste_bin",
    6: "chamber_door_latch",
}

ACTION_CLASSES = {
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


def compute_dataset_statistics(data_root: Optional[Path] = None) -> Dict[str, object]:
    """Compute and print comprehensive dataset statistics."""
    root = data_root or (_PROJECT_ROOT / "data")
    stats: Dict[str, object] = {}

    print("\n" + "=" * 70)
    print("ISRO BAS HAR -- DATASET INVENTORY & COMPOSITION REPORT")
    print("=" * 70)

    print(f"Data Directory: {root.resolve()}")

    # 1. Raw Videos Inventory
    raw_video_dir = root / "raw" / "videos"
    raw_meta_dir = root / "raw" / "metadata"

    video_files = list(raw_video_dir.glob("*.mp4")) + list(raw_video_dir.glob("*.avi"))
    meta_files = list(raw_meta_dir.glob("*.json"))

    actors: Set[str] = set()
    total_raw_duration_sec = 0.0
    total_raw_frames = 0

    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
            actors.add(d.get("actor_id", "UNKNOWN"))
            total_raw_duration_sec += float(d.get("effective_duration_sec", 0.0))
            total_raw_frames += int(d.get("recorded_frames", 0))
        except Exception:
            pass

    stats["raw_videos_count"] = len(video_files)
    stats["metadata_count"] = len(meta_files)
    stats["unique_actors"] = list(actors)
    stats["total_raw_duration_sec"] = total_raw_duration_sec
    stats["total_raw_frames"] = total_raw_frames

    print("\n[1] RAW VIDEO RECORDINGS:")
    print(f"  * Video Files Found      : {len(video_files)}")
    print(f"  * Metadata Manifests     : {len(meta_files)}")
    print(f"  * Unique Actors Logged   : {len(actors)} {sorted(list(actors)) if actors else '(None)'}")
    print(f"  * Total Recorded Footage : {total_raw_duration_sec / 60.0:.2f} minutes ({total_raw_frames} frames)")

    # 2. Object Detection Dataset
    obj_root = root / "object_detection"
    obj_counts = {"train": 0, "val": 0, "test": 0}
    class_bbox_counts: Counter = Counter()

    for split in ("train", "val", "test"):
        img_dir = obj_root / "images" / split
        lbl_dir = obj_root / "labels" / split

        if img_dir.exists():
            imgs = [f for f in img_dir.glob("*.jpg") if "contact_sheet" not in f.name]
            imgs += [f for f in img_dir.glob("*.png") if "contact_sheet" not in f.name]
            obj_counts[split] = len(imgs)

        if lbl_dir.exists():
            for lf in lbl_dir.glob("*.txt"):
                try:
                    for line in lf.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            class_bbox_counts[cid] += 1
                except Exception:
                    pass

    stats["object_images_per_split"] = obj_counts
    stats["object_bbox_counts"] = dict(class_bbox_counts)

    total_obj_images = sum(obj_counts.values())
    total_bboxes = sum(class_bbox_counts.values())

    print("\n[2] OBJECT DETECTION DATASET:")
    print(f"  * Total Images Labeled   : {total_obj_images} (Train: {obj_counts['train']} | Val: {obj_counts['val']} | Test: {obj_counts['test']})")
    print(f"  * Total Bounding Boxes   : {total_bboxes}")
    print("  * Class Breakdown:")
    for cid in range(7):
        cname = OBJECT_CLASSES[cid]
        cnt = class_bbox_counts.get(cid, 0)
        pct = (cnt / total_bboxes * 100.0) if total_bboxes > 0 else 0.0
        print(f"      [{cid}] {cname:<20} : {cnt:>5} boxes ({pct:>5.1f}%)")

    # 3. Temporal Action HAR Dataset
    har_manifest_dir = root / "temporal_har" / "manifests"
    har_features_dir = root / "temporal_har" / "features"

    action_segments_count: Counter = Counter()
    action_duration_sec: Dict[int, float] = defaultdict(float)
    manifest_files = list(har_manifest_dir.glob("*.json"))

    for mf in manifest_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
            for seg in d.get("segments", []):
                aid = seg.get("action_id", 0)
                action_segments_count[aid] += 1
                action_duration_sec[aid] += float(seg.get("duration_sec", 0.0))
        except Exception:
            pass

    npz_files = list(har_features_dir.glob("*.npz"))
    total_windows = 0
    for npz in npz_files:
        try:
            import numpy as np
            with np.load(npz) as data:
                total_windows += len(data.get("windows", []))
        except Exception:
            pass

    stats["temporal_manifests_count"] = len(manifest_files)
    stats["temporal_action_segments"] = dict(action_segments_count)
    stats["temporal_feature_archives"] = len(npz_files)
    stats["temporal_60frame_windows"] = total_windows

    total_segments = sum(action_segments_count.values())
    total_action_sec = sum(action_duration_sec.values())

    print("\n[3] TEMPORAL HAR ACTION DATASET:")
    print(f"  * Action Manifests       : {len(manifest_files)}")
    print(f"  * Total Action Intervals : {total_segments} ({total_action_sec / 60.0:.2f} minutes labeled)")
    print(f"  * Pre-Extracted Windows  : {total_windows} tensors (60 x 154 float32)")
    print("  * Action Breakdown:")
    for aid in range(10):
        aname = ACTION_CLASSES[aid]
        cnt = action_segments_count.get(aid, 0)
        dur = action_duration_sec.get(aid, 0.0)
        pct = (dur / total_action_sec * 100.0) if total_action_sec > 0 else 0.0
        print(f"      [{aid:>2}] {aname:<24} : {cnt:>4} segments ({dur:>6.1f}s, {pct:>5.1f}%)")

    # 4. Sequence Validation Test Trials
    seq_manifest_dir = root / "sequence_validation" / "manifests"
    seq_files = list(seq_manifest_dir.glob("*.json"))
    stats["sequence_trials_count"] = len(seq_files)

    print("\n[4] SEQUENCE VALIDATION BENCHMARKS:")
    print(f"  * Test Trial Manifests   : {len(seq_files)}")
    print("=" * 70 + "\n")


    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISRO BAS HAR — Dataset Statistics Reporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir", default="data", help="Data root folder (default: data)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_dataset_statistics(data_root=_PROJECT_ROOT / args.data_dir)
